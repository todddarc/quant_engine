"""
Tests for trade_filters module.
"""

import pytest
import pandas as pd
import numpy as np
from quant_engine.trade_filters import small_trade_mask, apply_no_trade_band


class TestSmallTradeMask:
    """Test small trade mask functionality."""
    
    def test_small_trade_mask_basic_weight_filtering(self):
        """Test basic weight-based filtering."""
        delta_w = pd.Series({'A': 0.001, 'B': 0.01, 'C': 0.0001})
        mask = small_trade_mask(delta_w, min_weight=0.005)
        
        expected = pd.Series({'A': True, 'B': False, 'C': True})
        pd.testing.assert_series_equal(mask, expected)
    
    def test_small_trade_mask_with_notional_filtering(self):
        """Test notional-based filtering with prices and AUM."""
        delta_w = pd.Series({'A': 0.001, 'B': 0.01, 'C': 0.0001})
        prices = pd.Series({'A': 100.0, 'B': 50.0, 'C': 200.0})
        aum = 1000000.0  # $1M
        
        # With min_notional = $5000, trades below this should be frozen
        # Notional calculation: dw.abs() * aum * pr / pr.clip(lower=EPS) ≈ dw.abs() * aum
        # A: 0.001 * 1M = $1,000 (below threshold)
        # B: 0.01 * 1M = $10,000 (above threshold) 
        # C: 0.0001 * 1M = $100 (below threshold, also below min_weight)
        mask = small_trade_mask(delta_w, prices=prices, aum=aum, min_notional=5000.0)
        
        expected = pd.Series({'A': True, 'B': False, 'C': True})
        pd.testing.assert_series_equal(mask, expected)
    
    def test_small_trade_mask_missing_prices(self):
        """Test behavior when prices are missing."""
        delta_w = pd.Series({'A': 0.001, 'B': 0.01, 'C': 0.0001})
        prices = pd.Series({'A': 100.0, 'B': np.nan, 'C': 200.0})
        aum = 1000000.0
        
        mask = small_trade_mask(delta_w, prices=prices, aum=aum, min_notional=5000.0)
        
        # A: below min_weight (0.001 < 0.0005 default), B: frozen due to missing price, C: below min_weight
        expected = pd.Series({'A': True, 'B': True, 'C': True})
        pd.testing.assert_series_equal(mask, expected)
    
    def test_small_trade_mask_infinite_values(self):
        """Test handling of infinite values."""
        delta_w = pd.Series({'A': 0.001, 'B': np.inf, 'C': -np.inf, 'D': 0.01})
        mask = small_trade_mask(delta_w, min_weight=0.005)
        
        expected = pd.Series({'A': True, 'B': True, 'C': True, 'D': False})
        pd.testing.assert_series_equal(mask, expected)
    
    def test_small_trade_mask_nan_values(self):
        """Test handling of NaN values."""
        delta_w = pd.Series({'A': 0.001, 'B': np.nan, 'C': 0.01})
        mask = small_trade_mask(delta_w, min_weight=0.005)
        
        expected = pd.Series({'A': True, 'B': True, 'C': False})
        pd.testing.assert_series_equal(mask, expected)
    
    def test_small_trade_mask_zero_prices(self):
        """Test handling of zero prices."""
        delta_w = pd.Series({'A': 0.001, 'B': 0.01})
        prices = pd.Series({'A': 0.0, 'B': 100.0})
        aum = 1000000.0
        
        mask = small_trade_mask(delta_w, prices=prices, aum=aum, min_notional=5000.0)
        
        # A should be frozen due to zero price
        expected = pd.Series({'A': True, 'B': False})
        pd.testing.assert_series_equal(mask, expected)
    
    def test_small_trade_mask_no_notional_params(self):
        """Test behavior when notional parameters are not provided."""
        delta_w = pd.Series({'A': 0.001, 'B': 0.01, 'C': np.nan})
        mask = small_trade_mask(delta_w, min_weight=0.005)
        
        expected = pd.Series({'A': True, 'B': False, 'C': True})
        pd.testing.assert_series_equal(mask, expected)
    
    def test_small_trade_mask_custom_min_weight(self):
        """Test with custom minimum weight threshold."""
        delta_w = pd.Series({'A': 0.001, 'B': 0.01, 'C': 0.02})
        mask = small_trade_mask(delta_w, min_weight=0.015)
        
        expected = pd.Series({'A': True, 'B': True, 'C': False})
        pd.testing.assert_series_equal(mask, expected)


class TestApplyNoTradeBand:
    """Test no-trade band application."""
    
    def test_apply_no_trade_band_basic(self):
        """Test basic no-trade band application."""
        prev_w = pd.Series({'A': 0.1, 'B': 0.2, 'C': 0.3})
        new_w = pd.Series({'A': 0.11, 'B': 0.19, 'C': 0.31})
        
        frozen_w, mask, stats = apply_no_trade_band(prev_w, new_w, min_weight=0.02)
        
        # All trades are small (< 0.02), so should be frozen
        expected_frozen = pd.Series({'A': 0.1, 'B': 0.2, 'C': 0.3})
        expected_mask = pd.Series({'A': True, 'B': True, 'C': True})
        
        pd.testing.assert_series_equal(frozen_w, expected_frozen)
        pd.testing.assert_series_equal(mask, expected_mask)
        assert stats['n_frozen'] == 3
        assert stats['turnover_before'] == pytest.approx(0.015, abs=1e-6)
        assert stats['turnover_after'] == pytest.approx(0.0, abs=1e-6)
    
    def test_apply_no_trade_band_mixed_trades(self):
        """Test with mix of large and small trades."""
        prev_w = pd.Series({'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4})
        new_w = pd.Series({'A': 0.11, 'B': 0.25, 'C': 0.31, 'D': 0.33})
        
        frozen_w, mask, stats = apply_no_trade_band(prev_w, new_w, min_weight=0.02)
        
        # A: 0.01 (small), B: 0.05 (large), C: 0.01 (small), D: -0.07 (large)
        expected_frozen = pd.Series({'A': 0.1, 'B': 0.25, 'C': 0.3, 'D': 0.33})
        expected_mask = pd.Series({'A': True, 'B': False, 'C': True, 'D': False})
        
        pd.testing.assert_series_equal(frozen_w, expected_frozen)
        pd.testing.assert_series_equal(mask, expected_mask)
        assert stats['n_frozen'] == 2
    
    def test_apply_no_trade_band_with_notional_filtering(self):
        """Test with notional-based filtering."""
        prev_w = pd.Series({'A': 0.1, 'B': 0.2})
        new_w = pd.Series({'A': 0.11, 'B': 0.21})
        prices = pd.Series({'A': 100.0, 'B': 50.0})
        aum = 1000000.0
        
        frozen_w, mask, stats = apply_no_trade_band(
            prev_w, new_w, prices=prices, aum=aum, 
            min_weight=0.005, min_notional=5000.0
        )
        
        # A: delta=0.01, notional=10,000 (above threshold), weight=0.01 (above threshold)
        # B: delta=0.01, notional=10,000 (above threshold), weight=0.01 (above threshold)
        # Both should pass both thresholds, so not frozen
        expected_frozen = pd.Series({'A': 0.11, 'B': 0.21})
        expected_mask = pd.Series({'A': False, 'B': False})
        
        pd.testing.assert_series_equal(frozen_w, expected_frozen)
        pd.testing.assert_series_equal(mask, expected_mask)
    
    def test_apply_no_trade_band_different_indices(self):
        """Test with different indices in prev_w and new_w."""
        prev_w = pd.Series({'A': 0.1, 'B': 0.2, 'C': 0.3})
        new_w = pd.Series({'B': 0.25, 'C': 0.31, 'D': 0.4})
        
        frozen_w, mask, stats = apply_no_trade_band(prev_w, new_w, min_weight=0.02)
        
        # Should handle union of indices
        # A: delta = 0.0 - 0.1 = -0.1 (large trade, not frozen)
        # B: delta = 0.25 - 0.2 = 0.05 (large trade, not frozen)  
        # C: delta = 0.31 - 0.3 = 0.01 (small trade, frozen)
        # D: delta = 0.4 - 0.0 = 0.4 (large trade, not frozen)
        expected_frozen = pd.Series({'A': 0.0, 'B': 0.25, 'C': 0.3, 'D': 0.4})
        expected_mask = pd.Series({'A': False, 'B': False, 'C': True, 'D': False})
        
        pd.testing.assert_series_equal(frozen_w, expected_frozen)
        pd.testing.assert_series_equal(mask, expected_mask)
    
    def test_apply_no_trade_band_missing_values(self):
        """Test with missing values in weights."""
        prev_w = pd.Series({'A': 0.1, 'B': np.nan, 'C': 0.3})
        new_w = pd.Series({'A': 0.11, 'B': 0.2, 'C': np.nan})
        
        frozen_w, mask, stats = apply_no_trade_band(prev_w, new_w, min_weight=0.02)
        
        # Missing values should be filled with 0.0
        # A: delta = 0.11 - 0.1 = 0.01 (small trade, frozen)
        # B: delta = 0.2 - 0.0 = 0.2 (large trade, not frozen)
        # C: delta = 0.0 - 0.3 = -0.3 (large trade, not frozen)
        expected_frozen = pd.Series({'A': 0.1, 'B': 0.2, 'C': 0.0})
        expected_mask = pd.Series({'A': True, 'B': False, 'C': False})
        
        pd.testing.assert_series_equal(frozen_w, expected_frozen)
        pd.testing.assert_series_equal(mask, expected_mask)
    
    def test_apply_no_trade_band_turnover_calculation(self):
        """Test turnover calculation accuracy."""
        prev_w = pd.Series({'A': 0.1, 'B': 0.2})
        new_w = pd.Series({'A': 0.15, 'B': 0.25})
        
        frozen_w, mask, stats = apply_no_trade_band(prev_w, new_w, min_weight=0.02)
        
        # Turnover before: 0.5 * (0.05 + 0.05) = 0.05
        # Turnover after: 0.5 * (0.05 + 0.05) = 0.05 (no trades frozen)
        assert stats['turnover_before'] == pytest.approx(0.05, abs=1e-6)
        assert stats['turnover_after'] == pytest.approx(0.05, abs=1e-6)
    
    def test_apply_no_trade_band_empty_series(self):
        """Test with empty series."""
        prev_w = pd.Series(dtype=float)
        new_w = pd.Series(dtype=float)
        
        frozen_w, mask, stats = apply_no_trade_band(prev_w, new_w, min_weight=0.02)
        
        assert len(frozen_w) == 0
        assert len(mask) == 0
        assert stats['n_frozen'] == 0
        assert stats['turnover_before'] == 0.0
        assert stats['turnover_after'] == 0.0


class TestTradeFiltersIntegration:
    """Integration tests for trade filters."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from delta calculation to frozen weights."""
        # Simulate portfolio weights
        prev_w = pd.Series({
            'AAPL': 0.05, 'MSFT': 0.04, 'GOOGL': 0.03, 'AMZN': 0.02, 'TSLA': 0.01
        })
        new_w = pd.Series({
            'AAPL': 0.052, 'MSFT': 0.038, 'GOOGL': 0.031, 'AMZN': 0.019, 'TSLA': 0.011
        })
        prices = pd.Series({
            'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0, 'AMZN': 100.0, 'TSLA': 200.0
        })
        aum = 10000000.0  # $10M
        
        frozen_w, mask, stats = apply_no_trade_band(
            prev_w, new_w, prices=prices, aum=aum,
            min_weight=0.005, min_notional=10000.0
        )
        
        # Verify results are reasonable
        assert len(frozen_w) == 5
        assert len(mask) == 5
        assert stats['n_frozen'] >= 0
        assert stats['turnover_after'] <= stats['turnover_before']
        
        # Check that frozen weights are either new_w or prev_w
        for ticker in frozen_w.index:
            if mask[ticker]:
                assert frozen_w[ticker] == prev_w[ticker]
            else:
                assert frozen_w[ticker] == new_w[ticker]
    
    def test_edge_case_extreme_values(self):
        """Test with extreme values."""
        prev_w = pd.Series({'A': 0.0, 'B': 1.0, 'C': 0.5})
        new_w = pd.Series({'A': 0.001, 'B': 0.999, 'C': 0.501})
        
        frozen_w, mask, stats = apply_no_trade_band(prev_w, new_w, min_weight=0.002)
        
        # Should handle extreme values gracefully
        assert all(np.isfinite(frozen_w))
        assert all(np.isfinite(mask))
        assert np.isfinite(stats['turnover_before'])
        assert np.isfinite(stats['turnover_after']) 
