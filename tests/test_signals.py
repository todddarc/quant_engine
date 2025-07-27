"""
Tests for signals module.
"""

import pytest
import pandas as pd
import numpy as np
from src.engine.signals import (
    generate_momentum_signal, generate_value_signal, winsorize_signals,
    z_score_signals, sector_neutralize, combine_signals, calculate_signal_ic,
    momentum_12m_1m_gap, value_ep
)


class TestValueEp:
    """Test value_ep function."""
    
    def test_value_enforces_availability_lag(self):
        """Test that availability lag is properly enforced."""
        # Create test data
        asof_dt = pd.Timestamp('2023-12-31')
        
        # Fundamentals with different availability dates
        fundamentals_data = [
            # This should be ignored: available_asof > t
            {'report_dt': '2023-10-01', 'available_asof': '2024-01-15', 'ticker': 'AAPL', 'eps_ttm': 10.0, 'book_value_ps': 50.0},
            # This should be used: available_asof <= t and lag >= 60 days
            {'report_dt': '2023-10-01', 'available_asof': '2023-10-31', 'ticker': 'AAPL', 'eps_ttm': 5.0, 'book_value_ps': 25.0},
            # This should be ignored: lag < 60 days
            {'report_dt': '2023-12-01', 'available_asof': '2023-12-15', 'ticker': 'AAPL', 'eps_ttm': 15.0, 'book_value_ps': 75.0},
        ]
        
        fundamentals_df = pd.DataFrame(fundamentals_data)
        
        # Current prices
        prices_data = [
            {'asof_dt': '2023-12-31', 'ticker': 'AAPL', 'close': 100.0},
        ]
        prices_df = pd.DataFrame(prices_data)
        
        result = value_ep(fundamentals_df, prices_df, asof_dt, min_lag_days=60)
        
        # Should use the second fundamental (eps_ttm=5.0, price=100.0)
        expected_ep = 5.0 / 100.0  # 0.05
        assert result['AAPL'] == expected_ep
        assert result.name == "val_ep"
    
    def test_value_computation_basic(self):
        """Test basic E/P computation."""
        asof_dt = pd.Timestamp('2023-12-31')
        
        # Simple case: eps_ttm=5, price=100
        fundamentals_data = [
            {'report_dt': '2023-10-01', 'available_asof': '2023-10-31', 'ticker': 'AAPL', 'eps_ttm': 5.0, 'book_value_ps': 25.0},
        ]
        fundamentals_df = pd.DataFrame(fundamentals_data)
        
        prices_data = [
            {'asof_dt': '2023-12-31', 'ticker': 'AAPL', 'close': 100.0},
        ]
        prices_df = pd.DataFrame(prices_data)
        
        result = value_ep(fundamentals_df, prices_df, asof_dt, min_lag_days=60)
        
        # E/P = 5/100 = 0.05
        assert result['AAPL'] == 0.05
        assert result.name == "val_ep"
    
    def test_value_drops_missing(self):
        """Test that tickers missing fundamentals or prices are excluded."""
        asof_dt = pd.Timestamp('2023-12-31')
        
        # Fundamentals for AAPL only
        fundamentals_data = [
            {'report_dt': '2023-10-01', 'available_asof': '2023-10-31', 'ticker': 'AAPL', 'eps_ttm': 5.0, 'book_value_ps': 25.0},
        ]
        fundamentals_df = pd.DataFrame(fundamentals_data)
        
        # Prices for both AAPL and MSFT, but MSFT has no fundamentals
        prices_data = [
            {'asof_dt': '2023-12-31', 'ticker': 'AAPL', 'close': 100.0},
            {'asof_dt': '2023-12-31', 'ticker': 'MSFT', 'close': 200.0},
        ]
        prices_df = pd.DataFrame(prices_data)
        
        result = value_ep(fundamentals_df, prices_df, asof_dt, min_lag_days=60)
        
        # Should only have AAPL (MSFT has no fundamentals)
        assert len(result) == 1
        assert 'AAPL' in result.index
        assert 'MSFT' not in result.index
        assert result.name == "val_ep"
    
    def test_value_negative_eps(self):
        """Test that negative eps_ttm is allowed."""
        asof_dt = pd.Timestamp('2023-12-31')
        
        # Negative eps_ttm
        fundamentals_data = [
            {'report_dt': '2023-10-01', 'available_asof': '2023-10-31', 'ticker': 'AAPL', 'eps_ttm': -2.0, 'book_value_ps': 25.0},
        ]
        fundamentals_df = pd.DataFrame(fundamentals_data)
        
        prices_data = [
            {'asof_dt': '2023-12-31', 'ticker': 'AAPL', 'close': 100.0},
        ]
        prices_df = pd.DataFrame(prices_data)
        
        result = value_ep(fundamentals_df, prices_df, asof_dt, min_lag_days=60)
        
        # E/P = -2/100 = -0.02
        assert result['AAPL'] == -0.02
        assert result.name == "val_ep"
    
    def test_value_multiple_records_picks_latest(self):
        """Test that when multiple records meet PIT rule, the latest available_asof is picked."""
        asof_dt = pd.Timestamp('2023-12-31')
        
        # Multiple fundamentals for AAPL with different available_asof dates
        fundamentals_data = [
            {'report_dt': '2023-09-01', 'available_asof': '2023-09-30', 'ticker': 'AAPL', 'eps_ttm': 3.0, 'book_value_ps': 15.0},
            {'report_dt': '2023-10-01', 'available_asof': '2023-10-31', 'ticker': 'AAPL', 'eps_ttm': 5.0, 'book_value_ps': 25.0},
            {'report_dt': '2023-11-01', 'available_asof': '2023-11-30', 'ticker': 'AAPL', 'eps_ttm': 4.0, 'book_value_ps': 20.0},
        ]
        fundamentals_df = pd.DataFrame(fundamentals_data)
        
        prices_data = [
            {'asof_dt': '2023-12-31', 'ticker': 'AAPL', 'close': 100.0},
        ]
        prices_df = pd.DataFrame(prices_data)
        
        result = value_ep(fundamentals_df, prices_df, asof_dt, min_lag_days=30)
        
        # Should pick the latest available_asof (2023-11-30) with eps_ttm=4.0
        expected_ep = 4.0 / 100.0  # 0.04
        assert result['AAPL'] == expected_ep
        assert result.name == "val_ep"


class TestMomentum12m1mGap:
    """Test momentum_12m_1m_gap function."""
    
    def test_momentum_gap_enforced(self):
        """Test that gap period is properly enforced."""
        # Create test data with 300 business days of history
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        base_prices = np.linspace(100, 200, 300)  # Monotonic uptrend
        
        # Create DataFrame with base prices
        data = []
        for date, price in zip(dates, base_prices):
            data.append({
                'asof_dt': date,
                'ticker': 'AAPL',
                'close': price
            })
        
        prices_df = pd.DataFrame(data)
        
        # Calculate momentum asof the last business day BEFORE manipulation
        asof_dt = dates[-1]
        result1 = momentum_12m_1m_gap(prices_df, asof_dt, lookback=252, gap=21)
        
        # Calculate the actual t_lookback_gap date for manipulation
        t_lookback_gap_date = asof_dt - pd.offsets.BusinessDay(252 + 21)
        
        # Now manipulate the price at t-lookback-gap (should affect result)
        prices_df.loc[prices_df['asof_dt'] == t_lookback_gap_date, 'close'] = 50  # Much lower price
        
        result2 = momentum_12m_1m_gap(prices_df, asof_dt, lookback=252, gap=21)
        
        # Results should be different because we changed the start price
        assert result1['AAPL'] != result2['AAPL']
        
        # Now test that manipulation within gap window doesn't affect result
        # Calculate a date clearly within the gap window (not at the boundary)
        t_gap_within_date = asof_dt - pd.offsets.BusinessDay(10)  # 10 business days back (within 21-day gap)
        
        # Manipulate a price within the last 21 business days (gap window)
        prices_df.loc[prices_df['asof_dt'] == t_gap_within_date, 'close'] = 999  # Extreme value
        
        result3 = momentum_12m_1m_gap(prices_df, asof_dt, lookback=252, gap=21)
        
        # Result should be the same as result2 because gap window is enforced
        assert result2['AAPL'] == result3['AAPL']
    
    def test_momentum_positive_for_uptrend(self):
        """Test that monotonic uptrend produces positive momentum."""
        # Create synthetic monotonic uptrend
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        prices = np.linspace(100, 200, 300)  # Linear uptrend
        
        data = [{'asof_dt': date, 'ticker': 'AAPL', 'close': price} 
                for date, price in zip(dates, prices)]
        prices_df = pd.DataFrame(data)
        
        asof_dt = dates[-1]
        result = momentum_12m_1m_gap(prices_df, asof_dt, lookback=252, gap=21)
        
        # Momentum should be positive for uptrend
        assert result['AAPL'] > 0
        assert result.name == "mom_12_1"
    
    def test_momentum_zero_for_flat(self):
        """Test that flat prices produce near-zero momentum."""
        # Create synthetic flat prices
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        prices = [100.0] * 300  # Flat prices
        
        data = [{'asof_dt': date, 'ticker': 'AAPL', 'close': price} 
                for date, price in zip(dates, prices)]
        prices_df = pd.DataFrame(data)
        
        asof_dt = dates[-1]
        result = momentum_12m_1m_gap(prices_df, asof_dt, lookback=252, gap=21)
        
        # Momentum should be approximately zero for flat prices
        assert abs(result['AAPL']) < 1e-10
        assert result.name == "mom_12_1"
    
    def test_momentum_requires_history(self):
        """Test that ticker with insufficient history is not returned."""
        # Create data with only 100 business days (less than lookback=252)
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        prices = np.linspace(100, 150, 100)
        
        data = [{'asof_dt': date, 'ticker': 'AAPL', 'close': price} 
                for date, price in zip(dates, prices)]
        prices_df = pd.DataFrame(data)
        
        asof_dt = dates[-1]
        result = momentum_12m_1m_gap(prices_df, asof_dt, lookback=252, gap=21)
        
        # Should return empty series for insufficient history
        assert len(result) == 0
        assert result.name == "mom_12_1"
    
    def test_momentum_multiple_tickers(self):
        """Test momentum calculation with multiple tickers."""
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        
        data = []
        # AAPL: uptrend
        aapl_prices = np.linspace(100, 200, 300)
        for date, price in zip(dates, aapl_prices):
            data.append({'asof_dt': date, 'ticker': 'AAPL', 'close': price})
        
        # MSFT: downtrend
        msft_prices = np.linspace(200, 100, 300)
        for date, price in zip(dates, msft_prices):
            data.append({'asof_dt': date, 'ticker': 'MSFT', 'close': price})
        
        prices_df = pd.DataFrame(data)
        
        asof_dt = dates[-1]
        result = momentum_12m_1m_gap(prices_df, asof_dt, lookback=252, gap=21)
        
        # Should have both tickers
        assert len(result) == 2
        assert 'AAPL' in result.index
        assert 'MSFT' in result.index
        
        # AAPL should have positive momentum (uptrend)
        assert result['AAPL'] > 0
        
        # MSFT should have negative momentum (downtrend)
        assert result['MSFT'] < 0
        
        # Should be sorted by index
        assert list(result.index) == sorted(result.index)


class TestMomentumSignal:
    """Test momentum signal generation."""
    
    def test_generate_momentum_signal_basic(self):
        """Test basic momentum signal generation."""
        pytest.skip("not implemented")
    
    def test_generate_momentum_signal_with_gap(self):
        """Test momentum signal with gap period."""
        pytest.skip("not implemented")
    
    def test_generate_momentum_signal_insufficient_data(self):
        """Test momentum signal with insufficient historical data."""
        pytest.skip("not implemented")


class TestValueSignal:
    """Test value signal generation."""
    
    def test_generate_value_signal_basic(self):
        """Test basic value signal generation."""
        pytest.skip("not implemented")
    
    def test_generate_value_signal_missing_data(self):
        """Test value signal with missing fundamental data."""
        pytest.skip("not implemented")


class TestSignalProcessing:
    """Test signal processing functions."""
    
    def test_winsorize_signals(self):
        """Test signal winsorization."""
        pytest.skip("not implemented")
    
    def test_z_score_signals(self):
        """Test signal z-scoring."""
        pytest.skip("not implemented")
    
    def test_sector_neutralize(self):
        """Test sector neutralization."""
        pytest.skip("not implemented")


class TestSignalCombination:
    """Test signal combination functionality."""
    
    def test_combine_signals_equal_weights(self):
        """Test signal combination with equal weights."""
        pytest.skip("not implemented")
    
    def test_combine_signals_custom_weights(self):
        """Test signal combination with custom weights."""
        pytest.skip("not implemented")
    
    def test_combine_signals_missing_data(self):
        """Test signal combination with missing data."""
        pytest.skip("not implemented")


class TestSignalValidation:
    """Test signal validation functionality."""
    
    def test_calculate_signal_ic_basic(self):
        """Test basic IC calculation."""
        pytest.skip("not implemented")
    
    def test_calculate_signal_ic_rolling_window(self):
        """Test rolling window IC calculation."""
        pytest.skip("not implemented") 