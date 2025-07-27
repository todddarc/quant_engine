"""
Tests for prep module.
"""

import pytest
import pandas as pd
import numpy as np
from src.engine.prep import (
    prepare_data, calculate_returns, align_data_to_date,
    handle_missing_data, create_universe_mask, calculate_forward_returns,
    winsorize, zscore, sector_neutralize
)


class TestWinsorize:
    """Test winsorization functionality."""
    
    def test_winsorize_clamps_extremes(self):
        """Test that winsorize clamps values to quantile boundaries."""
        # Create series with clear outliers
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], 
                        index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        
        result = winsorize(data, p_low=0.1, p_high=0.9)
        
        # Check that extremes are clamped
        assert result.max() <= data.quantile(0.9)
        assert result.min() >= data.quantile(0.1)
        # Check that middle values are unchanged
        assert result['C'] == 3
        assert result['D'] == 4
        assert result['E'] == 5
        assert result['F'] == 6
        assert result['G'] == 7
        assert result['H'] == 8
    
    def test_winsorize_preserves_nans(self):
        """Test that winsorize preserves NaN positions."""
        data = pd.Series([1, np.nan, 3, 4, 5, np.nan, 7, 8, 9, 10],
                        index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        
        result = winsorize(data, p_low=0.1, p_high=0.9)
        
        # Check NaN positions are preserved
        assert pd.isna(result['B'])
        assert pd.isna(result['F'])
        # Check non-NaN values are processed
        assert not np.isnan(result['A'])
        assert not np.isnan(result['C'])
    
    def test_winsorize_insufficient_data(self):
        """Test winsorize returns original when insufficient data."""
        data = pd.Series([1, 2], index=['A', 'B'])
        
        result = winsorize(data, p_low=0.1, p_high=0.9)
        
        # Should return original series
        pd.testing.assert_series_equal(result, data.astype(float))
    
    def test_winsorize_invalid_quantiles(self):
        """Test winsorize returns original when p_low >= p_high."""
        data = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        
        result = winsorize(data, p_low=0.9, p_high=0.1)
        
        # Should return original series
        pd.testing.assert_series_equal(result, data.astype(float))


class TestZscore:
    """Test z-score standardization functionality."""
    
    def test_zscore_center_and_scale(self):
        """Test that zscore centers and scales non-NaN values."""
        data = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        
        result = zscore(data)
        
        # Check mean ≈ 0 and std ≈ 1 on non-NaN values
        non_nan_result = result.dropna()
        assert abs(non_nan_result.mean()) < 1e-6
        assert abs(non_nan_result.std(ddof=0) - 1.0) < 1e-6
    
    def test_zscore_degenerate(self):
        """Test zscore handles constant series correctly."""
        data = pd.Series([5, 5, 5, 5, 5], index=['A', 'B', 'C', 'D', 'E'])
        
        result = zscore(data)
        
        # Should return all zeros (not NaNs)
        assert all(result == 0)
        assert not result.isna().any()
    
    def test_zscore_preserves_nans(self):
        """Test that zscore preserves NaN positions."""
        data = pd.Series([1, np.nan, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        
        result = zscore(data)
        
        # Check NaN position is preserved
        assert pd.isna(result['B'])
        # Check non-NaN values are processed
        assert not np.isnan(result['A'])
        assert not np.isnan(result['C'])
    
    def test_zscore_empty_series(self):
        """Test zscore handles empty series."""
        data = pd.Series(dtype=float)
        
        result = zscore(data)
        
        # Should return empty series
        assert len(result) == 0
        assert result.dtype == float


class TestSectorNeutralize:
    """Test sector neutralization functionality."""
    
    def test_sector_neutralize_removes_between_sector_means(self):
        """Test that sector neutralization removes between-sector mean differences."""
        # Create data with different sector means
        data = pd.Series({
            'A': 10, 'B': 12, 'C': 14,  # Tech sector (mean = 12)
            'D': 20, 'E': 22, 'F': 24,  # Finance sector (mean = 22)
            'G': 30, 'H': 32, 'I': 34   # Energy sector (mean = 32)
        })
        
        sectors = pd.Series({
            'A': 'Tech', 'B': 'Tech', 'C': 'Tech',
            'D': 'Finance', 'E': 'Finance', 'F': 'Finance',
            'G': 'Energy', 'H': 'Energy', 'I': 'Energy'
        })
        
        result = sector_neutralize(data, sectors)
        
        # Check that sector means are approximately 0
        tech_mean = result[['A', 'B', 'C']].mean()
        finance_mean = result[['D', 'E', 'F']].mean()
        energy_mean = result[['G', 'H', 'I']].mean()
        
        assert abs(tech_mean) < 1e-6
        assert abs(finance_mean) < 1e-6
        assert abs(energy_mean) < 1e-6
    
    def test_sector_neutralize_small_sector_stability(self):
        """Test sector neutralization handles small sectors gracefully."""
        data = pd.Series({
            'A': 10, 'B': 12, 'C': 14,  # Tech sector
            'D': 20  # Single name in Finance
        })
        
        sectors = pd.Series({
            'A': 'Tech', 'B': 'Tech', 'C': 'Tech',
            'D': 'Finance'
        })
        
        result = sector_neutralize(data, sectors)
        
        # Should not crash and should center single-name sector to 0
        assert result['D'] == 0
        # Tech sector should be z-scored
        tech_data = result[['A', 'B', 'C']]
        assert abs(tech_data.mean()) < 1e-6
        assert abs(tech_data.std(ddof=0) - 1.0) < 1e-6
    
    def test_sector_neutralize_aligns_index(self):
        """Test sector neutralization aligns index correctly with dict input."""
        data = pd.Series({
            'A': 10, 'B': 12, 'C': 14, 'D': 16, 'E': 18
        })
        
        sectors_dict = {
            'A': 'Tech', 'B': 'Tech', 'C': 'Finance', 'D': 'Finance'
            # 'E' missing from sectors
        }
        
        result = sector_neutralize(data, sectors_dict)
        
        # Should only include tickers with sector labels
        expected_index = ['A', 'B', 'C', 'D']
        assert list(result.index) == expected_index
        assert 'E' not in result.index
    
    def test_sector_neutralize_constant_sector(self):
        """Test sector neutralization handles constant sectors correctly."""
        data = pd.Series({
            'A': 10, 'B': 10, 'C': 10,  # Constant Tech sector
            'D': 20, 'E': 22, 'F': 24   # Variable Finance sector
        })
        
        sectors = pd.Series({
            'A': 'Tech', 'B': 'Tech', 'C': 'Tech',
            'D': 'Finance', 'E': 'Finance', 'F': 'Finance'
        })
        
        result = sector_neutralize(data, sectors)
        
        # Constant sector should be centered to 0
        tech_data = result[['A', 'B', 'C']]
        assert all(tech_data == 0)
        # Variable sector should be z-scored
        finance_data = result[['D', 'E', 'F']]
        assert abs(finance_data.mean()) < 1e-6
        assert abs(finance_data.std(ddof=0) - 1.0) < 1e-6


class TestPrepareData:
    """Test data preparation functionality."""
    
    def test_prepare_data_basic(self):
        """Test basic data preparation."""
        pytest.skip("not implemented")
    
    def test_prepare_data_point_in_time(self):
        """Test point-in-time data preparation."""
        pytest.skip("not implemented")


class TestCalculateReturns:
    """Test return calculation functionality."""
    
    def test_calculate_returns_daily(self):
        """Test daily return calculation."""
        pytest.skip("not implemented")
    
    def test_calculate_returns_multiple_periods(self):
        """Test return calculation for multiple periods."""
        pytest.skip("not implemented")


class TestAlignDataToDate:
    """Test data alignment functionality."""
    
    def test_align_data_to_date_basic(self):
        """Test basic data alignment."""
        pytest.skip("not implemented")
    
    def test_align_data_to_date_missing_data(self):
        """Test data alignment with missing data."""
        pytest.skip("not implemented")


class TestHandleMissingData:
    """Test missing data handling."""
    
    def test_handle_missing_data_drop(self):
        """Test missing data handling with drop method."""
        pytest.skip("not implemented")
    
    def test_handle_missing_data_forward_fill(self):
        """Test missing data handling with forward fill."""
        pytest.skip("not implemented")


class TestCreateUniverseMask:
    """Test universe mask creation."""
    
    def test_create_universe_mask_basic(self):
        """Test basic universe mask creation."""
        pytest.skip("not implemented")
    
    def test_create_universe_mask_with_filters(self):
        """Test universe mask with price and volume filters."""
        pytest.skip("not implemented")


class TestCalculateForwardReturns:
    """Test forward return calculation."""
    
    def test_calculate_forward_returns_basic(self):
        """Test basic forward return calculation."""
        pytest.skip("not implemented")
    
    def test_calculate_forward_returns_multiple_periods(self):
        """Test forward return calculation for multiple periods."""
        pytest.skip("not implemented") 