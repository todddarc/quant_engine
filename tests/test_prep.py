"""
Tests for prep module.
"""

import pytest
import pandas as pd
import numpy as np
from quant_engine.prep import winsorize, zscore, sector_neutralize


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
        """Test winsorize handles small datasets."""
        data = pd.Series([1, 2], index=['A', 'B'])
        
        result = winsorize(data, p_low=0.1, p_high=0.9)
        
        # Should still apply winsorization even with small data
        assert result.max() <= data.quantile(0.9)
        assert result.min() >= data.quantile(0.1)
    
    def test_winsorize_invalid_quantiles(self):
        """Test winsorize handles invalid quantile order."""
        data = pd.Series([1, 2, 3, 4, 5], index=['A', 'B', 'C', 'D', 'E'])
        
        result = winsorize(data, p_low=0.9, p_high=0.1)
        
        # Should still apply winsorization (pandas handles the order)
        assert result.max() <= data.quantile(0.9)
        assert result.min() >= data.quantile(0.1)


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
        # Tech sector should be z-scored (using ddof=1 for sample std)
        tech_data = result[['A', 'B', 'C']]
        assert abs(tech_data.mean()) < 1e-6
        assert abs(tech_data.std(ddof=1) - 1.0) < 1e-6
    
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

        # Should include all tickers, with NaN for missing sectors
        expected_index = ['A', 'B', 'C', 'D', 'E']
        assert list(result.index) == expected_index
        # 'E' should be NaN since it has no sector
        assert pd.isna(result['E'])
    
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
        # Variable sector should be z-scored (using ddof=1 for sample std)
        finance_data = result[['D', 'E', 'F']]
        assert abs(finance_data.mean()) < 1e-6
        assert abs(finance_data.std(ddof=1) - 1.0) < 1e-6


 
