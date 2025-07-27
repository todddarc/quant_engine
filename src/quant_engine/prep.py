"""
Data preparation utilities for signal processing and portfolio optimization.

Provides functions for winsorization, z-scoring, and sector neutralization
of financial signals with proper handling of missing data and edge cases.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional


def winsorize(series: pd.Series, p_low: float, p_high: float) -> pd.Series:
    """
    Winsorize a series by capping values at specified percentiles.
    
    Args:
        series: Input series to winsorize
        p_low: Lower percentile (e.g., 0.01 for 1st percentile)
        p_high: Upper percentile (e.g., 0.99 for 99th percentile)
        
    Returns:
        Winsorized series with same index as input
        
    Notes:
        - Handles NaN values by excluding them from percentile calculation
        - Returns NaN for any NaN inputs
        - Uses pandas quantile method for percentile calculation
    """
    if series.empty:
        return series
    
    # Calculate percentiles excluding NaN values
    low_val = series.quantile(p_low)
    high_val = series.quantile(p_high)
    
    # Apply winsorization
    result = series.copy()
    result = result.clip(lower=low_val, upper=high_val)
    
    return result


def zscore(series: pd.Series, ddof: int = 0) -> pd.Series:
    """
    Calculate z-score (standardized) values for a series.
    
    Args:
        series: Input series to standardize
        ddof: Delta degrees of freedom for std calculation (0 for population, 1 for sample)
        
    Returns:
        Z-scored series with same index as input
        
    Notes:
        - Z-score = (x - mean) / std
        - Returns NaN for any NaN inputs
        - Returns NaN if std is zero (all values identical)
        - Uses pandas mean() and std() methods
    """
    if series.empty:
        return series
    
    mean_val = series.mean()
    std_val = series.std(ddof=ddof)
    
    if std_val == 0:
        # All values are identical, return zeros
        return pd.Series(0.0, index=series.index)
    
    result = (series - mean_val) / std_val
    return result


def sector_neutralize(
    series: pd.Series,
    sectors_map: Union[pd.Series, Dict],
    method: str = "within_sector_z"
) -> pd.Series:
    """
    Neutralize a series by sector using specified method.
    
    Args:
        series: Input series to neutralize (indexed by ticker)
        sectors_map: Sector mapping (Series or Dict with ticker -> sector)
        method: Neutralization method ("demean", "within_sector_z")
        
    Returns:
        Sector-neutralized series with same index as input
        
    Notes:
        - "demean": Subtract sector mean from each value
        - "within_sector_z": Z-score within each sector
        - Handles missing sectors by returning original values
        - Returns NaN for tickers not in sectors_map
    """
    if series.empty:
        return series
    
    # Convert sectors_map to Series if it's a dict
    if isinstance(sectors_map, dict):
        sectors_ser = pd.Series(sectors_map)
    else:
        sectors_ser = sectors_map
    
    # Align indices
    common_idx = series.index.intersection(sectors_ser.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float, index=series.index)
    
    series_aligned = series.loc[common_idx]
    sectors_aligned = sectors_ser.loc[common_idx]
    
    result = pd.Series(index=series.index, dtype=float)
    
    if method == "demean":
        # Subtract sector mean from each value
        for sector in sectors_aligned.unique():
            sector_mask = sectors_aligned == sector
            sector_data = series_aligned[sector_mask]
            if len(sector_data) > 0:
                sector_mean = sector_data.mean()
                result.loc[sector_data.index] = sector_data - sector_mean
                
    elif method == "within_sector_z":
        # Z-score within each sector
        for sector in sectors_aligned.unique():
            sector_mask = sectors_aligned == sector
            sector_data = series_aligned[sector_mask]
            if len(sector_data) > 1:  # Need at least 2 values for std
                sector_mean = sector_data.mean()
                sector_std = sector_data.std(ddof=1)
                if sector_std > 0:
                    result.loc[sector_data.index] = (sector_data - sector_mean) / sector_std
                else:
                    # All values in sector are identical
                    result.loc[sector_data.index] = 0.0
            elif len(sector_data) == 1:
                # Single value in sector, set to 0
                result.loc[sector_data.index] = 0.0
                
    else:
        raise ValueError(f"Unknown neutralization method: {method}")
    
    return result 