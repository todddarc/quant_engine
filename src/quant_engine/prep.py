"""
Data preparation module for preprocessing and feature engineering.

Handles data alignment, return calculations, and signal preprocessing steps.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union


def winsorize(series: pd.Series, p_low: float, p_high: float) -> pd.Series:
    """
    Clip values to empirical quantiles computed on non-NaN entries.
    
    Args:
        series: Input series to winsorize
        p_low: Lower quantile threshold (0-1)
        p_high: Upper quantile threshold (0-1)
        
    Returns:
        Winsorized series with same index and dtype=float
        
    Notes:
        - Does not modify NaNs
        - Returns original series if p_low >= p_high or insufficient data (n<3)
    """
    if p_low >= p_high:
        return series.astype(float)
    
    # Get non-NaN values
    non_nan_series = series.dropna()
    if len(non_nan_series) < 3:
        return series.astype(float)
    
    # Calculate quantiles
    low_threshold = non_nan_series.quantile(p_low)
    high_threshold = non_nan_series.quantile(p_high)
    
    # Apply winsorization
    result = series.copy().astype(float)
    result = result.clip(lower=low_threshold, upper=high_threshold)
    
    return result


def zscore(series: pd.Series, ddof: int = 0) -> pd.Series:
    """
    Standardize non-NaN values: (x - mean) / std.
    
    Args:
        series: Input series to standardize
        ddof: Delta degrees of freedom for std calculation
        
    Returns:
        Z-scored series with same index and dtype=float
        
    Notes:
        - Leaves NaNs as NaN
        - Returns zeros for entries where std ≈ 0 (<= 1e-12)
    """
    result = series.copy().astype(float)
    
    # Get non-NaN values
    non_nan_mask = ~series.isna()
    if not non_nan_mask.any():
        return result
    
    non_nan_values = series[non_nan_mask]
    mean_val = non_nan_values.mean()
    std_val = non_nan_values.std(ddof=ddof)
    
    # Center all non-NaN values
    result[non_nan_mask] = non_nan_values - mean_val
    
    # Scale if std is not approximately zero
    if std_val > 1e-12:
        result[non_nan_mask] = result[non_nan_mask] / std_val
    
    return result


def sector_neutralize(
    series: pd.Series,
    sectors_map: Union[pd.Series, Dict],
    method: str = "within_sector_z"
) -> pd.Series:
    """
    Neutralize sector effects from a series.
    
    Args:
        series: Input series to neutralize
        sectors_map: Mapping from ticker to sector (Series or dict)
        method: Neutralization method ("within_sector_z")
        
    Returns:
        Sector-neutralized series aligned to intersection of tickers with sector labels
        
    Notes:
        - For "within_sector_z": z-score within each sector
        - If sector has std ≈ 0 or size < 2, center to 0 instead
        - Preserves relative ranks within each sector
    """
    # Coerce sectors_map to Series and align to series index
    if isinstance(sectors_map, dict):
        sectors_series = pd.Series(sectors_map)
    else:
        sectors_series = sectors_map.copy()
    
    # Align sectors to series index and drop missing sectors
    aligned_sectors = sectors_series.reindex(series.index)
    valid_mask = ~aligned_sectors.isna()
    
    if not valid_mask.any():
        return pd.Series(dtype=float)
    
    # Get valid data
    valid_series = series[valid_mask]
    valid_sectors = aligned_sectors[valid_mask]
    
    if method == "within_sector_z":
        result = pd.Series(index=valid_series.index, dtype=float)
        
        for sector in valid_sectors.unique():
            sector_mask = valid_sectors == sector
            sector_data = valid_series[sector_mask]
            
            if len(sector_data) < 2:
                # Small sector: just center to 0
                result[sector_mask] = sector_data - sector_data.mean()
            else:
                # Apply z-score within sector
                sector_mean = sector_data.mean()
                sector_std = sector_data.std(ddof=0)
                
                if sector_std <= 1e-12:
                    # Near-constant sector: just center to 0
                    result[sector_mask] = sector_data - sector_mean
                else:
                    # Normal z-score
                    result[sector_mask] = (sector_data - sector_mean) / sector_std
        
        return result
    else:
        raise ValueError(f"Unknown neutralization method: {method}")


def prepare_data(prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame, 
                sectors_df: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    """
    Prepare aligned dataset for signal generation and optimization.
    
    Args:
        prices_df: Price data with asof_dt, ticker, close
        fundamentals_df: Fundamental data with point-in-time lags
        sectors_df: Sector mapping
        asof_date: Current date for point-in-time calculation
        
    Returns:
        Aligned DataFrame with all required columns
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("prepare_data not implemented")


def calculate_returns(prices_df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Calculate price returns over specified periods.
    
    Args:
        prices_df: Price data with asof_dt, ticker, close
        periods: Number of periods for return calculation
        
    Returns:
        DataFrame with return columns
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("calculate_returns not implemented")


def align_data_to_date(prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame,
                      sectors_df: pd.DataFrame, asof_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align all data to a specific date with proper point-in-time handling.
    
    Args:
        prices_df: Price data
        fundamentals_df: Fundamental data
        sectors_df: Sector mapping
        asof_date: Target date for alignment
        
    Returns:
        Tuple of aligned DataFrames
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("align_data_to_date not implemented")


def handle_missing_data(df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
    """
    Handle missing data using specified method.
    
    Args:
        df: Input DataFrame
        method: Method to use ('drop', 'forward_fill', 'interpolate')
        
    Returns:
        DataFrame with missing data handled
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("handle_missing_data not implemented")


def create_universe_mask(prices_df: pd.DataFrame, min_price: float = 5.0, 
                        min_volume: Optional[float] = None) -> pd.Series:
    """
    Create universe mask for investable securities.
    
    Args:
        prices_df: Price data
        min_price: Minimum price threshold
        min_volume: Minimum volume threshold (if available)
        
    Returns:
        Boolean series indicating investable securities
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("create_universe_mask not implemented")


def calculate_forward_returns(prices_df: pd.DataFrame, periods: int = 21) -> pd.DataFrame:
    """
    Calculate forward returns for signal validation.
    
    Args:
        prices_df: Price data
        periods: Number of periods to look forward
        
    Returns:
        DataFrame with forward return columns
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("calculate_forward_returns not implemented") 