"""
Pre-trade validation checks module.

Implements comprehensive safety checks for data quality, schema validation,
turnover limits, and sector exposure constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union


def check_schema(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str = "schema"
) -> Dict[str, Dict[str, str]]:
    """
    Verify required columns exist in df. BLOCK if any missing.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        name: Name for the check result
        
    Returns:
        Dict with check result: {name: {"status": "PASS"|"BLOCK", "details": "..."}}
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        details = f"Missing columns: {', '.join(missing_cols)}"
        status = "BLOCK"
    else:
        details = f"All required columns present: {', '.join(required_cols)}"
        status = "PASS"
    
    return {name: {"status": status, "details": details}}


def check_missingness(
    df: pd.DataFrame,
    max_rate: float = 0.005,
    name: str = "missingness"
) -> Dict[str, Dict[str, str]]:
    """
    Compute NA share per column. BLOCK if any column's NA rate > max_rate.
    
    Args:
        df: DataFrame to check
        max_rate: Maximum allowed NA rate per column
        name: Name for the check result
        
    Returns:
        Dict with check result: {name: {"status": "PASS"|"BLOCK", "details": "..."}}
    """
    na_rates = df.isna().mean()
    violating_cols = na_rates[na_rates > max_rate]
    
    if len(violating_cols) > 0:
        worst_col = violating_cols.idxmax()
        worst_rate = violating_cols.max()
        details = f"Column '{worst_col}' has {worst_rate:.3f} NA rate (max: {max_rate:.3f})"
        status = "BLOCK"
    else:
        max_na_rate = na_rates.max() if len(na_rates) > 0 else 0.0
        details = f"Max NA rate: {max_na_rate:.3f} (threshold: {max_rate:.3f})"
        status = "PASS"
    
    return {name: {"status": status, "details": details}}


def check_turnover(
    prev_w: pd.Series,
    new_w: pd.Series,
    cap: float = 0.10,
    name: str = "turnover"
) -> Dict[str, Dict[str, str]]:
    """
    Turnover = 0.5 * sum(|new_w - prev_w_aligned|) over shared tickers.
    
    Args:
        prev_w: Prior weights series
        new_w: New weights series
        cap: Maximum allowed turnover
        name: Name for the check result
        
    Returns:
        Dict with check result: {name: {"status": "PASS"|"BLOCK", "details": "..."}}
    """
    # Handle duplicates by summing weights
    prev_w_clean = prev_w.groupby(prev_w.index).sum()
    new_w_clean = new_w.groupby(new_w.index).sum()
    
    # Ensure weights are finite
    prev_w_clean = prev_w_clean.replace([np.inf, -np.inf], np.nan)
    new_w_clean = new_w_clean.replace([np.inf, -np.inf], np.nan)
    
    # Get all unique tickers
    all_tickers = prev_w_clean.index.union(new_w_clean.index)
    
    # Align both series to all tickers, filling missing with 0
    prev_w_aligned = prev_w_clean.reindex(all_tickers, fill_value=0.0)
    new_w_aligned = new_w_clean.reindex(all_tickers, fill_value=0.0)
    
    # Calculate turnover
    turnover = 0.5 * np.sum(np.abs(new_w_aligned - prev_w_aligned))
    
    if turnover > cap + 1e-8:
        details = f"Turnover {turnover:.4f} exceeds cap {cap:.4f}"
        status = "BLOCK"
    else:
        details = f"Turnover {turnover:.4f} within cap {cap:.4f}"
        status = "PASS"
    
    return {name: {"status": status, "details": details}}


def check_sector_exposure(
    new_w: pd.Series,
    sectors_map: Union[pd.Series, dict],
    cap: float = 0.25,
    name: str = "sector_exposure"
) -> Dict[str, Dict[str, str]]:
    """
    Sum weights by sector and ensure each sector sum <= cap + 1e-8.
    
    Args:
        new_w: New weights series
        sectors_map: Mapping from ticker to sector (Series or dict)
        cap: Maximum allowed sector weight
        name: Name for the check result
        
    Returns:
        Dict with check result: {name: {"status": "PASS"|"BLOCK", "details": "..."}}
    """
    # Handle duplicates by summing weights
    new_w_clean = new_w.groupby(new_w.index).sum()
    
    # Convert sectors_map to Series if it's a dict
    if isinstance(sectors_map, dict):
        sectors_series = pd.Series(sectors_map)
    else:
        sectors_series = sectors_map.copy()
    
    # Align sectors_map to new_w index
    sectors_aligned = sectors_series.reindex(new_w_clean.index)
    
    # Count tickers without sector labels
    tickers_without_sector = sectors_aligned.isna().sum()
    
    # Calculate sector weights (only for tickers with sector labels)
    sector_weights = new_w_clean[sectors_aligned.notna()].groupby(sectors_aligned[sectors_aligned.notna()]).sum()
    
    # Check for violations
    violating_sectors = sector_weights[sector_weights > cap + 1e-8]
    
    if len(violating_sectors) > 0:
        max_sector = violating_sectors.idxmax()
        max_exposure = violating_sectors.max()
        details = f"Sector '{max_sector}' exposure {max_exposure:.3f} exceeds cap {cap:.3f}"
        status = "BLOCK"
    else:
        max_exposure = sector_weights.max() if len(sector_weights) > 0 else 0.0
        details = f"Max sector exposure: {max_exposure:.3f} (cap: {cap:.3f})"
        if tickers_without_sector > 0:
            details += f", {tickers_without_sector} tickers without sector labels"
        status = "PASS"
    
    return {name: {"status": status, "details": details}}


def aggregate_checks(check_results: Dict[str, Dict[str, str]]) -> Tuple[bool, Dict[str, Dict[str, str]]]:
    """
    Aggregate all check results into a final ok_to_trade boolean.
    
    Args:
        check_results: Dict of check results
        
    Returns:
        Tuple of (ok_to_trade, check_results)
    """
    ok_to_trade = all(result["status"] == "PASS" for result in check_results.values())
    return ok_to_trade, check_results


def validate_data(prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame,
                 sectors_df: pd.DataFrame, holdings_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate data quality and schema compliance.
    
    Args:
        prices_df: Price data
        fundamentals_df: Fundamental data
        sectors_df: Sector mapping
        holdings_df: Prior holdings
        
    Returns:
        Dict with validation results for each check
        
    Raises:
        ValueError: If critical validation fails
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("validate_data not implemented")


def check_data_missingness(df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, float]:
    """
    Check for excessive missing data in DataFrame.
    
    Args:
        df: DataFrame to check
        threshold: Maximum allowed missing fraction
        
    Returns:
        Dict mapping columns to missing fractions
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("check_data_missingness not implemented")


def check_schema_drift(df: pd.DataFrame, expected_schema: Dict[str, type], 
                      name: str) -> bool:
    """
    Check for schema drift in DataFrame.
    
    Args:
        df: DataFrame to check
        expected_schema: Expected column types
        name: Data source name for error messages
        
    Returns:
        True if schema is valid
        
    Raises:
        ValueError: If schema drift detected
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("check_schema_drift not implemented")


def check_extreme_values(series: pd.Series, method: str = 'iqr', 
                        multiplier: float = 3.0) -> List[Any]:
    """
    Check for extreme values using specified method.
    
    Args:
        series: Series to check
        method: Method to use ('iqr', 'zscore', 'percentile')
        multiplier: Multiplier for outlier detection
        
    Returns:
        List of extreme values found
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("check_extreme_values not implemented")


def run_all_checks(prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame,
                  sectors_df: pd.DataFrame, holdings_df: pd.DataFrame,
                  new_weights: pd.Series, prior_weights: pd.Series,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all pre-trade validation checks.
    
    Args:
        prices_df: Price data
        fundamentals_df: Fundamental data
        sectors_df: Sector mapping
        holdings_df: Prior holdings
        new_weights: New portfolio weights
        prior_weights: Prior portfolio weights
        config: Configuration parameters
        
    Returns:
        Dict with all check results
        
    Raises:
        ValueError: If any critical check fails
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("run_all_checks not implemented") 