"""
Risk model module for covariance estimation and risk management.

Implements shrinkage covariance estimation with diagonal loading for portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from scipy import linalg
from datetime import date
import logging


def returns_from_prices(
    prices_df: pd.DataFrame,
    asof_dt: Union[str, pd.Timestamp, date],
    lookback_days: int = 60,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Returns a ticker x lookback_days DataFrame of simple returns for the trailing window ending at asof_dt.

    Args:
        prices_df: DataFrame with columns ["asof_dt", "ticker", price_col]
        asof_dt: End date for the lookback window
        lookback_days: Number of trading days to look back
        price_col: Column name for price data (default "close")

    Returns:
        DataFrame with tickers as index and returns as columns (1..lookback_days, oldest to newest)

    Notes:
        - Point-in-time: uses only rows with asof_dt <= asof_dt
        - Returns r_{t-1->t} = P_t / P_{t-1} - 1 over the last lookback_days trading steps
        - Includes ONLY tickers with FULL window (lookback_days consecutive returns)
        - Handles duplicates by dropping duplicate ["asof_dt", "ticker"] (keep last)
    """
    # Coerce asof_dt to Timestamp
    asof_timestamp = pd.Timestamp(asof_dt)
    
    # Clean and prepare data
    prices_clean = prices_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    prices_clean["asof_dt"] = pd.to_datetime(prices_clean["asof_dt"])
    
    # Point-in-time filter: use only data up to asof_dt
    prices_filtered = prices_clean[prices_clean["asof_dt"] <= asof_timestamp]
    
    if prices_filtered.empty:
        return pd.DataFrame()
    
    # Get all unique dates and sort them
    all_dates = sorted(prices_filtered["asof_dt"].unique())
    
    # Find the index of asof_dt in the sorted dates
    try:
        end_idx = all_dates.index(asof_timestamp)
    except ValueError:
        # asof_dt not in data, find the closest date <= asof_dt
        valid_dates = [d for d in all_dates if d <= asof_timestamp]
        if not valid_dates:
            return pd.DataFrame()
        end_idx = all_dates.index(max(valid_dates))
    
    # Check if we have enough data
    if end_idx < lookback_days:
        return pd.DataFrame()
    
    # Get the window of dates we need
    start_idx = end_idx - lookback_days + 1
    window_dates = all_dates[start_idx:end_idx + 1]
    
    # Get prices for all tickers in the window
    window_prices = prices_filtered[prices_filtered["asof_dt"].isin(window_dates)]
    
    # Pivot to get ticker x date matrix
    price_matrix = window_prices.pivot(index="ticker", columns="asof_dt", values=price_col)
    
    # Check which tickers have full data (no NaNs)
    tickers_with_full_data = price_matrix.dropna().index
    
    if len(tickers_with_full_data) == 0:
        return pd.DataFrame()
    
    # Get the subset with full data
    price_matrix_full = price_matrix.loc[tickers_with_full_data]
    
    # Sort tickers for consistent output
    price_matrix_full = price_matrix_full.sort_index()
    
    # Calculate returns: r_{t-1->t} = P_t / P_{t-1} - 1
    returns_matrix = price_matrix_full.pct_change(axis=1).iloc[:, 1:]  # Drop first column (NaN)
    
    # Rename columns to 1..n_returns (oldest to newest)
    n_returns = len(returns_matrix.columns)
    returns_matrix.columns = range(1, n_returns + 1)
    
    logging.info(f"returns_from_prices: shape={returns_matrix.shape}, asof_dt={asof_dt}")
    
    return returns_matrix


def shrink_cov(
    returns: pd.DataFrame,   # ticker x lookback_days from returns_from_prices
    lam: float = 0.3,        # 0 => sample covariance, 1 => diagonal (variances only)
    diag_load: float = 1e-4,
    min_var: float = 1e-10
) -> pd.DataFrame:
    """
    Computes a shrunk, diagonally-loaded covariance matrix.

    Args:
        returns: DataFrame with tickers as index and returns as columns
        lam: Shrinkage parameter (0-1), 0=sample cov, 1=diagonal only
        diag_load: Diagonal loading value
        min_var: Minimum variance to enforce on diagonal

    Returns:
        Covariance matrix as DataFrame with same ticker order as input

    Notes:
        - Let S be the sample covariance (rowvars=tickers)
        - Let D be diag(diag(S)) (diagonal matrix of sample variances)
        - Σ = (1 - lam) * S + lam * D
        - Diagonal loading: Σ <- Σ + diag_load * I
        - Enforce minimum variance on diagonal: clip diag to at least min_var
        - Ensure symmetry: Σ = (Σ + Σ.T)/2
    """
    if returns.empty:
        return pd.DataFrame()
    
    # Convert to numpy for calculations
    returns_array = returns.values
    
    # Calculate sample covariance matrix
    # Note: pandas cov() uses ddof=1 by default, but we want population covariance
    sample_cov = np.cov(returns_array, rowvar=True, ddof=0)
    
    # Create diagonal matrix of sample variances
    diag_var = np.diag(np.diag(sample_cov))
    
    # Apply shrinkage: Σ = (1 - lam) * S + lam * D
    shrunk_cov = (1 - lam) * sample_cov + lam * diag_var
    
    # Add diagonal loading: Σ <- Σ + diag_load * I
    shrunk_cov += diag_load * np.eye(shrunk_cov.shape[0])
    
    # Enforce minimum variance on diagonal
    np.fill_diagonal(shrunk_cov, np.maximum(np.diag(shrunk_cov), min_var))
    
    # Ensure symmetry: Σ = (Σ + Σ.T)/2
    shrunk_cov = (shrunk_cov + shrunk_cov.T) / 2
    
    # Convert back to DataFrame with same ticker order
    cov_df = pd.DataFrame(
        shrunk_cov,
        index=returns.index,
        columns=returns.index
    )
    
    logging.info(f"shrink_cov: shape={cov_df.shape}, lam={lam}, diag_load={diag_load}")
    
    return cov_df


def estimate_covariance(returns_df: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """
    Estimate covariance matrix using rolling window of returns.
    
    Args:
        returns_df: DataFrame with daily returns
        lookback_days: Number of days to use for covariance estimation
        
    Returns:
        Covariance matrix as DataFrame
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("estimate_covariance not implemented")


def apply_shrinkage(cov_matrix: pd.DataFrame, shrinkage_lambda: float = 0.3,
                   diagonal_load: float = 1e-4) -> pd.DataFrame:
    """
    Apply shrinkage to covariance matrix with diagonal loading.
    
    Args:
        cov_matrix: Input covariance matrix
        shrinkage_lambda: Shrinkage parameter (0-1)
        diagonal_load: Small value to add to diagonal for conditioning
        
    Returns:
        Shrunk covariance matrix
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("apply_shrinkage not implemented")


def calculate_portfolio_risk(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """
    Calculate portfolio risk (volatility) given weights and covariance matrix.
    
    Args:
        weights: Portfolio weights series
        cov_matrix: Covariance matrix
        
    Returns:
        Portfolio volatility (annualized)
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("calculate_portfolio_risk not implemented")


def calculate_marginal_contribution(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Calculate marginal contribution to risk for each asset.
    
    Args:
        weights: Portfolio weights series
        cov_matrix: Covariance matrix
        
    Returns:
        Series with marginal contribution to risk
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("calculate_marginal_contribution not implemented")


def validate_covariance_matrix(cov_matrix: pd.DataFrame) -> bool:
    """
    Validate that covariance matrix is positive definite and well-conditioned.
    
    Args:
        cov_matrix: Covariance matrix to validate
        
    Returns:
        True if matrix is valid
        
    Raises:
        ValueError: If matrix is not positive definite
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("validate_covariance_matrix not implemented")


def calculate_correlation_matrix(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix from covariance matrix.
    
    Args:
        cov_matrix: Covariance matrix
        
    Returns:
        Correlation matrix
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("calculate_correlation_matrix not implemented") 