"""
Data I/O helpers for quant engine.

Provides small, reusable functions for loading and writing CSV data
with minimal validation and type coercion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional


def load_prices(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load price data from CSV with validation and cleaning.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with columns ["asof_dt", "ticker", "close"]
        
    Raises:
        ValueError: If required columns missing or data invalid
    """
    # Read CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ["asof_dt", "ticker", "close"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Coerce types
    df["asof_dt"] = pd.to_datetime(df["asof_dt"]).dt.date
    df["asof_dt"] = pd.to_datetime(df["asof_dt"])
    df["ticker"] = df["ticker"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    
    # Drop duplicates and invalid data
    df = df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last")
    df = df.dropna(subset=["close"])
    df = df[np.isfinite(df["close"])]
    
    # Sort and return
    df = df.sort_values(["asof_dt", "ticker"]).reset_index(drop=True)
    return df


def load_fundamentals(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load fundamental data from CSV with validation and cleaning.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with fundamental data
        
    Raises:
        ValueError: If required columns missing or data invalid
    """
    # Read CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ["report_dt", "available_asof", "ticker", "eps_ttm", "book_value_ps"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Coerce types
    df["report_dt"] = pd.to_datetime(df["report_dt"]).dt.date
    df["report_dt"] = pd.to_datetime(df["report_dt"])
    df["available_asof"] = pd.to_datetime(df["available_asof"]).dt.date
    df["available_asof"] = pd.to_datetime(df["available_asof"])
    df["ticker"] = df["ticker"].astype(str)
    df["eps_ttm"] = pd.to_numeric(df["eps_ttm"], errors="coerce")
    df["book_value_ps"] = pd.to_numeric(df["book_value_ps"], errors="coerce")
    
    # Keep valid rows
    df = df[df["available_asof"] >= df["report_dt"]]
    df = df.dropna(subset=["eps_ttm", "book_value_ps"])
    df = df[np.isfinite(df["eps_ttm"]) & np.isfinite(df["book_value_ps"])]
    
    # Sort and return
    df = df.sort_values(["ticker", "available_asof"]).reset_index(drop=True)
    return df


def load_sectors(path: Union[str, Path]) -> pd.Series:
    """
    Load sector data from CSV and return as Series.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Series with ticker index and sector values
        
    Raises:
        ValueError: If required columns missing
    """
    # Read CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ["ticker", "sector"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Coerce types and clean
    df["ticker"] = df["ticker"].astype(str)
    df["sector"] = df["sector"].astype(str)
    df = df.dropna(subset=["sector"])
    df = df.drop_duplicates(subset=["ticker"], keep="last")
    
    # Return as Series
    return df.set_index("ticker")["sector"]


def load_holdings(path: Union[str, Path], asof: Union[str, pd.Timestamp]) -> pd.Series:
    """
    Load holdings data for a specific date.
    
    Args:
        path: Path to CSV file
        asof: Date to filter for
        
    Returns:
        Series with ticker index and normalized weights
        
    Raises:
        ValueError: If required columns missing or no data for asof
    """
    # Read CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ["asof_dt", "ticker", "weight"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Coerce types
    df["asof_dt"] = pd.to_datetime(df["asof_dt"]).dt.date
    df["asof_dt"] = pd.to_datetime(df["asof_dt"])
    df["ticker"] = df["ticker"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    
    # Filter to asof date
    asof_dt = pd.to_datetime(asof).date()
    asof_dt = pd.to_datetime(asof_dt)
    df = df[df["asof_dt"] == asof_dt]
    
    if df.empty:
        raise ValueError(f"No holdings data found for date: {asof}")
    
    # Group by ticker and sum weights
    holdings = df.groupby("ticker")["weight"].sum()
    
    # Clean weights
    holdings = holdings.clip(lower=0)  # Clip negatives to 0
    holdings = holdings.dropna()  # Drop NaNs
    
    # Renormalize to sum=1
    if holdings.sum() > 0:
        holdings = holdings / holdings.sum()
    
    return holdings


def write_holdings(outdir: Union[str, Path], asof: Union[str, pd.Timestamp], 
                  weights: pd.Series) -> Path:
    """
    Write holdings data to CSV.
    
    Args:
        outdir: Output directory
        asof: Date for holdings
        weights: Series with ticker index and weights
        
    Returns:
        Path to written file
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    asof_dt = pd.to_datetime(asof).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "asof_dt": asof_dt,
        "ticker": weights.index,
        "weight": weights.values
    })
    
    # Sort by ticker and write
    df = df.sort_values("ticker")
    outpath = outdir / f"holdings_{asof_dt}.csv"
    df.to_csv(outpath, index=False)
    
    return outpath


def write_trades(outdir: Union[str, Path], asof: Union[str, pd.Timestamp],
                new_w: pd.Series, prev_w: pd.Series) -> Path:
    """
    Write trade data to CSV.
    
    Args:
        outdir: Output directory
        asof: Date for trades
        new_w: New weights Series
        prev_w: Previous weights Series
        
    Returns:
        Path to written file
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Align tickers
    all_tickers = new_w.index.union(prev_w.index)
    new_aligned = new_w.reindex(all_tickers, fill_value=0.0)
    prev_aligned = prev_w.reindex(all_tickers, fill_value=0.0)
    
    # Calculate deltas
    delta = new_aligned - prev_aligned
    
    # Prepare data
    asof_dt = pd.to_datetime(asof).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "asof_dt": asof_dt,
        "ticker": delta.index,
        "delta_weight": delta.values
    })
    
    # Sort by absolute delta descending, then by ticker
    df["abs_delta"] = df["delta_weight"].abs()
    df = df.sort_values(["abs_delta", "ticker"], ascending=[False, True])
    df = df.drop("abs_delta", axis=1)
    
    # Write file
    outpath = outdir / f"trades_{asof_dt}.csv"
    df.to_csv(outpath, index=False)
    
    return outpath


def unique_dates(prices_df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Get unique dates from prices DataFrame.
    
    Args:
        prices_df: DataFrame with asof_dt column
        
    Returns:
        Sorted DatetimeIndex of unique dates
    """
    return pd.DatetimeIndex(sorted(prices_df["asof_dt"].unique()))


def next_day_exists(prices_df: pd.DataFrame, asof: Union[str, pd.Timestamp]) -> bool:
    """
    Check if there is a trading day after the given date.
    
    Args:
        prices_df: DataFrame with asof_dt column
        asof: Date to check
        
    Returns:
        True if there is a trading day strictly after asof
    """
    asof_dt = pd.to_datetime(asof)
    unique_dates = pd.DatetimeIndex(sorted(prices_df["asof_dt"].unique()))
    
    return (unique_dates > asof_dt).any() 