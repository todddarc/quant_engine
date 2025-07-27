"""
Signal generation module for momentum and value alpha signals.

Implements momentum (252-day lookback with 21-day gap) and value (E/P) signals
with winsorization, z-scoring, and sector neutralization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union
from datetime import date


def value_ep(fundamentals_df: pd.DataFrame, prices_df: pd.DataFrame, asof_dt: Union[date, str, pd.Timestamp], 
             min_lag_days: int = 60) -> pd.Series:
    """
    Calculate value signal using E/P ratio with point-in-time discipline.
    
    Args:
        fundamentals_df: DataFrame with columns ["report_dt", "available_asof", "ticker", "eps_ttm", "book_value_ps"]
        prices_df: DataFrame with columns ["asof_dt", "ticker", "close"]
        asof_dt: Current date (str, date, or Timestamp)
        min_lag_days: Minimum lag days required for fundamentals to be available
        
    Returns:
        Series with E/P values indexed by ticker, name="val_ep"
        
    Raises:
        ValueError: If required columns are missing
    """
    # Validate input
    required_fund_cols = ["report_dt", "available_asof", "ticker", "eps_ttm", "book_value_ps"]
    required_price_cols = ["asof_dt", "ticker", "close"]
    
    if not all(col in fundamentals_df.columns for col in required_fund_cols):
        raise ValueError(f"fundamentals_df must contain columns: {required_fund_cols}")
    
    if not all(col in prices_df.columns for col in required_price_cols):
        raise ValueError(f"prices_df must contain columns: {required_price_cols}")
    
    # Coerce asof_dt to Timestamp
    asof_timestamp = pd.Timestamp(asof_dt)
    
    # Clean and prepare data
    fundamentals_clean = fundamentals_df.drop_duplicates(subset=["available_asof", "ticker"], keep="last").copy()
    prices_clean = prices_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    
    # Coerce date fields to Timestamp
    fundamentals_clean["available_asof"] = pd.to_datetime(fundamentals_clean["available_asof"])
    fundamentals_clean["report_dt"] = pd.to_datetime(fundamentals_clean["report_dt"])
    prices_clean["asof_dt"] = pd.to_datetime(prices_clean["asof_dt"])
    
    # Point-in-time rule: filter fundamentals
    # available_asof <= t AND (t - available_asof).days >= min_lag_days
    fundamentals_filtered = fundamentals_clean[
        (fundamentals_clean["available_asof"] <= asof_timestamp) &
        ((asof_timestamp - fundamentals_clean["available_asof"]).dt.days >= min_lag_days)
    ]
    
    # Get current prices
    current_prices = prices_clean[prices_clean["asof_dt"] == asof_timestamp]
    
    if current_prices.empty:
        return pd.Series(dtype=float, name="val_ep")
    
    # For each ticker, get the most recent available fundamental record
    results = {}
    for ticker in current_prices["ticker"].unique():
        # Get current price for this ticker
        ticker_price = current_prices[current_prices["ticker"] == ticker]
        if ticker_price.empty:
            continue
        current_price = ticker_price.iloc[0]["close"]
        
        # Get fundamentals for this ticker
        ticker_fundamentals = fundamentals_filtered[fundamentals_filtered["ticker"] == ticker]
        if ticker_fundamentals.empty:
            continue
        
        # Get the most recent available fundamental record (max available_asof)
        latest_fundamental = ticker_fundamentals.sort_values("available_asof").iloc[-1]
        eps_ttm = latest_fundamental["eps_ttm"]
        
        # Compute E/P ratio
        if current_price != 0:  # Avoid division by zero
            ep_ratio = eps_ttm / current_price
            results[ticker] = ep_ratio
    
    s = pd.Series(results, name="val_ep")
    return s.sort_index()


def momentum_12m_1m_gap(prices_df: pd.DataFrame, asof_dt: Union[date, str, pd.Timestamp], 
                       lookback: int = 252, gap: int = 21) -> pd.Series:
    """
    Calculate momentum signal with lookback and gap periods.
    
    Args:
        prices_df: DataFrame with columns ["asof_dt", "ticker", "close"]
        asof_dt: Current date (str, date, or Timestamp)
        lookback: Number of days to look back for momentum calculation
        gap: Number of days to skip (avoid short-term reversal)
        
    Returns:
        Series with momentum values indexed by ticker, name="mom_12_1"
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["asof_dt", "ticker", "close"]
    if not all(col in prices_df.columns for col in required_cols):
        raise ValueError(f"prices_df must contain columns: {required_cols}")
    
    asof_timestamp = pd.Timestamp(asof_dt)
    prices_clean = prices_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    prices_clean["asof_dt"] = pd.to_datetime(prices_clean["asof_dt"])
    
    # Use business days for gap calculations
    t_gap = asof_timestamp - pd.offsets.BusinessDay(gap)
    t_lookback_gap = asof_timestamp - pd.offsets.BusinessDay(gap + lookback)
    
    results = {}
    for ticker, tdf in prices_clean.groupby("ticker"):
        tdf = tdf.sort_values("asof_dt")
        # Find last price on or before t_gap
        p_t_gap = tdf[tdf["asof_dt"] <= t_gap]
        if p_t_gap.empty:
            continue
        p_t_gap_val = p_t_gap.iloc[-1]["close"]
        t_gap_actual = p_t_gap.iloc[-1]["asof_dt"]
        # Find last price on or before t_lookback_gap
        p_t_lookback_gap = tdf[tdf["asof_dt"] <= t_lookback_gap]
        if p_t_lookback_gap.empty:
            continue
        p_t_lookback_gap_val = p_t_lookback_gap.iloc[-1]["close"]
        t_lookback_gap_actual = p_t_lookback_gap.iloc[-1]["asof_dt"]
        # Require at least `lookback` unique trading days between the two dates (inclusive)
        n_trading_days = tdf[(tdf["asof_dt"] >= t_lookback_gap_actual) & (tdf["asof_dt"] <= t_gap_actual)]["asof_dt"].nunique()
        if n_trading_days < lookback:
            continue
        momentum = p_t_gap_val / p_t_lookback_gap_val - 1
        results[ticker] = momentum
    s = pd.Series(results, name="mom_12_1")
    return s.sort_index() 