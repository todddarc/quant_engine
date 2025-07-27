"""
Utility functions for the quant engine.

Common helper functions for logging, configuration, and data manipulation.
"""

import pandas as pd
import numpy as np
import logging
import yaml
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from scipy import stats
from datetime import date


@dataclass
class Config:
    """Configuration dataclass for engine parameters."""
    
    # Signal parameters
    momentum_lookback_days: int = 252
    momentum_gap_days: int = 21
    signal_weights: Dict[str, float] = None
    
    # Risk parameters
    covariance_lookback_days: int = 60
    shrinkage_lambda: float = 0.3
    diagonal_load: float = 1e-4
    
    # Optimization constraints
    per_name_cap: float = 0.05
    sector_cap: float = 0.25
    turnover_cap: float = 0.10
    
    # Data parameters
    min_price: float = 5.0
    winsorize_percentile: float = 0.025
    
    def __post_init__(self):
        if self.signal_weights is None:
            self.signal_weights = {'momentum': 0.5, 'value': 0.5}


def setup_logging(level: str = "INFO", fmt: Optional[str] = None) -> None:
    """
    Configure root logger once. Safe to call multiple times.
    - level: "DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL"
    - fmt default: "%(asctime)s %(levelname)s %(name)s - %(message)s"
    """
    # Map level string, default to INFO
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Remove existing handlers on root to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set default format if not provided
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(level=log_level, format=fmt)


def load_config(config_path: Path) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Config dataclass instance
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("load_config not implemented")


def save_results(weights: pd.Series, trades: pd.DataFrame, report: str,
                output_dir: Path) -> None:
    """
    Save portfolio results to files.
    
    Args:
        weights: Final portfolio weights
        trades: Trade recommendations
        report: Text report
        output_dir: Output directory
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("save_results not implemented")


def calculate_portfolio_metrics(weights: pd.Series, returns: pd.Series,
                              cov_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        weights: Portfolio weights
        returns: Asset returns
        cov_matrix: Covariance matrix
        
    Returns:
        Dict with portfolio metrics
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("calculate_portfolio_metrics not implemented")


def generate_report(signal_ic: pd.Series, decile_returns: pd.DataFrame,
                   portfolio_metrics: Dict[str, float],
                   check_results: Dict[str, Any]) -> str:
    """
    Generate text report with performance and validation results.
    
    Args:
        signal_ic: Signal IC series
        decile_returns: Decile performance data
        portfolio_metrics: Portfolio performance metrics
        check_results: Validation check results
        
    Returns:
        Formatted report string
        
    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError("generate_report not implemented")


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Fail fast if required keys are missing. Raise ValueError with a helpful message.
    Required (by our code path):
      data.prices_path, data.fundamentals_path, data.sectors_path, data.holdings_path
      signals.momentum.lookback, signals.momentum.gap
      signals.value.min_lag_days
      risk.cov_lookback_days, risk.shrink_lambda, risk.diag_load
      optimization.w_max, optimization.sector_cap, optimization.turnover_cap, optimization.risk_aversion
      paths.output_dir  (we allow a default if absent)
    Tolerate legacy blocks; only check keys we actually read.
    """
    missing_keys = []
    
    # Check data paths
    data_cfg = cfg.get("data", {})
    required_data = ["prices_path", "fundamentals_path", "sectors_path", "holdings_path"]
    for key in required_data:
        if key not in data_cfg:
            missing_keys.append(f"data.{key}")
    
    # Check signals config
    signals_cfg = cfg.get("signals", {})
    momentum_cfg = signals_cfg.get("momentum", {})
    value_cfg = signals_cfg.get("value", {})
    
    required_momentum = ["lookback", "gap"]
    for key in required_momentum:
        if key not in momentum_cfg:
            missing_keys.append(f"signals.momentum.{key}")
    
    if "min_lag_days" not in value_cfg:
        missing_keys.append("signals.value.min_lag_days")
    
    # Check risk config
    risk_cfg = cfg.get("risk", {})
    required_risk = ["cov_lookback_days", "shrink_lambda", "diag_load"]
    for key in required_risk:
        if key not in risk_cfg:
            missing_keys.append(f"risk.{key}")
    
    # Check optimization config
    opt_cfg = cfg.get("optimization", {})
    required_opt = ["w_max", "sector_cap", "turnover_cap", "risk_aversion"]
    for key in required_opt:
        if key not in opt_cfg:
            missing_keys.append(f"optimization.{key}")
    
    # paths.output_dir is optional (has default)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")


def set_random_seed(seed: int) -> None:
    """
    Set numpy and python's random seeds (and PYTHONHASHSEED env) for determinism.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) 


def compute_next_period_returns(
    prices_df: pd.DataFrame,
    asof_dt: Union[date, str, pd.Timestamp]
) -> pd.Series:
    """
    Compute next-period returns for a given as-of date.
    
    Args:
        prices_df: DataFrame with columns ["asof_dt", "ticker", "close"]
        asof_dt: Current date (str, date, or Timestamp)
        
    Returns:
        Series with next-period returns indexed by ticker
        
    Notes:
        - Returns r_{t→t+1} = close_{t+1}/close_t - 1 for the next trading day
        - Uses only tickers with both t and t+1 prices
        - Returns empty Series if no next day in data
    """
    # Coerce asof_dt to Timestamp
    asof_timestamp = pd.Timestamp(asof_dt)
    
    # Clean and prepare data
    prices_clean = prices_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    prices_clean["asof_dt"] = pd.to_datetime(prices_clean["asof_dt"])
    
    # Get current prices
    current_prices = prices_clean[prices_clean["asof_dt"] == asof_timestamp]
    if current_prices.empty:
        return pd.Series(dtype=float)
    
    # Get next day prices
    all_dates = prices_clean["asof_dt"].unique()
    all_dates = sorted(all_dates)
    
    # Find next trading day
    current_idx = None
    for i, date in enumerate(all_dates):
        if date == asof_timestamp:
            current_idx = i
            break
    
    if current_idx is None or current_idx == len(all_dates) - 1:
        return pd.Series(dtype=float)
    
    next_date = all_dates[current_idx + 1]
    next_prices = prices_clean[prices_clean["asof_dt"] == next_date]
    
    if next_prices.empty:
        return pd.Series(dtype=float)
    
    # Merge current and next prices
    merged = current_prices.merge(
        next_prices[["ticker", "close"]], 
        on="ticker", 
        suffixes=("_t", "_t1")
    )
    
    # Calculate returns
    merged["return"] = merged["close_t1"] / merged["close_t"] - 1
    
    result = merged.set_index("ticker")["return"]
    result.index.name = None  # Remove index name for consistency
    result.name = None  # Remove series name for consistency
    return result


def cross_sectional_ic(
    signal: pd.Series,
    next_ret: pd.Series,
    method: str = "spearman"
) -> float:
    """
    Compute cross-sectional information coefficient between signal and next returns.
    
    Args:
        signal: Signal series indexed by ticker
        next_ret: Next-period returns series indexed by ticker
        method: Correlation method ("spearman" or "pearson")
        
    Returns:
        Information coefficient (float)
        
    Notes:
        - Aligns on intersection of tickers and drops NaNs
        - Returns np.nan if < 3 paired observations
    """
    # Align series on intersection of tickers
    aligned_data = pd.DataFrame({
        "signal": signal,
        "next_ret": next_ret
    }).dropna()
    
    if len(aligned_data) < 3:
        return np.nan
    
    if method == "spearman":
        return stats.spearmanr(aligned_data["signal"], aligned_data["next_ret"])[0]
    elif method == "pearson":
        return np.corrcoef(aligned_data["signal"], aligned_data["next_ret"])[0, 1]
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def compute_ic_series(
    signals_hist_df: pd.DataFrame,
    future_returns_df: pd.DataFrame,
    method: str = "spearman"
) -> pd.DataFrame:
    """
    Compute information coefficient time series.
    
    Args:
        signals_hist_df: DataFrame with columns ["asof_dt", "ticker", "signal"]
        future_returns_df: DataFrame with columns ["asof_dt", "ticker", "ret_fwd1"]
        method: Correlation method ("spearman" or "pearson")
        
    Returns:
        DataFrame with columns ["asof_dt", "ic"] sorted by date
    """
    # Clean and prepare data
    signals_clean = signals_hist_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    returns_clean = future_returns_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    
    # Coerce dates to Timestamp
    signals_clean["asof_dt"] = pd.to_datetime(signals_clean["asof_dt"])
    returns_clean["asof_dt"] = pd.to_datetime(returns_clean["asof_dt"])
    
    # Get common dates
    signal_dates = set(signals_clean["asof_dt"].unique())
    return_dates = set(returns_clean["asof_dt"].unique())
    common_dates = sorted(signal_dates.intersection(return_dates))
    
    ic_results = []
    
    for date in common_dates:
        # Get signal and returns for this date
        date_signals = signals_clean[signals_clean["asof_dt"] == date].set_index("ticker")["signal"]
        date_returns = returns_clean[returns_clean["asof_dt"] == date].set_index("ticker")["ret_fwd1"]
        
        # Compute IC
        ic = cross_sectional_ic(date_signals, date_returns, method=method)
        ic_results.append({"asof_dt": date, "ic": ic})
    
    if not ic_results:
        return pd.DataFrame(columns=["asof_dt", "ic"])
    
    return pd.DataFrame(ic_results).sort_values("asof_dt")


def summarize_ic(
    ic_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute summary statistics for information coefficient series.
    
    Args:
        ic_df: DataFrame with columns ["asof_dt", "ic"]
        
    Returns:
        Dict with mean_ic, std_ic, t_stat, and hit_rate
    """
    ic_series = ic_df["ic"].dropna()
    
    if len(ic_series) == 0:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "t_stat": np.nan,
            "hit_rate": np.nan
        }
    
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    n = len(ic_series)
    
    # Calculate t-statistic
    if std_ic > 0:
        t_stat = mean_ic / (std_ic / np.sqrt(n))
    else:
        t_stat = np.nan
    
    # Calculate hit rate
    hit_rate = (ic_series > 0).mean()
    
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "t_stat": t_stat,
        "hit_rate": hit_rate
    }


def decile_portfolio_returns(
    signals_hist_df: pd.DataFrame,
    future_returns_df: pd.DataFrame,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Compute decile portfolio returns.
    
    Args:
        signals_hist_df: DataFrame with columns ["asof_dt", "ticker", "signal"]
        future_returns_df: DataFrame with columns ["asof_dt", "ticker", "ret_fwd1"]
        n_deciles: Number of deciles (default 10)
        
    Returns:
        DataFrame with columns ["decile", "mean_ret"] including L-S row
    """
    # Clean and prepare data
    signals_clean = signals_hist_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    returns_clean = future_returns_df.drop_duplicates(subset=["asof_dt", "ticker"], keep="last").copy()
    
    # Coerce dates to Timestamp
    signals_clean["asof_dt"] = pd.to_datetime(signals_clean["asof_dt"])
    returns_clean["asof_dt"] = pd.to_datetime(returns_clean["asof_dt"])
    
    # Get common dates
    signal_dates = set(signals_clean["asof_dt"].unique())
    return_dates = set(returns_clean["asof_dt"].unique())
    common_dates = sorted(signal_dates.intersection(return_dates))
    
    all_decile_returns = []
    
    for date in common_dates:
        # Get signal and returns for this date
        date_data = signals_clean[signals_clean["asof_dt"] == date].copy()
        date_data = date_data.merge(
            returns_clean[returns_clean["asof_dt"] == date][["ticker", "ret_fwd1"]],
            on="ticker",
            how="inner"
        )
        
        # Drop NaNs
        date_data = date_data.dropna(subset=["signal", "ret_fwd1"])
        
        if len(date_data) < n_deciles:
            continue
        
        # Handle ties deterministically: rank first, then qcut
        date_data["rank"] = date_data["signal"].rank(method="first")
        date_data["decile"] = pd.qcut(date_data["rank"], n_deciles, labels=False) + 1
        
        # Compute average return per decile
        decile_returns = date_data.groupby("decile")["ret_fwd1"].mean().reset_index()
        decile_returns["asof_dt"] = date
        all_decile_returns.append(decile_returns)
    
    if not all_decile_returns:
        return pd.DataFrame(columns=["decile", "mean_ret"])
    
    # Combine all dates and compute average across dates
    combined = pd.concat(all_decile_returns, axis=0)
    avg_by_decile = combined.groupby("decile")["ret_fwd1"].mean().reset_index()
    avg_by_decile.columns = ["decile", "mean_ret"]
    
    # Add long-short row
    if len(avg_by_decile) >= 2:
        ls_return = avg_by_decile.iloc[-1]["mean_ret"] - avg_by_decile.iloc[0]["mean_ret"]
        ls_row = pd.DataFrame({"decile": ["L-S"], "mean_ret": [ls_return]})
        avg_by_decile = pd.concat([avg_by_decile, ls_row], ignore_index=True)
    
    return avg_by_decile 