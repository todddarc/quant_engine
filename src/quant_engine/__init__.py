"""
Quant Engine - A minimal quantitative equity portfolio construction engine.

This package provides a complete pipeline from signal generation to portfolio optimization
with strict point-in-time discipline and comprehensive risk management.
"""

__version__ = "0.1.0"

from .data_io import (
    load_prices, load_fundamentals, load_sectors, load_holdings,
    write_holdings, write_trades, unique_dates, next_day_exists
)
from .signals import momentum_12m_1m_gap, value_ep
from .prep import winsorize, zscore, sector_neutralize
from .risk import returns_from_prices, shrink_cov, validate_covariance_matrix, marginal_risk_contribution
from .optimize import mean_variance_opt
from .checks import validate_data, check_turnover, check_sector_exposure, check_schema, check_missingness, aggregate_checks
from .run_day import run, main
from .utils import compute_next_period_returns, cross_sectional_ic, compute_ic_series, summarize_ic, decile_portfolio_returns

__all__ = [
    "load_prices",
    "load_fundamentals", 
    "load_sectors",
    "load_holdings",
    "write_holdings",
    "write_trades",
    "unique_dates",
    "next_day_exists",
    "momentum_12m_1m_gap",
    "value_ep",
    "winsorize",
    "zscore", 
    "sector_neutralize",
    "returns_from_prices",
    "shrink_cov",
    "validate_covariance_matrix",
    "marginal_risk_contribution",
    "mean_variance_opt",
    "validate_data",
    "check_turnover",
    "check_sector_exposure",
    "check_schema",
    "check_missingness",
    "aggregate_checks",
    "run",
    "main",
    "compute_next_period_returns",
    "cross_sectional_ic",
    "compute_ic_series",
    "summarize_ic",
    "decile_portfolio_returns",
] 