"""
Quant Engine - A minimal quantitative equity portfolio construction engine.

This package provides a complete pipeline from signal generation to portfolio optimization
with strict point-in-time discipline and comprehensive risk management.
"""

__version__ = "0.1.0"

from .data_io import load_prices, load_fundamentals, load_sectors, load_prior_holdings
from .signals import generate_momentum_signal, generate_value_signal, combine_signals, momentum_12m_1m_gap
from .prep import prepare_data, calculate_returns, winsorize, zscore, sector_neutralize
from .risk import estimate_covariance, apply_shrinkage, returns_from_prices, shrink_cov
from .optimize import optimize_portfolio, apply_constraints, mean_variance_opt
from .checks import validate_data, check_turnover, check_sector_exposure, check_schema, check_missingness, aggregate_checks
from .run_day import run, main
from .utils import compute_next_period_returns, cross_sectional_ic, compute_ic_series, summarize_ic, decile_portfolio_returns

__all__ = [
    "load_prices",
    "load_fundamentals", 
    "load_sectors",
    "load_prior_holdings",
    "generate_momentum_signal",
    "generate_value_signal",
    "combine_signals",
    "momentum_12m_1m_gap",
    "prepare_data",
    "calculate_returns",
    "winsorize",
    "zscore", 
    "sector_neutralize",
    "estimate_covariance",
    "apply_shrinkage",
    "returns_from_prices",
    "shrink_cov",
    "optimize_portfolio",
    "apply_constraints",
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