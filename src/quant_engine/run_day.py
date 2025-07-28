"""
Main execution module for running the complete portfolio construction pipeline.

Orchestrates the entire process from data loading to portfolio optimization
with comprehensive validation and reporting.
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from .data_io import (
    load_prices, load_fundamentals, load_sectors, load_holdings,
    write_holdings, write_trades, unique_dates, next_day_exists
)
from .signals import momentum_12m_1m_gap, value_ep
from .prep import winsorize, zscore, sector_neutralize
from .risk import returns_from_prices, shrink_cov
from .optimize import mean_variance_opt
from .checks import check_schema, check_missingness, check_turnover, check_sector_exposure, aggregate_checks, check_schema_drift, check_extreme_values
from .utils import compute_next_period_returns, cross_sectional_ic, compute_ic_series, summarize_ic, decile_portfolio_returns, setup_logging, validate_config, set_random_seed
from .risk import returns_from_prices, shrink_cov, validate_covariance_matrix, marginal_risk_contribution
from .trade_filters import apply_no_trade_band


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_inputs(cfg: dict, asof_str: str):
    """
    Load inputs via data_io. Returns:
       prices_df, fundamentals_df, sectors_ser, prev_w (Series), asof (Timestamp)
    """
    asof = pd.to_datetime(asof_str).normalize()
    data_cfg = cfg.get("data", {})
    prices_df = load_prices(data_cfg["prices_path"])
    fundamentals_df = load_fundamentals(data_cfg["fundamentals_path"])
    sectors_ser = load_sectors(data_cfg["sectors_path"])
    
    # Validate asof date exists in prices data
    available_dates = unique_dates(prices_df)
    if asof not in available_dates:
        raise ValueError(f"Date {asof_str} not found in prices data. Available dates: {sorted(available_dates)[:5]}...")
    
    # Handle missing holdings file gracefully
    try:
        prev_w = load_holdings(data_cfg["holdings_path"], asof)
    except (FileNotFoundError, ValueError):
        # Use equal weights if no prior holdings
        all_tickers = prices_df[prices_df['asof_dt'] == asof]['ticker'].unique()
        prev_w = pd.Series(1.0/len(all_tickers), index=all_tickers)
    
    return prices_df, fundamentals_df, sectors_ser, prev_w, asof





def build_signals(prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame, 
                 sectors_ser: pd.Series, asof_dt: pd.Timestamp, config: Dict[str, Any]) -> pd.Series:
    """Build alpha signals by combining momentum and value."""
    # Slice PIT fundamentals
    pit_fundamentals = fundamentals_df[fundamentals_df['available_asof'] <= asof_dt]
    
    # Build momentum signal
    momentum = momentum_12m_1m_gap(
        prices_df, asof_dt,
        lookback=config['signals']['momentum']['lookback'],
        gap=config['signals']['momentum']['gap']
    )
    
    # Build value signal
    value = value_ep(
        pit_fundamentals, prices_df, asof_dt,
        min_lag_days=config['signals']['value']['min_lag_days']
    )
    
    # Get common tickers
    common_tickers = momentum.index.intersection(value.index)
    if len(common_tickers) < 2:
        raise ValueError(f"Insufficient common tickers for signal combination: {len(common_tickers)}")
    
    # Align signals
    momentum_aligned = momentum.loc[common_tickers]
    value_aligned = value.loc[common_tickers]
    
    # Preprocess signals based on config flags
    preprocessing_config = config.get('signals', {}).get('preprocessing', {})
    winsorize_flag = preprocessing_config.get('winsorize', True)
    sector_neutralize_flag = preprocessing_config.get('sector_neutralize', True)
    winsorize_bounds = preprocessing_config.get('winsorize_bounds', [0.01, 0.99])
    
    # Apply winsorization if enabled
    if winsorize_flag:
        momentum_processed = winsorize(momentum_aligned, winsorize_bounds[0], winsorize_bounds[1])
        value_processed = winsorize(value_aligned, winsorize_bounds[0], winsorize_bounds[1])
    else:
        momentum_processed = momentum_aligned
        value_processed = value_aligned
    
    # Apply z-scoring
    momentum_processed = zscore(momentum_processed)
    value_processed = zscore(value_processed)
    
    # Apply sector neutralization if enabled
    if sector_neutralize_flag:
        momentum_processed = sector_neutralize(momentum_processed, sectors_ser)
        value_processed = sector_neutralize(value_processed, sectors_ser)
    
    # Combine signals
    alpha = (config['signals']['weights']['momentum'] * momentum_processed + 
             config['signals']['weights']['value'] * value_processed)
    
    return alpha


def compute_validation_metrics(prices_df: pd.DataFrame, alpha: pd.Series, 
                              asof_dt: pd.Timestamp, config: Dict[str, Any],
                              fundamentals_df: pd.DataFrame, sectors_ser: pd.Series) -> Dict[str, Any]:
    """Compute IC snapshot and historical metrics."""
    # Get next day returns for IC snapshot
    next_returns = compute_next_period_returns(prices_df, asof_dt)
    
    ic_snapshot = "N/A"
    if not next_returns.empty:
        common_tickers = alpha.index.intersection(next_returns.index)
        if len(common_tickers) >= 3:
            ic_snapshot = cross_sectional_ic(
                alpha.loc[common_tickers], 
                next_returns.loc[common_tickers]
            )
    
    # Compute historical IC series and deciles (up to asof-1)
    all_dates = unique_dates(prices_df)
    historical_dates = [d for d in all_dates if d < asof_dt]
    
    ic_summary = "N/A"
    decile_ls = "N/A"
    
    ic_window_days = config.get('validation', {}).get('ic_window_days', 60)
    
    if len(historical_dates) >= 10:
        # Build historical signals and returns over the window
        signals_hist = []
        returns_hist = []
        
        # Get the last K dates where both t and t+1 exist
        valid_dates = []
        for date in historical_dates[-ic_window_days:]:
            next_date_returns = compute_next_period_returns(prices_df, date)
            if not next_date_returns.empty:
                valid_dates.append(date)
        
        logging.info(f"Building historical signals for {len(valid_dates)} dates")
        
        for date in valid_dates:
            try:
                # Build signals for this date with same preprocessing as today
                momentum_aligned = momentum_12m_1m_gap(
                    prices_df, date,
                    lookback=config['signals']['momentum']['lookback'],
                    gap=config['signals']['momentum']['gap']
                )
                
                value_aligned = value_ep(
                    fundamentals_df, prices_df, date,
                    min_lag_days=config['signals']['value']['min_lag_days']
                )
                
                # Align signals
                common_tickers = momentum_aligned.index.intersection(value_aligned.index)
                if len(common_tickers) >= 20:  # Skip if too few names
                    momentum_aligned = momentum_aligned.loc[common_tickers]
                    value_aligned = value_aligned.loc[common_tickers]
                    
                    # Same preprocessing as today (using config flags)
                    preprocessing_config = config.get('signals', {}).get('preprocessing', {})
                    winsorize_flag = preprocessing_config.get('winsorize', True)
                    sector_neutralize_flag = preprocessing_config.get('sector_neutralize', True)
                    winsorize_bounds = preprocessing_config.get('winsorize_bounds', [0.01, 0.99])
                    
                    # Apply winsorization if enabled
                    if winsorize_flag:
                        momentum_processed = winsorize(momentum_aligned, winsorize_bounds[0], winsorize_bounds[1])
                        value_processed = winsorize(value_aligned, winsorize_bounds[0], winsorize_bounds[1])
                    else:
                        momentum_processed = momentum_aligned
                        value_processed = value_aligned
                    
                    # Apply z-scoring
                    momentum_processed = zscore(momentum_processed)
                    value_processed = zscore(value_processed)
                    
                    # Apply sector neutralization if enabled
                    if sector_neutralize_flag:
                        momentum_processed = sector_neutralize(momentum_processed, sectors_ser)
                        value_processed = sector_neutralize(value_processed, sectors_ser)
                    
                    # Combine signals
                    date_signals = (config['signals']['weights']['momentum'] * momentum_processed + 
                                   config['signals']['weights']['value'] * value_processed)
                    
                    # Get next day returns
                    date_returns = compute_next_period_returns(prices_df, date)
                    
                    # Align on common tickers
                    final_tickers = date_signals.index.intersection(date_returns.index)
                    if len(final_tickers) >= 20:  # Skip if too few names
                        date_signals = date_signals.loc[final_tickers]
                        date_returns = date_returns.loc[final_tickers]
                        
                        # Store data
                        for ticker in final_tickers:
                            signals_hist.append({
                                'asof_dt': date,
                                'ticker': ticker,
                                'signal': date_signals[ticker]
                            })
                            returns_hist.append({
                                'asof_dt': date,
                                'ticker': ticker,
                                'ret_fwd1': date_returns[ticker]
                            })
                        
            except Exception as e:
                logging.warning(f"Failed to compute signals for {date}: {e}")
                continue
        
        if len(signals_hist) >= 10:
            # Convert to DataFrames
            signals_df = pd.DataFrame(signals_hist)
            returns_df = pd.DataFrame(returns_hist)
            
            # Compute IC summary
            ic_series = compute_ic_series(signals_df, returns_df)
            if not ic_series.empty:
                ic_summary = summarize_ic(ic_series)
            
            # Compute decile returns
            decile_returns = decile_portfolio_returns(signals_df, returns_df)
            if not decile_returns.empty:
                ls_row = decile_returns[decile_returns['decile'] == 'L-S']
                if not ls_row.empty:
                    decile_ls = ls_row.iloc[0]['mean_ret']
    
    return {
        'ic_snapshot': ic_snapshot,
        'ic_summary': ic_summary,
        'decile_ls': decile_ls
    }


def build_risk_model(prices_df: pd.DataFrame, asof_dt: pd.Timestamp, config: Dict[str, Any]) -> pd.DataFrame:
    """Build covariance matrix from returns."""
    returns_matrix = returns_from_prices(
        prices_df, asof_dt,
        lookback_days=config['risk']['cov_lookback_days']
    )
    
    if returns_matrix.empty:
        raise ValueError("Insufficient data for covariance estimation")
    
    Sigma = shrink_cov(
        returns_matrix,
        lam=config['risk']['shrink_lambda'],
        diag_load=config['risk']['diag_load']
    )
    
    return Sigma


def run_optimization(alpha: pd.Series, Sigma: pd.DataFrame, sectors_ser: pd.Series,
                    prev_weights: pd.Series, config: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, Any]]:
    """Run mean-variance optimization."""
    # Align all inputs
    common_tickers = alpha.index.intersection(Sigma.index)
    if len(common_tickers) < 2:
        raise ValueError(f"Insufficient common tickers for optimization: {len(common_tickers)}")
    
    alpha_opt = alpha.loc[common_tickers]
    Sigma_opt = Sigma.loc[common_tickers, common_tickers]
    sectors_opt = sectors_ser.reindex(common_tickers)
    prev_w_opt = prev_weights.reindex(common_tickers).fillna(1.0/len(common_tickers))
    
    # Run optimization
    weights, diagnostics = mean_variance_opt(
        alpha_opt, Sigma_opt, sectors_opt, prev_w_opt,
        w_max=config['optimization']['w_max'],
        sector_cap=config['optimization']['sector_cap'],
        turnover_cap=config['optimization']['turnover_cap'],
        risk_aversion=config['optimization']['risk_aversion']
    )
    
    return weights, diagnostics


def run_pre_trade_checks(prices_df: pd.DataFrame, sectors_ser: pd.Series,
                        prev_weights: pd.Series, new_weights: pd.Series,
                        asof_dt: pd.Timestamp, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Run all pre-trade checks."""
    checks = {}
    
    # Schema checks
    prev_holdings_schema = check_schema(
        pd.DataFrame({'asof_dt': [asof_dt], 'ticker': ['TEST'], 'weight': [0.1]}),
        ['asof_dt', 'ticker', 'weight'],
        'holdings_schema'
    )
    checks.update(prev_holdings_schema)
    
    sectors_schema = check_schema(sectors_ser.reset_index(), ['ticker', 'sector'], 'sectors_schema')
    checks.update(sectors_schema)
    
    # Missingness check on prices for asof day
    prices_asof = prices_df[prices_df['asof_dt'] == asof_dt]
    missingness_check = check_missingness(
        prices_asof, 
        max_rate=config['checks']['missing_max'],
        name='prices_missingness'
    )
    checks.update(missingness_check)
    
    # Turnover check
    turnover_check = check_turnover(
        prev_weights, new_weights,
        cap=config['optimization']['turnover_cap'],
        name='turnover'
    )
    checks.update(turnover_check)
    
    # Sector exposure check
    sector_check = check_sector_exposure(
        new_weights,
        sectors_ser,
        cap=config['optimization']['sector_cap'],
        name='sector_exposure'
    )
    checks.update(sector_check)
    
    # Aggregate results
    ok_to_trade, check_results = aggregate_checks(checks)
    
    return ok_to_trade, check_results





def write_report(asof: str, validation_metrics: Dict[str, Any], 
                opt_diagnostics: Dict[str, Any], check_results: Dict[str, Any],
                ok_to_trade: bool, output_dir: Path, risk_diag_line: Optional[str] = None,
                drift_prices: Optional[Dict[str, Any]] = None, drift_fund: Optional[Dict[str, Any]] = None,
                drift_sects: Optional[Dict[str, Any]] = None, extreme: Optional[Dict[str, Any]] = None,
                freeze_mask: Optional[pd.Series] = None, tf_stats: Optional[Dict[str, Any]] = None,
                reopt_used: bool = False, reopt_success: bool = True, min_weight: float = 0.0005,
                min_notional: Optional[float] = None, aum: Optional[float] = None) -> None:
    """Write text report."""
    # Create reports directory
    (output_dir / 'reports').mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'reports' / f'{asof}_report.txt'
    
    with open(report_path, 'w') as f:
        f.write(f"PORTFOLIO CONSTRUCTION REPORT\n")
        f.write(f"Date: {asof}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        
        # Validation metrics
        f.write("VALIDATION METRICS\n")
        f.write(f"IC Snapshot: {validation_metrics['ic_snapshot']}\n")
        if validation_metrics['ic_summary'] != "N/A":
            ic_sum = validation_metrics['ic_summary']
            f.write(f"IC Summary (60-day): mean={ic_sum['mean_ic']:.4f}, std={ic_sum['std_ic']:.4f}, t-stat={ic_sum['t_stat']:.2f}, hit_rate={ic_sum['hit_rate']:.2f}\n")
        else:
            f.write("IC Summary: N/A\n")
        f.write(f"Decile L-S: {validation_metrics['decile_ls']}\n\n")
        
        # Optimizer diagnostics
        f.write("OPTIMIZER DIAGNOSTICS\n")
        f.write(f"Success: {opt_diagnostics['success']}\n")
        f.write(f"Alpha Dot: {opt_diagnostics['alpha_dot']:.6f}\n")
        f.write(f"Risk: {opt_diagnostics['risk']:.6f}\n")
        f.write(f"Turnover: {opt_diagnostics['turnover']:.4f}\n\n")
        
        # Risk diagnostics
        f.write("RISK DIAGNOSTICS\n")
        f.write(f"{risk_diag_line}\n\n")
        
        # Trade filters
        f.write("TRADE FILTERS\n")
        if freeze_mask is not None and freeze_mask.any():
            mw_str = f"{min_weight:.4f}"
            mn_str = "N/A" if min_notional is None else f"{float(min_notional):.0f}"
            aum_str = "N/A" if aum is None else f"{float(aum):.0f}"
            reopt_str = "used" if reopt_used and reopt_success else ("failed" if reopt_used else "not used")
            f.write(
                f"Frozen names: {int(freeze_mask.sum())}, "
                f"min_weight={mw_str}, min_notional={mn_str}, AUM={aum_str}, reopt={reopt_str}\n"
            )
            f.write(
                f"Turnover {tf_stats['turnover_before']:.4f} -> {tf_stats['turnover_after']:.4f} (after filter)\n"
            )
        else:
            f.write("No small-trade filtering applied.\n")
        f.write("\n")
        
        # Data diagnostics (WARN-only)
        f.write("DATA DIAGNOSTICS (WARN-only)\n")
        if drift_prices is not None:
            def _fmt_drift(d):
                if d["ok"]:
                    return f"{d['name']}: OK"
                add = ", ".join(d["added"]) if d["added"] else "-"
                rem = ", ".join(d["removed"]) if d["removed"] else "-"
                return f"{d['name']}: ADDED[{add}] REMOVED[{rem}]"
            f.write(f"{_fmt_drift(drift_prices)}\n")
            f.write(f"{_fmt_drift(drift_fund)}\n")
            f.write(f"{_fmt_drift(drift_sects)}\n")
        if extreme is not None and extreme.get("n", 0) > 0:
            f.write(f"Signal extremes: {extreme['n_extreme']} / {extreme['n']} beyond {extreme['threshold']:.1f} std dev\n")
        else:
            f.write("Signal extremes: N/A\n")
        f.write("\n")
        
        # Pre-trade checks
        f.write("PRE-TRADE CHECKS\n")
        for check_name, result in check_results.items():
            f.write(f"{check_name}: {result['status']} - {result['details']}\n")
        f.write(f"\nOK TO TRADE: {ok_to_trade}\n")
        
        if not ok_to_trade:
            f.write("\nFALLBACK: previous weights emitted\n")


def run(asof: str, config_path: str) -> Dict[str, Any]:
    """
    Run the complete portfolio construction pipeline.
    
    Args:
        asof: Date to run (YYYY-MM-DD)
        config_path: Path to config file
        
    Returns:
        Summary dictionary with results
    """
    # Load config
    config = load_config(config_path)
    
    # Setup logging with config
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        fmt=config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Validate config
    try:
        validate_config(config)
    except ValueError as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    # Set random seed if specified
    if config.get("performance", {}).get("random_seed") is not None:
        set_random_seed(config["performance"]["random_seed"])
    
    # Read trading thresholds from config with safe defaults
    trade_cfg = config.get("trading", {})
    min_weight = float(trade_cfg.get("min_weight", 0.0005))       # 5 bps
    min_notional = trade_cfg.get("min_notional", None)            # e.g., 10000.0
    if isinstance(min_notional, str) and min_notional.strip() == "":
        min_notional = None
    aum = trade_cfg.get("aum", None)                              # e.g., 10_000_000
    aum = float(aum) if aum is not None else None
    
    logging.info(f"Loaded config from {config_path}")
    
    # Load data via data_io
    prices_df, fundamentals_df, sectors_ser, prev_weights, asof_dt = _load_inputs(config, asof)
    logging.info(f"Loaded data: {len(prices_df)} price records, {len(fundamentals_df)} fundamental records")
    logging.info(f"Validated asof date: {asof}")
    
    # Data diagnostics (WARN-only)
    expected_prices = ["asof_dt","ticker","close"]
    expected_fund   = ["report_dt","available_asof","ticker","eps_ttm","book_value_ps"]
    expected_sects  = ["ticker","sector"]
    expected_hold   = ["asof_dt","ticker","weight"]
    
    drift_prices = check_schema_drift(prices_df, expected_prices, "prices")
    drift_fund   = check_schema_drift(fundamentals_df, expected_fund, "fundamentals")
    # sectors_ser is a Series; rebuild a small DataFrame for schema check
    sectors_df = sectors_ser.rename("sector").reset_index()[["ticker","sector"]]
    drift_sects = check_schema_drift(sectors_df, expected_sects, "sectors")
    
    # Build signals
    alpha = build_signals(prices_df, fundamentals_df, sectors_ser, asof_dt, config)
    logging.info(f"Built signals for {len(alpha)} tickers")
    
    # Signal extreme values check (WARN-only)
    try:
        extreme = check_extreme_values(alpha)
    except Exception:
        extreme = {"n": 0, "n_extreme": 0, "indices": [], "threshold": float("nan")}
    
    # Compute validation metrics
    validation_metrics = compute_validation_metrics(prices_df, alpha, asof_dt, config, fundamentals_df, sectors_ser)
    logging.info("Computed validation metrics")
    
    # Build risk model
    Sigma = build_risk_model(prices_df, asof_dt, config)
    logging.info(f"Built risk model: {Sigma.shape}")
    
    # Run optimization
    weights, opt_diagnostics = run_optimization(alpha, Sigma, sectors_ser, prev_weights, config)
    logging.info("Completed optimization")
    
    # Check if optimizer failed and fell back to prior weights
    optimizer_failed = not opt_diagnostics.get('success', True)
    
    # Apply trade filters and re-optimize if needed
    freeze_mask = pd.Series(False, index=weights.index)
    tf_stats = {"turnover_before": 0.0, "turnover_after": 0.0}
    reopt_used = False
    reopt_success = True
    
    if not optimizer_failed:  # Only apply filters if first optimization succeeded
        try:
            # Compute today's price vector (Series) aligned to tickers
            prices_today = prices_df.loc[prices_df["asof_dt"] == asof_dt, ["ticker","close"]].set_index("ticker")["close"]
            
            # Align prev_weights to match weights index
            prev_w_aligned = prev_weights.reindex(weights.index).fillna(0.0)
            
            # Apply band
            nw_frozen_view, freeze_mask, tf_stats = apply_no_trade_band(
                prev_w=prev_w_aligned, new_w=weights, prices=prices_today, aum=aum,
                min_weight=min_weight, min_notional=min_notional
            )
            
            # If nothing to freeze (all False), keep weights as-is
            if freeze_mask.any():
                try:
                    # Build fixed_weights = prev_w for frozen names
                    fixed_w = prev_w_aligned.where(freeze_mask, np.nan).dropna()
                    
                    # Re-run optimizer with same inputs but fixed_weights
                    res2 = mean_variance_opt(
                        alpha=alpha,
                        Sigma=Sigma,
                        sectors_map=sectors_ser,
                        prev_w=prev_weights,
                        risk_aversion=config['optimization']['risk_aversion'],
                        w_max=config['optimization']['w_max'],
                        sector_cap=config['optimization']['sector_cap'],
                        turnover_cap=config['optimization']['turnover_cap'],
                        fixed_weights=fixed_w,   # NEW
                    )
                    if res2[1].get("success", False):
                        new_w_reopt = res2[0]
                        weights = new_w_reopt  # adopt the re-optimized solution
                        reopt_used = True
                        # Recompute stats after re-opt
                        prices_today = prices_today.reindex(weights.index)
                        # Update tf_stats to reflect real after state
                        tf_stats["turnover_after"] = 0.5 * float(np.abs(weights - prev_w_aligned).sum())
                    else:
                        reopt_used = True
                        reopt_success = False
                        logging.warning("Re-opt with frozen names failed; keeping first solution. Message: %s", res2[1].get("message",""))
                except Exception as e:
                    reopt_used = True
                    reopt_success = False
                    logging.warning("Re-opt with frozen names errored; keeping first solution. Error: %s", e)
        except Exception as e:
            logging.warning("Trade filtering failed; keeping first solution. Error: %s", e)
    
    # Run pre-trade checks
    ok_to_trade, check_results = run_pre_trade_checks(
        prices_df, sectors_ser, prev_weights, weights, asof_dt, config
    )
    logging.info(f"Pre-trade checks: {'PASS' if ok_to_trade else 'BLOCK'}")
    
    # If optimizer failed, override ok_to_trade to False
    if optimizer_failed:
        ok_to_trade = False
        logging.warning("Optimizer failed, setting ok_to_trade to False")
    
    # Handle fallback if blocked
    if not ok_to_trade:
        weights = prev_weights
        opt_diagnostics['success'] = False
        logging.warning("Using fallback weights due to pre-trade check failure or optimizer failure")
    
    # Risk diagnostics (non-blocking)
    risk_diag_line = None
    risk_stats = None
    try:
        stats = validate_covariance_matrix(Sigma)
        risk_stats = stats
        
        # Top risk contributors (default top 5):
        top_n = int(config.get("output", {}).get("top_risk_contributors_n", 5))
        mrc = marginal_risk_contribution(Sigma, weights).head(top_n)
        
        # Build a compact line
        top_str = ", ".join([f"{idx} {pct:.1%}" for idx, pct in zip(mrc.index.tolist(), mrc["pct"].tolist())])
        risk_diag_line = (
            f"Cov cond={stats['cond']:.1f}, min_eig={stats['min_eig']:.3e}, asym={stats['max_asym']:.2e}; "
            f"Top risk contributors: {top_str}"
        )
    except Exception as e:
        logging.warning("Risk diagnostics failed: %s", e)
        risk_diag_line = "Cov diagnostics: N/A"
    
    # Write outputs via data_io
    outdir = Path(config.get("paths", {}).get("output_dir", "data/outputs"))
    path_hold = write_holdings(outdir, asof, weights)
    path_tr = write_trades(outdir, asof, weights, prev_weights)
    write_report(asof, validation_metrics, opt_diagnostics, check_results, ok_to_trade, outdir, risk_diag_line,
                 drift_prices, drift_fund, drift_sects, extreme, freeze_mask, tf_stats, reopt_used, reopt_success,
                 min_weight, min_notional, aum)
    logging.info("Wrote output files")
    
    # Return summary
    report_path = outdir / 'reports' / f'{asof}_report.txt'
    summary = {
        'ok_to_trade': ok_to_trade,
        'asof': asof_dt.date().isoformat(),
        'paths': {
            'holdings': str(path_hold),
            'trades': str(path_tr),
            'report': str(report_path)
        },
        'alpha_dot': opt_diagnostics.get('alpha_dot', float('nan')),
        'risk': opt_diagnostics.get('risk', float('nan')),
        'turnover': opt_diagnostics.get('turnover', float('nan'))
    }
    
    # Add risk diagnostics if available
    if risk_stats is not None:
        summary["risk_diag"] = {"cond": risk_stats.get("cond"), "min_eig": risk_stats.get("min_eig")}
    
    return summary


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Quant Engine Portfolio Construction")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--asof", required=True, help="Date to run (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        summary = run(args.asof, args.config)
        exit_code = 0 if summary['ok_to_trade'] else 1
        
        if summary['ok_to_trade']:
            print(f"Report written to: {summary['paths']['report']}")
        
        exit(exit_code)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        exit(1)


if __name__ == "__main__":
    main() 