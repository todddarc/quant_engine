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
from .checks import check_schema, check_missingness, check_turnover, check_sector_exposure, aggregate_checks
from .utils import compute_next_period_returns, cross_sectional_ic, compute_ic_series, summarize_ic, decile_portfolio_returns


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
    
    # Preprocess signals
    momentum_processed = sector_neutralize(
        zscore(winsorize(momentum_aligned, 0.01, 0.99)),
        sectors_ser
    )
    
    value_processed = sector_neutralize(
        zscore(winsorize(value_aligned, 0.01, 0.99)),
        sectors_ser
    )
    
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
                    
                    # Same preprocessing as today
                    momentum_processed = sector_neutralize(
                        zscore(winsorize(momentum_aligned, 0.01, 0.99)),
                        sectors_ser
                    )
                    
                    value_processed = sector_neutralize(
                        zscore(winsorize(value_aligned, 0.01, 0.99)),
                        sectors_ser
                    )
                    
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
                ok_to_trade: bool, output_dir: Path) -> None:
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
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load config
    config = load_config(config_path)
    logging.info(f"Loaded config from {config_path}")
    
    # Load data via data_io
    prices_df, fundamentals_df, sectors_ser, prev_weights, asof_dt = _load_inputs(config, asof)
    logging.info(f"Loaded data: {len(prices_df)} price records, {len(fundamentals_df)} fundamental records")
    logging.info(f"Validated asof date: {asof}")
    
    # Build signals
    alpha = build_signals(prices_df, fundamentals_df, sectors_ser, asof_dt, config)
    logging.info(f"Built signals for {len(alpha)} tickers")
    
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
    
    # Write outputs via data_io
    outdir = Path(config.get("paths", {}).get("output_dir", "data/outputs"))
    path_hold = write_holdings(outdir, asof, weights)
    path_tr = write_trades(outdir, asof, weights, prev_weights)
    write_report(asof, validation_metrics, opt_diagnostics, check_results, ok_to_trade, outdir)
    logging.info("Wrote output files")
    
    # Return summary
    report_path = outdir / 'reports' / f'{asof}_report.txt'
    return {
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