#!/usr/bin/env python3
"""
Deterministic synthetic data generator for quant engine testing.

Generates realistic price, fundamental, sector, and holdings data with
point-in-time correctness and survivorship bias simulation.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any


# =============================================================================
# PARAMETERS (easy to tweak)
# =============================================================================

N_TICKERS = 40
N_SECTORS = 10
N_DAYS = 500  # trading days
START_DATE = "2022-01-03"  # a Monday (business day)
SEED = 7

# Factor model parameters
MARKET_MEAN = 0.0002
MARKET_STD = 0.01
SECTOR_STD = 0.003  # Reduced from 0.005
IDIO_STD = 0.008   # Reduced from 0.01

# Skill injection parameters
BETA_MOM = 0.035  # Increased from 0.02
BETA_VAL = 0.018  # Increased from 0.01
EPSILON_SIG = 0.01  # idiosyncratic daily std (~1%)

# Sector names
SECTOR_NAMES = [
    "Tech", "Health", "Energy", "Financials", "Industrials",
    "Consumer", "Utilities", "Materials", "Comm", "RealEstate"
]

# Delisting parameters
N_DELISTINGS = 5
DELISTING_START_RATIO = 1/3  # Start delistings after 1/3 of the period
DELISTING_END_BUFFER = 30    # Stop delistings 30 days before end


def generate_calendar() -> pd.DatetimeIndex:
    """Generate business day calendar."""
    return pd.bdate_range(START_DATE, periods=N_DAYS)


def generate_universe() -> Tuple[List[str], Dict[str, str]]:
    """Generate ticker universe and sector assignments."""
    # Create tickers T000, T001, ..., T039
    tickers = [f"T{i:03d}" for i in range(N_TICKERS)]
    
    # Assign sectors round-robin
    sectors = {}
    for i, ticker in enumerate(tickers):
        sector_idx = i % N_SECTORS
        sectors[ticker] = SECTOR_NAMES[sector_idx]
    
    return tickers, sectors


def generate_delistings(tickers: List[str]) -> Dict[str, int]:
    """Generate random delisting dates for some tickers."""
    np.random.seed(SEED)  # Ensure determinism
    
    # Choose tickers to delist
    delist_tickers = np.random.choice(tickers, size=N_DELISTINGS, replace=False)
    
    # Generate delisting dates
    start_idx = int(N_DAYS * DELISTING_START_RATIO)
    end_idx = N_DAYS - DELISTING_END_BUFFER
    
    delistings = {}
    for ticker in delist_tickers:
        delist_idx = np.random.randint(start_idx, end_idx)
        delistings[ticker] = delist_idx
    
    return delistings


def compute_mom_proxy(prices_wide: pd.DataFrame, day_idx: int) -> pd.Series:
    """
    Compute momentum proxy using (t-21) and (t-252) relative to day_idx-1.
    Returns z-scored Series across alive tickers.
    """
    if day_idx < 252:  # Need at least 252 days of history
        return pd.Series(dtype=float)
    
    # Get prices at t-21 and t-252 (relative to day_idx-1)
    t_gap = day_idx - 21
    t_lookback = day_idx - 252
    
    if t_gap < 0 or t_lookback < 0:
        return pd.Series(dtype=float)
    
    # Get prices at these dates
    prices_gap = prices_wide.iloc[t_gap]
    prices_lookback = prices_wide.iloc[t_lookback]
    
    # Compute momentum: P(t-gap) / P(t-lookback) - 1
    momentum = (prices_gap / prices_lookback) - 1
    
    # Z-score across tickers
    momentum_clean = momentum.dropna()
    if len(momentum_clean) < 3:
        return pd.Series(dtype=float)
    
    mean_mom = momentum_clean.mean()
    std_mom = momentum_clean.std()
    
    if std_mom == 0:
        return pd.Series(0.0, index=momentum.index)
    
    mom_z = (momentum - mean_mom) / std_mom
    return mom_z


def compute_val_proxy(fundamentals_df: pd.DataFrame, prices_on_day: pd.Series, day_date: pd.Timestamp) -> pd.Series:
    """
    Compute value proxy using PIT fundamentals.
    For each ticker, pick the latest fundamentals row with available_asof <= day_date,
    compute E/P = eps_ttm / price_t, then z-score across alive tickers; NaNs -> 0.
    """
    # Convert available_asof to datetime for comparison
    fundamentals_df = fundamentals_df.copy()
    fundamentals_df['available_asof'] = pd.to_datetime(fundamentals_df['available_asof'])
    
    # Filter fundamentals to PIT data
    pit_fundamentals = fundamentals_df[fundamentals_df['available_asof'] <= day_date].copy()
    
    if pit_fundamentals.empty:
        return pd.Series(dtype=float)
    
    # For each ticker, get the latest available fundamental
    latest_fundamentals = pit_fundamentals.loc[pit_fundamentals.groupby('ticker')['available_asof'].idxmax()]
    
    # Compute E/P for each ticker
    value_ratios = []
    for _, row in latest_fundamentals.iterrows():
        ticker = row['ticker']
        if ticker in prices_on_day.index:
            price = prices_on_day[ticker]
            if price > 0:
                ep_ratio = row['eps_ttm'] / price
                value_ratios.append({'ticker': ticker, 'ep_ratio': ep_ratio})
    
    if not value_ratios:
        return pd.Series(dtype=float)
    
    value_df = pd.DataFrame(value_ratios)
    
    # Z-score across tickers
    mean_ep = value_df['ep_ratio'].mean()
    std_ep = value_df['ep_ratio'].std()
    
    if std_ep == 0:
        return pd.Series(0.0, index=prices_on_day.index)
    
    value_z = (value_df['ep_ratio'] - mean_ep) / std_ep
    return pd.Series(value_z.values, index=value_df['ticker'])


def get_tickers_alive_on(prices_wide: pd.DataFrame, date: pd.Timestamp, delistings: Dict[str, int], calendar: pd.DatetimeIndex) -> List[str]:
    """Get list of tickers that are alive on a given date."""
    date_idx = calendar.get_loc(date)
    
    alive_tickers = []
    for ticker in prices_wide.columns:
        if ticker not in delistings or delistings[ticker] > date_idx:
            alive_tickers.append(ticker)
    
    return alive_tickers


def generate_prices(calendar: pd.DatetimeIndex, tickers: List[str], 
                   sectors: Dict[str, str], delistings: Dict[str, int],
                   fundamentals_df: pd.DataFrame, log_alpha_gt: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate price data using factor model with skill injection and delistings."""
    np.random.seed(SEED)  # Ensure determinism
    
    # Initialize prices near 100 with small dispersion
    prices_dict = {}
    for i, ticker in enumerate(tickers):
        prices_dict[ticker] = 100 + np.random.normal(0, 5)  # Small cross-sectional dispersion
    
    # Create wide format for easier manipulation
    prices_wide = pd.DataFrame(index=calendar, columns=tickers)
    
    # Set initial prices
    for ticker in tickers:
        prices_wide.loc[calendar[0], ticker] = prices_dict[ticker]
    
    # Initialize alpha logging list
    alpha_log_rows = []
    
    # Generate factor model returns with skill injection
    for t in range(1, len(calendar)):
        date_t = calendar[t]
        date_tm1 = calendar[t-1]
        
        # Determine active tickers (exclude delisted)
        active = get_tickers_alive_on(prices_wide, date_tm1, delistings, calendar)
        
        if len(active) == 0:
            continue
        
        # Build proxies from information available at t-1
        mom_z = compute_mom_proxy(prices_wide.loc[:, active], day_idx=t)
        val_z = compute_val_proxy(fundamentals_df, 
                                prices_on_day=prices_wide.loc[date_tm1, active],
                                day_date=date_tm1)
        
        # Align proxies to active tickers; fill missing with 0
        mom_z = mom_z.reindex(active).astype(float).fillna(0.0)
        val_z = val_z.reindex(active).astype(float).fillna(0.0)
        
        # Generate components
        market_t = np.random.normal(loc=MARKET_MEAN, scale=MARKET_STD)
        sector_shocks = {s: np.random.normal(loc=0.0, scale=SECTOR_STD) for s in SECTOR_NAMES}
        idio = np.random.normal(loc=0.0, scale=EPSILON_SIG, size=len(active))
        
        # Predictive alpha term based on t-1 signals
        alpha_term = BETA_MOM * mom_z.values + BETA_VAL * val_z.values
        
        # Log alpha ground truth if requested
        if log_alpha_gt:
            for i, ticker in enumerate(active):
                alpha_log_rows.append({
                    "asof_dt": date_t.date().isoformat(),
                    "ticker": ticker,
                    "alpha_gt_used": float(alpha_term[i]),
                    "mom_z_tm1": float(mom_z.loc[ticker]) if ticker in mom_z.index else np.nan,
                    "val_z_tm1": float(val_z.loc[ticker]) if ticker in val_z.index else np.nan,
                    "beta_mom": float(BETA_MOM),
                    "beta_val": float(BETA_VAL),
                })
        
        # Final return for active tickers
        r_t = market_t + np.array([sector_shocks[sectors[ticker]] for ticker in active]) + alpha_term + idio
        
        # Update prices: P_t = P_{t-1} * (1 + r_t), with floor at 1.0
        new_prices = np.maximum(1.0, prices_wide.loc[date_tm1, active].values * (1.0 + r_t))
        prices_wide.loc[date_t, active] = new_prices
    
    # Convert to long format
    prices_data = []
    for date in calendar:
        for ticker in tickers:
            if pd.notna(prices_wide.loc[date, ticker]):
                prices_data.append({
                    'asof_dt': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'close': prices_wide.loc[date, ticker]
                })
    
    prices_df = pd.DataFrame(prices_data)
    prices_df = prices_df.drop_duplicates(subset=['asof_dt', 'ticker'], keep='last')
    
    return prices_df, alpha_log_rows


def generate_fundamentals(calendar: pd.DatetimeIndex, tickers: List[str]) -> pd.DataFrame:
    """Generate fundamental data with PIT correctness."""
    np.random.seed(SEED)  # Ensure determinism
    
    fundamentals_data = []
    
    # Generate quarterly report dates (starting a few quarters before first price date)
    first_date = calendar[0]
    last_date = calendar[-1]
    
    # Start from 3 quarters before first date
    start_quarter = pd.Timestamp(first_date) - pd.DateOffset(months=9)
    start_quarter = start_quarter.replace(day=1).to_period('Q').asfreq('Q').end_time
    
    # Generate all quarter ends in range
    quarter_ends = pd.date_range(start_quarter, last_date, freq='QE')
    
    for ticker in tickers:
        # Initialize fundamental values
        eps_ttm = 5.0 + np.random.normal(0, 2)  # Start around 5 with dispersion
        book_value_ps = 50.0 + np.random.normal(0, 10)  # Start around 50 with dispersion
        
        for quarter_end in quarter_ends:
            # Add drift and noise to fundamentals
            eps_drift = np.random.normal(0, 0.1)  # Small drift
            book_drift = np.random.normal(0, 0.5)  # Small drift
            
            eps_ttm += eps_drift
            book_value_ps += book_drift
            
            # Add some noise
            eps_ttm += np.random.normal(0, 0.2)
            book_value_ps += np.random.normal(0, 1.0)
            
            # Ensure reasonable bounds
            eps_ttm = max(eps_ttm, -2.0)  # Allow small negatives
            book_value_ps = max(book_value_ps, 5.0)  # Keep positive
            
            # Generate reporting lag (60-90 days)
            lag_days = np.random.randint(60, 91)
            available_asof = quarter_end + pd.Timedelta(days=lag_days)
            
            # Only include if available_asof is within our price window
            if available_asof <= last_date:
                fundamentals_data.append({
                    'report_dt': quarter_end.strftime('%Y-%m-%d'),
                    'available_asof': available_asof.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'eps_ttm': eps_ttm,
                    'book_value_ps': book_value_ps
                })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    fundamentals_df = fundamentals_df.drop_duplicates(subset=['report_dt', 'ticker'], keep='last')
    
    return fundamentals_df


def generate_sectors(tickers: List[str], sectors: Dict[str, str]) -> pd.DataFrame:
    """Generate sector mapping data."""
    sectors_data = []
    for ticker in tickers:
        sectors_data.append({
            'ticker': ticker,
            'sector': sectors[ticker]
        })
    
    sectors_df = pd.DataFrame(sectors_data)
    sectors_df = sectors_df.drop_duplicates(subset=['ticker'], keep='last')
    
    return sectors_df


def generate_holdings_prev(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Generate prior holdings for the last available trading day."""
    np.random.seed(SEED)  # Ensure determinism
    
    # Get last date
    last_date = prices_df['asof_dt'].max()
    
    # Get tickers that exist on last date
    last_date_prices = prices_df[prices_df['asof_dt'] == last_date]
    active_tickers = last_date_prices['ticker'].tolist()
    
    # Start with equal weights
    n_active = len(active_tickers)
    equal_weight = 1.0 / n_active
    
    # Add small noise and ensure non-negative
    holdings_data = []
    for ticker in active_tickers:
        noise = np.random.normal(0, equal_weight * 0.1)  # 10% noise
        weight = max(equal_weight + noise, 0.0)
        holdings_data.append({
            'asof_dt': last_date,
            'ticker': ticker,
            'weight': weight
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    # Renormalize to sum to 1
    total_weight = holdings_df['weight'].sum()
    holdings_df['weight'] = holdings_df['weight'] / total_weight
    
    return holdings_df


def generate(outdir: str = "data", log_alpha_gt: bool = True, alpha_gt_path: str = "data/alpha_gt.csv") -> Dict[str, Any]:
    """Generate all synthetic data files."""
    # Set seed for determinism
    np.random.seed(SEED)
    
    # Create output directory
    out_path = Path(outdir)
    out_path.mkdir(exist_ok=True)
    
    # Generate calendar
    calendar = generate_calendar()
    
    # Generate universe
    tickers, sectors = generate_universe()
    
    # Generate delistings
    delistings = generate_delistings(tickers)
    
    # Generate all datasets
    fundamentals_df = generate_fundamentals(calendar, tickers)
    prices_df, alpha_log_rows = generate_prices(calendar, tickers, sectors, delistings, fundamentals_df, log_alpha_gt)
    sectors_df = generate_sectors(tickers, sectors)
    holdings_df = generate_holdings_prev(prices_df)
    
    # Write files
    prices_df.to_csv(out_path / 'prices.csv', index=False)
    fundamentals_df.to_csv(out_path / 'fundamentals.csv', index=False)
    sectors_df.to_csv(out_path / 'sectors.csv', index=False)
    holdings_df.to_csv(out_path / 'holdings_prev.csv', index=False)
    
    # Write alpha ground truth if requested
    if log_alpha_gt and alpha_log_rows:
        alpha_df = pd.DataFrame(alpha_log_rows)
        # Ensure proper dtypes
        alpha_df['asof_dt'] = alpha_df['asof_dt'].astype(str)
        alpha_df['ticker'] = alpha_df['ticker'].astype(str)
        alpha_df['alpha_gt_used'] = alpha_df['alpha_gt_used'].astype(float)
        alpha_df['mom_z_tm1'] = alpha_df['mom_z_tm1'].astype(float)
        alpha_df['val_z_tm1'] = alpha_df['val_z_tm1'].astype(float)
        alpha_df['beta_mom'] = alpha_df['beta_mom'].astype(float)
        alpha_df['beta_val'] = alpha_df['beta_val'].astype(float)
        
        # Sort by asof_dt, ticker
        alpha_df = alpha_df.sort_values(['asof_dt', 'ticker'])
        
        # Create parent directories if needed
        alpha_gt_path = Path(alpha_gt_path)
        alpha_gt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to CSV
        alpha_df.to_csv(alpha_gt_path, index=False)
        print(f"Alpha ground truth written to: {alpha_gt_path}")
    
    # Generate summary
    last_date = prices_df['asof_dt'].max()
    active_tickers = prices_df[prices_df['asof_dt'] == last_date]['ticker'].nunique()
    delisted_tickers = set(tickers) - set(prices_df[prices_df['asof_dt'] == last_date]['ticker'].unique())
    
    summary = {
        'total_tickers': len(tickers),
        'active_tickers': active_tickers,
        'delisted_tickers': len(delisted_tickers),
        'total_days': len(calendar),
        'price_date_range': (prices_df['asof_dt'].min(), prices_df['asof_dt'].max()),
        'fundamentals_date_range': (fundamentals_df['available_asof'].min(), fundamentals_df['available_asof'].max()),
        'delistings': delistings
    }
    
    # Print summary
    print("=== SYNTHETIC DATA GENERATION SUMMARY ===")
    print(f"Total tickers: {summary['total_tickers']}")
    print(f"Active on last day: {summary['active_tickers']}")
    print(f"Delisted: {summary['delisted_tickers']}")
    print(f"Total trading days: {summary['total_days']}")
    print(f"Price date range: {summary['price_date_range'][0]} to {summary['price_date_range'][1]}")
    print(f"Fundamentals available range: {summary['fundamentals_date_range'][0]} to {summary['fundamentals_date_range'][1]}")
    print(f"Delisted tickers: {sorted(delisted_tickers)}")
    print(f"Output directory: {out_path.absolute()}")
    
    return summary


def main():
    global BETA_MOM, BETA_VAL
    
    parser = argparse.ArgumentParser(description="Generate synthetic data for quant engine testing")
    parser.add_argument("--outdir", default="data", help="Output directory (default: data)")
    parser.add_argument("--beta-mom", type=float, default=BETA_MOM, help=f"Momentum beta (default: {BETA_MOM})")
    parser.add_argument("--beta-val", type=float, default=BETA_VAL, help=f"Value beta (default: {BETA_VAL})")
    parser.add_argument("--log-alpha-gt", action="store_true", default=True, help="Log alpha ground truth (default: True)")
    parser.add_argument("--no-log-alpha-gt", dest="log_alpha_gt", action="store_false", help="Disable alpha ground truth logging")
    parser.add_argument("--alpha-gt-path", default="data/alpha_gt.csv", help="Path for alpha ground truth CSV (default: data/alpha_gt.csv)")
    args = parser.parse_args()
    
    # Update global parameters
    BETA_MOM = args.beta_mom
    BETA_VAL = args.beta_val
    
    generate(args.outdir, args.log_alpha_gt, args.alpha_gt_path)


if __name__ == "__main__":
    main() 