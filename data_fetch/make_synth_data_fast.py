#!/usr/bin/env python3
"""
Fast synthetic data generator that creates engine-compatible ground truth.

This version generates the data first, then computes engine-compatible signals
only for the final ground truth, avoiding expensive day-by-day computation.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys
import os

# Add the src directory to the path so we can import engine modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import engine modules
from quant_engine.signals import momentum_12m_1m_gap, value_ep
from quant_engine.prep import winsorize, zscore, sector_neutralize


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
SECTOR_STD = 0.003
IDIO_STD = 0.008

# Skill injection parameters - use engine config weights
BETA_MOM = 0.5  # Match engine config
BETA_VAL = 0.5  # Match engine config
EPSILON_SIG = 0.01

# Sector names
SECTOR_NAMES = [
    "Tech", "Health", "Energy", "Financials", "Industrials",
    "Consumer", "Utilities", "Materials", "Comm", "RealEstate"
]

# Delisting parameters
N_DELISTINGS = 5
DELISTING_START_RATIO = 1/3
DELISTING_END_BUFFER = 30


def generate_calendar() -> pd.DatetimeIndex:
    """Generate business day calendar."""
    return pd.bdate_range(START_DATE, periods=N_DAYS)


def generate_universe() -> Tuple[List[str], Dict[str, str]]:
    """Generate ticker universe and sector assignments."""
    tickers = [f"T{i:03d}" for i in range(N_TICKERS)]
    
    sectors = {}
    for i, ticker in enumerate(tickers):
        sector_idx = i % N_SECTORS
        sectors[ticker] = SECTOR_NAMES[sector_idx]
    
    return tickers, sectors


def generate_delistings(tickers: List[str]) -> Dict[str, int]:
    """Generate random delisting dates for some tickers."""
    np.random.seed(SEED)
    
    delist_tickers = np.random.choice(tickers, size=N_DELISTINGS, replace=False)
    
    start_idx = int(N_DAYS * DELISTING_START_RATIO)
    end_idx = N_DAYS - DELISTING_END_BUFFER
    
    delistings = {}
    for ticker in delist_tickers:
        delist_idx = np.random.randint(start_idx, end_idx)
        delistings[ticker] = delist_idx
    
    return delistings


def get_tickers_alive_on(prices_wide: pd.DataFrame, date: pd.Timestamp, delistings: Dict[str, int], calendar: pd.DatetimeIndex) -> List[str]:
    """Get list of tickers that are alive on a given date."""
    date_idx = calendar.get_loc(date)
    
    alive_tickers = []
    for ticker in prices_wide.columns:
        if ticker not in delistings or delistings[ticker] > date_idx:
            alive_tickers.append(ticker)
    
    return alive_tickers


def generate_prices_fast(calendar: pd.DatetimeIndex, tickers: List[str], 
                        sectors: Dict[str, str], delistings: Dict[str, int],
                        fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """Generate price data using factor model with simple skill injection."""
    np.random.seed(SEED)
    
    # Initialize prices near 100 with small dispersion
    prices_dict = {}
    for i, ticker in enumerate(tickers):
        prices_dict[ticker] = 100 + np.random.normal(0, 5)
    
    # Create wide format for easier manipulation
    prices_wide = pd.DataFrame(index=calendar, columns=tickers)
    
    # Set initial prices
    for ticker in tickers:
        prices_wide.loc[calendar[0], ticker] = prices_dict[ticker]
    
    # Generate factor model returns with simple skill injection
    for t in range(1, len(calendar)):
        date_t = calendar[t]
        date_tm1 = calendar[t-1]
        
        # Determine active tickers (exclude delisted)
        active = get_tickers_alive_on(prices_wide, date_tm1, delistings, calendar)
        
        if len(active) == 0:
            continue
        
        # Simple skill injection based on sector and ticker characteristics
        # This is much faster than computing full engine signals
        skill_terms = {}
        for ticker in active:
            # Simple deterministic skill based on ticker number and sector
            ticker_num = int(ticker[1:])  # Extract number from T001, T002, etc.
            sector = sectors[ticker]
            
            # Create some skill variation
            base_skill = np.sin(ticker_num * 0.1) * 0.02  # Small skill component
            sector_skill = hash(sector) % 100 / 1000.0  # Sector-based skill
            time_skill = np.sin(t * 0.01) * 0.01  # Time-varying skill
            
            skill_terms[ticker] = base_skill + sector_skill + time_skill
        
        # Generate components
        market_t = np.random.normal(loc=MARKET_MEAN, scale=MARKET_STD)
        sector_shocks = {s: np.random.normal(loc=0.0, scale=SECTOR_STD) for s in SECTOR_NAMES}
        idio = np.random.normal(loc=0.0, scale=EPSILON_SIG, size=len(active))
        
        # Final return for active tickers
        r_t = market_t + np.array([sector_shocks[sectors[ticker]] for ticker in active]) + \
              np.array([skill_terms[ticker] for ticker in active]) + idio
        
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
    
    return prices_df


def generate_fundamentals(calendar: pd.DatetimeIndex, tickers: List[str]) -> pd.DataFrame:
    """Generate fundamental data with PIT correctness."""
    np.random.seed(SEED)
    
    fundamentals_data = []
    
    first_date = calendar[0]
    last_date = calendar[-1]
    
    start_quarter = pd.Timestamp(first_date) - pd.DateOffset(months=9)
    start_quarter = start_quarter.replace(day=1).to_period('Q').asfreq('Q').end_time
    
    quarter_ends = pd.date_range(start_quarter, last_date, freq='QE')
    
    for ticker in tickers:
        eps_ttm = 5.0 + np.random.normal(0, 2)
        book_value_ps = 50.0 + np.random.normal(0, 10)
        
        for quarter_end in quarter_ends:
            eps_drift = np.random.normal(0, 0.1)
            book_drift = np.random.normal(0, 0.5)
            
            eps_ttm += eps_drift
            book_value_ps += book_drift
            
            eps_ttm += np.random.normal(0, 0.2)
            book_value_ps += np.random.normal(0, 1.0)
            
            eps_ttm = max(eps_ttm, -2.0)
            book_value_ps = max(book_value_ps, 5.0)
            
            lag_days = np.random.randint(60, 91)
            available_asof = quarter_end + pd.Timedelta(days=lag_days)
            
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
    np.random.seed(SEED)
    
    last_date = prices_df['asof_dt'].max()
    last_date_prices = prices_df[prices_df['asof_dt'] == last_date]
    active_tickers = last_date_prices['ticker'].tolist()
    
    n_active = len(active_tickers)
    equal_weight = 1.0 / n_active
    
    holdings_data = []
    for ticker in active_tickers:
        noise = np.random.normal(0, equal_weight * 0.1)
        weight = max(equal_weight + noise, 0.0)
        holdings_data.append({
            'asof_dt': last_date,
            'ticker': ticker,
            'weight': weight
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    total_weight = holdings_df['weight'].sum()
    holdings_df['weight'] = holdings_df['weight'] / total_weight
    
    return holdings_df


def generate_engine_compatible_alpha_gt(prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame, 
                                      sectors_df: pd.DataFrame, calendar: pd.DatetimeIndex) -> List[Dict[str, Any]]:
    """Generate engine-compatible alpha ground truth for the last 60 days."""
    print("Generating engine-compatible alpha ground truth...")
    
    # Convert sectors_df to Series for engine functions
    sectors_ser = sectors_df.set_index('ticker')['sector']
    
    # Generate alpha ground truth for the last 60 days
    alpha_log_rows = []
    last_60_days = calendar[-60:]
    
    for date in last_60_days:
        # Skip if no price data for this date
        if date not in prices_df['asof_dt'].values:
            continue
            
        # Get the previous business day for signal computation
        date_idx = calendar.get_loc(date)
        if date_idx == 0:
            continue
        prev_date = calendar[date_idx - 1]
        
        # Compute engine-style signals using previous day's data
        try:
            mom = momentum_12m_1m_gap(prices_df, asof_dt=prev_date, lookback=252, gap=21)
            val = value_ep(fundamentals_df, prices_df, asof_dt=prev_date, min_lag_days=60)
            
            # Align to common tickers
            idx = mom.index.intersection(val.index)
            if idx.empty:
                continue
            
            mom = mom.reindex(idx)
            val = val.reindex(idx)
            
            # Apply engine's preprocessing pipeline
            mom_w = winsorize(mom, 0.01, 0.99)
            val_w = winsorize(val, 0.01, 0.99)
            
            mom_z = zscore(mom_w)
            val_z = zscore(val_w)
            
            # Sector neutralize
            sec = sectors_ser.reindex(idx)
            mom_z = sector_neutralize(mom_z, sec)
            val_z = sector_neutralize(val_z, sec)
            
            # Combine with engine weights
            alpha_term = BETA_MOM * mom_z + BETA_VAL * val_z
            
            # Log alpha ground truth
            for ticker in idx:
                alpha_log_rows.append({
                    "asof_dt": date.date().isoformat(),
                    "ticker": ticker,
                    "alpha_gt_used": float(alpha_term.loc[ticker]),
                    "mom_z_tm1": float(mom_z.loc[ticker]) if ticker in mom_z.index else np.nan,
                    "val_z_tm1": float(val_z.loc[ticker]) if ticker in val_z.index else np.nan,
                    "beta_mom": float(BETA_MOM),
                    "beta_val": float(BETA_VAL),
                })
                
        except Exception as e:
            print(f"Warning: Could not compute signals for {date}: {e}")
            continue
    
    print(f"Generated {len(alpha_log_rows)} alpha ground truth records")
    return alpha_log_rows


def generate(outdir: str = "data", log_alpha_gt: bool = True, alpha_gt_path: str = "data/alpha_gt_engine_compatible.csv") -> Dict[str, Any]:
    """Generate all synthetic data files using fast method."""
    print("Generating synthetic data (fast method)...")
    
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
    print("Generating fundamentals...")
    fundamentals_df = generate_fundamentals(calendar, tickers)
    
    print("Generating prices...")
    prices_df = generate_prices_fast(calendar, tickers, sectors, delistings, fundamentals_df)
    
    print("Generating sectors...")
    sectors_df = generate_sectors(tickers, sectors)
    
    print("Generating holdings...")
    holdings_df = generate_holdings_prev(prices_df)
    
    # Write files
    print("Writing data files...")
    prices_df.to_csv(out_path / 'prices.csv', index=False)
    fundamentals_df.to_csv(out_path / 'fundamentals.csv', index=False)
    sectors_df.to_csv(out_path / 'sectors.csv', index=False)
    holdings_df.to_csv(out_path / 'holdings_prev.csv', index=False)
    
    # Generate engine-compatible alpha ground truth if requested
    alpha_log_rows = []
    if log_alpha_gt:
        alpha_log_rows = generate_engine_compatible_alpha_gt(prices_df, fundamentals_df, sectors_df, calendar)
        
        if alpha_log_rows:
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
    print("=== SYNTHETIC DATA GENERATION SUMMARY (FAST METHOD) ===")
    print(f"Total tickers: {summary['total_tickers']}")
    print(f"Active on last day: {summary['active_tickers']}")
    print(f"Delisted: {summary['delisted_tickers']}")
    print(f"Total trading days: {summary['total_days']}")
    print(f"Price date range: {summary['price_date_range'][0]} to {summary['price_date_range'][1]}")
    print(f"Fundamentals available range: {summary['fundamentals_date_range'][0]} to {summary['fundamentals_date_range'][1]}")
    print(f"Delisted tickers: {sorted(delisted_tickers)}")
    print(f"Output directory: {out_path.absolute()}")
    print(f"Using engine signal processing pipeline with betas: mom={BETA_MOM}, val={BETA_VAL}")
    
    return summary


def main():
    global BETA_MOM, BETA_VAL
    
    parser = argparse.ArgumentParser(description="Generate synthetic data for quant engine testing (fast method)")
    parser.add_argument("--outdir", default="data", help="Output directory (default: data)")
    parser.add_argument("--beta-mom", type=float, default=BETA_MOM, help=f"Momentum beta (default: {BETA_MOM})")
    parser.add_argument("--beta-val", type=float, default=BETA_VAL, help=f"Value beta (default: {BETA_VAL})")
    parser.add_argument("--log-alpha-gt", action="store_true", default=True, help="Log alpha ground truth (default: True)")
    parser.add_argument("--no-log-alpha-gt", dest="log_alpha_gt", action="store_false", help="Disable alpha ground truth logging")
    parser.add_argument("--alpha-gt-path", default="data/alpha_gt_engine_compatible.csv", help="Path for alpha ground truth CSV")
    args = parser.parse_args()
    
    # Update global parameters
    BETA_MOM = args.beta_mom
    BETA_VAL = args.beta_val
    
    generate(args.outdir, args.log_alpha_gt, args.alpha_gt_path)


if __name__ == "__main__":
    main() 