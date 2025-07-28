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

from pandas.tseries.offsets import BDay
from quant_engine.signals import momentum_12m_1m_gap, value_ep
from quant_engine.prep import zscore, winsorize


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _zscore_vec(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    mu = np.nanmean(y)
    sd = np.nanstd(y, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(y)
    return (y - mu) / sd

def _proj_perp(X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Return the residual of projecting y onto columns of X (per-day cross-section).
    (I - P_X) y, with ridge eps for stability.
    X: n x p, y: n
    """
    if X.size == 0 or y.size == 0:
        return y
    XtX = X.T @ X
    # ridge for invertibility
    XtX.flat[::XtX.shape[0]+1] += eps
    coef = np.linalg.solve(XtX, X.T @ y)
    y_hat = X @ coef
    return y - y_hat

def _build_engine_features_tminus1(prices_df, fundamentals_df, asof_t,
                                   lookback=252, gap=21,
                                   min_lag=60, winsor=(0.0, 1.0)):
    """
    Build z_mom and z_val exactly like the engine, at t-1.
    - momentum_12m_1m_gap(prices, asof=t-1, lookback, gap)
    - value_ep(fundamentals, prices, asof=t-1, min_lag_days)
    - optional winsorize BEFORE cross-sectional zscore (match engine if needed)
    """
    tm1 = pd.to_datetime(asof_t) - BDay(1)
    mom = momentum_12m_1m_gap(prices_df, asof_dt=tm1,
                              lookback=lookback, gap=gap)
    val = value_ep(fundamentals_df, prices_df, asof_dt=tm1,
                   min_lag_days=min_lag)
    idx = mom.index.intersection(val.index)
    if idx.empty:
        return idx, pd.Series(dtype=float), pd.Series(dtype=float), np.empty((0,2))
    mom = mom.reindex(idx)
    val = val.reindex(idx)
    if winsor is not None:
        lo, hi = float(winsor[0]), float(winsor[1])
        mom = winsorize(mom, lo, hi)
        val = winsorize(val, lo, hi)
    mom_z = zscore(mom)
    val_z = zscore(val)
    X = np.column_stack([mom_z.values, val_z.values])  # n x 2
    return idx, mom_z, val_z, X


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
MARKET_STD = 0.02    # Further increased - more market noise
SECTOR_STD = 0.008   # Further increased - more sector noise

# === Hard-mode DGP knobs (tweak to tune difficulty) ===
CAPTURE = 0.15        # expected daily projection R^2 (0.2 hard, 0.8 easy) - FURTHER REDUCED
ALPHA_SCALE = 0.008   # scales alpha vs noise (affects IC, not capture R^2) - FURTHER REDUCED
W_MOM, W_VAL = 0.50, 0.50   # engine mixing used to form z_eng
WINSOR_PCTS = (0.0, 1.0)    # (low, high) winsor for raw mom/val before zscore; (0,1) = off

# === Ensure we compute features exactly like the engine ===
LOOKBACK_DAYS = 252
GAP_DAYS = 21
MIN_LAG_DAYS = 60     # interpret consistently with value_ep

# Idiosyncratic noise
EPSILON_SIG = 0.025  # idiosyncratic daily std (~2.5%) - FURTHER INCREASED

# Sector names
SECTOR_NAMES = [
    "Tech", "Health", "Energy", "Financials", "Industrials",
    "Consumer", "Utilities", "Materials", "Comm", "RealEstate"
]

# Delisting parameters
N_DELISTINGS = 5  # No delistings for short test periods
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
        
        # === 1) Build engine features exactly like the engine, at t-1 ===
        # Convert wide prices to long format for engine functions
        prices_long = []
        for ticker in active:
            for i in range(t):
                if pd.notna(prices_wide.iloc[i][ticker]):
                    prices_long.append({
                        'asof_dt': calendar[i].strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'close': prices_wide.iloc[i][ticker]
                    })
        prices_df = pd.DataFrame(prices_long)
        
        idx, mom_z, val_z, X = _build_engine_features_tminus1(
            prices_df, fundamentals_df, asof_t=date_t,
            lookback=LOOKBACK_DAYS, gap=GAP_DAYS, min_lag=MIN_LAG_DAYS,
            winsor=WINSOR_PCTS
        )
        if len(idx) == 0:
            # no names to inject today; continue with shocks and price update as usual
            alpha_term = np.zeros(len(active))
            if log_alpha_gt:
                for i, ticker in enumerate(active):
                    alpha_log_rows.append({
                        "asof_dt": date_t.date().isoformat(),
                        "ticker": ticker,
                        "alpha_gt_used": 0.0,
                        "mom_z_tm1": np.nan,
                        "val_z_tm1": np.nan,
                        "z_eng_tm1": np.nan,
                        "u_perp_tm1": np.nan,
                        "capture_tgt": float(CAPTURE),
                    })
        else:
            # Combined "engine-spanned" direction (inside span of X)
            z_eng = _zscore_vec(W_MOM * mom_z.reindex(idx).values + W_VAL * val_z.reindex(idx).values)
            
            # === 2) Build a simple "hard" raw vector h_raw outside your engine ===
            # Keep it simple but realistic: interaction + square + mild style
            rng = np.random.RandomState(SEED + int(pd.to_datetime(date_t).strftime("%Y%m%d")))
            inter = (mom_z.reindex(idx).values * val_z.reindex(idx).values)
            sqmom = (mom_z.reindex(idx).values ** 2)
            style = rng.normal(size=len(idx))
            h_raw = 0.5 * inter + 0.3 * sqmom + 0.2 * style
            
            # If you have a sectors Series (ticker->sector) available locally as sectors_ser,
            # you can add a tiny weekly sector tilt to h_raw. If not, skip this block.
            try:
                sectors_ser = pd.Series(sectors)
                sec = sectors_ser.reindex(idx).fillna("UNK")
                uniq_secs = np.unique(sec.values)
                week_id = int(pd.to_datetime(date_t).strftime("%G%V"))
                star_sec = uniq_secs[week_id % len(uniq_secs)]
                tilt = (sec.values == star_sec).astype(float)
                h_raw = h_raw + 0.2 * tilt
            except Exception:
                pass  # ok to proceed without sector tilt
            
            # === 3) Orthogonalize to engine span and standardize ===
            u_raw = _proj_perp(X, h_raw, eps=1e-6)   # residual ⟂ span{z_mom, z_val}
            u = _zscore_vec(u_raw)
            
            # === 4) Mix with capture knob and scale ===
            c = float(CAPTURE)
            c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
            alpha_true_unit = np.sqrt(c) * z_eng + np.sqrt(1.0 - c) * u
            alpha_true = float(ALPHA_SCALE) * alpha_true_unit
            
            # Map back to active tickers
            alpha_term = np.zeros(len(active))
            for i, ticker in enumerate(active):
                if ticker in idx:
                    ticker_idx = list(idx).index(ticker)
                    alpha_term[i] = alpha_true[ticker_idx]
            
            # === 6) Log ground truth for diagnostics ===
            if log_alpha_gt:
                for i, ticker in enumerate(active):
                    if ticker in idx:
                        ticker_idx = list(idx).index(ticker)
                        alpha_log_rows.append({
                            "asof_dt": date_t.date().isoformat(),
                            "ticker": ticker,
                            "alpha_gt_used": float(alpha_true[ticker_idx]),
                            "mom_z_tm1": float(mom_z.loc[ticker]),
                            "val_z_tm1": float(val_z.loc[ticker]),
                            "z_eng_tm1": float(z_eng[ticker_idx]),
                            "u_perp_tm1": float(u[ticker_idx]),
                            "capture_tgt": float(c),
                        })
                    else:
                        alpha_log_rows.append({
                            "asof_dt": date_t.date().isoformat(),
                            "ticker": ticker,
                            "alpha_gt_used": 0.0,
                            "mom_z_tm1": np.nan,
                            "val_z_tm1": np.nan,
                            "z_eng_tm1": np.nan,
                            "u_perp_tm1": np.nan,
                            "capture_tgt": float(c),
                        })
        
        # Generate components
        market_t = np.random.normal(loc=MARKET_MEAN, scale=MARKET_STD)
        sector_shocks = {s: np.random.normal(loc=0.0, scale=SECTOR_STD) for s in SECTOR_NAMES}
        idio = np.random.normal(loc=0.0, scale=EPSILON_SIG, size=len(active))
        
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
        alpha_df['z_eng_tm1'] = alpha_df['z_eng_tm1'].astype(float)
        alpha_df['u_perp_tm1'] = alpha_df['u_perp_tm1'].astype(float)
        alpha_df['capture_tgt'] = alpha_df['capture_tgt'].astype(float)
        
        # Sort by asof_dt, ticker
        alpha_df = alpha_df.sort_values(['asof_dt', 'ticker'])
        
        # Create parent directories if needed
        alpha_gt_path = Path(alpha_gt_path)
        alpha_gt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to CSV
        alpha_df.to_csv(alpha_gt_path, index=False)
        print(f"Alpha ground truth written to: {alpha_gt_path}")
        
        # quick consistency check (tiny sample)
        try:
            sample = alpha_df.tail(200)
            # Correlate alpha_gt_used with z_eng_tm1 cross-sectionally for the last day
            last_day = sample["asof_dt"].iloc[-1]
            sub = sample[sample["asof_dt"] == last_day]
            if not sub.empty:
                corr = pd.Series(sub["alpha_gt_used"]).corr(pd.Series(sub["z_eng_tm1"]), method="spearman")
                print(f"[Sanity] Spearman(engine vs alpha) on last day {last_day}: {corr:.3f}")
        except Exception as e:
            print(f"[Sanity] Alignment check skipped: {e}")
    
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
    parser = argparse.ArgumentParser(description="Generate synthetic data for quant engine testing")
    parser.add_argument("--outdir", default="data", help="Output directory (default: data)")
    parser.add_argument("--log-alpha-gt", action="store_true", default=True, help="Log alpha ground truth (default: True)")
    parser.add_argument("--no-log-alpha-gt", dest="log_alpha_gt", action="store_false", help="Disable alpha ground truth logging")
    parser.add_argument("--alpha-gt-path", default="data/alpha_gt.csv", help="Path for alpha ground truth CSV (default: data/alpha_gt.csv)")
    args = parser.parse_args()
    
    generate(args.outdir, args.log_alpha_gt, args.alpha_gt_path)


if __name__ == "__main__":
    main() 