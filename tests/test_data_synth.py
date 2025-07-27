"""
Tests for synthetic data generator.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from data_fetch.make_synth_data import generate


def test_prices_schema_and_monotonic_dates(tmp_path):
    """Test prices.csv schema and date monotonicity."""
    # Generate data
    summary = generate(str(tmp_path))
    
    # Load prices
    prices_df = pd.read_csv(tmp_path / 'prices.csv')
    
    # Check schema
    expected_cols = ['asof_dt', 'ticker', 'close']
    assert list(prices_df.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(prices_df.columns)}"
    
    # Check no negative prices
    assert (prices_df['close'] > 0).all(), "Found negative or zero prices"
    
    # Check date monotonicity per ticker
    prices_df['asof_dt'] = pd.to_datetime(prices_df['asof_dt'])
    for ticker in prices_df['ticker'].unique():
        ticker_data = prices_df[prices_df['ticker'] == ticker].sort_values('asof_dt')
        assert ticker_data['asof_dt'].is_monotonic_increasing, f"Dates not monotonic for ticker {ticker}"
    
    # Check summary stats
    assert summary['total_tickers'] == 40
    assert summary['total_days'] == 500
    assert summary['active_tickers'] < summary['total_tickers']  # Some delistings


def test_fundamentals_pit_fields(tmp_path):
    """Test fundamentals.csv PIT correctness."""
    # Generate data
    summary = generate(str(tmp_path))
    
    # Load fundamentals
    fundamentals_df = pd.read_csv(tmp_path / 'fundamentals.csv')
    
    # Check schema
    expected_cols = ['report_dt', 'available_asof', 'ticker', 'eps_ttm', 'book_value_ps']
    assert list(fundamentals_df.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(fundamentals_df.columns)}"
    
    # Convert dates
    fundamentals_df['report_dt'] = pd.to_datetime(fundamentals_df['report_dt'])
    fundamentals_df['available_asof'] = pd.to_datetime(fundamentals_df['available_asof'])
    
    # Check PIT correctness: available_asof >= report_dt
    assert (fundamentals_df['available_asof'] >= fundamentals_df['report_dt']).all(), \
        "Found available_asof < report_dt"
    
    # Load prices to get max date
    prices_df = pd.read_csv(tmp_path / 'prices.csv')
    prices_df['asof_dt'] = pd.to_datetime(prices_df['asof_dt'])
    max_price_date = prices_df['asof_dt'].max()
    
    # Check all available_asof <= max price date
    assert (fundamentals_df['available_asof'] <= max_price_date).all(), \
        "Found available_asof > max price date"
    
    # Check fundamentals date range in summary
    assert summary['fundamentals_date_range'][0] <= summary['fundamentals_date_range'][1]


def test_delistings_present(tmp_path):
    """Test that delistings are present."""
    # Generate data
    summary = generate(str(tmp_path))
    
    # Load prices
    prices_df = pd.read_csv(tmp_path / 'prices.csv')
    prices_df['asof_dt'] = pd.to_datetime(prices_df['asof_dt'])
    
    # Get last date
    last_date = prices_df['asof_dt'].max()
    last_date_prices = prices_df[prices_df['asof_dt'] == last_date]
    
    # Check that some tickers are missing on last date (delisted)
    active_tickers = set(last_date_prices['ticker'].unique())
    all_tickers = set(prices_df['ticker'].unique())
    delisted_tickers = all_tickers - active_tickers
    
    assert len(delisted_tickers) >= 2, f"Expected at least 2 delistings, got {len(delisted_tickers)}"
    
    # Check summary matches
    assert summary['delisted_tickers'] == len(delisted_tickers)
    assert summary['active_tickers'] == len(active_tickers)


def test_holdings_prev_valid(tmp_path):
    """Test holdings_prev.csv validity."""
    # Generate data
    summary = generate(str(tmp_path))
    
    # Load holdings
    holdings_df = pd.read_csv(tmp_path / 'holdings_prev.csv')
    
    # Check schema
    expected_cols = ['asof_dt', 'ticker', 'weight']
    assert list(holdings_df.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(holdings_df.columns)}"
    
    # Load prices to get last date
    prices_df = pd.read_csv(tmp_path / 'prices.csv')
    prices_df['asof_dt'] = pd.to_datetime(prices_df['asof_dt'])
    last_date = prices_df['asof_dt'].max()
    
    # Check holdings date matches last price date
    holdings_date = pd.to_datetime(holdings_df['asof_dt'].iloc[0])
    assert holdings_date == last_date, f"Holdings date {holdings_date} != last price date {last_date}"
    
    # Check weights sum to 1
    weight_sum = holdings_df['weight'].sum()
    assert abs(weight_sum - 1.0) < 1e-8, f"Weights sum to {weight_sum}, expected 1.0"
    
    # Check all weights non-negative
    assert (holdings_df['weight'] >= 0).all(), "Found negative weights"
    
    # Check all tickers in holdings exist in prices on that date
    last_date_prices = prices_df[prices_df['asof_dt'] == last_date]
    active_tickers = set(last_date_prices['ticker'].unique())
    holdings_tickers = set(holdings_df['ticker'].unique())
    
    assert holdings_tickers.issubset(active_tickers), \
        f"Holdings tickers {holdings_tickers - active_tickers} not in active tickers"


def test_sectors_schema(tmp_path):
    """Test sectors.csv schema."""
    # Generate data
    generate(str(tmp_path))
    
    # Load sectors
    sectors_df = pd.read_csv(tmp_path / 'sectors.csv')
    
    # Check schema
    expected_cols = ['ticker', 'sector']
    assert list(sectors_df.columns) == expected_cols, f"Expected columns {expected_cols}, got {list(sectors_df.columns)}"
    
    # Check all tickers have sectors
    assert sectors_df['ticker'].nunique() == 40, "Expected 40 tickers"
    assert sectors_df['sector'].nunique() == 10, "Expected 10 sectors"
    
    # Check no missing sectors
    assert sectors_df['sector'].notna().all(), "Found missing sector values"


def test_deterministic_output(tmp_path):
    """Test that output is deterministic."""
    # Generate data twice
    summary1 = generate(str(tmp_path / 'test1'))
    summary2 = generate(str(tmp_path / 'test2'))
    
    # Check summaries are identical
    for key in ['total_tickers', 'active_tickers', 'delisted_tickers', 'total_days']:
        assert summary1[key] == summary2[key], f"Summary mismatch for {key}"
    
    # Check files are identical
    for filename in ['prices.csv', 'fundamentals.csv', 'sectors.csv', 'holdings_prev.csv']:
        file1 = tmp_path / 'test1' / filename
        file2 = tmp_path / 'test2' / filename
        
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False), \
            f"Files {filename} are not identical"


def test_return_statistics_realism(tmp_path):
    """Test that return statistics are realistic (including skill injection)."""
    # Generate data
    generate(str(tmp_path))
    
    # Load prices
    prices_df = pd.read_csv(tmp_path / 'prices.csv')
    prices_df['asof_dt'] = pd.to_datetime(prices_df['asof_dt'])
    
    # Calculate returns
    prices_pivot = prices_df.pivot(index='asof_dt', columns='ticker', values='close')
    returns = prices_pivot.pct_change(fill_method=None).dropna()
    
    # Check return statistics are reasonable
    mean_returns = returns.mean()
    std_returns = returns.std()
    
    # Mean returns should be reasonable (accounting for skill injection)
    # With BETA_MOM=0.035 and BETA_VAL=0.018, we expect some drift
    assert abs(mean_returns.mean()) < 0.005, f"Mean returns too large: {mean_returns.mean()}"
    
    # Std returns should be reasonable (around 0.01-0.02)
    assert 0.005 < std_returns.mean() < 0.05, f"Std returns outside reasonable range: {std_returns.mean()}"
    
    # Check no extreme outliers
    assert (returns.abs() < 0.5).all().all(), "Found extreme returns (>50%)"


def test_skill_injection_present(tmp_path):
    """Test that skill injection is actually working in the synthetic data."""
    # Generate data
    generate(str(tmp_path))
    
    # Load prices
    prices_df = pd.read_csv(tmp_path / 'prices.csv')
    prices_df['asof_dt'] = pd.to_datetime(prices_df['asof_dt'])
    
    # Calculate returns
    prices_pivot = prices_df.pivot(index='asof_dt', columns='ticker', values='close')
    returns = prices_pivot.pct_change(fill_method=None).dropna()
    
    # Check that mean returns are not zero (indicating skill injection is working)
    mean_returns = returns.mean()
    assert abs(mean_returns.mean()) > 0.0001, f"Mean returns too small, skill injection may not be working: {mean_returns.mean()}"
    
    # Check that some tickers have positive mean returns and some have negative
    # (indicating the skill injection creates cross-sectional variation)
    positive_means = (mean_returns > 0.001).sum()
    negative_means = (mean_returns < -0.001).sum()
    assert positive_means > 0, "No tickers with positive mean returns"
    assert negative_means > 0, "No tickers with negative mean returns"


def test_fundamentals_realism(tmp_path):
    """Test that fundamentals are realistic."""
    # Generate data
    generate(str(tmp_path))
    
    # Load fundamentals
    fundamentals_df = pd.read_csv(tmp_path / 'fundamentals.csv')
    
    # Check eps_ttm is reasonable
    eps_values = fundamentals_df['eps_ttm']
    assert eps_values.min() >= -5, f"EPS too negative: {eps_values.min()}"
    assert eps_values.max() < 50, f"EPS too high: {eps_values.max()}"
    
    # Check book_value_ps is positive and reasonable
    book_values = fundamentals_df['book_value_ps']
    assert book_values.min() > 0, f"Book value not positive: {book_values.min()}"
    assert book_values.max() < 500, f"Book value too high: {book_values.max()}"
    
    # Check reporting lags are reasonable (60-90 days)
    fundamentals_df['report_dt'] = pd.to_datetime(fundamentals_df['report_dt'])
    fundamentals_df['available_asof'] = pd.to_datetime(fundamentals_df['available_asof'])
    lags = (fundamentals_df['available_asof'] - fundamentals_df['report_dt']).dt.days
    
    assert lags.min() >= 60, f"Reporting lag too short: {lags.min()}"
    assert lags.max() <= 90, f"Reporting lag too long: {lags.max()}"


def test_sector_balance(tmp_path):
    """Test that sectors are balanced."""
    # Generate data
    generate(str(tmp_path))
    
    # Load sectors
    sectors_df = pd.read_csv(tmp_path / 'sectors.csv')
    
    # Check sector distribution
    sector_counts = sectors_df['sector'].value_counts()
    
    # Should have 4 tickers per sector (40 tickers / 10 sectors)
    assert sector_counts.min() == 4, f"Min sector count: {sector_counts.min()}"
    assert sector_counts.max() == 4, f"Max sector count: {sector_counts.max()}"
    
    # Check all expected sectors present
    expected_sectors = [
        "Tech", "Health", "Energy", "Financials", "Industrials",
        "Consumer", "Utilities", "Materials", "Comm", "RealEstate"
    ]
    actual_sectors = set(sectors_df['sector'].unique())
    assert actual_sectors == set(expected_sectors), f"Sector mismatch: {actual_sectors} vs {expected_sectors}" 