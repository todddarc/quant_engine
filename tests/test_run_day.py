"""
Tests for run_day module.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.quant_engine.run_day import run


def create_test_data(tmp_path: Path) -> None:
    """Create minimal test data files."""
    # Create prices data: 5 tickers, 40 business days
    tickers = ['A', 'B', 'C', 'D', 'E']
    dates = pd.date_range('2023-12-01', periods=40, freq='B')  # Start from December
    
    prices_data = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            # Simple increasing prices with some noise
            base_price = 100 + i * 10 + (date - dates[0]).days * 0.1
            price = base_price + np.random.normal(0, 1)
            prices_data.append({
                'asof_dt': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': max(price, 1.0)  # Ensure positive prices
            })
    
    prices_df = pd.DataFrame(prices_data)
    prices_df.to_csv(tmp_path / 'prices.csv', index=False)
    
    # Create fundamentals data
    fundamentals_data = []
    for ticker in tickers:
        for quarter in ['2023-09-30', '2023-06-30', '2023-03-31']:
            fundamentals_data.append({
                'report_dt': quarter,
                'available_asof': pd.Timestamp(quarter) + pd.Timedelta(days=30),
                'ticker': ticker,
                'eps_ttm': 5.0 + np.random.normal(0, 1),
                'book_value_ps': 50.0 + np.random.normal(0, 5)
            })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    fundamentals_df.to_csv(tmp_path / 'fundamentals.csv', index=False)
    
    # Create sectors data
    sectors_data = [
        {'ticker': 'A', 'sector': 'Tech'},
        {'ticker': 'B', 'sector': 'Tech'},
        {'ticker': 'C', 'sector': 'Finance'},
        {'ticker': 'D', 'sector': 'Finance'},
        {'ticker': 'E', 'sector': 'Tech'}
    ]
    sectors_df = pd.DataFrame(sectors_data)
    sectors_df.to_csv(tmp_path / 'sectors.csv', index=False)
    
    # Create prior holdings data
    holdings_data = []
    for ticker in tickers:
        holdings_data.append({
            'asof_dt': '2023-12-15',  # Use a date in the middle of our range
            'ticker': ticker,
            'weight': 0.2  # Equal weights
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    holdings_df.to_csv(tmp_path / 'holdings.csv', index=False)


def create_test_config(tmp_path: Path) -> str:
    """Create a test configuration file."""
    config = {
        'data': {
            'prices_path': str(tmp_path / 'prices.csv'),
            'fundamentals_path': str(tmp_path / 'fundamentals.csv'),
            'sectors_path': str(tmp_path / 'sectors.csv'),
            'holdings_path': str(tmp_path / 'holdings.csv')
        },
        'signals': {
            'momentum': {
                'lookback': 10,  # Smaller for test
                'gap': 2
            },
            'value': {
                'min_lag_days': 30
            },
            'weights': {
                'momentum': 0.5,
                'value': 0.5
            }
        },
        'risk': {
            'cov_lookback_days': 15,  # Smaller for test
            'shrink_lambda': 0.3,
            'diag_load': 1e-4
        },
        'optimization': {
            'w_max': 0.6,  # Loose constraints for test
            'sector_cap': 0.8,
            'turnover_cap': 0.5,
            'risk_aversion': 8.0
        },
        'checks': {
            'missing_max': 0.5
        },
        'paths': {
            'output_dir': str(tmp_path)
        }
    }
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


def test_run_day_success(tmp_path):
    """Test successful pipeline run."""
    # Create test data with more history
    tickers = ['A', 'B', 'C', 'D', 'E']
    dates = pd.date_range('2023-11-01', periods=60, freq='B')  # More history
    
    prices_data = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            # Simple increasing prices with some noise
            base_price = 100 + i * 10 + (date - dates[0]).days * 0.1
            price = base_price + np.random.normal(0, 1)
            prices_data.append({
                'asof_dt': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': max(price, 1.0)  # Ensure positive prices
            })
    
    prices_df = pd.DataFrame(prices_data)
    prices_df.to_csv(tmp_path / 'prices.csv', index=False)
    
    # Create fundamentals data
    fundamentals_data = []
    for ticker in tickers:
        for quarter in ['2023-09-30', '2023-06-30', '2023-03-31']:
            fundamentals_data.append({
                'report_dt': quarter,
                'available_asof': pd.Timestamp(quarter) + pd.Timedelta(days=30),
                'ticker': ticker,
                'eps_ttm': 5.0 + np.random.normal(0, 1),
                'book_value_ps': 50.0 + np.random.normal(0, 5)
            })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    fundamentals_df.to_csv(tmp_path / 'fundamentals.csv', index=False)
    
    # Create sectors data
    sectors_data = [
        {'ticker': 'A', 'sector': 'Tech'},
        {'ticker': 'B', 'sector': 'Tech'},
        {'ticker': 'C', 'sector': 'Finance'},
        {'ticker': 'D', 'sector': 'Finance'},
        {'ticker': 'E', 'sector': 'Tech'}
    ]
    sectors_df = pd.DataFrame(sectors_data)
    sectors_df.to_csv(tmp_path / 'sectors.csv', index=False)
    
    # Create prior holdings data
    holdings_data = []
    for ticker in tickers:
        holdings_data.append({
            'asof_dt': '2023-12-15',  # Use a date in the middle of our range
            'ticker': ticker,
            'weight': 0.2  # Equal weights
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    holdings_df.to_csv(tmp_path / 'holdings.csv', index=False)
    
    # Create config with very small parameters
    config = {
        'data': {
            'prices_path': str(tmp_path / 'prices.csv'),
            'fundamentals_path': str(tmp_path / 'fundamentals.csv'),
            'sectors_path': str(tmp_path / 'sectors.csv'),
            'holdings_path': str(tmp_path / 'holdings.csv')
        },
        'signals': {
            'momentum': {
                'lookback': 5,  # Very small for test
                'gap': 1
            },
            'value': {
                'min_lag_days': 30
            },
            'weights': {
                'momentum': 0.5,
                'value': 0.5
            }
        },
        'risk': {
            'cov_lookback_days': 10,  # Very small for test
            'shrink_lambda': 0.3,
            'diag_load': 1e-4
        },
        'optimization': {
            'w_max': 0.6,  # Loose constraints for test
            'sector_cap': 0.8,
            'turnover_cap': 0.5,
            'risk_aversion': 8.0
        },
        'checks': {
            'missing_max': 0.5
        },
        'paths': {
            'output_dir': str(tmp_path)
        }
    }
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline for a date that has next-day returns
    asof = '2023-12-15'  # This date should have next-day returns
    
    summary = run(asof, str(config_path))
    
    # Check summary structure
    assert 'ok_to_trade' in summary
    assert 'ic_snapshot' in summary
    assert 'risk' in summary
    assert 'turnover' in summary
    assert 'paths' in summary
    
    # Check that pipeline succeeded
    assert summary['ok_to_trade'] is True
    
    # Check output files exist
    assert Path(summary['paths']['holdings']).exists()
    assert Path(summary['paths']['trades']).exists()
    assert Path(summary['paths']['report']).exists()
    
    # Check holdings file
    holdings_df = pd.read_csv(summary['paths']['holdings'])
    assert len(holdings_df) > 0
    assert 'ticker' in holdings_df.columns
    assert 'weight' in holdings_df.columns
    assert abs(holdings_df['weight'].sum() - 1.0) < 1e-6
    assert all(holdings_df['weight'] >= 0)
    
    # Check trades file
    trades_df = pd.read_csv(summary['paths']['trades'])
    assert len(trades_df) > 0
    assert 'ticker' in trades_df.columns
    assert 'delta_weight' in trades_df.columns
    
    # Check report file
    report_path = Path(summary['paths']['report'])
    assert report_path.exists()
    with open(report_path, 'r') as f:
        report_content = f.read()
        assert 'PORTFOLIO CONSTRUCTION REPORT' in report_content
        assert 'OK TO TRADE: True' in report_content


def test_run_day_blocked_by_turnover(tmp_path):
    """Test pipeline blocked by turnover constraint."""
    # Create test data with more history
    tickers = ['A', 'B', 'C', 'D', 'E']
    dates = pd.date_range('2023-11-01', periods=60, freq='B')  # More history
    
    prices_data = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            base_price = 100 + i * 10 + (date - dates[0]).days * 0.1
            price = base_price + np.random.normal(0, 1)
            prices_data.append({
                'asof_dt': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': max(price, 1.0)
            })
    
    prices_df = pd.DataFrame(prices_data)
    prices_df.to_csv(tmp_path / 'prices.csv', index=False)
    
    # Create fundamentals data
    fundamentals_data = []
    for ticker in tickers:
        for quarter in ['2023-09-30', '2023-06-30', '2023-03-31']:
            fundamentals_data.append({
                'report_dt': quarter,
                'available_asof': pd.Timestamp(quarter) + pd.Timedelta(days=30),
                'ticker': ticker,
                'eps_ttm': 5.0 + np.random.normal(0, 1),
                'book_value_ps': 50.0 + np.random.normal(0, 5)
            })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    fundamentals_df.to_csv(tmp_path / 'fundamentals.csv', index=False)
    
    # Create sectors data
    sectors_data = [
        {'ticker': 'A', 'sector': 'Tech'},
        {'ticker': 'B', 'sector': 'Tech'},
        {'ticker': 'C', 'sector': 'Finance'},
        {'ticker': 'D', 'sector': 'Finance'},
        {'ticker': 'E', 'sector': 'Tech'}
    ]
    sectors_df = pd.DataFrame(sectors_data)
    sectors_df.to_csv(tmp_path / 'sectors.csv', index=False)
    
    # Create prior holdings data
    holdings_data = []
    for ticker in tickers:
        holdings_data.append({
            'asof_dt': '2023-12-15',
            'ticker': ticker,
            'weight': 0.2
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    holdings_df.to_csv(tmp_path / 'holdings.csv', index=False)
    
    # Create config with very restrictive turnover cap
    config = {
        'data': {
            'prices_path': str(tmp_path / 'prices.csv'),
            'fundamentals_path': str(tmp_path / 'fundamentals.csv'),
            'sectors_path': str(tmp_path / 'sectors.csv'),
            'holdings_path': str(tmp_path / 'holdings.csv')
        },
        'signals': {
            'momentum': {
                'lookback': 5,
                'gap': 1
            },
            'value': {
                'min_lag_days': 30
            },
            'weights': {
                'momentum': 0.5,
                'value': 0.5
            }
        },
        'risk': {
            'cov_lookback_days': 10,
            'shrink_lambda': 0.3,
            'diag_load': 1e-4
        },
        'optimization': {
            'w_max': 0.6,
            'sector_cap': 0.8,
            'turnover_cap': 0.0,  # Impossible to satisfy
            'risk_aversion': 8.0
        },
        'checks': {
            'missing_max': 0.5
        },
        'paths': {
            'output_dir': str(tmp_path)
        }
    }
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    asof = '2023-12-15'
    summary = run(asof, str(config_path))
    
    # Check that pipeline was blocked
    assert summary['ok_to_trade'] is False
    
    # Check output files still exist
    assert Path(summary['paths']['holdings']).exists()
    assert Path(summary['paths']['trades']).exists()
    assert Path(summary['paths']['report']).exists()
    
    # Check report contains fallback message
    report_path = Path(summary['paths']['report'])
    with open(report_path, 'r') as f:
        report_content = f.read()
        assert 'OK TO TRADE: False' in report_content
        assert 'FALLBACK: previous weights emitted' in report_content


def test_run_day_invalid_date(tmp_path):
    """Test pipeline with invalid date."""
    create_test_data(tmp_path)
    config_path = create_test_config(tmp_path)
    
    # Try to run with a date that doesn't exist in prices
    asof = '2025-01-01'  # Date well outside our test range
    
    with pytest.raises(ValueError, match="Date 2025-01-01 not found"):
        run(asof, config_path)


def test_run_day_no_prior_holdings(tmp_path):
    """Test pipeline when no prior holdings exist."""
    # Create test data with more history
    tickers = ['A', 'B', 'C', 'D', 'E']
    dates = pd.date_range('2023-11-01', periods=60, freq='B')  # More history
    
    prices_data = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            base_price = 100 + i * 10 + (date - dates[0]).days * 0.1
            price = base_price + np.random.normal(0, 1)
            prices_data.append({
                'asof_dt': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': max(price, 1.0)
            })
    
    prices_df = pd.DataFrame(prices_data)
    prices_df.to_csv(tmp_path / 'prices.csv', index=False)
    
    # Create fundamentals data
    fundamentals_data = []
    for ticker in tickers:
        for quarter in ['2023-09-30', '2023-06-30', '2023-03-31']:
            fundamentals_data.append({
                'report_dt': quarter,
                'available_asof': pd.Timestamp(quarter) + pd.Timedelta(days=30),
                'ticker': ticker,
                'eps_ttm': 5.0 + np.random.normal(0, 1),
                'book_value_ps': 50.0 + np.random.normal(0, 5)
            })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    fundamentals_df.to_csv(tmp_path / 'fundamentals.csv', index=False)
    
    # Create sectors data
    sectors_data = [
        {'ticker': 'A', 'sector': 'Tech'},
        {'ticker': 'B', 'sector': 'Tech'},
        {'ticker': 'C', 'sector': 'Finance'},
        {'ticker': 'D', 'sector': 'Finance'},
        {'ticker': 'E', 'sector': 'Tech'}
    ]
    sectors_df = pd.DataFrame(sectors_data)
    sectors_df.to_csv(tmp_path / 'sectors.csv', index=False)
    
    # Don't create holdings file - let the pipeline handle it
    
    # Create config
    config = {
        'data': {
            'prices_path': str(tmp_path / 'prices.csv'),
            'fundamentals_path': str(tmp_path / 'fundamentals.csv'),
            'sectors_path': str(tmp_path / 'sectors.csv'),
            'holdings_path': str(tmp_path / 'holdings.csv')  # This file won't exist
        },
        'signals': {
            'momentum': {
                'lookback': 5,
                'gap': 1
            },
            'value': {
                'min_lag_days': 30
            },
            'weights': {
                'momentum': 0.5,
                'value': 0.5
            }
        },
        'risk': {
            'cov_lookback_days': 10,
            'shrink_lambda': 0.3,
            'diag_load': 1e-4
        },
        'optimization': {
            'w_max': 0.6,
            'sector_cap': 0.8,
            'turnover_cap': 0.5,
            'risk_aversion': 8.0
        },
        'checks': {
            'missing_max': 0.5
        },
        'paths': {
            'output_dir': str(tmp_path)
        }
    }
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    asof = '2023-12-15'
    summary = run(asof, str(config_path))
    
    # Should still succeed with equal weights
    assert summary['ok_to_trade'] is True
    
    # Check holdings file was created
    assert Path(summary['paths']['holdings']).exists()
    
    # Check that weights sum to 1
    holdings_df = pd.read_csv(summary['paths']['holdings'])
    assert abs(holdings_df['weight'].sum() - 1.0) < 1e-6


def test_run_day_insufficient_data(tmp_path):
    """Test pipeline with insufficient data for signals."""
    # Create minimal data that won't support signal generation
    tickers = ['A', 'B']
    dates = pd.date_range('2023-01-01', periods=5, freq='B')  # Too few dates
    
    # Create minimal prices
    prices_data = []
    for date in dates:
        for ticker in tickers:
            prices_data.append({
                'asof_dt': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': 100.0
            })
    
    prices_df = pd.DataFrame(prices_data)
    prices_df.to_csv(tmp_path / 'prices.csv', index=False)
    
    # Create minimal fundamentals
    fundamentals_data = []
    for ticker in tickers:
        fundamentals_data.append({
            'report_dt': '2023-03-31',
            'available_asof': '2023-04-30',
            'ticker': ticker,
            'eps_ttm': 5.0,
            'book_value_ps': 50.0
        })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    fundamentals_df.to_csv(tmp_path / 'fundamentals.csv', index=False)
    
    # Create sectors
    sectors_data = [{'ticker': 'A', 'sector': 'Tech'}, {'ticker': 'B', 'sector': 'Tech'}]
    sectors_df = pd.DataFrame(sectors_data)
    sectors_df.to_csv(tmp_path / 'sectors.csv', index=False)
    
    # Create holdings
    holdings_data = [{'asof_dt': '2023-01-05', 'ticker': 'A', 'weight': 0.5},
                    {'asof_dt': '2023-01-05', 'ticker': 'B', 'weight': 0.5}]
    holdings_df = pd.DataFrame(holdings_data)
    holdings_df.to_csv(tmp_path / 'holdings.csv', index=False)
    
    config_path = create_test_config(tmp_path)
    
    # Should fail due to insufficient data for momentum signal
    asof = '2023-01-05'
    with pytest.raises(ValueError, match="Insufficient common tickers"):
        run(asof, config_path) 