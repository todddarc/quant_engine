"""
CLI smoke tests for quant_engine.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import subprocess
import sys
from pathlib import Path
from quant_engine.run_day import run


def create_minimal_test_data(tmp_path: Path) -> None:
    """Create minimal test data files for fast CLI testing."""
    # Create prices data: 3 tickers, 30 business days
    tickers = ['A', 'B', 'C']
    dates = pd.date_range('2023-12-01', periods=30, freq='B')
    
    prices_data = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            # Simple increasing prices with minimal noise
            base_price = 100 + i * 10 + (date - dates[0]).days * 0.05
            price = base_price + np.random.normal(0, 0.5)
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
        for quarter in ['2023-09-30', '2023-06-30']:
            fundamentals_data.append({
                'report_dt': quarter,
                'available_asof': pd.Timestamp(quarter) + pd.Timedelta(days=30),
                'ticker': ticker,
                'eps_ttm': 5.0 + np.random.normal(0, 0.5),
                'book_value_ps': 50.0 + np.random.normal(0, 2)
            })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    fundamentals_df.to_csv(tmp_path / 'fundamentals.csv', index=False)
    
    # Create sectors data
    sectors_data = [
        {'ticker': 'A', 'sector': 'Tech'},
        {'ticker': 'B', 'sector': 'Finance'},
        {'ticker': 'C', 'sector': 'Tech'}
    ]
    sectors_df = pd.DataFrame(sectors_data)
    sectors_df.to_csv(tmp_path / 'sectors.csv', index=False)
    
    # Create prior holdings data
    holdings_data = []
    for ticker in tickers:
        holdings_data.append({
            'asof_dt': '2023-12-15',
            'ticker': ticker,
            'weight': 1.0 / len(tickers)  # Equal weights
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    holdings_df.to_csv(tmp_path / 'holdings.csv', index=False)


def create_minimal_config(tmp_path: Path) -> str:
    """Create a minimal test configuration file."""
    config = {
        'data': {
            'prices_path': str(tmp_path / 'prices.csv'),
            'fundamentals_path': str(tmp_path / 'fundamentals.csv'),
            'sectors_path': str(tmp_path / 'sectors.csv'),
            'holdings_path': str(tmp_path / 'holdings.csv')
        },
        'signals': {
            'momentum': {
                'lookback': 8,  # Very small for fast test
                'gap': 1
            },
            'value': {
                'min_lag_days': 20
            },
            'weights': {
                'momentum': 0.5,
                'value': 0.5
            }
        },
        'risk': {
            'cov_lookback_days': 10,  # Very small for fast test
            'shrink_lambda': 0.3,
            'diag_load': 1e-4
        },
        'optimization': {
            'w_max': 0.8,  # Very loose constraints
            'sector_cap': 0.9,
            'turnover_cap': 0.8,
            'risk_aversion': 5.0
        },
        'checks': {
            'missing_max': 0.8  # Very permissive
        },
        'paths': {
            'output_dir': str(tmp_path)
        }
    }
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


def test_run_function_smoke(tmp_path):
    """Test the run function directly with minimal data."""
    # Create test data and config
    create_minimal_test_data(tmp_path)
    config_path = create_minimal_config(tmp_path)
    
    # Run the function
    result = run("2023-12-20", config_path)
    
    # Assert basic structure
    assert isinstance(result, dict)
    assert "ok_to_trade" in result
    assert result["ok_to_trade"] in {True, False}
    assert "alpha_dot" in result
    assert "risk" in result
    assert "turnover" in result
    assert "asof" in result
    assert "paths" in result
    
    # Assert paths structure
    assert isinstance(result["paths"], dict)
    assert "holdings" in result["paths"]
    assert "trades" in result["paths"]
    assert "report" in result["paths"]
    
    # Assert numeric values are reasonable
    assert isinstance(result["alpha_dot"], (int, float))
    assert isinstance(result["risk"], (int, float))
    assert isinstance(result["turnover"], (int, float))
    assert result["risk"] >= 0  # Risk should be non-negative
    assert result["turnover"] >= 0  # Turnover should be non-negative


def test_cli_subprocess_smoke(tmp_path):
    """Test CLI execution via subprocess."""
    # Create test data and config
    create_minimal_test_data(tmp_path)
    config_path = create_minimal_config(tmp_path)
    
    # Run via subprocess from project root (where src/ is located)
    cmd = [
        sys.executable, "-m", "src.quant_engine.run_day",
        "--config", config_path,
        "--asof", "2023-12-20"
    ]
    
    # Run from project root, not tmp_path
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    
    # Assert return code is 0 or 1 (success or blocked, not error)
    assert result.returncode in {0, 1}
    
    # Assert some basic output
    assert "INFO" in result.stdout or "ERROR" in result.stderr or result.returncode == 0
    
    # If successful, should mention report path
    if result.returncode == 0:
        assert "Report written to:" in result.stdout


def test_cli_subprocess_invalid_args(tmp_path):
    """Test CLI with invalid arguments returns error."""
    # Run with missing required arguments
    cmd = [sys.executable, "-m", "src.quant_engine.run_day", "--asof", "2023-12-20"]
    
    # Run from project root
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    
    # Should return error code
    assert result.returncode != 0
    
    # Should show usage/help
    assert "usage:" in result.stderr or "error:" in result.stderr 
