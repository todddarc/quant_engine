"""
Tests for utils module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.quant_engine.utils import (
    compute_next_period_returns, cross_sectional_ic, compute_ic_series,
    summarize_ic, decile_portfolio_returns
)


class TestComputeNextPeriodReturns:
    """Test next-period returns computation."""
    
    def test_compute_next_period_returns_basic(self):
        """Test basic next-period returns calculation."""
        # Create 2-day, 2-ticker price data
        prices_df = pd.DataFrame({
            "asof_dt": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "close": [100.0, 200.0, 105.0, 210.0]
        })
        
        result = compute_next_period_returns(prices_df, "2023-01-01")
        
        # Expected returns: (105/100 - 1) = 0.05, (210/200 - 1) = 0.05
        expected = pd.Series({"AAPL": 0.05, "MSFT": 0.05})
        pd.testing.assert_series_equal(result, expected, check_dtype=False, rtol=1e-10)
    
    def test_compute_next_period_returns_no_next_day(self):
        """Test when there's no next day in data."""
        prices_df = pd.DataFrame({
            "asof_dt": ["2023-01-01", "2023-01-01"],
            "ticker": ["AAPL", "MSFT"],
            "close": [100.0, 200.0]
        })
        
        result = compute_next_period_returns(prices_df, "2023-01-01")
        
        assert len(result) == 0
        assert result.dtype == float
    
    def test_compute_next_period_returns_missing_tickers(self):
        """Test when some tickers are missing on next day."""
        prices_df = pd.DataFrame({
            "asof_dt": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "close": [100.0, 200.0, 105.0]
        })
        
        result = compute_next_period_returns(prices_df, "2023-01-01")
        
        # Only AAPL should be included
        assert len(result) == 1
        assert "AAPL" in result.index
        assert abs(result["AAPL"] - 0.05) < 1e-10


class TestCrossSectionalIc:
    """Test cross-sectional information coefficient computation."""
    
    def test_cross_sectional_ic_random_noise_approx_zero(self):
        """Test that random signal vs returns gives |IC| < 0.15."""
        np.random.seed(42)
        
        # Create random signal and returns
        n_tickers = 100
        signal = pd.Series(np.random.randn(n_tickers), index=[f"T{i}" for i in range(n_tickers)])
        returns = pd.Series(np.random.randn(n_tickers), index=[f"T{i}" for i in range(n_tickers)])
        
        ic = cross_sectional_ic(signal, returns, method="spearman")
        
        assert abs(ic) < 0.15
    
    def test_cross_sectional_ic_perfect_correlation(self):
        """Test perfect correlation gives IC = 1."""
        n_tickers = 10
        signal = pd.Series(range(n_tickers), index=[f"T{i}" for i in range(n_tickers)])
        returns = pd.Series(range(n_tickers), index=[f"T{i}" for i in range(n_tickers)])
        
        ic = cross_sectional_ic(signal, returns, method="spearman")
        
        assert abs(ic - 1.0) < 1e-6
    
    def test_cross_sectional_ic_insufficient_data(self):
        """Test returns np.nan when < 3 observations."""
        signal = pd.Series([1, 2], index=["A", "B"])
        returns = pd.Series([0.1, 0.2], index=["A", "B"])
        
        ic = cross_sectional_ic(signal, returns)
        
        assert np.isnan(ic)
    
    def test_cross_sectional_ic_mismatched_index(self):
        """Test handles mismatched ticker indices."""
        signal = pd.Series([1, 2, 3], index=["A", "B", "C"])
        returns = pd.Series([0.1, 0.2], index=["A", "B"])
        
        ic = cross_sectional_ic(signal, returns)
        
        # Should align on intersection and compute IC
        # With only 2 points, we need at least 3 for valid correlation
        assert np.isnan(ic)  # Should return NaN for < 3 observations


class TestComputeIcSeries:
    """Test IC time series computation."""
    
    def test_ic_series_monotonic_positive(self):
        """Test that monotonic signal vs returns gives positive ICs."""
        # Create data where higher signal implies higher return
        dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
        tickers = ["A", "B", "C"]
        
        signals_data = []
        returns_data = []
        
        for date in dates:
            for i, ticker in enumerate(tickers):
                signals_data.append({
                    "asof_dt": date,
                    "ticker": ticker,
                    "signal": i  # Monotonic signal
                })
                returns_data.append({
                    "asof_dt": date,
                    "ticker": ticker,
                    "ret_fwd1": i * 0.1  # Monotonic returns
                })
        
        signals_df = pd.DataFrame(signals_data)
        returns_df = pd.DataFrame(returns_data)
        
        ic_df = compute_ic_series(signals_df, returns_df)
        
        # All ICs should be positive
        assert all(ic_df["ic"] > 0)
        assert len(ic_df) == 3  # One IC per date
    
    def test_ic_series_empty_data(self):
        """Test handles empty data gracefully."""
        signals_df = pd.DataFrame(columns=["asof_dt", "ticker", "signal"])
        returns_df = pd.DataFrame(columns=["asof_dt", "ticker", "ret_fwd1"])
        
        ic_df = compute_ic_series(signals_df, returns_df)
        
        assert len(ic_df) == 0
        assert list(ic_df.columns) == ["asof_dt", "ic"]


class TestSummarizeIc:
    """Test IC summary statistics computation."""
    
    def test_summarize_ic_basic(self):
        """Test basic IC summary computation."""
        ic_df = pd.DataFrame({
            "asof_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "ic": [0.1, 0.2, 0.3]
        })
        
        summary = summarize_ic(ic_df)
        
        assert "mean_ic" in summary
        assert "std_ic" in summary
        assert "t_stat" in summary
        assert "hit_rate" in summary
        assert abs(summary["mean_ic"] - 0.2) < 1e-10
        assert summary["hit_rate"] == 1.0  # All positive
    
    def test_summarize_ic_with_nans(self):
        """Test handles NaNs in IC series."""
        ic_df = pd.DataFrame({
            "asof_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "ic": [0.1, np.nan, 0.3]
        })
        
        summary = summarize_ic(ic_df)
        
        assert summary["mean_ic"] == 0.2  # Average of 0.1 and 0.3
        assert summary["hit_rate"] == 1.0  # Both non-NaN values are positive
    
    def test_summarize_ic_empty(self):
        """Test handles empty IC series."""
        ic_df = pd.DataFrame(columns=["asof_dt", "ic"])
        
        summary = summarize_ic(ic_df)
        
        assert all(np.isnan(v) for v in summary.values())


class TestDecilePortfolioReturns:
    """Test decile portfolio returns computation."""
    
    def test_deciles_equalish_bins_single_date(self):
        """Test that deciles produce roughly equal bins."""
        # Create single date with 100 names
        n_tickers = 100
        signals_data = []
        returns_data = []
        
        for i in range(n_tickers):
            ticker = f"T{i}"
            signals_data.append({
                "asof_dt": "2023-01-01",
                "ticker": ticker,
                "signal": i  # Monotonic signal
            })
            returns_data.append({
                "asof_dt": "2023-01-01",
                "ticker": ticker,
                "ret_fwd1": i * 0.001  # Small returns
            })
        
        signals_df = pd.DataFrame(signals_data)
        returns_df = pd.DataFrame(returns_data)
        
        result = decile_portfolio_returns(signals_df, returns_df, n_deciles=10)
        
        # Should have 10 deciles + L-S row
        assert len(result) == 11
        # Check that deciles are 1-10
        decile_numbers = result[result["decile"] != "L-S"]["decile"]
        assert all(1 <= d <= 10 for d in decile_numbers)
    
    def test_deciles_long_short_sign(self):
        """Test that L-S return has correct sign for monotonic relationship."""
        # Create data where higher signal implies higher return
        dates = ["2023-01-01", "2023-01-02"]
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        
        signals_data = []
        returns_data = []
        
        for date in dates:
            for i, ticker in enumerate(tickers):
                signals_data.append({
                    "asof_dt": date,
                    "ticker": ticker,
                    "signal": i  # Monotonic signal
                })
                returns_data.append({
                    "asof_dt": date,
                    "ticker": ticker,
                    "ret_fwd1": i * 0.01  # Monotonic returns
                })
        
        signals_df = pd.DataFrame(signals_data)
        returns_df = pd.DataFrame(returns_data)
        
        result = decile_portfolio_returns(signals_df, returns_df, n_deciles=5)
        
        # Find L-S row
        ls_row = result[result["decile"] == "L-S"]
        assert len(ls_row) == 1
        assert ls_row.iloc[0]["mean_ret"] > 0  # Should be positive for monotonic relationship
    
    def test_deciles_insufficient_data(self):
        """Test handles insufficient data for deciles."""
        # Create data with fewer tickers than deciles
        signals_df = pd.DataFrame({
            "asof_dt": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "ticker": ["A", "B", "C"],
            "signal": [1, 2, 3]
        })
        returns_df = pd.DataFrame({
            "asof_dt": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "ticker": ["A", "B", "C"],
            "ret_fwd1": [0.1, 0.2, 0.3]
        })
        
        result = decile_portfolio_returns(signals_df, returns_df, n_deciles=10)
        
        # Should return empty DataFrame
        assert len(result) == 0 