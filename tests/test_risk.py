"""
Tests for risk module.
"""

import pytest
import pandas as pd
import numpy as np
from quant_engine.risk import (
    validate_covariance_matrix, returns_from_prices, shrink_cov,
    marginal_risk_contribution
)


class TestReturnsFromPrices:
    """Test returns matrix construction from prices."""
    
    def test_returns_windowing_and_pit(self):
        """Test returns windowing and point-in-time discipline."""
        # Create a tiny prices DF for two tickers over 65 business days
        dates = pd.date_range("2023-01-01", periods=65, freq="B")
        tickers = ["AAPL", "MSFT"]
        
        # Create known pattern: AAPL grows linearly, MSFT grows exponentially
        prices_data = []
        for i, date in enumerate(dates):
            for j, ticker in enumerate(tickers):
                if ticker == "AAPL":
                    price = 100 + i * 0.5  # Linear growth
                else:  # MSFT
                    price = 200 * (1.01 ** i)  # Exponential growth
                prices_data.append({
                    "asof_dt": date,
                    "ticker": ticker,
                    "close": price
                })
        
        prices_df = pd.DataFrame(prices_data)
        asof_dt = dates[-1]  # Last available date
        
        result = returns_from_prices(prices_df, asof_dt, lookback_days=60)
        
        # Check shape: should be (n_tickers_with_full_window, 59) since we lose one column in returns calculation
        assert result.shape == (2, 59)
        
        # Check that the last column equals (P_t / P_{t-1} - 1) at t=asof_dt
        # For AAPL: linear growth, so return should be approximately constant
        aapl_returns = result.loc["AAPL"]
        # For linear growth: price = 100 + i * 0.5, so return ≈ 0.5 / (100 + i * 0.5)
        # At the end, this is approximately 0.5 / (100 + 64 * 0.5) ≈ 0.0038
        assert abs(aapl_returns.iloc[-1] - 0.0038) < 0.001  # Approximate return
        
        # For MSFT: exponential growth, so return should be approximately 0.01
        msft_returns = result.loc["MSFT"]
        # For exponential growth: price = 200 * (1.01)^i, so return ≈ 0.01
        assert abs(msft_returns.iloc[-1] - 0.01) < 0.001  # Approximate return
        
        # Check column names: should be 1..59
        assert list(result.columns) == list(range(1, 60))
        
        # Check index: should be sorted tickers
        assert list(result.index) == ["AAPL", "MSFT"]
    
    def test_returns_requires_full_window(self):
        """Test that tickers missing data are excluded."""
        # Create data where one ticker is missing a few days
        dates = pd.date_range("2023-01-01", periods=65, freq="B")
        tickers = ["AAPL", "MSFT"]
        
        prices_data = []
        for i, date in enumerate(dates):
            for j, ticker in enumerate(tickers):
                # Skip a few days for MSFT to create gaps
                if ticker == "MSFT" and 30 <= i <= 32:
                    continue
                
                price = 100 + i * 0.5 if ticker == "AAPL" else 200 + i * 0.3
                prices_data.append({
                    "asof_dt": date,
                    "ticker": ticker,
                    "close": price
                })
        
        prices_df = pd.DataFrame(prices_data)
        asof_dt = dates[-1]
        
        result = returns_from_prices(prices_df, asof_dt, lookback_days=60)
        
        # Only AAPL should have full window, MSFT should be excluded
        assert len(result) == 1
        assert "AAPL" in result.index
        assert "MSFT" not in result.index
    
    def test_returns_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create data with fewer days than lookback
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        prices_data = []
        for i, date in enumerate(dates):
            prices_data.append({
                "asof_dt": date,
                "ticker": "AAPL",
                "close": 100 + i * 0.5
            })
        
        prices_df = pd.DataFrame(prices_data)
        asof_dt = dates[-1]
        
        result = returns_from_prices(prices_df, asof_dt, lookback_days=60)
        
        # Should return empty DataFrame
        assert result.empty


class TestShrinkCov:
    """Test covariance shrinkage functionality."""
    
    def test_shrink_cov_shapes_and_symmetry(self):
        """Test covariance matrix shapes and symmetry."""
        np.random.seed(42)
        
        # Build a random-but-seeded returns matrix (5 tickers x 60)
        n_tickers, n_days = 5, 60
        returns_data = np.random.randn(n_tickers, n_days) * 0.02  # 2% daily vol
        
        returns_df = pd.DataFrame(
            returns_data,
            index=[f"T{i}" for i in range(n_tickers)],
            columns=range(1, n_days + 1)
        )
        
        cov_matrix = shrink_cov(returns_df)
        
        # Check shape
        assert cov_matrix.shape == (n_tickers, n_tickers)
        
        # Check symmetry: max|Σ-Σ.T| < 1e-12
        symmetry_error = np.max(np.abs(cov_matrix.values - cov_matrix.values.T))
        assert symmetry_error < 1e-12
        
        # Check diagonal strictly > 0
        assert all(np.diag(cov_matrix.values) > 0)
        
        # Check index/columns match
        assert list(cov_matrix.index) == list(cov_matrix.columns)
        assert list(cov_matrix.index) == [f"T{i}" for i in range(n_tickers)]
    
    def test_shrink_lambda_extremes(self):
        """Test shrinkage with extreme lambda values."""
        np.random.seed(42)
        
        # Create returns matrix
        returns_data = np.random.randn(3, 60) * 0.02
        returns_df = pd.DataFrame(
            returns_data,
            index=["A", "B", "C"],
            columns=range(1, 61)
        )
        
        # Test lam=0 (sample covariance)
        cov_sample = shrink_cov(returns_df, lam=0.0, diag_load=0.0)
        
        # Test lam=1 (diagonal only)
        cov_diag = shrink_cov(returns_df, lam=1.0, diag_load=0.0)
        
        # With lam=1, off-diagonals should be approximately 0
        off_diag_elements = cov_diag.values[~np.eye(3, dtype=bool)]
        assert np.allclose(off_diag_elements, 0, atol=1e-10)
        
        # With lam=0, should be close to sample covariance
        sample_cov_np = np.cov(returns_data, rowvar=True, ddof=0)
        sample_cov_df = pd.DataFrame(sample_cov_np, index=["A", "B", "C"], columns=["A", "B", "C"])
        assert np.allclose(cov_sample.values, sample_cov_df.values, atol=1e-10)
    
    def test_diag_load_improves_conditioning(self):
        """Test that diagonal loading improves conditioning."""
        # Construct nearly collinear returns for two tickers
        np.random.seed(42)
        base_returns = np.random.randn(60) * 0.02
        
        # Ticker B is almost a multiple of ticker A (nearly collinear)
        returns_data = np.array([
            base_returns,
            base_returns * 1.001 + np.random.randn(60) * 0.0001,  # Almost collinear
            np.random.randn(60) * 0.02  # Independent third ticker
        ])
        
        returns_df = pd.DataFrame(
            returns_data,
            index=["A", "B", "C"],
            columns=range(1, 61)
        )
        
        # Compute covariance without diagonal loading
        cov_no_load = shrink_cov(returns_df, lam=0.3, diag_load=0.0)
        
        # Compute covariance with diagonal loading
        cov_with_load = shrink_cov(returns_df, lam=0.3, diag_load=1e-4)
        
        # Calculate condition numbers
        cond_no_load = np.linalg.cond(cov_no_load.values)
        cond_with_load = np.linalg.cond(cov_with_load.values)
        
        # Diagonal loading should improve conditioning (lower condition number)
        assert cond_with_load < cond_no_load
    
    def test_empty_returns(self):
        """Test handling of empty returns DataFrame."""
        empty_returns = pd.DataFrame()
        result = shrink_cov(empty_returns)
        assert result.empty 
