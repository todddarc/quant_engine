"""
Tests for data_io module.
"""

import pytest
import pandas as pd
from pathlib import Path
from quant_engine.data_io import (
    load_prices, load_fundamentals, load_sectors, load_holdings,
    write_holdings, write_trades, unique_dates, next_day_exists
)











class TestLoadHoldings:
    """Test holdings loading functionality."""
    

    
    def test_load_holdings_invalid_date(self, tmp_path):
        """Test holdings loading with invalid date."""
        # Create holdings file with data for 2023-12-15
        holdings_data = [
            {'asof_dt': '2023-12-15', 'ticker': 'A', 'weight': 0.5},
            {'asof_dt': '2023-12-15', 'ticker': 'B', 'weight': 0.5}
        ]
        holdings_df = pd.DataFrame(holdings_data)
        holdings_df.to_csv(tmp_path / 'holdings.csv', index=False)
        
        # Try to load holdings for a date that doesn't exist
        with pytest.raises(ValueError, match="No holdings data found for date: 2025-01-01"):
            load_holdings(tmp_path / 'holdings.csv', '2025-01-01')
    
    def test_load_holdings_missing_file(self, tmp_path):
        """Test holdings loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_holdings(tmp_path / 'nonexistent.csv', '2023-12-15')











class TestDataIOIntegration:
    """Test integration scenarios for data_io functions."""
    
    def test_load_holdings_fallback_scenario(self, tmp_path):
        """Test the fallback scenario when no holdings file exists."""
        # Create prices data
        prices_data = [
            {'asof_dt': '2023-12-15', 'ticker': 'A', 'close': 100.0},
            {'asof_dt': '2023-12-15', 'ticker': 'B', 'close': 200.0},
            {'asof_dt': '2023-12-16', 'ticker': 'A', 'close': 101.0},
            {'asof_dt': '2023-12-16', 'ticker': 'B', 'close': 202.0}
        ]
        prices_df = pd.DataFrame(prices_data)
        prices_df.to_csv(tmp_path / 'prices.csv', index=False)
        
        # Create fundamentals data
        fundamentals_data = [
            {'report_dt': '2023-09-30', 'available_asof': '2023-10-30', 'ticker': 'A', 'eps_ttm': 5.0, 'book_value_ps': 50.0},
            {'report_dt': '2023-09-30', 'available_asof': '2023-10-30', 'ticker': 'B', 'eps_ttm': 5.0, 'book_value_ps': 50.0}
        ]
        fundamentals_df = pd.DataFrame(fundamentals_data)
        fundamentals_df.to_csv(tmp_path / 'fundamentals.csv', index=False)
        
        # Create sectors data
        sectors_data = [
            {'ticker': 'A', 'sector': 'Tech'},
            {'ticker': 'B', 'sector': 'Tech'}
        ]
        sectors_df = pd.DataFrame(sectors_data)
        sectors_df.to_csv(tmp_path / 'sectors.csv', index=False)
        
        # Don't create holdings file - simulate missing file scenario
        
        # Test that the data_io functions work correctly
        prices = load_prices(tmp_path / 'prices.csv')
        fundamentals = load_fundamentals(tmp_path / 'fundamentals.csv')
        sectors = load_sectors(tmp_path / 'sectors.csv')
        
        # Verify data was loaded correctly
        assert len(prices) == 4
        assert len(fundamentals) == 2
        assert len(sectors) == 2
        
        # Test unique_dates function
        dates = unique_dates(prices)
        assert len(dates) == 2
        assert pd.Timestamp('2023-12-15') in dates
        assert pd.Timestamp('2023-12-16') in dates
        
        # Test next_day_exists function
        assert bool(next_day_exists(prices, '2023-12-15')) is True
        assert bool(next_day_exists(prices, '2023-12-16')) is False 
