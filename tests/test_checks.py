"""
Tests for checks module.
"""

import pytest
import pandas as pd
import numpy as np
from src.quant_engine.checks import (
    validate_data, check_turnover, check_sector_exposure,
    check_data_missingness, check_schema_drift, check_extreme_values,
    run_all_checks, check_schema, check_missingness, aggregate_checks
)


class TestValidateData:
    """Test data validation functionality."""
    
    def test_validate_data_valid(self):
        """Test validation of valid data."""
        pytest.skip("not implemented")
    
    def test_validate_data_invalid_schema(self):
        """Test validation with invalid schema."""
        pytest.skip("not implemented")
    
    def test_validate_data_missing_required(self):
        """Test validation with missing required data."""
        pytest.skip("not implemented")


class TestCheckTurnover:
    """Test turnover checking functionality."""
    
    def test_check_turnover_acceptable(self):
        """Test turnover check with acceptable levels."""
        pytest.skip("not implemented")
    
    def test_check_turnover_excessive(self):
        """Test turnover check with excessive levels."""
        pytest.skip("not implemented")
    
    def test_check_turnover_zero(self):
        """Test turnover check with zero turnover."""
        pytest.skip("not implemented")


class TestCheckSectorExposure:
    """Test sector exposure checking."""
    
    def test_check_sector_exposure_acceptable(self):
        """Test sector exposure check with acceptable changes."""
        pytest.skip("not implemented")
    
    def test_check_sector_exposure_excessive(self):
        """Test sector exposure check with excessive changes."""
        pytest.skip("not implemented")


class TestCheckDataMissingness:
    """Test missing data checking."""
    
    def test_check_data_missingness_acceptable(self):
        """Test missing data check with acceptable levels."""
        pytest.skip("not implemented")
    
    def test_check_data_missingness_excessive(self):
        """Test missing data check with excessive levels."""
        pytest.skip("not implemented")


class TestCheckSchemaDrift:
    """Test schema drift checking."""
    
    def test_check_schema_drift_valid(self):
        """Test schema drift check with valid schema."""
        pytest.skip("not implemented")
    
    def test_check_schema_drift_invalid(self):
        """Test schema drift check with invalid schema."""
        pytest.skip("not implemented")


class TestCheckExtremeValues:
    """Test extreme value checking."""
    
    def test_check_extreme_values_iqr(self):
        """Test extreme value check using IQR method."""
        pytest.skip("not implemented")
    
    def test_check_extreme_values_zscore(self):
        """Test extreme value check using z-score method."""
        pytest.skip("not implemented")
    
    def test_check_extreme_values_percentile(self):
        """Test extreme value check using percentile method."""
        pytest.skip("not implemented")


class TestRunAllChecks:
    """Test running all validation checks."""
    
    def test_run_all_checks_pass(self):
        """Test running all checks with passing results."""
        pytest.skip("not implemented")
    
    def test_run_all_checks_fail(self):
        """Test running all checks with failing results."""
        pytest.skip("not implemented")
    
    def test_run_all_checks_mixed(self):
        """Test running all checks with mixed results."""
        pytest.skip("not implemented") 


def test_check_schema_pass_and_block():
    """Test schema check with passing and blocking cases."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    
    # Test passing case
    ok = check_schema(df, ["a", "b"])
    assert ok["schema"]["status"] == "PASS"
    
    # Test blocking case
    bad = check_schema(df, ["a", "b", "c"])
    assert bad["schema"]["status"] == "BLOCK"
    assert "c" in bad["schema"]["details"]


def test_check_missingness_thresholds():
    """Test missingness check with different thresholds."""
    df = pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [1, 2, 3, 4]})
    
    # Test passing case (30% NA rate allowed)
    r1 = check_missingness(df, max_rate=0.30)
    assert r1["missingness"]["status"] == "PASS"
    
    # Test blocking case (20% NA rate threshold)
    r2 = check_missingness(df, max_rate=0.20)
    assert r2["missingness"]["status"] == "BLOCK"
    assert "x" in r2["missingness"]["details"]


def test_check_turnover_cap_enforced():
    """Test turnover check with cap enforcement."""
    idx = ["A", "B", "C", "D"]
    prev_w = pd.Series([0.25, 0.25, 0.25, 0.25], index=idx)
    
    # Shift 10% of the book from C to A => turnover = 0.5*(0.10+0.10)=0.10
    new_w = pd.Series([0.35, 0.25, 0.15, 0.25], index=idx)
    ok = check_turnover(prev_w, new_w, cap=0.10)
    assert ok["turnover"]["status"] == "PASS"
    
    # Exceed cap slightly
    new_w2 = pd.Series([0.36, 0.25, 0.14, 0.25], index=idx)  # turnover = 0.11
    bad = check_turnover(prev_w, new_w2, cap=0.10)
    assert bad["turnover"]["status"] == "BLOCK"


def test_check_turnover_handles_new_and_missing_names():
    """Test turnover check with new and missing tickers."""
    prev_w = pd.Series({"A": 0.5, "B": 0.5})
    new_w = pd.Series({"B": 0.3, "C": 0.7})  # A->0, C from 0
    
    # Turnover = 0.5*(|0-0.5|+|0.3-0.5|+|0.7-0|) = 0.5*(0.5+0.2+0.7)=0.7
    res = check_turnover(prev_w, new_w, cap=0.8)
    assert res["turnover"]["status"] == "PASS"


def test_check_sector_exposure_caps():
    """Test sector exposure check with different caps."""
    new_w = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.70})
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Health", "D": "Energy"})
    
    # With cap=0.60, Energy violates (0.70)
    bad = check_sector_exposure(new_w, sectors, cap=0.60)
    assert bad["sector_exposure"]["status"] == "BLOCK"
    
    # With cap=0.75, all pass
    ok = check_sector_exposure(new_w, sectors, cap=0.75)
    assert ok["sector_exposure"]["status"] == "PASS"


def test_aggregate_checks_logic():
    """Test aggregation of check results."""
    checks = {
        "schema": {"status": "PASS", "details": ""},
        "missingness": {"status": "PASS", "details": ""},
        "turnover": {"status": "BLOCK", "details": "over cap"}
    }
    
    ok, out = aggregate_checks(checks)
    assert ok is False
    assert out["turnover"]["status"] == "BLOCK"
    
    # Test all passing
    checks_all_pass = {
        "schema": {"status": "PASS", "details": ""},
        "missingness": {"status": "PASS", "details": ""}
    }
    ok, out = aggregate_checks(checks_all_pass)
    assert ok is True


def test_check_schema_empty_dataframe():
    """Test schema check with empty DataFrame."""
    df = pd.DataFrame()
    result = check_schema(df, ["a", "b"])
    assert result["schema"]["status"] == "BLOCK"
    assert "a" in result["schema"]["details"]


def test_check_missingness_empty_dataframe():
    """Test missingness check with empty DataFrame."""
    df = pd.DataFrame()
    result = check_missingness(df, max_rate=0.1)
    assert result["missingness"]["status"] == "PASS"


def test_check_turnover_duplicate_indices():
    """Test turnover check handles duplicate indices."""
    prev_w = pd.Series([0.3, 0.2, 0.5], index=["A", "A", "B"])  # Duplicate A
    new_w = pd.Series([0.4, 0.1, 0.5], index=["A", "B", "C"])
    
    # Should sum duplicate A weights: prev_w["A"] = 0.3 + 0.2 = 0.5
    result = check_turnover(prev_w, new_w, cap=0.5)
    # Turnover = 0.5*(|0.4-0.5|+|0.1-0.0|+|0.5-0.0|) = 0.5*(0.1+0.1+0.5) = 0.35
    assert result["turnover"]["status"] == "PASS"


def test_check_sector_exposure_dict_input():
    """Test sector exposure check with dict input."""
    new_w = pd.Series({"A": 0.3, "B": 0.2, "C": 0.5})  # Tech: 0.5, Health: 0.5
    sectors_dict = {"A": "Tech", "B": "Tech", "C": "Health"}
    
    result = check_sector_exposure(new_w, sectors_dict, cap=0.5)
    assert result["sector_exposure"]["status"] == "PASS"


def test_check_sector_exposure_missing_sectors():
    """Test sector exposure check with missing sector labels."""
    new_w = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
    sectors = pd.Series({"A": "Tech", "B": np.nan, "C": "Health"})  # B has no sector
    
    result = check_sector_exposure(new_w, sectors, cap=0.5)
    assert result["sector_exposure"]["status"] == "PASS"
    assert "1 tickers without sector labels" in result["sector_exposure"]["details"] 