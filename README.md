# Quant Engine

A quantitative equity portfolio construction engine for systematic trading strategies.

## Overview

The Quant Engine provides a complete pipeline for quantitative equity portfolio construction, featuring:

- **Signal Generation**: Momentum and value factor signals with configurable weights
- **Risk Management**: Covariance estimation with shrinkage and diagonal loading
- **Portfolio Optimization**: Mean-variance optimization with realistic constraints
- **Trade Filtering**: Small trade filtering with re-optimization
- **Point-in-Time Discipline**: Proper handling of fundamental data reporting lags
- **Validation Framework**: Pre-trade checks and data diagnostics

## Project Structure

```
quant_engine/
├── src/quant_engine/     # Core engine modules
│   ├── data_io.py        # Data loading and validation
│   ├── signals.py        # Signal generation
│   ├── prep.py           # Data preparation
│   ├── risk.py           # Risk model
│   ├── optimize.py       # Portfolio optimization
│   ├── checks.py         # Pre-trade validation
│   ├── trade_filters.py  # Trade filtering
│   ├── utils.py          # Utilities
│   └── run_day.py        # Main execution pipeline

├── tests/                # Test suite
├── data/                 # Input data files
├── data_fetch/           # Synthetic data generation
├── configs/              # Configuration files
└── README.md
```

## Quick Start

### Installation

```bash
pip install numpy pandas scipy pyyaml pytest
pip install -e .
```

### Usage

#### Command Line Interface (CLI)

The quant engine provides a convenient CLI for running portfolio construction:

```bash
# Basic usage with CLI command
qe-run --asof 2023-11-30 --config configs/config.yaml

# Alternative: direct Python module execution
python -m quant_engine.run_day --asof 2023-11-30 --config configs/config.yaml

# Check CLI help
qe-run --help
```

#### Data Generation

```bash
# Generate synthetic test data
python data_fetch/make_synth_data.py

# Run alpha capture analysis
python scripts/alpha_capture.py --config configs/config.yaml --alpha-gt-path data/alpha_gt.csv --asof 2023-11-30 --window 60
```

#### Testing

```bash
# Run all tests (excluding slow synthetic data tests)
pytest

# Run all tests including slow ones
pytest -m "not slow"

# Run specific test file
pytest tests/test_run_day.py

# Run tests with verbose output
pytest -v
```

### CLI Arguments

The `qe-run` command accepts the following arguments:

```bash
qe-run --asof DATE --config CONFIG_FILE [OPTIONS]

Arguments:
  --asof DATE           Date to run portfolio construction (YYYY-MM-DD format)
  --config CONFIG_FILE  Path to YAML configuration file

Examples:
  qe-run --asof 2023-11-30 --config configs/config.yaml
  qe-run --asof 2024-01-15 --config configs/config_simple.yaml
```

### Configuration

Edit `configs/config.yaml` to customize:
- Signal parameters and weights
- Risk model settings
- Optimization constraints
- Trading filters
- Validation thresholds

The configuration file uses YAML format for easy readability and modification.

### Input Data

Required CSV files in `data/` directory:
- `prices.csv`: Daily close prices (asof_dt, ticker, close)
- `fundamentals.csv`: Fundamental data with reporting lags
- `sectors.csv`: Sector classifications
- `holdings_prev.csv`: Prior portfolio weights

### Typical Workflow

1. **Generate synthetic data** (for testing):
   ```bash
   python data_fetch/make_synth_data.py
   ```

2. **Run portfolio construction**:
   ```bash
   qe-run --asof 2023-11-30 --config configs/config.yaml
   ```

3. **Analyze alpha capture** (optional):
   ```bash
   python scripts/alpha_capture.py --config configs/config.yaml --alpha-gt-path data/alpha_gt.csv --asof 2023-11-30 --window 60
   ```

4. **Check results** in `data/outputs/` and `reports/`

### Output

Generated files in `data/outputs/`:
- `holdings.csv`: Final portfolio weights
- `trades.csv`: Trade recommendations
- `reports/YYYY-MM-DD_report.txt`: Performance and validation report

## Features

### Signal Generation
- **Momentum**: 252-day lookback with 21-day gap
- **Value**: E/P ratio with fundamental data lag handling
- **Processing**: Winsorization, z-scoring, sector neutralization

### Risk Management
- **Covariance**: 60-day rolling window with shrinkage
- **Stability**: Diagonal loading for numerical conditioning
- **Diagnostics**: Condition number, eigenvalues, risk contributions

### Portfolio Optimization
- **Mean-Variance**: Risk-averse optimization with constraints
- **Fixed Weights**: Support for freezing specific tickers
- **Constraints**: Long-only, position limits, sector caps, turnover limits

### Trade Filtering
- **Small Trade Detection**: Configurable minimum weight and notional thresholds
- **Re-optimization**: Automatic re-optimization with frozen weights
- **Non-blocking**: Graceful fallback if re-optimization fails

### Validation
- **Pre-trade Checks**: Schema validation, missingness, turnover, sector exposure
- **Data Diagnostics**: Schema drift detection, extreme value flagging
- **Risk Diagnostics**: Covariance matrix health, top risk contributors

## Troubleshooting

### Common CLI Issues

**Command not found: `qe-run`**
```bash
# Make sure the package is installed in editable mode
pip install -e .

# Or use the direct Python module
python -m quant_engine.run_day --asof 2023-11-30 --config configs/config.yaml
```

**Date not found in prices data**
```bash
# Check available dates in your data
python -c "import pandas as pd; df = pd.read_csv('data/prices.csv'); print(sorted(df['asof_dt'].unique())[-5:])"
```

**Configuration file not found**
```bash
# Verify the config file exists
ls configs/
# Use absolute path if needed
qe-run --asof 2023-11-30 --config /full/path/to/configs/config.yaml
```

### Data Requirements

Ensure your input data files have the correct format:
- `prices.csv`: columns `asof_dt`, `ticker`, `close`
- `fundamentals.csv`: columns `asof_dt`, `ticker`, `field`, `value`, `available_asof`
- `sectors.csv`: columns `ticker`, `sector`
- `holdings_prev.csv`: columns `ticker`, `weight`

## Requirements

- Python 3.10+
- numpy, pandas, scipy, pyyaml, pytest 