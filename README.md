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

```bash
# Run tests
pytest tests/

# Execute portfolio construction
python -m quant_engine.run_day --asof 2024-01-15 --config configs/config.yaml

# Generate synthetic test data
python data_fetch/make_synth_data.py
```

### Configuration

Edit `configs/config.yaml` to customize:
- Signal parameters and weights
- Risk model settings
- Optimization constraints
- Trading filters
- Validation thresholds

### Input Data

Required CSV files in `data/` directory:
- `prices.csv`: Daily close prices (asof_dt, ticker, close)
- `fundamentals.csv`: Fundamental data with reporting lags
- `sectors.csv`: Sector classifications
- `holdings_prev.csv`: Prior portfolio weights

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

## Requirements

- Python 3.10+
- numpy, pandas, scipy, pyyaml, pytest 