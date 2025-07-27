# Quant Engine

A quantitative equity portfolio construction engine for systematic trading strategies.

## Overview

The Quant Engine provides a complete pipeline for quantitative equity portfolio construction, featuring:

- **Signal Generation**: Momentum and value factor signals with configurable weights
- **Risk Management**: Covariance estimation with shrinkage and diagonal loading
- **Portfolio Optimization**: Mean-variance optimization with realistic constraints
- **Point-in-Time Discipline**: Proper handling of fundamental data reporting lags
- **Validation Framework**: Comprehensive pre-trade checks and safety validations

## Project Structure

```
quant_engine/
├── src/quant_engine/     # Core engine modules (new package name)
│   ├── data_io.py        # Data loading and validation
│   ├── signals.py        # Signal generation
│   ├── prep.py           # Data preparation
│   ├── risk.py           # Risk model
│   ├── optimize.py       # Portfolio optimization
│   ├── checks.py         # Pre-trade validation
│   ├── utils.py          # Utilities
│   └── run_day.py        # Main execution pipeline
├── src/engine/           # Backward compatibility shim
├── tests/                # Unit tests
├── data/                 # Input data files
├── reports/              # Output files
├── configs/              # Configuration files
└── README.md
```

## Quick Start

### Installation

```bash
pip install numpy pandas scipy pyyaml pytest
```

### Usage

```bash
# Run tests
python -m pytest tests/

# Execute portfolio construction (multiple options)
python -m src.quant_engine.run_day --asof 2024-01-15 --config configs/config.yaml
python -m src.engine.run_day --asof 2024-01-15 --config configs/config.yaml  # legacy path
qe-run --asof 2024-01-15 --config configs/config.yaml  # console script (after installation)
```

### Configuration

Edit `configs/config.yaml` to customize:
- Signal parameters and weights
- Risk model settings
- Optimization constraints
- Validation thresholds

### Input Data

Required CSV files in `data/` directory:
- `prices.csv`: Daily close prices
- `fundamentals.csv`: Fundamental data with reporting lags
- `sectors.csv`: Sector classifications
- `holdings_prior.csv`: Prior portfolio weights

### Output

Generated files:
- `holdings.csv`: Final portfolio weights
- `trades.csv`: Trade recommendations
- `report.txt`: Performance and validation report

## Design Philosophy

### Signal Design
- **Momentum**: 252-day lookback with 21-day gap
- **Value**: E/P ratio for simplicity
- **Combination**: Configurable weights via YAML

### Risk Management
- **Covariance**: 60-day rolling window
- **Shrinkage**: λ=0.3 to diagonal
- **Stability**: Diagonal loading (1e-4)

### Optimization
- **Constraints**: Long-only with realistic caps
- **Fallback**: Prior weights if infeasible
- **Objective**: Mean-variance optimization

### Point-in-Time Discipline
- **Reporting lags**: Enforced via report_lag_days
- **Data alignment**: Strict date-based alignment
- **No look-ahead**: All calculations use only available data

## Requirements

- Python 3.10+
- numpy, pandas, scipy, pyyaml, pytest
- Deterministic execution with seeded random states 