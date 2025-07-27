# Quant Engine

A quantitative equity portfolio construction engine for systematic trading strategies.

## Overview

The Quant Engine provides a complete pipeline for quantitative equity portfolio construction, featuring:

- **Signal Generation**: Momentum and value factor signals with configurable weights
- **Risk Management**: Covariance estimation with shrinkage, diagonal loading, and comprehensive diagnostics
- **Portfolio Optimization**: Mean-variance optimization with realistic constraints and fixed weights support
- **Trade Filtering**: Intelligent small trade filtering with re-optimization
- **Point-in-Time Discipline**: Proper handling of fundamental data reporting lags
- **Validation Framework**: Comprehensive pre-trade checks, data diagnostics, and safety validations
- **Comprehensive Testing**: 98+ unit tests covering all core functionality

## Project Structure

```
quant_engine/
├── src/quant_engine/     # Core engine modules
│   ├── data_io.py        # Data loading and validation
│   ├── signals.py        # Signal generation (momentum, value)
│   ├── prep.py           # Data preparation (winsorize, zscore, sector neutralize)
│   ├── risk.py           # Risk model (covariance, shrinkage, diagnostics)
│   ├── optimize.py       # Portfolio optimization (mean-variance with fixed weights)
│   ├── checks.py         # Pre-trade validation and data diagnostics
│   ├── trade_filters.py  # Small trade filtering and re-optimization
│   ├── utils.py          # Utilities (logging, config validation, IC analysis)
│   └── run_day.py        # Main execution pipeline

├── tests/                # Comprehensive test suite (98+ tests)
├── data/                 # Input data files
├── data_fetch/           # Synthetic data generation
├── reports/              # Output files
├── configs/              # Configuration files
└── README.md
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install numpy pandas scipy pyyaml pytest

# Install package in development mode
pip install -e .
```

### Usage

```bash
# Run all tests (98+ tests)
pytest tests/

# Execute portfolio construction
python -m quant_engine.run_day --asof 2024-01-15 --config configs/config.yaml
qe-run --asof 2024-01-15 --config configs/config.yaml  # console script (after installation)

# Generate synthetic test data
python data_fetch/make_synth_data.py
```

### Configuration

Edit `configs/config.yaml` to customize:

**Core Settings:**
- Signal parameters and weights (momentum lookback, value lag)
- Risk model settings (covariance window, shrinkage, diagonal loading)
- Optimization constraints (position limits, sector caps, turnover)
- Validation thresholds (missing data, turnover limits)

**Advanced Features:**
- Trading filters (min_weight, min_notional, AUM)
- Logging configuration (level, format)
- Performance settings (random seed for reproducibility)
- Output customization (risk contributors count)

### Input Data

Required CSV files in `data/` directory:
- `prices.csv`: Daily close prices with asof_dt, ticker, close columns
- `fundamentals.csv`: Fundamental data with reporting lags (eps_ttm, book_value_ps)
- `sectors.csv`: Sector classifications
- `holdings_prev.csv`: Prior portfolio weights

### Output

Generated files in `data/outputs/`:
- `holdings.csv`: Final portfolio weights
- `trades.csv`: Trade recommendations
- `reports/YYYY-MM-DD_report.txt`: Comprehensive performance and validation report

## Key Features

### Signal Generation
- **Momentum**: 252-day lookback with 21-day gap (configurable)
- **Value**: E/P ratio with proper fundamental data lag handling
- **Processing**: Winsorization, z-scoring, sector neutralization
- **Combination**: Configurable weights via YAML

### Risk Management
- **Covariance Estimation**: 60-day rolling window with shrinkage (λ=0.3)
- **Stability**: Diagonal loading (1e-4) for numerical conditioning
- **Diagnostics**: Condition number, eigenvalues, asymmetry checks
- **Risk Contributions**: Marginal and component risk analysis

### Portfolio Optimization
- **Mean-Variance**: Risk-averse optimization with realistic constraints
- **Fixed Weights**: Support for freezing specific tickers at their weights
- **Constraints**: Long-only, position limits, sector caps, turnover limits
- **Fallback**: Prior weights if optimization fails

### Trade Filtering
- **Small Trade Detection**: Configurable minimum weight and notional thresholds
- **Re-optimization**: Automatic re-optimization with frozen weights
- **Non-blocking**: Graceful fallback if re-optimization fails
- **Reporting**: Comprehensive trade filter statistics

### Validation & Diagnostics
- **Pre-trade Checks**: Schema validation, missingness, turnover, sector exposure
- **Data Diagnostics**: Schema drift detection, extreme value flagging
- **Risk Diagnostics**: Covariance matrix health, top risk contributors
- **Trade Filter Reporting**: Frozen names, turnover impact, re-optimization status

### Point-in-Time Discipline
- **Reporting Lags**: Enforced via available_asof field
- **Data Alignment**: Strict date-based alignment
- **No Look-ahead**: All calculations use only available data
- **Fundamental Timing**: Proper handling of earnings announcement delays

## Advanced Usage

### Trade Filtering Configuration

```yaml
trading:
  aum: 10000000          # Assets under management (for notional filtering)
  min_weight: 0.0005     # Minimum trade size as weight (5 bps)
  min_notional: 10000    # Minimum trade size in currency ($10k floor)
```

### Fixed Weights Optimization

```python
from quant_engine.optimize import mean_variance_opt

# Freeze specific tickers at their previous weights
fixed_weights = pd.Series([0.05, 0.03], index=['AAPL', 'MSFT'])
weights, diagnostics = mean_variance_opt(
    alpha, Sigma, sectors_map, prev_w,
    fixed_weights=fixed_weights,  # New parameter
    # ... other parameters
)
```

### Custom Trade Filters

```python
from quant_engine.trade_filters import apply_no_trade_band

# Apply custom trade filtering
new_w_frozen, freeze_mask, stats = apply_no_trade_band(
    prev_w, new_w, prices, aum,
    min_weight=0.001,  # 10 bps
    min_notional=5000  # $5k minimum
)
```

## Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_optimize.py
pytest tests/test_trade_filters.py
pytest tests/test_risk.py

# Run with coverage
pytest --cov=src/quant_engine tests/
```

**Test Coverage:**
- **98+ tests** covering all core functionality
- **Unit tests** for individual functions and classes
- **Integration tests** for end-to-end workflows
- **Edge case handling** for robustness
- **Performance validation** for optimization algorithms

## Requirements

- **Python**: 3.10+
- **Core Dependencies**: numpy, pandas, scipy, pyyaml
- **Testing**: pytest
- **Development**: pip install -e . for editable installation

## Recent Updates

### Latest Features (v2.0+)
- **Trade Filtering**: Intelligent small trade filtering with re-optimization
- **Fixed Weights**: Support for freezing specific tickers during optimization
- **Risk Diagnostics**: Comprehensive covariance matrix analysis and risk contributions
- **Data Diagnostics**: Schema drift detection and extreme value monitoring
- **Enhanced Logging**: Configurable logging with structured output
- **Comprehensive Testing**: 98+ tests with full coverage of core functionality

### Performance Improvements
- **Optimized Algorithms**: Improved numerical stability in optimization
- **Memory Efficiency**: Better handling of large datasets
- **Error Handling**: Robust fallback mechanisms for edge cases
- **Reproducibility**: Deterministic execution with seeded random states 