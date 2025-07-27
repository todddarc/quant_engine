# Quant Engine

A minimal quantitative equity portfolio construction engine for IPM interview demonstration.

## Project Charter

### Goals
- Demonstrate complete quantitative equity portfolio construction pipeline
- Show point-in-time discipline with proper fundamental data lag handling (report_lag_days)
- Validate alpha signal quality through IC analysis and decile performance
- Generate tradeable portfolio weights with comprehensive risk management
- Provide clear, auditable output with performance metrics and safety checks

### Non-Goals
- Real-time data feeds or market connectivity
- Complex factor models or alternative data sources
- Transaction cost modeling or implementation shortfall
- Multi-period optimization or dynamic rebalancing
- Performance attribution or factor decomposition
- Production-grade error handling or monitoring
- B/M ratio implementation (eps_ttm and book_value_ps provided for future use)

### Data Contracts

#### prices.csv
- asof_dt: datetime64[ns]
- ticker: str
- close: float64

#### fundamentals.csv
- report_dt: datetime64[ns]
- ticker: str
- eps_ttm: float64
- book_value_ps: float64
- report_lag_days: int64

#### sectors.csv
- ticker: str
- sector: str

#### holdings_prior.csv
- ticker: str
- weight: float64

### Constraints
- Python 3.10+ only
- numpy, pandas, scipy dependencies only
- Deterministic execution with seeded random states
- Strict point-in-time discipline (no look-ahead bias)
- Comprehensive unit tests with pytest
- Simple CLI interface with YAML configuration: `python main.py --asof <YYYY-MM-DD>`
- No hard-coded paths or parameters
- Signal weights configurable in YAML (default: 0.5 momentum, 0.5 value)
- Risk model: 60-day covariance, shrinkage λ=0.3, diagonal load 1e-4
- Optimization constraints: sum=1, per-name cap 5%, per-sector cap 25%, turnover cap 10%

### Definition of Done
- **Artifacts**: holdings.csv, trades.csv, report.txt
- **Demo script**: `python main.py --asof 2024-01-15` runs end-to-end on ./data/*.csv
- **Tests**: All core modules have >80% test coverage
- **Documentation**: README explains design choices and usage
- **Validation**: Engine passes all pre-trade checks on sample data
- **CLI**: Accepts --asof parameter for daily rebalancing simulation

## Project Structure

```
quant_engine/
├── src/engine/           # Main engine modules
│   ├── __init__.py       # Package initialization
│   ├── data_io.py        # Data loading and validation
│   ├── signals.py        # Signal generation (momentum, value)
│   ├── prep.py           # Data preparation and alignment
│   ├── risk.py           # Risk model and covariance estimation
│   ├── optimize.py       # Portfolio optimization
│   ├── checks.py         # Pre-trade validation checks
│   ├── utils.py          # Utility functions and configuration
│   └── run_day.py        # Main execution pipeline
├── tests/                # Unit tests
│   ├── test_data_io.py
│   ├── test_signals.py
│   ├── test_prep.py
│   ├── test_risk.py
│   ├── test_optimize.py
│   └── test_checks.py
├── data/                 # Input data files
├── reports/              # Output files
├── configs/              # Configuration files
│   └── config.yaml       # Main configuration
└── README.md             # This file
```

## Implementation Plan

### Milestone 1: Scaffold (1 hour) ✅
- [x] Project structure and directories
- [x] Module stubs with type hints and docstrings
- [x] Test skeletons with pytest.skip
- [x] Configuration YAML with placeholders
- [x] README with Project Charter

### Milestone 2: Data Loading & Validation (1.5 hours)
- [ ] CSV readers with schema validation
- [ ] Point-in-time lag enforcement
- [ ] Data quality checks
- *Acceptance*: Loads sample data without errors, validates reporting lags

### Milestone 3: Signal Generation (2 hours)
- [ ] Momentum signal (252-day lookback with 21-day gap)
- [ ] Value signal (E/P ratio)
- [ ] Signal processing (winsorization, z-scoring, sector neutralization)
- [ ] Configurable weights from YAML
- *Acceptance*: Generates signals for sample data, passes basic sanity checks

### Milestone 4: Signal Validation (1.5 hours)
- [ ] Daily IC calculation (Spearman)
- [ ] Decile analysis and performance summary
- [ ] Signal quality metrics
- *Acceptance*: Produces IC report and decile performance metrics

### Milestone 5: Risk Model (1 hour)
- [ ] 60-day covariance estimation
- [ ] Shrinkage to diagonal (λ=0.3)
- [ ] Diagonal loading (1e-4)
- *Acceptance*: Generates positive definite covariance matrix

### Milestone 6: Optimization (1.5 hours)
- [ ] Mean-variance optimization with constraints
- [ ] Per-name cap (5%), per-sector cap (25%), turnover cap (10%)
- [ ] Fallback to prior weights if infeasible
- *Acceptance*: Solves for feasible weights within all constraints

### Milestone 7: Pre-trade Checks (1 hour)
- [ ] Schema validation, missingness, turnover spike checks
- [ ] Sector exposure delta validation
- [ ] Comprehensive safety checks
- *Acceptance*: Blocks on violations, logs warnings for acceptable issues

### Milestone 8: Integration & CLI (1 hour)
- [ ] End-to-end pipeline integration
- [ ] CLI with --asof parameter
- [ ] Output generation (holdings.csv, trades.csv, report.txt)
- *Acceptance*: `python main.py --asof 2024-01-15` produces all outputs

## Usage

### Setup
```bash
# Install dependencies
pip install numpy pandas scipy pyyaml pytest

# Run tests
python -m pytest tests/

# Run portfolio construction
python src/engine/run_day.py --asof 2024-01-15
```

### Configuration
Edit `configs/config.yaml` to adjust:
- Signal parameters (lookback periods, weights)
- Risk model settings (covariance lookback, shrinkage)
- Optimization constraints (caps, turnover limits)
- Validation thresholds
- Output preferences

### Input Data
Place the following CSV files in the `data/` directory:
- `prices.csv`: Daily close prices
- `fundamentals.csv`: Fundamental data with reporting lags
- `sectors.csv`: Sector classifications
- `holdings_prior.csv`: Prior day portfolio weights

### Output
The engine generates:
- `holdings.csv`: Final portfolio weights
- `trades.csv`: Trade recommendations
- `report.txt`: Performance and validation report

## Design Choices

### Signal Design
- **Momentum**: 252-day lookback with 21-day gap to avoid short-term reversal
- **Value**: E/P ratio for simplicity and interpretability
- **Combination**: Equal weights by default, configurable via YAML

### Risk Management
- **Covariance**: 60-day rolling window for stability
- **Shrinkage**: λ=0.3 to diagonal for robustness
- **Diagonal loading**: 1e-4 for numerical stability

### Optimization
- **Constraints**: Long-only with realistic caps
- **Fallback**: Prior weights if optimization infeasible
- **Objective**: Mean-variance with configurable risk aversion

### Point-in-Time Discipline
- **Reporting lags**: Enforced via report_lag_days field
- **Data alignment**: Strict date-based alignment
- **No look-ahead**: All calculations use only available data

## TODO Checklist

### Core Implementation
- [ ] Implement data_io.py functions
- [ ] Implement signals.py functions
- [ ] Implement prep.py functions
- [ ] Implement risk.py functions
- [ ] Implement optimize.py functions
- [ ] Implement checks.py functions
- [ ] Implement utils.py functions
- [ ] Implement run_day.py main pipeline

### Testing
- [ ] Write comprehensive unit tests
- [ ] Add integration tests
- [ ] Test edge cases and error conditions
- [ ] Achieve >80% test coverage

### Documentation
- [ ] Add detailed docstrings
- [ ] Create usage examples
- [ ] Document configuration options
- [ ] Add troubleshooting guide

### Validation
- [ ] Test with sample data
- [ ] Validate point-in-time discipline
- [ ] Verify optimization constraints
- [ ] Check output formats

### Final Steps
- [ ] Create sample data generator
- [ ] Add demo script
- [ ] Performance optimization
- [ ] Final testing and validation 