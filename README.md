# DeFiHunter Trade Intelligence System

A production-grade Python research and live-scanning system designed for high-selectivity DeFi perpetual futures trading.

## Overview

DeFiHunter aims to identify high-quality trading opportunities by evaluating:
- **Market Regime**: Overall BTC and ETH context (e.g., trend, chop, risk-on).
- **Sector Regime**: DeFi specific momentum via AAVE and UNI versus ETH.
- **Relative EMA Leadership**: Log-normalized deviation tracking to find coins genuinely leading anchors, not just rising with a macro tide.
- **Rule & ML Ranking**: A multi-tiered ranking system applying hard filters, deterministic scoring, and machine-learning based expectation evaluation.
- **Strict Risk Module**: Non-adaptive bounds ensure safety regardless of signal confidence.

## Architecture

This project strictly separates data, configurations, models, and execution engines:
- `defihunter/core`: Pydantic models and YAML configuration loaders.
- `defihunter/data`: OHLCV processing, technical indicator rendering, and universe filtering.
- `defihunter/engines`: The heart of the logic (Leadership Engine, ML Ranker, Risk, Rules, Regime).
- `defihunter/execution`: Live scanners and rigorous R-multiple backtesting suites.

## Setup

1. **Prerequisites**: Python 3.11+, and standard data science libraries (Pandas, Numpy, Scikit-Learn).
2. **Install requirements**: (Assume a standard `requirements.txt` or `poetry` environment)
   ```bash
   pip install pandas numpy pydantic pyyaml scikit-learn
   ```

## Usage

Use the provided `cli.py` to interact with the system endpoints.

### Live Scanning
Run a live market scan based on current prices:
```bash
python cli.py scan --config configs/default.yaml
```

### Backtesting
Run the historical simulation module:
```bash
python cli.py backtest
```
To run the ablation study specifically measuring the impact of the Relative EMA Leadership features:
```bash
python cli.py backtest --ablation
```

### Walk-Forward Validation
Execute rolling train/test windows:
```bash
python cli.py walk_forward
```

## Contact / License
Internal use only. Not financial advice. Strict risk parameters must be audited by humans before API execution.
