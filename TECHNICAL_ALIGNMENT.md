# DefiHunter: Technical Alignment Audit & Execution Hardening

This document serves as the canonical source of truth for execution parity and operational hardening in the DefiHunter system.

## 1. Executive Summary: The Parity Gap
Historically, the DefiHunter backtest engine diverged from the live scanner by ignoring family-specific execution gates and using hardcoded 0.0 values for OI/Funding. This patch closes these gaps by unifying the risk-gating logic and operationalizing a cached asynchronous data pipeline.

## 2. Runtime Reality: Fully Aligned Modules

### Unified Family Execution Gates
Both the `Scanner` (live) and `BacktestEngine` (sim) now leverage the `get_family_execution()` helper to enforce:
- **`watch_only` Mode**: Ranking logic remains active for alpha discovery, but trade opening is inhibited at the execution level.
- **Risk Scaling**: `risk_pct_mult` and `stop_width_mult` are injected into the sizing calc, ensuring backtest equity results mirror live sizing.
- **Metric Veto Layer**: Trades are inhibited if `entry_readiness` or `leader_prob` fall below family-specific minimums.

### Asynchronous Data Parity
The `BinanceFuturesFetcher` has been hardened for parallel execution:
- **`merge_asof` Strategy**: Handles millisecond-level drift between Binance OHLCV, Funding, and OI datasets. Price bars are backward-merged with the last known state.
- **TTL Cache Layer**: Dramatically reduces "RateLimit" errors by sharing OI/Funding buffers across parallel symbol scans (Default: 600s TTL).
- **Column Integrity**: Full restoration of standard OHLCV features (taker volume, quote volume) ensures that the `features.py` pipeline receives identical inputs in both live and historical modes.

## 3. Intentional Differences (Known Limitations)
Where 100% parity is impossible or computationally expensive:
- **Daily Loss Killswitch**: 
    - **Live**: Uses real-time USD balance tracking.
    - **Backtest**: Uses an R-multiple approximation (`pnl_r * risk_per_trade_pct`) since absolute notional equity fluctuates differently in vectorized simulation.
- **Execution Latency**: Backtest assumes OHLCV-bar execution (Bar Close), while live/paper trading experiences millisecond latency on order routing.

## 4. Active Family Coverage
The following families are currently governed by this alignment patch:
- `defi_lending`, `defi_dex`, `defi_beta`, `defi_oracles` (watch_only test), `defi_social`, `defi_stable`.

---
**Status**: [REGRESSION-SAFE] alignment achieved. Verification harness integrated for future CI/CD safety.
