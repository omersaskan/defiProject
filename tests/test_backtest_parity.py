"""
Test: BacktestEngine.simulate produces trade_log with parity fields.
Smoke test that does NOT require network access — uses synthetic data.
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


def _build_synthetic_df(n_bars: int = 40, n_symbols: int = 2) -> pd.DataFrame:
    """Builds a minimal synthetic multi-symbol DataFrame for backtest smoke tests."""
    symbols = [f"SYN{i}.p" for i in range(n_symbols)]
    base_time = datetime(2024, 1, 1)
    rows = []
    for i in range(n_bars):
        ts = base_time + timedelta(hours=i)
        for sym in symbols:
            close = 100.0 + np.random.randn() * 0.5
            atr   = 1.5
            rows.append({
                "timestamp":     ts,
                "symbol":        sym,
                "family":        "defi_beta",
                "close":         close,
                "high":          close + 0.8,
                "low":           close - 0.8,
                "open":          close - 0.2,
                "volume":        1_000_000.0,
                "atr":           atr,
                # Signal every 5th bar for first symbol
                "entry_signal":  (i % 5 == 0 and sym == symbols[0]),
                "stop_price":    close - atr * 1.5,
                "tp1_price":     close + atr * 2.0,
                "tp2_price":     close + atr * 4.0,
                "ml_rank_score": 70.0,
                "leadership_decay": False,
                "historical_regime": "trend",
                "fakeout_risk":   20.0,
            })
    return pd.DataFrame(rows)


def _make_config():
    """Minimal fake config object."""
    class BT:
        fee_bps = 2.0
        slippage_bps = 1.0
        max_concurrent_positions = 3
        time_stop_bars = 20

    class Cfg:
        backtest = BT()

    return Cfg()


def test_simulate_returns_trade_log():
    from defihunter.execution.backtest import BacktestEngine
    df  = _build_synthetic_df(n_bars=60)
    eng = BacktestEngine(config=_make_config())
    res = eng.simulate(df)
    assert isinstance(res, dict), "simulate must return a dict"
    assert "total_trades" in res
    assert res["total_trades"] >= 0


def test_trade_log_has_parity_fields():
    """Parity: trade_log entries must contain exit_reason, partial_taken, mfe_r, giveback_r."""
    from defihunter.execution.backtest import BacktestEngine
    df  = _build_synthetic_df(n_bars=80, n_symbols=3)
    eng = BacktestEngine(config=_make_config())
    eng.simulate(df)

    if not eng.trade_log:
        pytest.skip("No trades generated — adjust synthetic data if needed")

    for entry in eng.trade_log:
        assert "exit_reason"   in entry, f"Missing exit_reason in log entry: {entry}"
        assert "partial_taken" in entry, f"Missing partial_taken in log entry: {entry}"
        assert "mfe_r"         in entry, f"Missing mfe_r in log entry: {entry}"
        assert "giveback_r"    in entry, f"Missing giveback_r in log entry: {entry}"
        assert "peak_price_seen" in entry, f"Missing peak_price_seen in log entry: {entry}"


def test_simulate_pnl_r_numeric():
    from defihunter.execution.backtest import BacktestEngine
    df  = _build_synthetic_df(n_bars=80)
    eng = BacktestEngine(config=_make_config())
    eng.simulate(df)
    for entry in eng.trade_log:
        assert isinstance(entry['pnl_r'], (int, float)), "pnl_r must be numeric"
        assert not np.isnan(entry['pnl_r']), "pnl_r must not be NaN"


def test_adaptive_stop_engine_basic():
    """AdaptiveStopEngine returns sensible stop < close for all families."""
    from defihunter.engines.adaptive_stop import AdaptiveStopEngine, FAMILY_ATR_MULT
    eng = AdaptiveStopEngine()
    for family in list(FAMILY_ATR_MULT.keys()) + ["unknown_family"]:
        row  = {"close": 100.0, "atr": 2.0}
        res  = eng.compute_stop(row, family=family, regime="trend", fakeout_risk=30.0)
        assert res["stop_price"] < 100.0, f"stop_price must be < close for {family}"
        assert res["tp1_price"]  > 100.0, f"tp1_price must be > close for {family}"
        assert res["tp2_price"]  > res["tp1_price"], f"tp2 must be > tp1 for {family}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
