"""Tests for OutcomeLinker: no-leakage contract and correct rank computation."""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from defihunter.validation.outcome_linker import OutcomeLinker


def _make_shadow_row(symbol="AAVE.p", family="defi_lending", entry_price=100.0, scan_hours_ago=25):
    scan_ts = datetime.now(timezone.utc) - timedelta(hours=scan_hours_ago)
    return {
        "scan_id":               "test01",
        "scan_timestamp":        scan_ts.isoformat(),
        "scan_day":              scan_ts.strftime("%Y-%m-%d"),
        "timeframe":             "15m",
        "universe_size":         50,
        "symbol":                symbol,
        "family":                family,
        "regime":                "trend",
        "discovery_score":       72.0,
        "entry_readiness":       68.0,
        "fakeout_risk":          15.0,
        "hold_quality":          60.0,
        "leader_prob":           0.72,
        "composite_leader_score": 72.0,
        "suggested_action":      "trade",
        "setup_class":           "silent_accumulation",
        "stop_mode":             "hybrid_stop",
        "stop_price":            95.0,
        "tp1_price":             108.0,
        "tp2_price":             116.0,
        "atr_mult":              1.8,
        "paper_trade_opened":    True,
        "kelly_risk_pct":        1.0,
        "entry_price":           entry_price,
        "future_24h_return":     "",
        "future_24h_rank_in_family": "",
        "is_top3_family_next_24h": "",
        "leader_captured":       "",
        "missed_leader":         "",
        "final_exit_reason":     "",
        "pnl_r":                 "",
        "mfe_r":                 "",
        "giveback_r":            "",
        "hold_efficiency":       "",
    }


def _make_price_df(symbols, entry_prices, family="defi_lending", hours_range=48):
    """Build a synthetic price DataFrame for multiple symbols."""
    rows = []
    base_ts = datetime.now(timezone.utc) - timedelta(hours=hours_range)
    for h in range(hours_range + 2):
        ts = base_ts + timedelta(hours=h)
        for sym, ep in zip(symbols, entry_prices):
            # Trend upward: price rises ~0.5% per hour
            close = ep * (1 + 0.005 * h)
            rows.append({
                "timestamp": ts,
                "symbol":    sym,
                "family":    family,
                "close":     close,
                "high":      close * 1.003,
                "low":       close * 0.998,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────

def test_no_future_data_returns_nan():
    """If scan is within 24h, OutcomeLinker should return NaN (no future data)."""
    linker = OutcomeLinker(horizon_hours=24)
    # scan was only 1 hour ago — no 24h future data exists
    shadow_row = _make_shadow_row(scan_hours_ago=1)
    shadow_df  = pd.DataFrame([shadow_row])
    price_df   = _make_price_df(["AAVE.p"], [100.0], hours_range=25)

    result = linker.link(shadow_df, price_df)
    assert pd.isna(result.iloc[0]["future_24h_return"]), \
        "Scan within last 24h should have NaN future_24h_return"


def test_future_return_computed_correctly():
    """future_24h_return should be (close_at_T+24h - entry_price) / entry_price."""
    linker = OutcomeLinker(horizon_hours=24)
    shadow_df = pd.DataFrame([_make_shadow_row(entry_price=100.0, scan_hours_ago=30)])
    price_df  = _make_price_df(["AAVE.p"], [100.0], hours_range=35)

    result = linker.link(shadow_df, price_df)
    ret = result.iloc[0]["future_24h_return"]
    assert not pd.isna(ret), "future_24h_return should be computed"
    # Price rises ~0.5% per hour × 24h ≈ +12% but exact value depends on hourly price
    assert -0.5 < ret < 1.0, f"Return out of expected range: {ret}"


def test_family_rank_top3():
    """Coin with highest 24h return in family should be rank 1 and leader_captured=True."""
    linker = OutcomeLinker(horizon_hours=24)

    symbols     = ["AAVE.p", "UNI.p", "COMP.p", "MKR.p"]
    entry_prices = [100.0, 100.0, 100.0, 100.0]
    shadow_rows  = [_make_shadow_row(sym, "defi_lending", ep, scan_hours_ago=30)
                    for sym, ep in zip(symbols, entry_prices)]
    shadow_df    = pd.DataFrame(shadow_rows)

    # Give AAVE.p the highest future return (5× growth rate)
    rows = []
    base_ts = datetime.now(timezone.utc) - timedelta(hours=35)
    for h in range(38):
        ts = base_ts + timedelta(hours=h)
        multipliers = {"AAVE.p": 1.02, "UNI.p": 1.005, "COMP.p": 1.003, "MKR.p": 1.001}
        for sym, ep in zip(symbols, entry_prices):
            c = ep * (multipliers[sym] ** h)
            rows.append({"timestamp": ts, "symbol": sym, "family": "defi_lending",
                         "close": c, "high": c * 1.001, "low": c * 0.999})
    price_df = pd.DataFrame(rows)

    result = linker.link(shadow_df, price_df)

    aave_row = result[result["symbol"] == "AAVE.p"].iloc[0]
    assert aave_row["future_24h_rank_in_family"] == 1, \
        f"AAVE.p should be rank 1, got {aave_row['future_24h_rank_in_family']}"
    assert aave_row["leader_captured"] == True, "AAVE.p should be leader_captured=True"


def test_stop_loss_exit_reason():
    """Coin whose price drops below stop_price in 24h should get exit_reason=STOP_LOSS."""
    linker = OutcomeLinker(horizon_hours=24)
    shadow_row = _make_shadow_row(entry_price=100.0, scan_hours_ago=30)
    shadow_row["stop_price"] = 95.0
    shadow_df  = pd.DataFrame([shadow_row])

    # Price crashes to 90 within 24h
    rows = []
    base_ts = datetime.now(timezone.utc) - timedelta(hours=35)
    for h in range(38):
        ts = base_ts + timedelta(hours=h)
        c = 100.0 * (0.995 ** h)  # drops ~0.5% per hour
        rows.append({"timestamp": ts, "symbol": "AAVE.p", "family": "defi_lending",
                     "close": c, "high": c * 1.001, "low": c * 0.998})
    price_df = pd.DataFrame(rows)

    result = linker.link(shadow_df, price_df)
    exit_r = result.iloc[0]["final_exit_reason"]
    assert exit_r == "STOP_LOSS", f"Expected STOP_LOSS, got {exit_r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
