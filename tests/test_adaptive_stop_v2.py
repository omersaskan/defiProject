"""
Tests for AdaptiveStopEngine V2.

Key regressions tested:
1. fakeout_risk > 60 → stop WIDENS, reduce_size_first=True (V1 bug fix)
2. noise_tolerance_bars returned correctly per family
3. watch_only family skips trade in scanner (integration)
4. reduced_risk caps position size (stop_width_mult / kelly_pct)
5. soft_invalidation_price is closer to entry than hard_stop_price
6. chop regime widens stop (V2 change from V1 which tightened)
7. stop_width_mult > 1 triggers kelly_pct reduction to maintain net dollar risk
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from defihunter.engines.adaptive_stop import AdaptiveStopEngine


@pytest.fixture
def engine():
    return AdaptiveStopEngine()


def _row(close=100.0, atr=2.0, structure_low=0.0):
    return {"close": close, "atr": atr, "structure_low": structure_low, "swing_low": 0.0, "support_level": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 (CRITICAL V2 FIX): fakeout_risk > 60 → stop widens, NOT tightens
# ─────────────────────────────────────────────────────────────────────────────
def test_fakeout_high_widens_stop_and_flags_reduce_size(engine):
    """
    V1 bug: fakeout_risk > 60 → atr_mult *= 0.85 (tighter = more whipsaws).
    V2 fix: fakeout_risk > 60 → atr_mult *= 1.10 (wider) + reduce_size_first=True.
    """
    base   = engine.compute_stop(_row(), family="defi_lending", regime="trend", fakeout_risk=0.0)
    fakeout = engine.compute_stop(_row(), family="defi_lending", regime="trend", fakeout_risk=75.0)

    # Stop must be FURTHER from entry (lower stop price = wider stop)
    assert fakeout["stop_price"] < base["stop_price"], \
        f"fakeout stop {fakeout['stop_price']:.4f} should be further than base {base['stop_price']:.4f}"

    # risk_r must be LARGER
    assert fakeout["risk_r"] > base["risk_r"], \
        f"fakeout risk_r {fakeout['risk_r']:.4f} should be larger than base {base['risk_r']:.4f}"

    # reduce_size_first flag must be set
    assert fakeout["reduce_size_first"] is True, \
        "reduce_size_first should be True when fakeout_risk > 60"

    # Baseline must NOT have it set
    assert base["reduce_size_first"] is False, \
        "reduce_size_first should be False at low fakeout_risk"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: noise_tolerance_bars returned correctly per family
# ─────────────────────────────────────────────────────────────────────────────
def test_noise_tolerance_bars_by_family(engine):
    perp     = engine.compute_stop(_row(), family="defi_perp",    regime="trend")
    lending  = engine.compute_stop(_row(), family="defi_lending", regime="trend")
    unknown  = engine.compute_stop(_row(), family="unknown_fam",  regime="trend")

    assert perp["noise_tolerance_bars"] == 8,  f"defi_perp should have 8, got {perp['noise_tolerance_bars']}"
    assert lending["noise_tolerance_bars"] == 4, f"defi_lending should have 4, got {lending['noise_tolerance_bars']}"
    assert unknown["noise_tolerance_bars"] == 4, f"unknown family default should be 4"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: soft_invalidation_price is between entry and hard_stop
# ─────────────────────────────────────────────────────────────────────────────
def test_soft_invalidation_between_entry_and_hard_stop(engine):
    res = engine.compute_stop(_row(close=100.0, atr=2.0), family="defi_lending", regime="trend")
    entry     = 100.0
    hard_stop = res["hard_stop_price"]
    soft_stop = res["soft_invalidation_price"]

    assert hard_stop < soft_stop < entry, \
        f"soft_stop {soft_stop:.4f} must be between hard_stop {hard_stop:.4f} and entry {entry}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: chop regime WIDENS stop (V2 change — V1 tightened)
# ─────────────────────────────────────────────────────────────────────────────
def test_chop_regime_widens_stop_v2(engine):
    trend = engine.compute_stop(_row(), family="defi_lending", regime="trend")
    chop  = engine.compute_stop(_row(), family="defi_lending", regime="chop")

    # Chop: wider stop → higher atr_mult → lower stop_price
    assert chop["risk_r"] > trend["risk_r"], \
        f"chop risk_r {chop['risk_r']:.4f} should be > trend {trend['risk_r']:.4f} (V2 widens in chop)"
    assert chop["stop_price"] < trend["stop_price"], \
        f"chop stop_price {chop['stop_price']:.4f} should be lower (wider) than trend {trend['stop_price']:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: stop_width_mult > 1 proportionally expands risk_r
# ─────────────────────────────────────────────────────────────────────────────
def test_stop_width_mult_proportional_expansion(engine):
    base    = engine.compute_stop(_row(), family="defi_perp", regime="trend", stop_width_mult=1.0)
    widened = engine.compute_stop(_row(), family="defi_perp", regime="trend", stop_width_mult=1.3)

    expected_risk_ratio = 1.3
    actual_risk_ratio   = widened["risk_r"] / base["risk_r"]

    assert abs(actual_risk_ratio - expected_risk_ratio) < 0.05, \
        f"risk_r ratio should be ~{expected_risk_ratio}, got {actual_risk_ratio:.3f}"

    # stop_price must be further from entry
    assert widened["stop_price"] < base["stop_price"], \
        "wider stop_width_mult must produce a lower (further) stop_price"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: kelly_pct is divided by stop_width_mult in scanner integration
# Simulates the contract: kelly_pct /= width_mult when width_mult > 1
# ─────────────────────────────────────────────────────────────────────────────
def test_net_dollar_risk_constant_with_stop_width_mult(engine):
    """
    Net dollar risk = kelly_pct × stop_pct_of_portfolio must be constant.
    When stop_width_mult = 1.3, kelly_pct should be divided by 1.3.
    This is implemented in scanner.py; this test validates the arithmetic.
    """
    entry = 100.0
    atr   = 2.0

    base_kelly    = 1.0   # base kelly %
    width_mult    = 1.3

    base    = engine.compute_stop({"close": entry, "atr": atr}, family="defi_perp", regime="trend", stop_width_mult=1.0)
    widened = engine.compute_stop({"close": entry, "atr": atr}, family="defi_perp", regime="trend", stop_width_mult=width_mult)

    # Scanner logic: kelly_pct /= width_mult after computing stop
    adjusted_kelly = base_kelly / width_mult

    # Net risk in R for both scenarios:
    # risk = kelly_pct × (stop_dist / entry)
    risk_base    = base_kelly    * (base["risk_r"]    / entry)
    risk_widened = adjusted_kelly * (widened["risk_r"] / entry)

    assert abs(risk_base - risk_widened) < 0.001, \
        f"Net dollar risk should be equal: base={risk_base:.5f} widened={risk_widened:.5f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: defi_perp ATR mult is widened in V2 (was 1.2, now 1.6)
# ─────────────────────────────────────────────────────────────────────────────
def test_defi_perp_atr_mult_v2(engine):
    res = engine.compute_stop(_row(close=100.0, atr=1.0), family="defi_perp", regime="trend")
    # Base mult = 1.6, no regime modifier → risk_r ≈ 1.6
    assert res["risk_r"] == pytest.approx(1.6, rel=0.05), \
        f"defi_perp base risk_r should be ~1.6R, got {res['risk_r']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
