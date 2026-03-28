"""
Integration test: scanner → AdaptiveStopEngine → open_position wiring.

Tests the end-to-end path:
  mock candidate → mock AdaptiveStopEngine → open_position(adaptive_stop_result=...) → PaperPosition saved

Does NOT call Binance API.
"""
import sys
import json
import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_STOP  = 95.0
EXPECTED_TP1   = 110.0
EXPECTED_TP2   = 120.0
EXPECTED_MODE  = "hybrid_stop"
EXPECTED_MULT  = 1.5

def _make_final_decision(symbol: str = "AAVE.p", entry_price: float = 100.0):
    from defihunter.core.models import FinalDecision
    return FinalDecision(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        final_trade_score=75.0,
        decision="trade",
        entry_price=entry_price,
        stop_price=entry_price * 0.95,   # legacy fallback value (should be overridden)
        tp1_price=entry_price * 1.05,
        tp2_price=entry_price * 1.10,
        discovery_score=70.0,
        entry_readiness=68.0,
        fakeout_risk=20.0,
        hold_quality=60.0,
        leader_prob=0.75,
        composite_leader_score=75.0,
        explanation={
            "family":       "defi_lending",
            "triggers":     ["MSB_BULL"],
            "kelly_risk_pct": 1.0,
        },
    )


def _deterministic_adaptive_stop(row, family, regime, fakeout_risk, atr_col='atr'):
    """Deterministic mock that always returns the EXPECTED_ constants."""
    return {
        "stop_price": EXPECTED_STOP,
        "tp1_price":  EXPECTED_TP1,
        "tp2_price":  EXPECTED_TP2,
        "stop_mode":  EXPECTED_MODE,
        "atr_mult":   EXPECTED_MULT,
        "risk_r":     5.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_open_position_uses_adaptive_stop_result():
    """
    Verifies that PaperTradeEngine.open_position writes adaptive stop values
    (not the legacy fallback values) to the PaperPosition state.
    """
    from defihunter.execution.paper_trade import PaperTradeEngine, PaperPortfolio

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        state_path = f.name

    try:
        engine = PaperTradeEngine(state_path=state_path)
        engine.portfolio = PaperPortfolio(balance_usd=10_000.0)

        decision = _make_final_decision(entry_price=100.0)
        adaptive_result = _deterministic_adaptive_stop(
            row={}, family="defi_lending", regime="trend", fakeout_risk=20.0
        )

        opened = engine.open_position(
            decision,
            risk_pct=1.0,
            adaptive_stop_result=adaptive_result,
        )

        assert opened is True, "open_position should return True"
        assert len(engine.portfolio.open_positions) == 1

        pos = engine.portfolio.open_positions[0]

        # Adaptive values must be written — NOT the legacy 0.95/1.05/1.10 fallbacks
        assert pos.stop_price == EXPECTED_STOP,  f"stop_price should be {EXPECTED_STOP}, got {pos.stop_price}"
        assert pos.tp1_price  == EXPECTED_TP1,   f"tp1_price should be {EXPECTED_TP1}, got {pos.tp1_price}"
        assert pos.tp2_price  == EXPECTED_TP2,   f"tp2_price should be {EXPECTED_TP2}, got {pos.tp2_price}"

        # Family context preserved
        assert pos.family == "defi_lending", f"family should be defi_lending, got {pos.family}"

    finally:
        os.unlink(state_path)


def test_open_position_fallback_when_adaptive_stop_none():
    """
    Verifies that when adaptive_stop_result=None, the legacy stop from
    the FinalDecision is used (safe fallback).
    """
    from defihunter.execution.paper_trade import PaperTradeEngine, PaperPortfolio

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        state_path = f.name

    try:
        engine = PaperTradeEngine(state_path=state_path)
        engine.portfolio = PaperPortfolio(balance_usd=10_000.0)

        entry = 100.0
        expected_stop = entry * 0.95  # legacy fallback
        decision = _make_final_decision(entry_price=entry)

        opened = engine.open_position(decision, risk_pct=1.0, adaptive_stop_result=None)

        assert opened is True
        pos = engine.portfolio.open_positions[0]
        assert abs(pos.stop_price - expected_stop) < 0.01, \
            f"Fallback stop should be ~{expected_stop}, got {pos.stop_price}"

    finally:
        os.unlink(state_path)


def test_adaptive_stop_result_written_to_saved_json():
    """
    Verifies that the saved paper_portfolio.json contains the adaptive stop values.
    """
    from defihunter.execution.paper_trade import PaperTradeEngine, PaperPortfolio

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        state_path = f.name

    try:
        engine = PaperTradeEngine(state_path=state_path)
        engine.portfolio = PaperPortfolio(balance_usd=10_000.0)

        decision = _make_final_decision(entry_price=100.0)
        adaptive_result = _deterministic_adaptive_stop(
            row={}, family="defi_lending", regime="trend", fakeout_risk=20.0
        )

        engine.open_position(decision, risk_pct=1.0, adaptive_stop_result=adaptive_result)

        # Read back from disk
        with open(state_path, 'r') as f:
            saved = json.load(f)

        positions = saved.get("open_positions", [])
        assert len(positions) == 1, "Expected 1 saved position"

        saved_pos = positions[0]
        assert saved_pos["stop_price"] == EXPECTED_STOP,  f"Saved stop_price mismatch"
        assert saved_pos["tp1_price"]  == EXPECTED_TP1,   f"Saved tp1_price mismatch"
        assert saved_pos["tp2_price"]  == EXPECTED_TP2,   f"Saved tp2_price mismatch"
        assert saved_pos["family"]     == "defi_lending", f"Saved family mismatch"

    finally:
        os.unlink(state_path)


def test_adaptive_stop_engine_called_with_correct_family_regime():
    """
    Verifies AdaptiveStopEngine.compute_stop receives the correct family and regime.
    Uses a spy on AdaptiveStopEngine.
    """
    from defihunter.engines.adaptive_stop import AdaptiveStopEngine
    from defihunter.execution.paper_trade import PaperTradeEngine, PaperPortfolio

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        state_path = f.name

    try:
        engine   = PaperTradeEngine(state_path=state_path)
        engine.portfolio = PaperPortfolio(balance_usd=10_000.0)
        decision = _make_final_decision(entry_price=100.0)

        # Use spy to record what arguments compute_stop was called with
        stop_engine = AdaptiveStopEngine()
        captured    = {}

        def spy_compute_stop(row, family, regime, fakeout_risk=0.0, atr_col='atr'):
            captured['family']       = family
            captured['regime']       = regime
            captured['fakeout_risk'] = fakeout_risk
            return _deterministic_adaptive_stop(row, family, regime, fakeout_risk)

        stop_engine.compute_stop = spy_compute_stop

        # Simulate what scanner does
        adaptive_result = stop_engine.compute_stop(
            row={'close': 100.0, 'atr': 2.0},
            family=decision.explanation.get('family', 'defi_beta'),
            regime='trend',
            fakeout_risk=decision.fakeout_risk,
        )
        engine.open_position(decision, risk_pct=1.0, adaptive_stop_result=adaptive_result)

        assert captured['family']       == 'defi_lending', f"Expected defi_lending, got {captured['family']}"
        assert captured['regime']       == 'trend',        f"Expected trend, got {captured['regime']}"
        assert captured['fakeout_risk'] == 20.0,           f"Expected 20.0, got {captured['fakeout_risk']}"

    finally:
        os.unlink(state_path)


def test_no_adaptive_stop_regression_on_legacy_path():
    """
    Regression test: existing PaperTradeEngine tests should still pass if
    adaptive_stop_result is never passed (backward-compat).
    """
    from defihunter.execution.paper_trade import PaperTradeEngine, PaperPortfolio

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        state_path = f.name

    try:
        engine   = PaperTradeEngine(state_path=state_path)
        engine.portfolio = PaperPortfolio(balance_usd=10_000.0)
        decision = _make_final_decision(entry_price=200.0)

        # Old-style call — no adaptive_stop_result kwarg at all
        opened = engine.open_position(decision, risk_pct=1.0)
        assert opened is True, "Legacy call signature must still work"
        assert len(engine.portfolio.open_positions) == 1

    finally:
        os.unlink(state_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
