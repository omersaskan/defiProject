"""Tests for ShadowLogger: schema completeness, leakage safety."""
import sys
import os
import csv
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from defihunter.validation.shadow_logger import ShadowLogger, SCAN_TIME_COLUMNS


def _make_decision(symbol="AAVE.p", decision="trade"):
    from defihunter.core.models import FinalDecision
    return FinalDecision(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        final_trade_score=72.0,
        decision=decision,
        entry_price=100.0,
        stop_price=96.0,
        tp1_price=108.0,
        tp2_price=116.0,
        discovery_score=70.0,
        entry_readiness=68.0,
        fakeout_risk=15.0,
        hold_quality=60.0,
        leader_prob=0.72,
        composite_leader_score=72.0,
        explanation={"family": "defi_lending", "triggers": ["MSB_BULL"], "kelly_risk_pct": 1.0},
    )


def test_schema_matches_columns():
    """Written CSV must have exactly all SCAN_TIME_COLUMNS as header."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        log_path = f.name
    os.unlink(log_path)  # delete so logger re-creates with header

    try:
        logger = ShadowLogger(log_path=log_path)
        d = _make_decision()
        logger.log_scan([d], regime="trend", universe_size=50)

        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        assert set(cols) == set(SCAN_TIME_COLUMNS), \
            f"CSV columns mismatch. Extra: {set(cols)-set(SCAN_TIME_COLUMNS)} Missing: {set(SCAN_TIME_COLUMNS)-set(cols)}"
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)


def test_future_columns_empty_at_scantime():
    """future_* columns must be empty string at scan-time (not filled by logger)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        log_path = f.name
    os.unlink(log_path)

    try:
        logger = ShadowLogger(log_path=log_path)
        d = _make_decision()
        logger.log_scan([d], regime="trend", universe_size=50)

        rows = list(csv.DictReader(open(log_path)))
        assert len(rows) == 1
        future_cols = [
            "future_24h_return", "future_24h_rank_in_family", "is_top3_family_next_24h",
            "leader_captured", "missed_leader", "final_exit_reason",
            "pnl_r", "mfe_r", "giveback_r", "hold_efficiency",
        ]
        for col in future_cols:
            assert rows[0][col] == "", \
                f"Leakage: '{col}' should be empty at scan-time, got '{rows[0][col]}'"
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)


def test_scan_time_fields_populated():
    """Core scan-time fields must all be non-empty."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        log_path = f.name
    os.unlink(log_path)

    try:
        logger = ShadowLogger(log_path=log_path)
        d = _make_decision(symbol="UNI.p", decision="trade")
        asr = {"stop_mode": "hybrid_stop", "stop_price": 95.0, "tp1_price": 108.0,
               "tp2_price": 116.0, "atr_mult": 1.8, "risk_r": 5.0}
        logger.log_scan(
            [d], regime="trend", universe_size=60,
            adaptive_stop_map={"UNI.p": asr},
            paper_opened_symbols={"UNI.p"},
            kelly_map={"UNI.p": 1.2},
        )
        rows = list(csv.DictReader(open(log_path)))
        assert rows[0]["symbol"]       == "UNI.p"
        assert rows[0]["regime"]       == "trend"
        assert rows[0]["stop_mode"]    == "hybrid_stop"
        assert rows[0]["paper_trade_opened"] == "True"
        assert rows[0]["atr_mult"]     == "1.8"
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)


def test_multiple_decisions_written():
    """Multiple decisions must write multiple rows."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        log_path = f.name
    os.unlink(log_path)

    try:
        logger = ShadowLogger(log_path=log_path)
        decisions = [_make_decision(sym) for sym in ["A.p", "B.p", "C.p"]]
        count = logger.log_scan(decisions, regime="chop", universe_size=40)
        assert count == 3
        rows = list(csv.DictReader(open(log_path)))
        assert len(rows) == 3
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
