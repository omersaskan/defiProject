"""
Test: FinalDecision top-level fields are populated by DecisionEngine.
"""
import sys
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_candidate() -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol":           "TEST.p",
        "timestamp":        pd.Timestamp.now(),
        "close":            1.0,
        # ── DiscoveryEngine inputs (required for score >= 55) ─────────────────
        "leader_prob":          0.8,     # ml_factor = 0.8 → 32 pts
        "family_heat":          0.1,     # heat at max of clip range → 100% → 35×0.6
        "family_breadth":       1.0,     # breadth = 1.0 → 35×0.4
        "peer_momentum":        0.05,    # at max clip → 25×0.5
        "peer_rank":            1.0,     # → 25×0.5  ⇒ total ≈ 92
        # ── EntryEngine inputs ────────────────────────────────────────────────
        "msb_bull":         True,
        "taker_surge":      False,
        "v_delta_score":    0.1,
        "ema20_dist":       0.01,
        "wick_exhaustion_flag": 0,
        "sweep_wick_flag":  False,
        # ── HoldQuality inputs ────────────────────────────────────────────────
        "holdability_score":             70.0,
        "trend_persistence_score":       10.0,
        "volume_persistence_score":      5.0,
        "close_to_high_persistence":     0.6,
        "exhaustion_risk_score":         5.0,
        # ── Stop/TP ───────────────────────────────────────────────────────────
        "atr":              0.02,
        "stop_price":       0.97,
        "tp1_price":        1.05,
        "tp2_price":        1.10,
        "family":           "defi_lending",
    }])


def test_top_level_discovery_score_populated():
    from defihunter.engines.decision import DecisionEngine
    de = DecisionEngine()
    df = _make_candidate()
    decisions = de.process_candidates(df)
    assert len(decisions) > 0, "DecisionEngine returned no decisions"
    d = decisions[0]
    assert d.discovery_score > 0, f"discovery_score not populated: {d.discovery_score}"


def test_top_level_entry_readiness_populated():
    from defihunter.engines.decision import DecisionEngine
    de = DecisionEngine()
    decisions = de.process_candidates(_make_candidate())
    assert len(decisions) > 0
    d = decisions[0]
    assert d.entry_readiness >= 0, f"entry_readiness not populated: {d.entry_readiness}"


def test_top_level_fakeout_risk_populated():
    from defihunter.engines.decision import DecisionEngine
    de = DecisionEngine()
    decisions = de.process_candidates(_make_candidate())
    assert len(decisions) > 0
    d = decisions[0]
    assert d.fakeout_risk >= 0, f"fakeout_risk not populated: {d.fakeout_risk}"


def test_top_level_composite_score_populated():
    from defihunter.engines.decision import DecisionEngine
    de = DecisionEngine()
    decisions = de.process_candidates(_make_candidate())
    assert len(decisions) > 0
    d = decisions[0]
    assert d.composite_leader_score >= 0, f"composite_leader_score not populated"
    # final_trade_score and composite_leader_score must match
    assert d.final_trade_score == d.composite_leader_score, (
        f"final_trade_score ({d.final_trade_score}) != "
        f"composite_leader_score ({d.composite_leader_score})"
    )


def test_top_level_leader_prob_populated():
    from defihunter.engines.decision import DecisionEngine
    de = DecisionEngine()
    decisions = de.process_candidates(_make_candidate())
    assert len(decisions) > 0
    d = decisions[0]
    assert 0.0 <= d.leader_prob <= 1.0, f"leader_prob out of range: {d.leader_prob}"


def test_explanation_still_has_family():
    """explanation dict must still carry family and triggers."""
    from defihunter.engines.decision import DecisionEngine
    de = DecisionEngine()
    decisions = de.process_candidates(_make_candidate())
    assert len(decisions) > 0
    d = decisions[0]
    assert 'family' in d.explanation, "family missing from explanation"
    assert 'triggers' in d.explanation, "triggers missing from explanation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
