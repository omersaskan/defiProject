"""
ShadowLogger — writes scan-time decision data to logs/shadow_log.csv.

LEAKAGE SAFETY:
- Only scan-time columns are written here.
- All future_* columns are left NaN and filled later by OutcomeLinker.
- This method is called from run_shadow_validation.py (historical sim) and
  optionally from scanner.py (live shadow mode).
"""
import os
import csv
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCAN_TIME_COLUMNS = [
    "scan_id",
    "scan_timestamp",
    "scan_day",
    "timeframe",
    "universe_size",
    "symbol",
    "family",
    "regime",
    "discovery_score",
    "entry_readiness",
    "fakeout_risk",
    "hold_quality",
    "leader_prob",
    "composite_leader_score",
    "suggested_action",
    "setup_class",
    # Adaptive stop fields
    "stop_mode",
    "stop_price",
    "tp1_price",
    "tp2_price",
    "atr_mult",
    # Execution
    "paper_trade_opened",
    "kelly_risk_pct",
    "entry_price",
    # Post-hoc columns (NaN at log-time, filled by OutcomeLinker)
    "future_24h_return",
    "future_24h_rank_in_family",
    "is_top3_family_next_24h",
    "leader_captured",
    "missed_leader",
    "final_exit_reason",
    "pnl_r",
    "mfe_r",
    "giveback_r",
    "hold_efficiency",
]


class ShadowLogger:
    """
    Appends one row per candidate decision to the shadow CSV log.
    Future outcome columns are left empty — filled by OutcomeLinker post-hoc.
    """

    def __init__(self, log_path: str = "logs/shadow_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(SCAN_TIME_COLUMNS)

    def log_scan(
        self,
        decisions: List[Any],
        regime: str,
        universe_size: int,
        timeframe: str = "15m",
        adaptive_stop_map: Optional[Dict[str, Dict]] = None,
        paper_opened_symbols: Optional[set] = None,
        kelly_map: Optional[Dict[str, float]] = None,
        setup_class_map: Optional[Dict[str, str]] = None,
        scan_timestamp: Optional[datetime] = None,
    ):
        """
        Append one row per decision to shadowlog.
        All future_* columns are empty (filled by OutcomeLinker separately).

        Args:
            decisions: list of FinalDecision objects
            regime: current regime label
            universe_size: total evaluated universe size
            timeframe: e.g. '15m'
            adaptive_stop_map: {symbol: adaptive_stop_result dict}
            paper_opened_symbols: set of symbols for which trade was opened this scan
            kelly_map: {symbol: kelly_risk_pct}
            setup_class_map: {symbol: setup_class string}
            scan_timestamp: override timestamp (for historical sim)
        """
        if adaptive_stop_map is None:
            adaptive_stop_map = {}
        if paper_opened_symbols is None:
            paper_opened_symbols = set()
        if kelly_map is None:
            kelly_map = {}
        if setup_class_map is None:
            setup_class_map = {}

        ts = scan_timestamp or datetime.now(timezone.utc)
        scan_id = str(uuid.uuid4())[:8]
        scan_day = ts.strftime("%Y-%m-%d")

        rows = []
        for d in decisions:
            stop_res = adaptive_stop_map.get(d.symbol, {})
            row = {
                "scan_id":                scan_id,
                "scan_timestamp":         ts.isoformat(),
                "scan_day":               scan_day,
                "timeframe":              timeframe,
                "universe_size":          universe_size,
                "symbol":                 d.symbol,
                "family":                 d.explanation.get("family", "unknown"),
                "regime":                 regime,
                "discovery_score":        round(d.discovery_score, 3),
                "entry_readiness":        round(d.entry_readiness, 3),
                "fakeout_risk":           round(d.fakeout_risk, 3),
                "hold_quality":           round(d.hold_quality, 3),
                "leader_prob":            round(d.leader_prob, 4),
                "composite_leader_score": round(d.composite_leader_score, 3),
                "suggested_action":       d.decision,
                "setup_class":            setup_class_map.get(d.symbol, ""),
                # Adaptive stop
                "stop_mode":              stop_res.get("stop_mode", "none"),
                "stop_price":             stop_res.get("stop_price", d.stop_price),
                "tp1_price":              stop_res.get("tp1_price", d.tp1_price),
                "tp2_price":              stop_res.get("tp2_price", d.tp2_price),
                "atr_mult":               stop_res.get("atr_mult", ""),
                # Execution
                "paper_trade_opened":     d.symbol in paper_opened_symbols,
                "kelly_risk_pct":         kelly_map.get(d.symbol, ""),
                "entry_price":            round(d.entry_price, 6),
                # Future outcomes — intentionally EMPTY at scan-time
                "future_24h_return":      "",
                "future_24h_rank_in_family": "",
                "is_top3_family_next_24h": "",
                "leader_captured":        "",
                "missed_leader":          "",
                "final_exit_reason":      "",
                "pnl_r":                  "",
                "mfe_r":                  "",
                "giveback_r":             "",
                "hold_efficiency":        "",
            }
            rows.append(row)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCAN_TIME_COLUMNS)
            writer.writerows(rows)

        return len(rows)
