import pandas as pd
from typing import List, Dict, Any
from defihunter.engines.discovery import DiscoveryEngine
from defihunter.engines.entry import EntryEngine
from defihunter.core.models import FinalDecision


class DecisionEngine:
    """
    Orchestrator for Layered Leader Trading.
    Phase 5: Discovery -> Entry -> Risk -> Hold.

    FinalDecision top-level fields are always populated directly.
    `explanation` carries only auxiliary context (family, triggers, etc.).
    """
    def __init__(self, top_n: int = 5):
        self.discovery = DiscoveryEngine(top_n=top_n)
        self.entry = EntryEngine(min_readiness=65.0)

    def process_candidates(self, candidates: pd.DataFrame) -> List[FinalDecision]:
        """
        1. Run Discovery to find top leaders.
        2. Filter only those with DiscoveryScore >= 55.
        3. Check EntryReadiness & FakeoutRisk.
        4. Calculate HoldQuality.
        5. Generate CompositeLeaderScore.
        6. Populate ALL top-level FinalDecision fields directly (not only explanation).
        """
        if candidates.empty:
            return []

        # Step 1: Discovery Scoring
        df_discovery = self.discovery.compute_discovery_scores(candidates)

        # Step 2: Threshold — lowered slightly to catch emerging leaders
        potential_leaders = df_discovery[df_discovery['discovery_score'] >= 55.0]

        final_decisions = []
        for _, row in potential_leaders.iterrows():
            # Step 3: Entry & Risk Analysis
            entry_metrics = self.entry.compute_entry_metrics(pd.DataFrame([row]))

            # Step 4: Hold Quality (0-100)
            h_ml = row.get('holdability_score', 50.0)

            h_trend = min(100.0, row.get('trend_persistence_score', 0) * 5)
            h_vol   = min(100.0, row.get('volume_persistence_score', 0) * 10)
            h_close = row.get('close_to_high_persistence', 0.5) * 100.0
            h_persist = (h_trend * 0.4 + h_vol * 0.3 + h_close * 0.3)

            exhaust  = row.get('exhaustion_risk_score', 0.0)
            h_quality = (h_ml * 0.6 + h_persist * 0.4) - (exhaust * 0.5)
            h_quality = max(0.0, min(100.0, h_quality))

            # Step 5: Composite Leader Score (0-100)  — the primary leaderboard metric
            discovery  = row['discovery_score']
            readiness  = entry_metrics['readiness']
            fakeout    = entry_metrics['fakeout_risk']
            leader_p   = float(row.get('leader_prob', 0.0))

            risk_penalty = fakeout * 0.5
            composite = (discovery * 0.4 + readiness * 0.4 + h_quality * 0.2) - risk_penalty
            composite = max(0.0, min(100.0, composite))

            # Determine Action
            action = "watch"
            if entry_metrics['can_entry'] and discovery >= 60:
                action = "trade"
            elif discovery >= 70:
                action = "urgent_watch"
            elif fakeout > 70:
                action = "avoid_fakeout"

            final_decisions.append(FinalDecision(
                symbol=row['symbol'],
                timestamp=row.get('timestamp', row.get('Timestamp')),
                # ── TOP-LEVEL FIELDS (primary source of truth) ──────────────
                final_trade_score=round(composite, 2),
                decision=action,
                entry_price=row.get('close', 0.0),
                stop_price=row.get('stop_price', 0.0) or (row.get('close', 0.0) * 0.98),
                tp1_price=row.get('tp1_price', 0.0) or (row.get('close', 0.0) * 1.05),
                tp2_price=row.get('tp2_price', 0.0) or (row.get('close', 0.0) * 1.10),
                discovery_score=round(discovery, 2),
                entry_readiness=round(readiness, 2),
                fakeout_risk=round(fakeout, 2),
                hold_quality=round(h_quality, 2),
                leader_prob=round(leader_p, 4),
                composite_leader_score=round(composite, 2),
                # ── EXPLANATION (auxiliary context only) ─────────────────────
                explanation={
                    "family":           row.get('family', 'beta'),
                    "triggers":         entry_metrics.get('triggers', []),
                    "ml_explanation":   row.get('ml_explanation', 'N/A'),
                    "rejection_reason": row.get('rejection_reason', ''),
                    "kelly_risk_pct":   row.get('kelly_risk_pct', 1.0),
                }
            ))

        # Sort by composite score descending
        final_decisions.sort(key=lambda x: x.final_trade_score, reverse=True)
        return final_decisions

    def aggregate_and_rank(self, candidates: pd.DataFrame) -> List[FinalDecision]:
        return self.process_candidates(candidates)
