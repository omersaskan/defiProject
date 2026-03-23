import pandas as pd
from typing import List, Dict, Any
from defihunter.engines.discovery import DiscoveryEngine
from defihunter.engines.entry import EntryEngine
from defihunter.core.models import FinalDecision

class DecisionEngine:
    """
    Orchestrator for Layered Leader Trading.
    Phase 5: Discovery -> Entry -> Risk -> Hold.
    """
    def __init__(self, top_n: int = 5):
        self.discovery = DiscoveryEngine(top_n=top_n)
        self.entry = EntryEngine(min_readiness=65.0)

    def process_candidates(self, candidates: pd.DataFrame) -> List[FinalDecision]:
        """
        1. Run Discovery to find top leaders.
        2. Filter only those that have DiscoveryScore > 60.
        3. Check EntryReadiness & FakeoutRisk.
        4. Calculate HoldQuality.
        5. Generate CompositeLeaderScore.
        """
        if candidates.empty:
            return []

        # Step 1: Discovery Scoring
        df_discovery = self.discovery.compute_discovery_scores(candidates)
        
        # Step 2: Select candidates for deep review (Discovery > 55)
        # We lowered slightly to ensure we don't miss emerging leaders
        potential_leaders = df_discovery[df_discovery['discovery_score'] >= 55.0]
        
        final_decisions = []
        for _, row in potential_leaders.iterrows():
            # Usually passed as latest bar or slice in scanner
            # For backtest or scanner integration, we assume row contains latest and necessary features
            
            # Step 3: Entry & Risk Analysis
            # In live scanner, we re-fetch micro TF, but here we use what's available
            # Note: Scanner will override if it does its own micro-tf fetch
            entry_metrics = self.entry.compute_entry_metrics(pd.DataFrame([row]))
            
            # Step 4: Hold Quality (0-100)
            # Combines advanced persistence features and ML holdability
            h_ml = row.get('holdability_score', 50.0)
            
            # Persistence components
            h_trend = min(100.0, row.get('trend_persistence_score', 0) * 5)
            h_vol = min(100.0, row.get('volume_persistence_score', 0) * 10)
            h_close = row.get('close_to_high_persistence', 0.5) * 100.0
            h_persist = (h_trend * 0.4 + h_vol * 0.3 + h_close * 0.3)
            
            # Penalty for trend exhaustion
            exhaust = row.get('exhaustion_risk_score', 0.0)
            
            h_quality = (h_ml * 0.6 + h_persist * 0.4) - (exhaust * 0.5)
            h_quality = max(0.0, min(100.0, h_quality))
            
            # Step 5: Composite Leader Score (0-100)
            # This is the "God Score" for the leaderboard
            discovery = row['discovery_score']
            readiness = entry_metrics['readiness']
            risk_penalty = entry_metrics['fakeout_risk'] * 0.5
            
            composite = (discovery * 0.4 + readiness * 0.4 + h_quality * 0.2) - risk_penalty
            composite = max(0.0, min(100.0, composite))
            
            # Determine Action
            action = "watch"
            if entry_metrics['can_entry'] and discovery >= 60:
                action = "trade"
            elif discovery >= 70:
                action = "urgent_watch"
            elif entry_metrics['fakeout_risk'] > 70:
                action = "avoid_fakeout"

            final_decisions.append(FinalDecision(
                symbol=row['symbol'],
                timestamp=row.get('timestamp', row.get('Timestamp')),
                final_trade_score=round(composite, 2),
                decision=action,
                entry_price=row.get('close', 0.0),
                stop_price=row.get('stop_price', 0.0) or (row.get('close', 0.0) * 0.98),
                tp1_price=row.get('tp1_price', 0.0) or (row.get('close', 0.0) * 1.05),
                explanation={
                    "family": row.get('family', 'beta'),
                    "discovery_score": round(discovery, 1),
                    "entry_readiness": round(readiness, 1),
                    "fakeout_risk": round(entry_metrics['fakeout_risk'], 1),
                    "hold_quality": round(h_quality, 1),
                    "leader_prob": round(row.get('leader_prob', 0), 2),
                    "composite_score": round(composite, 1),
                    "triggers": entry_metrics['triggers']
                }
            ))

        # Sort by composite score
        final_decisions.sort(key=lambda x: x.final_trade_score, reverse=True)
        return final_decisions

    def aggregate_and_rank(self, candidates: pd.DataFrame) -> List[FinalDecision]:
        return self.process_candidates(candidates)
