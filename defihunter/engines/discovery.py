import pandas as pd
from typing import Dict, List, Any
from defihunter.core.models import FinalDecision

class DiscoveryEngine:
    """
    The 'Who' Engine.
    Identifies which DeFi coins are likely to become daily leaders.
    Combines:
    - ML Rank Score (Cross-sectional ranking probability)
    - Leadership Score (Relative strength vs anchors)
    - Rule-based logic (Regime fit, macro buildup)
    """
    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def compute_discovery_scores(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates Discovery factors into a single DiscoveryScore (0-100).
        Discovery answers: "Is this a potential DeFi leader candidate?"
        """
        if candidates.empty:
            return candidates

        df = candidates.copy()

        # 1. ML Factor (0-1.0)
        df['ml_discovery_factor'] = (df['leader_prob'].fillna(0.5) if 'leader_prob' in df.columns else pd.Series(0.5, index=df.index))
        
        # 2. Family Factor (0-1.0)
        # Combines family heat (momentum) and breadth (participation)
        f_heat = (df['family_heat_score'] if 'family_heat_score' in df.columns else pd.Series(0.0, index=df.index)).fillna(0).clip(lower=0, upper=0.1) / 0.1
        f_breadth = (df['family_breadth_score'] if 'family_breadth_score' in df.columns else pd.Series(0.0, index=df.index)).fillna(0)
        df['family_factor'] = (f_heat * 0.6 + f_breadth * 0.4)

        # 3. Strength Factor (0-1.0)
        # Moves vs peers and total rule score
        p_mo = (df['peer_momentum'] if 'peer_momentum' in df.columns else pd.Series(0.0, index=df.index)).fillna(0).clip(lower=0, upper=0.05) / 0.05
        p_rank = (df['peer_rank'] if 'peer_rank' in df.columns else pd.Series(0.5, index=df.index)).fillna(0.5)
        df['strength_factor'] = (p_mo * 0.5 + p_rank * 0.5)

        # Weighted Discovery Score
        w_ml = 0.40
        w_fam = 0.35
        w_strength = 0.25

        df['discovery_score'] = (
            (df['ml_discovery_factor'] * w_ml) +
            (df['family_factor'] * w_fam) +
            (df['strength_factor'] * w_strength)
        ) * 100.0

        return df.sort_values(by='discovery_score', ascending=False)
