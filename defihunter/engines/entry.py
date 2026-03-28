import pandas as pd
import numpy as np
from typing import Dict, Any

class EntryEngine:
    """
    The 'When' Engine.
    Detects the optimal time to enter and assesses breakout risks.
    """
    def __init__(self, min_readiness: float = 65.0):
        self.min_readiness = min_readiness

    def evaluate_readiness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch evaluation for historical data (backtesting).
        Populates 'entry_readiness' and 'fakeout_risk' columns.
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 1. Component Flags (Vectorized)
        ready_score = pd.Series(0.0, index=df.index)
        if 'msb_bull' in df.columns:
            ready_score += df['msb_bull'].astype(float) * 40
        if 'taker_surge' in df.columns:
            ready_score += df['taker_surge'].astype(float) * 25
        if 'v_delta_score' in df.columns:
            ready_score += (df['v_delta_score'] > 0.08).astype(float) * 20
        if 'squeeze_release' in df.columns:
            ready_score += df['squeeze_release'].astype(float) * 15
            
        df['entry_readiness'] = ready_score.clip(upper=100.0)
        
        # 2. Fakeout Risk (Vectorized)
        risk_score = pd.Series(0.0, index=df.index)
        if 'breakout_quality' in df.columns and 'is_breakout_bar' in df.columns:
            risk_score += ((df['breakout_quality'] < 40) & df['is_breakout_bar'].astype(bool)).astype(float) * 40
        
        if 'exhaustion_risk_score' in df.columns:
            risk_score += df['exhaustion_risk_score'] * 0.5
            
        if 'upper_wick_ratio' in df.columns:
            risk_score += (df['upper_wick_ratio'] > 0.6).astype(float) * 20
            
        df['fakeout_risk'] = risk_score.clip(upper=100.0)
        
        return df

    def compute_entry_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates EntryReadiness and FakeoutRisk for a single coin.
        """
        if df.empty:
            return {"readiness": 0.0, "fakeout_risk": 0.0, "triggers": []}

        last_row = df.iloc[-1]
        
        # 1. Entry Readiness (0-100)
        # Focus on micro-catalysts and confirmation
        ready_score = 0.0
        triggers = []

        if last_row.get('msb_bull', False):
            ready_score += 40
            triggers.append("MSB_BULL")
        
        if last_row.get('taker_surge', False):
            ready_score += 25
            triggers.append("TAKER_SURGE")
        
        if last_row.get('v_delta_score', 0.0) > 0.08:
            ready_score += 20
            triggers.append("CVD_ACCEL")
            
        if last_row.get('squeeze_release', False):
            ready_score += 15
            triggers.append("SQUEEZE_RELEASE")

        # 2. Fakeout Risk (0-100)
        # Focus on exhaustion and poor breakout quality
        risk_score = 0.0
        
        # Poor breakout quality
        bq = last_row.get('breakout_quality', 0.0)
        if bq < 40 and last_row.get('is_breakout_bar', False):
            risk_score += 40
            
        # Exhaustion risk from persistence features
        exhaustion = last_row.get('exhaustion_risk_score', 0.0)
        risk_score += exhaustion * 0.5
        
        # High upper wick ratio
        if last_row.get('upper_wick_ratio', 0.0) > 0.6:
            risk_score += 20

        return {
            "readiness": min(100.0, ready_score),
            "fakeout_risk": min(100.0, risk_score),
            "triggers": triggers,
            "can_entry": ready_score >= self.min_readiness and risk_score < 60
        }
