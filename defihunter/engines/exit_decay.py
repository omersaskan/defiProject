import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class ExitDecayEngine:
    """
    GT-REDESIGN: Advanced Exit & Decay Engine.
    Handles leadership-decay, peer-relative decay, and climax/exhaustion exits.
    """
    def __init__(self, config: Any = None):
        self.config = config

    def evaluate_exit_signals(self, symbol: str, df: pd.DataFrame, family_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluates cumulative exit signals for a given symbol.
        Returns a dictionary with decay flags and reasons.
        """
        if df.empty:
            return {"exit_signal": False, "reason": "No data"}

        last_row = df.iloc[-1]
        
        # Mapping signals from feature pipeline (Phase 4 & 5)
        signals = {
            "leadership_decay": last_row.get("leadership_decay", False),
            "volume_climax": last_row.get("volume_climax_flag", last_row.get("decay_is_climax", False)),
            "wick_exhaustion": last_row.get("wick_exhaustion_flag", False),
            "peer_relative_decay": last_row.get("peer_relative_decay", False),
            "leadership_turn_down": last_row.get("leadership_turn_down", False),
            "family_cooling_flag": last_row.get("family_cooling_flag", False)
        }
        
        # 1. Manual Fallback: Peer-Relative Decay
        if not signals["peer_relative_decay"] and family_data is not None and not family_data.empty:
            family_avg_return = family_data['avg_return_1h'].iloc[-1] if 'avg_return_1h' in family_data.columns else family_data['close'].pct_change(4).iloc[-1]
            coin_return = df['close'].pct_change(4).iloc[-1]
            if coin_return < -0.01 and coin_return < family_avg_return - 0.02:
                signals["peer_relative_decay"] = True

        # 2. Manual Fallback: Leadership Turn Down
        if not signals["leadership_turn_down"]:
            rel_spread_cols = [c for c in df.columns if 'rel_spread' in c and 'slope' in c]
            if rel_spread_cols:
                slope_val = last_row.get(rel_spread_cols[0], 0)
                z_col = rel_spread_cols[0].replace('_slope_4', '_z_96')
                z_val = last_row.get(z_col, 0)
                if slope_val < 0 and z_val > 2.0:
                    signals["leadership_turn_down"] = True

        # 3. Manual Fallback: Family Cooling Flag
        if not signals["family_cooling_flag"] and family_data is not None and not family_data.empty:
            family_momentum = family_data['family_heat_accel'].iloc[-1] if 'family_heat_accel' in family_data.columns else family_data['close'].pct_change(12).iloc[-1]
            if family_momentum < -5.0:
                signals["family_cooling_flag"] = True

        # Combined Exit Logic
        exit_triggered = signals["leadership_decay"] or \
                         signals["peer_relative_decay"] or \
                         signals["leadership_turn_down"] or \
                         signals["family_cooling_flag"] or \
                         signals["volume_climax"] or \
                         signals["wick_exhaustion"]

        reasons = [k for k, v in signals.items() if v]
        
        return {
            "exit_signal": exit_triggered,
            "exit_reason": ", ".join(reasons) if reasons else "None",
            "signals": signals
        }
