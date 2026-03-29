import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from defihunter.utils.trade_utils import TradeUtils

class VectorizedBacktestEngine:
    """
    GT-Institutional: High-Performance Vectorized Backtester.
    Optimized for multi-year strategy iteration using vectorized SL/TP logic.
    Note: Some complex management signals (dynamic decay) are approximated.
    """
    def __init__(self, config: Any = None):
        self.config = config
        self.bt_cfg = getattr(config, "backtest", None) if config else None
        
    def run(self, df: pd.DataFrame, initial_equity: float = 10000.0) -> Dict[str, Any]:
        """
        Runs vectorized simulation on a multi-symbol dataframe.
        Expects df with columns: [timestamp, symbol, close, high, low, entry_signal, stop_price, tp1_price, tp2_price]
        """
        if df.empty:
            return {"error": "Empty dataframe"}
            
        df = df.copy().sort_values(['symbol', 'timestamp'])
        
        # 1. Entry Logic (Vectorized)
        # We assume immediate execution at 'close' of the signal bar
        df['is_entry'] = df['entry_signal'].fillna(False)
        
        # 2. Exit Logic (Vectorized approximations)
        # For a truly vectorized SL/TP/TS, we'd need to shift and compare.
        # This is complex for multi-exit (TP1/TP2). 
        # A simpler way is to find the FIRST bar where price hits SL or TP.
        
        # Simplified Vectorized version (assuming single TP for now for speed)
        # In a real institutional tool, we'd use 'apply' or custom Numba for TP1/TP2 logic.
        results = []
        
        for symbol, s_df in df.groupby('symbol'):
            s_df = s_df.reset_index(drop=True)
            entries = s_df[s_df['is_entry']].index
            
            for start_idx in entries:
                entry_row = s_df.iloc[start_idx]
                entry_p = entry_row['close']
                stop_p = entry_row['stop_price']
                tp1_p = entry_row['tp1_price']
                tp2_p = entry_row['tp2_price']
                
                # Look ahead for first exit
                future = s_df.iloc[start_idx + 1:]
                if future.empty: continue
                
                # Vectorized exit detection for THIS trade
                sl_hit = future['low'] <= stop_p
                tp1_hit = future['high'] >= tp1_p
                tp2_hit = future['high'] >= tp2_p
                
                # First event
                exit_idx = None
                reason = "TIMEOUT"
                exit_p = entry_p
                
                # Check for first SL hit
                sl_indices = future.index[sl_hit]
                tp1_indices = future.index[tp1_hit]
                tp2_indices = future.index[tp2_hit]
                
                first_sl = sl_indices[0] if not sl_indices.empty else 999999
                first_tp1 = tp1_indices[0] if not tp1_indices.empty else 999999
                
                if first_sl < first_tp1:
                    exit_idx = first_sl
                    exit_p = stop_p
                    reason = "STOP_LOSS"
                elif first_tp1 != 999999:
                    # Partial exit logic (TP1)
                    # For simplicity in 'vectorized' we simulate the TP2 or subsequent SL
                    # after SL is moved to breakeven
                    new_stop = entry_p
                    remaining = s_df.iloc[first_tp1 + 1:]
                    
                    sl2_hit = remaining['low'] <= new_stop
                    tp2_hit_rem = remaining['high'] >= tp2_p
                    
                    first_sl2 = sl2_hit.index[sl2_hit][0] if not sl2_hit.index[sl2_hit].empty else 999999
                    first_tp2 = tp2_hit_rem.index[tp2_hit_rem][0] if not tp2_hit_rem.index[tp2_hit_rem].empty else 999999
                    
                    if first_tp2 < first_sl2:
                        exit_idx = first_tp2
                        exit_p = tp2_p
                        reason = "TP2_FINAL"
                    elif first_sl2 != 999999:
                        exit_idx = first_sl2
                        exit_p = new_stop
                        reason = "BE_STOP"
                    else:
                        exit_idx = s_df.index[-1]
                        exit_p = s_df.iloc[-1]['close']
                        reason = "END_OF_DATA"
                else:
                    exit_idx = s_df.index[-1]
                    exit_p = s_df.iloc[-1]['close']
                    reason = "END_OF_DATA"
                
                # Calculate metrics
                fee_bps = getattr(self.bt_cfg, "fee_bps", 2.0) if self.bt_cfg else 2.0
                slippage_bps = getattr(self.bt_cfg, "slippage_bps", 1.0) if self.bt_cfg else 1.0
                
                pnl_r = TradeUtils.calculate_net_pnl_r(
                    entry_price=entry_p,
                    exit_price=exit_p,
                    stop_price=stop_p,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps
                )
                
                results.append({
                    "symbol": symbol,
                    "entry_time": entry_row['timestamp'],
                    "exit_time": s_df.iloc[exit_idx]['timestamp'],
                    "exit_reason": reason,
                    "pnl_r": pnl_r,
                    "bars_held": exit_idx - start_idx
                })
        
        if not results:
            return {"total_trades": 0, "win_rate": 0}
            
        res_df = pd.DataFrame(results)
        return {
            "total_trades": len(res_df),
            "win_rate": len(res_df[res_df['pnl_r'] > 0]) / len(res_df),
            "avg_pnl_r": res_df['pnl_r'].mean(),
            "sum_pnl_r": res_df['pnl_r'].sum(),
            "expectancy": res_df['pnl_r'].mean(),
            "avg_bars_held": res_df['bars_held'].mean()
        }
