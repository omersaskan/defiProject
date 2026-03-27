import pandas as pd
import numpy as np
from typing import Any, Dict, List

class BacktestEngine:
    def __init__(self, config: Any = None, use_ml_ranking: bool = False, use_leadership_features: bool = True):
        self.config = config
        self.use_ml_ranking = use_ml_ranking
        self.use_leadership_features = use_leadership_features
        self.trade_log = []
        
    def simulate(self, df: pd.DataFrame) -> dict:
        """
        Multi-symbol event-driven simulation (Patch 1).
        Processes timestamps in order, managing a global portfolio of open positions.
        """
        if df.empty or 'timestamp' not in df.columns or 'symbol' not in df.columns:
            return {"error": "Invalid dataframe for multi-symbol backtest."}

        # Ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)
        timestamps = sorted(df['timestamp'].unique())
        
        # Portfolio State
        open_positions = [] # List of dicts: {symbol, entry_time, entry_price, stop, tp1, tp2, r_dist, bars_held, highest_seen}
        self.trade_log = []
        
        # Resolve Config
        bt_cfg = getattr(self.config, 'backtest', None)
        fee_bps = getattr(bt_cfg, 'fee_bps', 2.0)
        slippage_bps = getattr(bt_cfg, 'slippage_bps', 1.0)
        max_pos = getattr(bt_cfg, 'max_concurrent_positions', 5)
        time_stop = getattr(bt_cfg, 'time_stop_bars', 24)
        
        total_costs_pct = (fee_bps + slippage_bps) / 10000.0

        for ts in timestamps:
            ts_data = df[df['timestamp'] == ts]
            
            # 1. Update existing positions
            still_open = []
            for pos in open_positions:
                row = ts_data[ts_data['symbol'] == pos['symbol']]
                if row.empty:
                    # No data for this symbol at this TS, assume it stays open (risk: hole in data)
                    still_open.append(pos)
                    continue
                
                bar = row.iloc[0]
                pos['bars_held'] += 1
                
                if bar['high'] > pos['highest_seen']:
                    pos['highest_seen'] = bar['high']
                
                exit_price = None
                reason = None
                
                # Exit Logic: SL / TP2 / Leadership Decay / Time
                if bar['low'] <= pos['stop_price']:
                    exit_price = pos['stop_price']
                    reason = "STOP_LOSS"
                elif bar['high'] >= pos['tp2_price']:
                    exit_price = pos['tp2_price']
                    reason = "TP2"
                elif bar.get('leadership_decay', False):
                    exit_price = bar['close']
                    reason = "LEADERSHIP_DECAY"
                elif pos['bars_held'] >= time_stop:
                    exit_price = bar['close']
                    reason = "TIME_EXIT"
                
                if exit_price is not None:
                    pnl_r = ((exit_price - pos['entry_price']) / pos['r_dist']) - (total_costs_pct * pos['entry_price'] / pos['r_dist'])
                    self.trade_log.append({
                        "symbol": pos['symbol'],
                        "entry_time": pos['entry_time'],
                        "exit_time": ts,
                        "outcome": reason,
                        "pnl_r": pnl_r,
                        "bars_held": pos['bars_held'],
                        "peak_pnl_r": (pos['highest_seen'] - pos['entry_price']) / pos['r_dist']
                    })
                else:
                    still_open.append(pos)
            
            open_positions = still_open
            
            # 2. Check for new entries (Multi-Symbol selection / Patch 1)
            candidates = ts_data[ts_data['entry_signal'] == True]
            if not candidates.empty and len(open_positions) < max_pos:
                rooms = max_pos - len(open_positions)
                # Selection prioritized by ML rank if available, else total_score
                score_col = 'ml_rank_score' if 'ml_rank_score' in candidates.columns else 'total_score'
                top_candidates = candidates.nlargest(rooms, score_col)
                
                for _, c in top_candidates.iterrows():
                    # Avoid duplicate entry if already open
                    if any(p['symbol'] == c['symbol'] for p in open_positions):
                        continue
                        
                    entry_p = c['close']
                    stop_p = c.get('stop_price', entry_p * 0.95)
                    r_dist = abs(entry_p - stop_p)
                    if r_dist <= 0: continue
                    
                    open_positions.append({
                        "symbol": c['symbol'],
                        "entry_time": ts,
                        "entry_price": entry_p,
                        "stop_price": stop_p,
                        "tp1_price": c.get('tp1_price', entry_p * 1.05),
                        "tp2_price": c.get('tp2_price', entry_p * 1.10),
                        "r_dist": r_dist,
                        "bars_held": 0,
                        "highest_seen": entry_p
                    })

        # Metrics Compilation
        if not self.trade_log:
            return {"win_rate": 0, "profit_factor": 0, "expectancy_r": 0, "total_trades": 0}
            
        t_df = pd.DataFrame(self.trade_log)
        win_rate = len(t_df[t_df['pnl_r'] > 0]) / len(t_df) if len(t_df) > 0 else 0
        expectancy = t_df['pnl_r'].mean()
        
        # Profit Factor
        pos_pnl = t_df[t_df['pnl_r'] > 0]['pnl_r'].sum()
        neg_pnl = abs(t_df[t_df['pnl_r'] < 0]['pnl_r'].sum())
        pf = pos_pnl / neg_pnl if neg_pnl > 0 else (99.0 if pos_pnl > 0 else 0)
        
        return {
            "win_rate": round(win_rate * 100, 2),
            "expectancy_r": round(expectancy, 2),
            "profit_factor": round(pf, 2),
            "total_trades": len(t_df),
            "avg_bars_held": round(t_df['bars_held'].mean(), 1)
        }

    def evaluate_ranking_quality(self, df: pd.DataFrame, bars_horizon: int = 24, k: int = 3) -> dict:
        """
        Patch 1: High-fidelity leader prediction metrics for family-relative engine.
        Calculates precision, capture rates, and rank correlation.
        """
        if df.empty or 'ml_rank_score' not in df.columns or 'close' not in df.columns:
            return {"error": "Missing ml_rank_score or close for ranking evaluation"}
            
        df = df.copy().sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        # Compute future return correctly per symbol
        df['future_return'] = df.groupby('symbol')['close'].shift(-bars_horizon) / df['close'] - 1
        df = df.dropna(subset=['future_return'])
        
        # Compute family relative ranks for evaluation
        if 'family' in df.columns:
            df['future_family_rank'] = df.groupby(['timestamp', 'family'])['future_return'].rank(ascending=False, method='min')
            
        correlations, precisions, capture_rates, family_ranks = [], [], [], []
        missed_count = 0
        
        ts_list = sorted(df['timestamp'].unique())
        for ts in ts_list:
            ts_data = df[df['timestamp'] == ts]
            if len(ts_data) < k: continue
            
            # 1. Rank Correlation (Rank Score vs Future Return)
            corr = ts_data['ml_rank_score'].corr(ts_data['future_return'], method='spearman')
            if not np.isnan(corr): correlations.append(corr)
            
            # 2. Top-K Precision and Recall
            top_k_pred = set(ts_data.nlargest(k, 'ml_rank_score')['symbol'])
            top_k_actual = set(ts_data.nlargest(k, 'future_return')['symbol'])
            precisions.append(len(top_k_pred & top_k_actual) / k)
            
            # 3. Leader Capture Rate (% of top-1 actual captured in top-k pred)
            best_actual = ts_data.nlargest(1, 'future_return')['symbol'].iloc[0]
            if best_actual in top_k_pred:
                capture_rates.append(1.0)
            else:
                capture_rates.append(0.0)
                missed_count += 1
            
            # 4. Avg Selected Future Family Rank
            if 'future_family_rank' in ts_data.columns:
                selected_ranks = ts_data[ts_data['symbol'].isin(top_k_pred)]['future_family_rank']
                if not selected_ranks.empty:
                    family_ranks.append(selected_ranks.mean())
            
        # Calculate derived metrics for overall performance
        t_df = pd.DataFrame(self.trade_log)
        hold_eff, exit_eff = 0, 0
        if not t_df.empty and 'peak_pnl_r' in t_df.columns:
            # hold_efficiency = avg R realized / avg peak R seen
            realized_trades = t_df[t_df['pnl_r'] > 0]
            if not realized_trades.empty:
                realized = realized_trades['pnl_r'].mean()
                peak = realized_trades['peak_pnl_r'].mean()
                if peak and peak > 0:
                    hold_eff = round((realized / peak) * 100, 1)
                
                # exit_efficiency = % of trades capturing at least 50% of peak R
                exit_eff_count = len(realized_trades[realized_trades['pnl_r'] >= realized_trades['peak_pnl_r'] * 0.5])
                exit_eff = round((exit_eff_count / len(realized_trades)) * 100, 1)

        return {
            "rank_correlation": round(np.nanmean(correlations), 3) if correlations else 0,
            "top_k_precision": round(np.nanmean(precisions) * 100, 1) if precisions else 0,
            "top_k_recall": round(np.nanmean(precisions) * 100, 1) if precisions else 0,
            "leader_capture_rate": round(np.nanmean(capture_rates) * 100, 1) if capture_rates else 0,
            "avg_selected_future_family_rank": round(np.nanmean(family_ranks), 2) if family_ranks else 0,
            "missed_leaders": missed_count,
            "hold_efficiency": hold_eff,
            "exit_efficiency": exit_eff,
            "n_timestamps": len(correlations)
        }

    def walk_forward_simulate(self, dataframe: pd.DataFrame, train_size: int = 500, test_size: int = 200, step_size: int = 100, train_ml: bool = False) -> list:
        # Legacy placeholder - maintained for structure
        return []
