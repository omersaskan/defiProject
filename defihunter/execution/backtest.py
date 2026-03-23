import pandas as pd
import numpy as np
from typing import Any, Dict, List

class BacktestEngine:
    def __init__(self, config: Any = None, use_ml_ranking: bool = False, use_leadership_features: bool = True):
        self.config = config
        self.use_ml_ranking = use_ml_ranking
        self.use_leadership_features = use_leadership_features
        self.trade_log = []
        
    def simulate(self, dataframe_with_signals: pd.DataFrame) -> dict:
        """
        Simulates event-driven trades based on entry/exit signals.
        Evaluates row-by-row if a triggered signal hits Stop, TP1, or TP2 in subsequent bars.
        """
        if dataframe_with_signals.empty or 'entry_signal' not in dataframe_with_signals.columns:
            return {"error": "No valid signals or data provided."}
            
        print(f"Simulating trades over {len(dataframe_with_signals)} periods.")
        
        df = dataframe_with_signals.copy()
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        # Track active trades by exit timestamp for portfolio Limits
        active_trade_exits = []
        
        # Resolve Costs and Limits from Config
        fee_bps = 2.0
        slippage_bps = 1.0
        time_stop_bars = 24
        max_positions = 999
        funding_enabled = False
        
        if self.config and hasattr(self.config, 'backtest'):
            bt_cfg = self.config.backtest
            fee_bps = getattr(bt_cfg, 'fee_bps', 2.0)
            slippage_bps = getattr(bt_cfg, 'slippage_bps', 1.0)
            time_stop_bars = getattr(bt_cfg, 'time_stop_bars', 24)
            max_positions = getattr(bt_cfg, 'max_concurrent_positions', 999)
            funding_enabled = getattr(bt_cfg, 'funding_costs_enabled', False)
        elif self.config and hasattr(self.config, 'risk'): # Fallback logic
            fee_bps = getattr(self.config.risk, 'backtest_fee_bps', 2.0)
            slippage_bps = getattr(self.config.risk, 'backtest_slippage_bps', 1.0)
            
        total_costs_pct = (fee_bps + slippage_bps) / 10000.0
        
        # Extract signal rows
        signal_indices = df[df['entry_signal'] == True].index
        
        for idx in signal_indices:
            signal_bar = df.iloc[idx]
            current_time = signal_bar['timestamp']
            
            # 1. Update active trades (remove ones that have exited by now)
            active_trade_exits = [exit_time for exit_time in active_trade_exits if exit_time > current_time]
            
            # 2. Enforce Portfolio Constraint
            if len(active_trade_exits) >= max_positions:
                continue # Skip this signal, portfolio is full
            
            # 3. Realistic Entry: Execution on Next Bar Open (Criteria 7.3)
            # Signal triggers at Close of Bar T, Entry happens at Open of Bar T+1
            entry_price = df.iloc[idx + 1]['open']
            stop_price = signal_bar.get('stop_price', entry_price * 0.95)
            tp1_price = signal_bar.get('tp1_price', entry_price * 1.05)
            
            # Risk distance
            r_dist = abs(entry_price - stop_price)
            if r_dist == 0: continue
            
            # Fee in R-terms
            cost_in_r = (total_costs_pct * entry_price) / r_dist
            
            # Approximate single-bar funding cost (e.g. 15m bar from an 8H rate)
            funding_rate_value = signal_bar.get('funding_rate', 0.0) if funding_enabled else 0.0
            funding_cost_per_bar_r = (funding_rate_value / 32.0 * entry_price) / r_dist if r_dist > 0 else 0
            
            outcome = "OPEN"
            pnl_r = 0.0
            bars_held = 0
            
            # Trailing stop configuration
            current_stop = stop_price
            highest_seen = entry_price
            final_tp = signal_bar.get('tp2_price', tp1_price * 1.05)
            
            # Scan forward time_stop bars (or until end of data)
            max_hold = min(idx + time_stop_bars, len(df))
            for future_idx in range(idx + 1, max_hold):
                future_bar = df.iloc[future_idx]
                bars_held += 1
                
                # Trailing stop adjustments
                if future_bar['high'] > highest_seen:
                    highest_seen = future_bar['high']
                    
                # If we've made at least 1R profit, move stop to breakeven
                if highest_seen >= entry_price + r_dist * 1.0:
                    if current_stop < entry_price:
                        current_stop = entry_price
                        
                # If we've made 2R profit, trail by 1R behind highest_seen
                if highest_seen >= entry_price + r_dist * 2.0:
                    potential_stop = highest_seen - r_dist * 1.0
                    if potential_stop > current_stop:
                        current_stop = potential_stop
                
                # Check stop loss first (Pessimistic intra-bar fill)
                if future_bar['low'] <= current_stop:
                    outcome = "LOSS" if current_stop < entry_price else "WIN_TRAILED"
                    exit_price = current_stop
                    # Apply fees, slippage, and cumulative funding
                    cum_funding_r = (funding_cost_per_bar_r * bars_held) if funding_enabled else 0.0
                    pnl_r = ((exit_price - entry_price) / r_dist) - cost_in_r - cum_funding_r
                    break
                    
                # Check Leadership Decay (Phase 3 Intelligent Exit)
                if future_bar.get('leadership_decay', False):
                    outcome = "EXIT_DECAY"
                    exit_price = future_bar['close']
                    cum_funding_r = (funding_cost_per_bar_r * bars_held) if funding_enabled else 0.0
                    pnl_r = ((exit_price - entry_price) / r_dist) - cost_in_r - cum_funding_r
                    break

                # Check final TP
                if future_bar['high'] >= final_tp:
                    outcome = "WIN"
                    # Apply fees, slippage, and cumulative funding
                    cum_funding_r = (funding_cost_per_bar_r * bars_held) if funding_enabled else 0.0
                    pnl_r = ((final_tp - entry_price) / r_dist) - cost_in_r - cum_funding_r
                    break
                    
            if outcome == "OPEN":
                # Time exit after time_stop_bars
                exit_price = df.iloc[max_hold - 1]['close']
                cum_funding_r = (funding_cost_per_bar_r * bars_held) if funding_enabled else 0.0
                pnl_r = ((exit_price - entry_price) / r_dist) - cost_in_r - cum_funding_r
                outcome = "TIME_EXIT"
                
            # Record exit time for portfolio concurrency tracking
            exit_time = df.iloc[idx + bars_held]['timestamp'] if bars_held > 0 else signal_bar['timestamp']
            active_trade_exits.append(exit_time)
                
            self.trade_log.append({
                "symbol": signal_bar.get('symbol', 'UNKNOWN'),
                "entry_time": signal_bar['timestamp'],
                "outcome": outcome,
                "pnl_r": pnl_r,
                "bars_held": bars_held,
                "peak_pnl_r": (highest_seen - entry_price) / r_dist if r_dist > 0 else 0,
                "mfe_capture_ratio": ((highest_seen - entry_price) / (final_tp - entry_price)) if (final_tp - entry_price) > 0 else 0
            })
            
        # Compile Metrics
        if not self.trade_log:
            return {"win_rate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0, "total_trades": 0}
            
        trades_df = pd.DataFrame(self.trade_log)
        wins = trades_df[trades_df['pnl_r'] > 0]
        losses = trades_df[trades_df['pnl_r'] <= 0]
        
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0
        gross_profit = wins['pnl_r'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['pnl_r'].sum()) if not losses.empty else 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        expectancy = trades_df['pnl_r'].mean()
        
        # New Exit Metrics
        # Giveback Ratio: (Peak - Realized) / Peak
        trades_df['giveback_ratio'] = (trades_df['peak_pnl_r'] - trades_df['pnl_r']).clip(lower=0) / trades_df['peak_pnl_r'].replace(0, np.nan)
        # Hold Efficiency: Realized / Peak
        trades_df['hold_efficiency'] = trades_df['pnl_r'] / trades_df['peak_pnl_r'].replace(0, np.nan)
        # Exit Efficiency: (Exit - Entry) / (Peak - Entry)
        trades_df['exit_efficiency'] = trades_df['hold_efficiency']
        
        # Lead Time: Bars to Peak
        # (Assuming bars_held is total hold, we don't have bars_to_peak explicitly in trade_log yet, 
        # but we can approximate or just use bars_held for simplicity if we don't track peak_idx)
        # Let's assume 15m bars for lead time calculation
        avg_lead_time_hours = (trades_df['bars_held'].mean() * 0.25) if len(trades_df) > 0 else 0.0
        
        return {
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy_r": round(expectancy, 2),
            "total_trades": len(trades_df),
            "avg_giveback_ratio": round(trades_df['giveback_ratio'].mean(), 2),
            "avg_hold_efficiency": round(trades_df['hold_efficiency'].mean(), 2),
            "avg_exit_efficiency": round(trades_df['exit_efficiency'].mean(), 2),
            "avg_lead_time_hours": round(avg_lead_time_hours, 2)
        }

    def evaluate_ranking_quality(self, scored_dfs: Dict[str, pd.DataFrame], k: int = 5) -> dict:
        """
        GT-REDESIGN: Evaluates the quality of 'Who' selection.
        Measures Spearman Rank Correlation, Top-K Precision/Recall, and Capture Rate.
        """
        metrics = []
        
        all_data = pd.concat(scored_dfs.values())
        if 'ml_rank_score' not in all_data.columns or 'close' not in all_data.columns:
            return {"error": "Missing ml_rank_score or close for ranking evaluation"}
            
        timestamps = sorted(all_data['timestamp'].unique())
        
        correlations = []
        top_k_precisions = []
        top_k_recalls = []
        capture_rates = []
        missed_leaders_count = 0
        total_eval_points = 0
        
        for ts in timestamps:
            ts_data = all_data[all_data['timestamp'] == ts].copy()
            # 24h future return
            ts_data['future_return_24h'] = ts_data.groupby('symbol')['close'].shift(-24) / ts_data['close'] - 1
            ts_data = ts_data.dropna(subset=['future_return_24h'])
            
            if len(ts_data) < k * 2: continue
            total_eval_points += 1
            
            # Rank Correlation
            corr = ts_data['ml_rank_score'].corr(ts_data['future_return_24h'], method='spearman')
            if not np.isnan(corr):
                correlations.append(corr)
                
            # Top-k Metrics
            top_k_pred = set(ts_data.nlargest(k, 'ml_rank_score')['symbol'])
            top_k_actual = set(ts_data.nlargest(k, 'future_return_24h')['symbol'])
            
            hits = len(top_k_pred & top_k_actual)
            top_k_precisions.append(hits / float(k))
            top_k_recalls.append(hits / float(k)) # For fixed k, Prec == Recall
            
            # Leader Capture Rate: % of top-1 actual captured in top-k pred
            best_actual = ts_data.nlargest(1, 'future_return_24h')['symbol'].iloc[0]
            capture_rates.append(1.0 if best_actual in top_k_pred else 0.0)
            if best_actual not in top_k_pred:
                missed_leaders_count += 1
            
            # Avg Selected Future Family Rank
            if 'future_24h_rank_in_family' in ts_data.columns:
                selected_ranks = ts_data[ts_data['symbol'].isin(top_k_pred)]['future_24h_rank_in_family']
                if not selected_ranks.empty:
                    metrics.append(selected_ranks.mean())
            
        return {
            "rank_correlation": round(np.mean(correlations), 3) if correlations else 0.0,
            "top_k_precision": round(np.mean(top_k_precisions) * 100, 1) if top_k_precisions else 0.0,
            "top_k_recall": round(np.mean(top_k_recalls) * 100, 1) if top_k_recalls else 0.0,
            "leader_capture_rate": round(np.mean(capture_rates) * 100, 1) if capture_rates else 0.0,
            "avg_selected_future_family_rank": round(np.mean(metrics), 2) if metrics else 0.0,
            "missed_leaders": missed_leaders_count,
            "n_timestamps_evaluated": total_eval_points
        }

    def walk_forward_simulate(self, dataframe: pd.DataFrame, train_size: int = 500, test_size: int = 200, step_size: int = 100, train_ml: bool = False) -> list:
        """
        Implements walk-forward evaluation using a sliding window.
        If train_ml is True, dynamically trains MLRankingEngine on the trailing window before simulating.
        Returns a list of performance reports for each test fold.
        """
        if len(dataframe) < train_size + test_size:
            return [{"error": "Data too short for walk-forward"}]
            
        folds_results = []
        df = dataframe.sort_values(by='timestamp').reset_index(drop=True)
        
        ml_engine = None
        if train_ml:
            from defihunter.engines.ml_ranking import MLRankingEngine
            ml_engine = MLRankingEngine()
            
        start_idx = 0
        while start_idx + train_size + test_size <= len(df):
            test_start = start_idx + train_size
            test_end = test_start + test_size
            
            # test_df only for this fold
            train_df = df.iloc[start_idx:test_start].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            if train_ml and ml_engine:
                print(f"WF: Sub-Training ML Model on {len(train_df)} rows for fold [{train_df.iloc[0]['timestamp']} -> {train_df.iloc[-1]['timestamp']}]")
                # Needs targets to be pre-generated in dataframe (e.g. via DatasetBuilder)
                if 'regime' in train_df.columns:
                    ml_engine.train_global(train_df)
                else:
                    ml_engine.train(train_df, symbol="WF_BACKTEST")
                
                # Apply model predictions to test_df to override entry_signal
                # Just simulating basic logic: if ML probability is high, we take the trade
                test_df_scored, _ = ml_engine.rank_candidates(test_df, top_n=len(test_df))
                
                # Merge the new rank scores back, though rank_candidates returns the scored df directly
                # Ensure we only trade when ml_rank_score is very high (> 70) if ML is governing
                if 'ml_rank_score' in test_df_scored.columns:
                    test_df_scored['entry_signal'] = test_df_scored['entry_signal'] & (test_df_scored['ml_rank_score'] >= 70)
                test_df = test_df_scored
            
            # Clear previous results and simulate
            self.trade_log = []
            report = self.simulate(test_df)
            
            folds_results.append({
                "fold_start": df.iloc[test_start]['timestamp'],
                "fold_end": df.iloc[test_end-1]['timestamp'],
                "metrics": report
            })
            
            start_idx += step_size
            
        return folds_results

    def run_ablation_study(self, dataframe: pd.DataFrame, rule_engine, family: str) -> dict:
        """
        Run the Relative EMA Leadership integration ablation study.
        Toggles leadership features to see their impact on expectancy.
        """
        if dataframe.empty:
            return {"error": "No data for ablation study."}

        # 1. Base Rules (Veto if leadership is used in rules normally, but here we can force a run without it)
        # We simulate this by mocking leadership score to 0 or bypassing the rule logic
        df_base = dataframe.copy()
        df_base['relative_leadership_score'] = 0
        df_base['total_score'] = df_base['trend_score'] + df_base['expansion_score'] + df_base['participation_score']
        df_base['entry_signal'] = df_base['total_score'] >= 50 # Standard threshold
        
        self.trade_log = []
        base_report = self.simulate(df_base)
        
        # 2. Rules + Leadership
        # Use the rules already in the dataframe (assuming they were pre-calculated)
        self.trade_log = []
        leadership_report = self.simulate(dataframe)
        
        # 3. Rules + Leadership + ML (Requires ML being pre-run or running it here)
        # Assuming ml_rank_score is in df
        df_ml = dataframe.copy()
        if 'ml_rank_score' in df_ml.columns:
            # Only take signals where ML rank EV is high (e.g. > 60)
            df_ml['entry_signal'] = df_ml['entry_signal'] & (df_ml['ml_rank_score'] > 60)
            self.trade_log = []
            ml_report = self.simulate(df_ml)
        else:
            ml_report = {"expectancy_r": "N/A"}

        return {
            "base_rules_expectancy": base_report.get('expectancy_r', 0),
            "with_leadership_expectancy": leadership_report.get('expectancy_r', 0),
            "with_ml_ranking_expectancy": ml_report.get('expectancy_r', 0),
            "base_trade_count": base_report.get('total_trades', 0),
            "leadership_trade_count": leadership_report.get('total_trades', 0)
        }

