import pandas as pd
import numpy as np
from typing import Any, Dict, List

from defihunter.engines.adaptive_stop import AdaptiveStopEngine


class BacktestEngine:
    """
    Multi-symbol event-driven backtester.

    Execution semantics match PaperTradeEngine:
      - Partial TP1 at tp1_price → 50% close, runner active, SL moved to breakeven
      - Trailing stop for runner: trails 20% reward distance behind peak
      - Enriched trade_log with partial_taken, mfe_r, giveback_r, exit_reason
    """

    def __init__(
        self,
        config: Any = None,
        use_ml_ranking: bool = False,
        use_leadership_features: bool = True,
    ):
        self.config = config
        self.use_ml_ranking = use_ml_ranking
        self.use_leadership_features = use_leadership_features
        self.trade_log: List[Dict] = []
        self._adaptive_stop = AdaptiveStopEngine()

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN SIMULATE
    # ─────────────────────────────────────────────────────────────────────────
    def simulate(self, df: pd.DataFrame) -> dict:
        """
        Multi-symbol event-driven simulation with full paper-trade parity.
        Processes timestamps in order, managing a global portfolio of open positions.
        """
        if df.empty or 'timestamp' not in df.columns or 'symbol' not in df.columns:
            return {"error": "Invalid dataframe for multi-symbol backtest."}

        df = df.sort_values('timestamp').reset_index(drop=True)
        timestamps = sorted(df['timestamp'].unique())

        # Portfolio State
        open_positions: List[Dict] = []
        self.trade_log = []

        # Config resolution
        bt_cfg        = getattr(self.config, 'backtest', None)
        fee_bps       = getattr(bt_cfg, 'fee_bps', 2.0)
        slippage_bps  = getattr(bt_cfg, 'slippage_bps', 1.0)
        max_pos       = getattr(bt_cfg, 'max_concurrent_positions', 5)
        time_stop     = getattr(bt_cfg, 'time_stop_bars', 24)
        total_costs   = (fee_bps + slippage_bps) / 10_000.0

        for ts in timestamps:
            ts_data = df[df['timestamp'] == ts]

            # ── 1. Update existing positions ──────────────────────────────────
            still_open = []
            for pos in open_positions:
                row = ts_data[ts_data['symbol'] == pos['symbol']]
                if row.empty:
                    still_open.append(pos)
                    continue

                bar = row.iloc[0]
                pos['bars_held'] += 1

                curr_high  = float(bar.get('high', bar['close']))
                curr_low   = float(bar.get('low',  bar['close']))
                curr_close = float(bar['close'])

                if curr_high > pos['highest_seen']:
                    pos['highest_seen'] = curr_high

                exit_price = None
                reason     = None

                # Leadership decay exit (fast-exit before SL)
                if bar.get('leadership_decay', False) and pos['status'] == 'runner':
                    exit_price = curr_close
                    reason     = "LEADERSHIP_DECAY"

                # Stop loss
                elif curr_low <= pos['stop_price']:
                    exit_price = pos['stop_price']
                    reason     = "STOP_LOSS"

                # Final TP (runner)
                elif pos['status'] == 'runner' and curr_high >= pos['tp2_price']:
                    exit_price = pos['tp2_price']
                    reason     = "TP2"

                # Time stop
                elif pos['bars_held'] >= time_stop:
                    exit_price = curr_close
                    reason     = "TIME_EXIT"

                # ── Partial TP1 (open → runner) ───────────────────────────────
                elif pos['status'] == 'open' and curr_high >= pos['tp1_price']:
                    # Realise 50 % at TP1
                    pos['partial_taken'] = True
                    pos['status']        = 'runner'
                    # Move SL to breakeven for runner
                    pos['stop_price']    = pos['entry_price']
                    still_open.append(pos)
                    continue

                # ── Trailing stop for runner ──────────────────────────────────
                else:
                    if pos['status'] == 'runner':
                        reward_dist      = pos['tp2_price'] - pos['entry_price']
                        activation_price = pos['entry_price'] + (reward_dist * 0.30)
                        if pos['highest_seen'] > activation_price:
                            trail_stop = pos['highest_seen'] - (reward_dist * 0.20)
                            if trail_stop > pos['stop_price']:
                                pos['stop_price'] = trail_stop
                    still_open.append(pos)
                    continue

                # ── Close position & log ──────────────────────────────────────
                r_dist   = pos['r_dist']
                cost_r   = (total_costs * pos['entry_price']) / r_dist if r_dist > 0 else 0
                pnl_r    = ((exit_price - pos['entry_price']) / r_dist) - cost_r if r_dist > 0 else 0
                mfe_r    = (pos['highest_seen'] - pos['entry_price']) / r_dist if r_dist > 0 else 0
                gvb_r    = (pos['highest_seen'] - exit_price) / r_dist if r_dist > 0 else 0

                self.trade_log.append({
                    "symbol":         pos['symbol'],
                    "family":         pos.get('family', ''),
                    "entry_time":     pos['entry_time'],
                    "exit_time":      ts,
                    "exit_reason":    reason,
                    "outcome":        reason,      # backward-compat alias
                    "partial_taken":  pos['partial_taken'],
                    "pnl_r":          round(pnl_r, 4),
                    "bars_held":      pos['bars_held'],
                    "peak_price_seen": pos['highest_seen'],
                    "peak_pnl_r":     round(mfe_r, 4),
                    "mfe_r":          round(mfe_r, 4),
                    "giveback_r":     round(max(gvb_r, 0), 4),
                })

            open_positions = still_open

            # ── 2. New entries ────────────────────────────────────────────────
            candidates = ts_data[ts_data.get('entry_signal', pd.Series(False, index=ts_data.index)) == True] \
                if 'entry_signal' in ts_data.columns \
                else ts_data.iloc[0:0]

            if not candidates.empty and len(open_positions) < max_pos:
                rooms      = max_pos - len(open_positions)
                score_col  = 'ml_rank_score' if 'ml_rank_score' in candidates.columns else \
                             'composite_leader_score' if 'composite_leader_score' in candidates.columns else \
                             'total_score'
                top_cands  = candidates.nlargest(rooms, score_col)

                for _, c in top_cands.iterrows():
                    if any(p['symbol'] == c['symbol'] for p in open_positions):
                        continue

                    entry_p  = float(c['close'])
                    family   = str(c.get('family', 'defi_beta'))
                    regime   = str(c.get('historical_regime', 'trend'))
                    fakeout  = float(c.get('fakeout_risk', 0.0))

                    # Adaptive stop/TP
                    if c.get('stop_price', 0.0) and float(c.get('stop_price', 0.0)) > 0:
                        stop_p  = float(c['stop_price'])
                        tp1_p   = float(c.get('tp1_price', entry_p * 1.05))
                        tp2_p   = float(c.get('tp2_price', entry_p * 1.10))
                    else:
                        stop_result = self._adaptive_stop.compute_stop(
                            c, family=family, regime=regime, fakeout_risk=fakeout
                        )
                        stop_p = stop_result['stop_price']
                        tp1_p  = stop_result['tp1_price']
                        tp2_p  = stop_result['tp2_price']

                    r_dist = abs(entry_p - stop_p)
                    if r_dist <= 0:
                        continue

                    open_positions.append({
                        "symbol":        c['symbol'],
                        "family":        family,
                        "entry_time":    ts,
                        "entry_price":   entry_p,
                        "stop_price":    stop_p,
                        "tp1_price":     tp1_p,
                        "tp2_price":     tp2_p,
                        "r_dist":        r_dist,
                        "bars_held":     0,
                        "highest_seen":  entry_p,
                        "status":        "open",   # open | runner
                        "partial_taken": False,
                    })

        # ── Metrics compilation ───────────────────────────────────────────────
        if not self.trade_log:
            return {"win_rate": 0, "profit_factor": 0, "expectancy_r": 0,
                    "total_trades": 0, "avg_bars_held": 0}

        t_df     = pd.DataFrame(self.trade_log)
        wins     = t_df[t_df['pnl_r'] > 0]
        losses   = t_df[t_df['pnl_r'] < 0]
        win_rate = len(wins) / len(t_df) if len(t_df) > 0 else 0
        expectancy = t_df['pnl_r'].mean()

        pos_pnl  = wins['pnl_r'].sum()
        neg_pnl  = abs(losses['pnl_r'].sum())
        pf       = pos_pnl / neg_pnl if neg_pnl > 0 else (99.0 if pos_pnl > 0 else 0)

        # Hold / exit efficiency
        mfe_col  = 'mfe_r' if 'mfe_r' in t_df.columns else 'peak_pnl_r'
        hold_eff = 0.0
        if not wins.empty and mfe_col in wins.columns:
            peaks = wins[mfe_col]
            realized = wins['pnl_r']
            valid = peaks[peaks > 0]
            if not valid.empty:
                hold_eff = round((realized[valid.index].mean() / valid.mean()) * 100, 1)

        return {
            "win_rate":      round(win_rate * 100, 2),
            "expectancy_r":  round(expectancy, 3),
            "profit_factor": round(pf, 2),
            "total_trades":  len(t_df),
            "avg_bars_held": round(t_df['bars_held'].mean(), 1),
            "partial_tp_count": int(t_df['partial_taken'].sum()),
            "hold_efficiency":  hold_eff,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # RANKING QUALITY
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_ranking_quality(
        self, df: pd.DataFrame, bars_horizon: int = 24, k: int = 3
    ) -> dict:
        """
        High-fidelity leader prediction metrics for family-relative engine.
        Calculates precision, leader capture rate, and rank correlation.
        """
        if df.empty or 'ml_rank_score' not in df.columns or 'close' not in df.columns:
            return {"error": "Missing ml_rank_score or close for ranking evaluation"}

        df = df.copy().sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        df['future_return'] = (
            df.groupby('symbol')['close'].shift(-bars_horizon) / df['close'] - 1
        )
        df = df.dropna(subset=['future_return'])

        if 'family' in df.columns:
            df['future_family_rank'] = df.groupby(['timestamp', 'family'])['future_return'].rank(
                ascending=False, method='min'
            )

        correlations, precisions, capture_rates, family_ranks = [], [], [], []
        missed_count = 0

        for ts in sorted(df['timestamp'].unique()):
            ts_data = df[df['timestamp'] == ts]
            if len(ts_data) < k:
                continue

            corr = ts_data['ml_rank_score'].corr(ts_data['future_return'], method='spearman')
            if not np.isnan(corr):
                correlations.append(corr)

            top_k_pred   = set(ts_data.nlargest(k, 'ml_rank_score')['symbol'])
            top_k_actual = set(ts_data.nlargest(k, 'future_return')['symbol'])
            precisions.append(len(top_k_pred & top_k_actual) / k)

            best_actual = ts_data.nlargest(1, 'future_return')['symbol'].iloc[0]
            if best_actual in top_k_pred:
                capture_rates.append(1.0)
            else:
                capture_rates.append(0.0)
                missed_count += 1

            if 'future_family_rank' in ts_data.columns:
                sel_ranks = ts_data[ts_data['symbol'].isin(top_k_pred)]['future_family_rank']
                if not sel_ranks.empty:
                    family_ranks.append(sel_ranks.mean())

        # Hold / exit efficiency from trade_log
        t_df = pd.DataFrame(self.trade_log)
        hold_eff, exit_eff = 0, 0
        if not t_df.empty and 'mfe_r' in t_df.columns:
            rcv = t_df[t_df['pnl_r'] > 0]
            if not rcv.empty:
                peak  = rcv['mfe_r'].mean()
                real  = rcv['pnl_r'].mean()
                if peak and peak > 0:
                    hold_eff = round((real / peak) * 100, 1)
                cap_50 = len(rcv[rcv['pnl_r'] >= rcv['mfe_r'] * 0.5])
                exit_eff = round((cap_50 / len(rcv)) * 100, 1)

        return {
            "rank_correlation":                round(np.nanmean(correlations), 3) if correlations else 0,
            "top_k_precision":                 round(np.nanmean(precisions) * 100, 1) if precisions else 0,
            "top_k_recall":                    round(np.nanmean(precisions) * 100, 1) if precisions else 0,
            "leader_capture_rate":             round(np.nanmean(capture_rates) * 100, 1) if capture_rates else 0,
            "avg_selected_future_family_rank": round(np.nanmean(family_ranks), 2) if family_ranks else 0,
            "missed_leaders":                  missed_count,
            "hold_efficiency":                 hold_eff,
            "exit_efficiency":                 exit_eff,
            "n_timestamps":                    len(correlations),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # LEGACY — kept for structural compatibility, not called by CLI
    # ─────────────────────────────────────────────────────────────────────────
    def walk_forward_simulate(
        self,
        dataframe: pd.DataFrame,
        train_size: int = 500,
        test_size: int = 200,
        step_size: int = 100,
        train_ml: bool = False,
    ) -> list:
        """[LEGACY — not called by CLI or app. Use run_backtest.py instead.]"""
        return []
