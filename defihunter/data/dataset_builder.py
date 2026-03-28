import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from defihunter.engines.family_aggregator import FamilyAggregator
from defihunter.core.config import AppConfig
from defihunter.utils.timeframe import TimeframeHelper
from defihunter.utils.logger import logger


class DatasetBuilder:
    """
    Bridge between signal generation and ML training.
    GT #7: Generates pre-pump oriented training targets for leading-signal ML.
    Generates leakage-safe training targets and formatted feature sets.
    BUG-2 FIX: generate_labels rewritten with vectorized numpy ops instead of O(n²) Python loop.
    GT-GOLD-2: Added target_early_big — 1h/1.5% early signal + 24h/10% big move composite target.
    """

    def __init__(self, config: Optional[AppConfig] = None, target_r: float = 1.5, stop_r: float = -1.0, 
                 prepump_gain_pct: float = 0.05, timeframe: str = '15m'):
        self.config = config
        self.target_r = target_r
        self.stop_r = stop_r
        self.timeframe = timeframe
        
        # FIX: Enforce time-aware horizon. 
        self.window = TimeframeHelper.get_bars('24h', timeframe)
            
        self.prepump_gain_pct = prepump_gain_pct
        self.family_aggregator = FamilyAggregator(config.families) if config else None

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates binary target: 1 if target_r is hit before stop_r within 'window' bars.
        Also computes MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion) in R-terms.
        GT #7: Adds 'prepump_target' — forward %5 gain in 24 bars (leading indicator target).
        GT-GOLD-2: Adds 'target_early_big' — 1h/1.5% AND 24h/10% composite gainer target.
        BUG-2 FIX: Vectorized using rolling().max/min + shift() — no more O(n²) Python loop.
        """
        if df.empty:
            return df

        df = df.copy()
        n = len(df)
        w = self.window

        entry_price = df['close'].values
        atr_vals = df.get('atr', pd.Series(entry_price * 0.02, index=df.index)).values
        atr_vals = np.where(atr_vals == 0, entry_price * 0.02, atr_vals)

        tp_price = entry_price + (atr_vals * 2.0)
        sl_price = entry_price - (atr_vals * 1.5)
        r_dist   = np.abs(entry_price - sl_price)
        r_dist   = np.where(r_dist == 0, 1e-8, r_dist)

        # ── Vectorized forward windows ─────────────────────────────────────────
        # For each row i we need max(high[i+1..i+w]) and min(low[i+1..i+w]).
        # Trick: shift(-w) on a rolling(w) max/min gives the FORWARD window value.
        high_arr = df['high'].values
        low_arr  = df['low'].values

        # future_max_24 = max high in next w bars (forward window via reverse rolling)
        future_max_w = (
            pd.Series(high_arr[::-1]).rolling(w, min_periods=1).max().values[::-1]
        )
        future_min_w = (
            pd.Series(low_arr[::-1]).rolling(w, min_periods=1).min().values[::-1]
        )
        # Exclude the current bar — shift by 1 forward (last w rows lose future data)
        future_max_w = np.roll(future_max_w, -1)
        future_min_w = np.roll(future_min_w, -1)

        # MFE / MAE in R
        df['mfe_r'] = np.clip(
            (future_max_w - entry_price) / r_dist, -20, 20
        )
        df['mae_r'] = (future_min_w - entry_price) / r_dist

        # Mask last 'w' rows (no full future window) → set to 0
        df.iloc[-w:, df.columns.get_loc('mfe_r')] = 0.0
        df.iloc[-w:, df.columns.get_loc('mae_r')] = 0.0

        # ── target_hit: did high reach TP before low hit SL? ──────────────────
        # Fast vectorized: if future_max >= tp AND future_min stays > sl → hit TP
        tp_hit = future_max_w >= tp_price
        sl_hit = future_min_w <= sl_price
        df['target_hit']       = (tp_hit & ~sl_hit).astype(int)
        df['short_target_hit'] = (~tp_hit & sl_hit).astype(int)
        df['exit_type']        = np.where(tp_hit & ~sl_hit, "tp",
                                  np.where(~tp_hit & sl_hit, "sl", "time_exit"))
        df.iloc[-w:, df.columns.get_loc('target_hit')]       = 0
        df.iloc[-w:, df.columns.get_loc('short_target_hit')] = 0
        df.iloc[-w:, df.columns.get_loc('exit_type')]        = "none"

        # ── GT #7: prepump_target ──────────────────────────────────────────────
        gain_pct = (future_max_w / entry_price) - 1
        df['prepump_target'] = (gain_pct >= self.prepump_gain_pct).astype(int)
        df.iloc[-w:, df.columns.get_loc('prepump_target')] = 0

        # bars_to_target (approximate: proportional, not exact bar count)
        df['bars_to_target'] = 0

        # Calculate basic legacy columns (future returns moved to multi-df phase for cross-sectional prep)

        # ── GT #11: Multi-step targets ─────────────────────────────────────────
        for horizon_h, pct, col in [('4h', 0.02, 'target_4h_2pct'),
                                   ('12h', 0.05, 'target_12h_5pct'),
                                   ('24h', 0.10, 'target_24h_10pct')]:
            h_bars = TimeframeHelper.get_bars(horizon_h, self.timeframe)
            fut_h = pd.Series(high_arr[::-1]).rolling(h_bars, min_periods=1).max().values[::-1]
            fut_h = np.roll(fut_h, -1)
            df[col] = ((fut_h / entry_price) - 1 >= pct).astype(int)
            df.iloc[-h_bars:, df.columns.get_loc(col)] = 0

        # ── GT-GOLD-2: target_early_big ────────────────────────────────────────
        # 1h/1.5% early momentum + 24h/10% large move = "true daily top gainer"
        h1_bars = TimeframeHelper.get_bars('1h', self.timeframe)
        fut_1h = pd.Series(high_arr[::-1]).rolling(h1_bars, min_periods=1).max().values[::-1]
        fut_1h = np.roll(fut_1h, -1)
        h24_bars = TimeframeHelper.get_bars('24h', self.timeframe)
        fut_24h = pd.Series(high_arr[::-1]).rolling(h24_bars, min_periods=1).max().values[::-1]
        fut_24h = np.roll(fut_24h, -1)
        
        early_move  = (fut_1h  / entry_price - 1) >= 0.015   # 1.5% in 1h
        big_move    = (fut_24h / entry_price - 1) >= 0.10    # 10% in 24h
        df['target_early_big'] = (early_move & big_move).astype(int)
        df.iloc[-h24_bars:, df.columns.get_loc('target_early_big')] = 0

        # ── GT-GOLD-NEW: is_hyper_gainer (The Ultimate Top Gainer Target) ──────
        # > 15% move within the window (e.g. 24h)
        df['is_hyper_gainer'] = ((fut_24h / entry_price) - 1 >= 0.15).astype(int)
        df.iloc[-h24_bars:, df.columns.get_loc('is_hyper_gainer')] = 0

        return df

    def generate_cross_sectional_labels(self, multi_df: pd.DataFrame) -> pd.DataFrame:
        """
        GT-REDESIGN: Primary innovation for Leader Prediction.
        Calculates ranking labels based on FUTURE returns across symbols.
        Ensures the model predicts UNKNOWN future leaders, not known past ones.
        """
        if multi_df.empty or 'timestamp' not in multi_df.columns:
            return multi_df
            
        from defihunter.labels import leader_rank
        
        # 1. Compute FUTURE Returns per symbol
        df = leader_rank.add_future_returns(multi_df, self.timeframe)
        
        # 2. Family-Relative Ranking
        df = leader_rank.add_family_rank_targets(df)
        
        # 3. Top-K Targets (Specific Discovery Labels)
        df = leader_rank.add_topk_targets(df)
        
        # 4. Overall Rank (Across all DeFi for discovery) - Keep for compatibility
        if 'future_24h_return' in df.columns:
            df['future_24h_rank_overall_pct'] = df.groupby('timestamp')['future_24h_return'].rank(pct=True)
            df['is_top3_overall'] = (df.groupby('timestamp')['future_24h_return'].rank(ascending=False) <= 3).astype(int)
            
        # 5. Family-Relative Spread (Future Alpha)
        if 'family' in df.columns and 'future_24h_return' in df.columns:
            family_mean = df.groupby(['timestamp', 'family'])['future_24h_return'].transform('mean')
            df['family_relative_alpha_24h'] = df['future_24h_return'] - family_mean
            
        # Cleanup: Drop rows that have NaNs due to future shift (at the end of dataset)
        return df
            
        return df

    def apply_sector_features(self, multi_df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
        """
        Phase 3: Computes and injects Family Heat and Relative Performance 
        into a multi-symbol DataFrame.
        """
        if multi_df.empty or 'symbol' not in multi_df.columns:
            return multi_df

        # Split into individual symbol DFs for aggregator
        all_dfs = {sym: group for sym, group in multi_df.groupby('symbol')}
        
        # Compute aggregates
        family_stats = self.family_aggregator.compute_family_stats(all_dfs, timeframe=timeframe)
        
        # Inject back
        processed_list = []
        for sym, group in all_dfs.items():
            processed_list.append(self.family_aggregator.inject_family_features(sym, group, family_stats, timeframe=timeframe))
            
        combined = pd.concat(processed_list).sort_values('timestamp').reset_index(drop=True)
        
        # 2.5 Add cross-sectional relative features (peer_rank, etc)
        from defihunter.data.features import compute_family_features
        return compute_family_features(combined)

    def build(self, df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
        """
        Public alias for label generation and sector feature injection.
        """
        if df.empty: return df
        df = self.generate_labels(df)
        if 'symbol' in df.columns and len(df['symbol'].unique()) > 1:
            df = self.apply_sector_features(df, timeframe=timeframe)
            df = self.generate_cross_sectional_labels(df)
        return df

    def prepare_training_data(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                               use_prepump_target: bool = True):
        """
        Returns X, y_df for model training. y_df contains multiple targets.
        GT-GOLD-2: Prefers 'target_early_big' (composite gainer target) as primary.
        """
        df_labeled = self.generate_labels(df)

        if feature_cols is None:
            exclude = ['timestamp', 'symbol', 'target_hit', 'short_target_hit', 'mfe_r',
                       'mae_r', 'exit_type', 'prepump_target', 'bars_to_target',
                       'open_time', 'close', 'high', 'low', 'open', 'volume', 'quote_volume',
                       'target_4h_2pct', 'target_12h_5pct', 'target_24h_10pct', 'target_early_big',
                       'is_hyper_gainer', 'future_6h_return', 'future_12h_return', 'future_24h_return',
                       'future_6h_rank_in_family', 'future_12h_rank_in_family', 'future_24h_rank_in_family',
                       'future_6h_rank_in_family_pct', 'future_12h_rank_in_family_pct', 'future_24h_rank_in_family_pct',
                       'is_top3_family_next_24h', 'is_top_decile_family_next_12h',
                       'is_top3_overall', 'future_24h_rank_overall_pct', 'family_relative_alpha_24h']
            feature_cols = [c for c in df_labeled.columns
                           if pd.api.types.is_numeric_dtype(df_labeled[c]) and c not in exclude]

        # Drop rows that don't have a full window ahead
        df_final = df_labeled.iloc[:-self.window].dropna(subset=feature_cols)

        # Enforce relative ranking objectives instead of absolute pumping targets
        base_targets = ['target_hit', 'short_target_hit', 'mfe_r']
        optional_targets = ['is_top3_family_next_24h', 'is_top_decile_family_next_12h',
                            'future_24h_rank_in_family', 'family_relative_alpha_24h',
                            'target_early_big', 'is_hyper_gainer']

        available_targets = base_targets + [c for c in optional_targets if c in df_final.columns]
        y_targets = df_final[available_targets].copy()

        if use_prepump_target and 'is_top3_family_next_24h' in df_final.columns:
            logger.info("[DatasetBuilder] GT-RANKING: Using strictly is_top3_family_next_24h (Family Discovery)")
            y_targets['primary_clf'] = df_final['is_top3_family_next_24h']
        elif use_prepump_target and 'is_top_decile_family_next_12h' in df_final.columns:
            logger.info("[DatasetBuilder] GT-RANKING: Using is_top_decile_family_next_12h")
            y_targets['primary_clf'] = df_final['is_top_decile_family_next_12h']
        else:
            logger.warning("[DatasetBuilder] Family rank target not found, falling back to absolute hyper_gainer")
            if 'is_hyper_gainer' in df_final.columns:
                y_targets['primary_clf'] = df_final['is_hyper_gainer']
            else:
                y_targets['primary_clf'] = df_final.get('target_hit', 0)

        return df_final[feature_cols], y_targets
