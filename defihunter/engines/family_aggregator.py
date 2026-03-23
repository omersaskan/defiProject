import pandas as pd
import numpy as np
from typing import Dict, List, Any
from defihunter.utils.timeframe import TimeframeHelper

class FamilyAggregator:
    """
    Computes sector-wide aggregates (Heat, Momentum, Rotation) 
    for DeFi Families.
    """
    def __init__(self, families: Dict[str, Any]):
        self.families = families
        self.aggregates = {}

    def compute_family_stats(self, all_dfs: Dict[str, pd.DataFrame], timeframe: str = '15m') -> Dict[str, pd.DataFrame]:
        """
        Calculates aggregate stats for each family and returns a map of {family_label: aggregate_df}.
        """
        family_stats = {}
        
        # Invert family mapping for lookup
        symbol_to_family = {}
        for f_label, f_config in self.families.items():
            for member in f_config.members:
                symbol_to_family[member] = f_label

        # Group dataframes by family
        family_groups = {}
        for sym, df in all_dfs.items():
            f_label = symbol_to_family.get(sym, 'other')
            if f_label not in family_groups:
                family_groups[f_label] = []
            
            df_sync = df.copy()
            if 'timestamp' in df_sync.columns:
                df_sync = df_sync.set_index('timestamp')
            family_groups[f_label].append(df_sync)

        # Compute aggregates per family
        for f_label, dfs in family_groups.items():
            if not dfs: continue
            
            h1_bars = TimeframeHelper.get_bars('1h', timeframe)
            h4_bars = TimeframeHelper.get_bars('4h', timeframe)
            
            returns_1h = pd.concat([df['close'].pct_change(h1_bars) for df in dfs], axis=1)
            vol_z     = pd.concat([df['volume_zscore'] if 'volume_zscore' in df.columns else pd.Series(0, index=df.index) for df in dfs], axis=1)
            
            agg_df = pd.DataFrame(index=returns_1h.index)
            agg_df['avg_return_1h'] = returns_1h.mean(axis=1)
            agg_df['avg_vol_z']    = vol_z.mean(axis=1)
            
            # GT-REDESIGN: Family Breadth (How many coins in family are positive?)
            agg_df['family_breadth'] = (returns_1h > 0).astype(int).sum(axis=1) / len(dfs)
            
            # Family Heat: Momentum score (0-100)
            heat = (agg_df['avg_return_1h'].clip(0, 0.05) / 0.05 * 40) + \
                   (agg_df['avg_vol_z'].clip(0, 3) / 3 * 30) + \
                   (agg_df['family_breadth'] * 30)
            agg_df['family_heat'] = heat.fillna(0)
            
            # GT-REDESIGN: Leadership Acceleration (Is the family heat accelerating?)
            agg_df['family_heat_accel'] = agg_df['family_heat'].diff(h1_bars)
            
            family_stats[f_label] = agg_df

        self.aggregates = family_stats
        return family_stats

    def inject_family_features(self, symbol: str, df: pd.DataFrame, family_stats: Dict[str, pd.DataFrame], timeframe: str = '15m') -> pd.DataFrame:
        """
        Injects family-wide features into a specific coin's DataFrame.
        """
        df = df.copy()
        symbol_to_family = {}
        for f_label, f_config in self.families.items():
            for member in f_config.members:
                symbol_to_family[member] = f_label
        
        f_label = symbol_to_family.get(symbol)
        if not f_label or f_label not in family_stats:
            df['family_heat'] = 0.0
            df['family_heat_accel'] = 0.0
            df['family_breadth'] = 0.0
            df['family_rel_return_1h'] = 0.0
            df['is_family_leader'] = False
            return df
            
        f_df = family_stats[f_label]
        df_idx = df.set_index('timestamp') if 'timestamp' in df.columns else df
        
        # Safe union of indices to prevent NaNs on mismatch
        common_idx = df_idx.index.intersection(f_df.index)
        
        df_idx.loc[common_idx, 'family_heat'] = f_df.loc[common_idx, 'family_heat']
        df_idx.loc[common_idx, 'family_heat_accel'] = f_df.loc[common_idx, 'family_heat_accel']
        df_idx.loc[common_idx, 'family_breadth'] = f_df.loc[common_idx, 'family_breadth']
        df_idx.loc[common_idx, 'family_avg_return_1h'] = f_df.loc[common_idx, 'avg_return_1h']
        
        h1_bars = TimeframeHelper.get_bars('1h', timeframe)
        df_idx['coin_return_1h'] = df_idx['close'].pct_change(h1_bars)
        df_idx['family_rel_return_1h'] = df_idx['coin_return_1h'] - df_idx['family_avg_return_1h']
        
        # GT-REDESIGN: Leadership detection
        # Coin is beating family average + family is hot
        df_idx['is_family_leader'] = (df_idx['family_rel_return_1h'] > 0.01) & (df_idx['family_heat'] > 50)
        
        # --- NEW CONTEXT FEATURES ---
        # 1. leader_vs_family_spread (Alias for clarity in decision engines)
        df_idx['leader_vs_family_spread'] = df_idx['family_rel_return_1h']
        
        # 2. family_leader_consistency_score (Rolling sum of leadership to detect true persistent leaders)
        leader_int = df_idx['is_family_leader'].astype(int)
        df_idx['family_leader_consistency_score'] = leader_int.rolling(12).sum() / 12.0
        
        # 3. peer_relative_decay (Is the spread shrinking or dropping?)
        df_idx['peer_relative_decay'] = (df_idx['family_rel_return_1h'].diff(2) < -0.01)
        
        # 4. leadership_turn_down (Was a leader recently, but lost the spread)
        was_leader = leader_int.rolling(4).max().shift(1) == 1
        df_idx['leadership_turn_down'] = was_leader & ~df_idx['is_family_leader']
        
        # 5. family_cooling_flag (Family heat is dropping fast)
        df_idx['family_cooling_flag'] = (df_idx['family_heat_accel'] < -5.0) & (df_idx['family_heat'] < 40)
        
        if 'timestamp' in df.columns:
            return df_idx.reset_index()
        return df_idx
        
        if 'timestamp' in df.columns:
            return df_idx.reset_index()
        return df_idx
