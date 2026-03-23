import pandas as pd
import numpy as np
from defihunter.utils.timeframe import TimeframeHelper

def add_future_returns(df: pd.DataFrame, timeframe: str, horizons=['6h', '12h', '24h']):
    """
    Computes forward-looking returns for ranking labels.
    """
    df = df.copy()
    for h_str in horizons:
        h_bars = TimeframeHelper.get_bars(h_str, timeframe)
        ret_col = f'future_{h_str}_return'
        # Shift close prices backwards to get 'close at T + h_bars'
        df[ret_col] = df.groupby('symbol')['close'].shift(-h_bars) / df['close'] - 1
    return df

def add_family_rank_targets(df: pd.DataFrame):
    """
    Ranks coins within their respective families based on future returns.
    """
    if 'family' not in df.columns:
        return df
        
    df = df.copy()
    horizons = ['6h', '12h', '24h']
    
    for h_str in horizons:
        ret_col = f'future_{h_str}_return'
        if ret_col not in df.columns:
            continue
            
        rank_col = f'future_{h_str}_rank_in_family'
        # Rank within family (1 = best gainer in future)
        df[rank_col] = df.groupby(['timestamp', 'family'])[ret_col].rank(ascending=False, method='min')
        
        # Percentile rank within family (0-1, 1 = best)
        df[f'{rank_col}_pct'] = df.groupby(['timestamp', 'family'])[ret_col].rank(pct=True)
    
    # is_top3_family_next_24h
    if 'future_24h_rank_in_family' in df.columns:
        df['is_top3_family_next_24h'] = (df['future_24h_rank_in_family'] <= 3).astype(int)
        
    # is_top_decile_family_next_12h
    if 'future_12h_rank_in_family_pct' in df.columns:
        df['is_top_decile_family_next_12h'] = (df['future_12h_rank_in_family_pct'] >= 0.9).astype(int)
        
    return df

def add_topk_targets(df: pd.DataFrame):
    """
    Legacy placeholder - now merged into add_family_rank_targets for efficiency.
    """
    return df
