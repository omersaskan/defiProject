import pandas as pd
import numpy as np
from typing import Optional, Dict
from defihunter.utils.timeframe import TimeframeHelper

def compute_log_spread(coin_close: pd.Series, coin_ema: pd.Series, anchor_close: pd.Series, anchor_ema: pd.Series) -> pd.Series:
    """
    Computes log-normalized relative EMA spread between coin and anchor.
    coin_dev   = log(close / ema_n)
    anchor_dev = log(anchor_close / anchor_ema_n)
    rel_spread = coin_dev - anchor_dev
    """
    coin_dev = np.log(coin_close / coin_ema)
    anchor_dev = np.log(anchor_close / anchor_ema)
    return coin_dev - anchor_dev

class LeadershipEngine:
    def __init__(self, anchors: list[str], ema_lengths: list[int]):
        self.anchors = anchors
        self.ema_lengths = ema_lengths
        
    def compute_rs_divergence(self, df: pd.DataFrame, btc_df: Optional[pd.DataFrame], timeframe: str = '15m') -> pd.DataFrame:
        """
        GT #2: Relative Strength Divergence.
        Now timeframe-aware.
        """
        df = df.copy()
        
        if btc_df is None or btc_df.empty or 'close' not in btc_df.columns:
            df['rs_divergence_bull'] = False
            df['rs_divergence_persistence'] = 0
            return df
        
        # Align lengths
        adf = btc_df.copy().reset_index(drop=True)
        if len(adf) > len(df):
            adf = adf.tail(len(df)).reset_index(drop=True)
        elif len(adf) < len(df):
            adf = adf.reindex(range(len(df))).ffill().bfill()
        
        # 1h return comparison (timeframe-aware)
        h1_bars = TimeframeHelper.get_bars('1h', timeframe)
        btc_return_h1 = adf['close'].pct_change(h1_bars)
        coin_return_h1 = df['close'].pct_change(h1_bars)
        
        # BTC negative but coin outperforming by at least 1%
        df['rs_divergence_bull'] = (btc_return_h1 < -0.005) & (coin_return_h1 > btc_return_h1 + 0.01)
        
        # Persistence
        is_div = df['rs_divergence_bull'].astype(int)
        df['rs_divergence_persistence'] = is_div.groupby((is_div != is_div.shift()).cumsum()).cumsum()
        
        # Extended divergence: BTC below its medium EMA while coin above its medium EMA
        medium_ema = 55 # can be dynamic
        if f'ema_{medium_ema}' in df.columns and f'ema_{medium_ema}' in adf.columns:
            btc_weak = adf['close'] < adf[f'ema_{medium_ema}']
            coin_strong = df['close'] > df[f'ema_{medium_ema}']
            df['rs_strong_divergence'] = btc_weak & coin_strong
        else:
            df['rs_strong_divergence'] = False
        
        return df

    def compute_leadership_decay(self, df: pd.DataFrame, anchor_name: str = 'btc', ema_length: int = 55) -> pd.DataFrame:
        """
        GT-REDESIGN: Leadership Decay & Climax Detection.
        Optimized for exiting leaders before a full crash.
        """
        spread_col = f'rel_spread_{anchor_name}_ema{ema_length}'
        if spread_col not in df.columns:
            df['leadership_decay'] = False
            return df
            
        # 1. Spread peak tracking (Rolling 24h peak)
        spread_peak = df[spread_col].rolling(window=96, min_periods=1).max()
        
        # 2. Basic Decay Conditions
        fall_from_peak = (spread_peak - df[spread_col]) / (spread_peak.abs() + 1e-8)
        slope_neg = df[f'{spread_col}_slope_4'] < 0
        z_low = df[f'{spread_col}_z_96'] < 0.5
        
        # 3. Climax / Exhaustion Signals (GT-REDESIGN)
        # Type A: Volume Climax (Extreme volume + weakening spread slope)
        vol_extreme = df['volume_zscore'] > 3.0 if 'volume_zscore' in df.columns else False
        slope_weakening = df[f'{spread_col}_acceleration'] < 0
        climax_type_a = vol_extreme & slope_weakening
        
        # Type B: Exhaustion Wick (Long upper wick at local peak)
        high_dist = df['high'] - df[['open', 'close']].max(axis=1)
        body_size = (df['close'] - df['open']).abs()
        wick_exhaustion = (high_dist > body_size * 2.5) & (df[spread_col] > spread_peak * 0.9)
        
        # Total Decay Flag
        df['leadership_decay'] = (fall_from_peak > 0.15) & (slope_neg | z_low | climax_type_a | wick_exhaustion)
        
        # Store sub-signals for explanation
        df['decay_fall_pct'] = fall_from_peak * 100
        df['decay_is_climax'] = climax_type_a | wick_exhaustion
        
        return df
        
    def add_leadership_features(self, df: pd.DataFrame, anchor_data: Dict[str, pd.DataFrame], timeframe: str = '15m') -> pd.DataFrame:
        """
        Adds rel_spread_* features and their derivatives.
        GT-REDESIGN: Timeframe-aware derivatives.
        """
        df = df.copy()
        
        for anchor in self.anchors:
            if anchor not in anchor_data:
                continue
                
            adf = anchor_data[anchor].copy()
            
            # Align anchor index
            if 'timestamp' in df.columns and 'timestamp' in adf.columns:
                coin_ts = df['timestamp'].reset_index(drop=True)
                adf = adf.sort_values('timestamp').reset_index(drop=True)
                if len(adf) != len(df):
                    if len(adf) > len(df):
                        adf = adf.tail(len(df)).reset_index(drop=True)
                    else:
                        adf = adf.reindex(range(len(df))).ffill().bfill()
            else:
                adf = adf.reset_index(drop=True)
                df  = df.reset_index(drop=True)
            
            for length in self.ema_lengths:
                if f'ema_{length}' not in df.columns or f'ema_{length}' not in adf.columns:
                    continue
                    
                coin_ema = df[f'ema_{length}'].reset_index(drop=True)
                anchor_ema = adf[f'ema_{length}'].reset_index(drop=True)
                coin_close = df['close'].reset_index(drop=True)
                anchor_close = adf['close'].reset_index(drop=True)
                
                anchor_name = anchor.split('.')[0].lower()
                spread_col = f'rel_spread_{anchor_name}_ema{length}'
                
                # Base Spread
                df[spread_col] = compute_log_spread(coin_close, coin_ema, anchor_close, anchor_ema).values
                
                # 1. Slope (timeframe-aware)
                h1_bars = TimeframeHelper.get_bars('1h', timeframe)
                h4_bars = TimeframeHelper.get_bars('4h', timeframe)
                df[f'{spread_col}_slope_4'] = df[spread_col].diff(h1_bars) / h1_bars
                df[f'{spread_col}_slope_12'] = df[spread_col].diff(h4_bars) / h4_bars
                
                # 2. Persistence
                is_positive = (df[spread_col] > 0).astype(int)
                df[f'{spread_col}_persistence'] = is_positive.groupby((is_positive != is_positive.shift()).cumsum()).cumsum()
                
                # 3. Acceleration
                df[f'{spread_col}_acceleration'] = df[f'{spread_col}_slope_4'].diff(h1_bars)
                
                # 4. Z-Score (24h lookback)
                z_bars = TimeframeHelper.get_bars('24h', timeframe)
                rolling_mean = df[spread_col].rolling(window=z_bars, min_periods=min(10, z_bars)).mean()
                rolling_std = df[spread_col].rolling(window=z_bars, min_periods=min(10, z_bars)).std() + 1e-8
                df[f'{spread_col}_z_96'] = (df[spread_col] - rolling_mean) / rolling_std
                
                # Leadership Decay check
                df = self.compute_leadership_decay(df, anchor_name, length)
                
        # GT #2: Compute RS Divergence against BTC anchor
        btc_anchor_key = next((k for k in anchor_data if k.upper().startswith('BTC')), None)
        btc_anchor_df = anchor_data.get(btc_anchor_key) if btc_anchor_key else None
        df = self.compute_rs_divergence(df, btc_anchor_df, timeframe)
        
        return df
