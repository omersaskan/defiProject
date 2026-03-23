import pandas as pd
import numpy as np
from defihunter.utils.timeframe import TimeframeHelper

def compute_ohlcv_features(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Computes basic OHLCV structural features and anomalies.
    """
    df = df.copy()
    df['range'] = df['high'] - df['low'] + 1e-8
    df['body_ratio'] = abs(df['close'] - df['open']) / df['range']
    
    max_body = df[['open', 'close']].max(axis=1)
    min_body = df[['open', 'close']].min(axis=1)
    
    df['upper_wick_ratio'] = (df['high'] - max_body) / df['range']
    df['lower_wick_ratio'] = (min_body - df['low']) / df['range']
    
    df['close_to_high'] = (df['high'] - df['close']) / df['range']
    df['close_to_low'] = (df['close'] - df['low']) / df['range']
    
    # Anomaly tracking for universe filtering (Timeframe-aware rolling window)
    lookback = TimeframeHelper.get_bars('12h', timeframe)
    df['max_recent_wick'] = df[['upper_wick_ratio', 'lower_wick_ratio']].max(axis=1).rolling(lookback, min_periods=1).max()
    df['bar_count'] = np.arange(len(df)) + 1
    
    return df

def compute_atr_and_emas(df: pd.DataFrame, timeframe: str = '15m', atr_period=14, ema_periods=[20, 55, 100]) -> pd.DataFrame:
    """
    Computes Average True Range (ATR) and multiple EMAs.
    """
    df = df.copy()
    
    # True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR
    df['atr'] = tr.rolling(window=atr_period).mean()
    
    # EMAs
    for period in ema_periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
    # Volatility Velocity (Is volatility expanding or shrinking?)
    diff_bars = TimeframeHelper.get_bars('1h', timeframe)
    df['volatility_velocity'] = df['atr'].diff(diff_bars) / (df['atr'].shift(diff_bars) + 1e-8)
        
    return df

def compute_zscores(df: pd.DataFrame, timeframe: str = '15m', window_duration='12h') -> pd.DataFrame:
    """
    Computes rolling Z-scores for volume and quote_volume.
    """
    df = df.copy()
    window = TimeframeHelper.get_bars(window_duration, timeframe)
    
    for col in ['volume', 'quote_volume']:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=window).mean().shift(1)
            rolling_std = df[col].rolling(window=window).std().shift(1) + 1e-8
            df[f'{col}_zscore'] = (df[col] - rolling_mean) / rolling_std
            
    return df

def compute_returns(df: pd.DataFrame, timeframe: str = '15m', horizons=['1h', '4h', '12h', '24h']) -> pd.DataFrame:
    """
    Computes percentage returns over strict backward horizons.
    GT-REDESIGN: Now uses TimeframeHelper to ensure horizons match time, not bars.
    """
    df = df.copy()
    for h_str in horizons:
        h_bars = TimeframeHelper.get_bars(h_str, timeframe)
        df[f'return_{h_str}'] = (df['close'] / df['close'].shift(h_bars)) - 1
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    GT #10: Time-aware features — DeFi pumps cluster around Asia open and US open.
    Cyclical encoding prevents the model from treating hour 23 as far from hour 0.
    """
    df = df.copy()
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        # Cyclical encoding — sin/cos to capture periodicity
        df['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df['day_sin']  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df['day_cos']  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        # Historical DeFi pump windows (UTC)
        df['is_asia_session']    = ((ts.dt.hour >= 0) & (ts.dt.hour < 6)).astype(int)
        df['is_us_open_session'] = ((ts.dt.hour >= 13) & (ts.dt.hour < 17)).astype(int)
        df['is_weekend']         = (ts.dt.dayofweek >= 5).astype(int)
    else:
        for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                    'is_asia_session', 'is_us_open_session', 'is_weekend']:
            df[col] = 0.0
    return df

def compute_participation_features(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Computes advanced participation metrics like volume MAs, CVD, OI deltas, and funding features.
    GT #12: Adds taker_surge — sudden market buy order explosion (pre-pump footprint).
    """
    df = df.copy()
    
    if 'volume' in df.columns:
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
    if 'taker_buy_volume' in df.columns and 'taker_sell_volume' in df.columns:
        df['volume_delta'] = df['taker_buy_volume'] - df['taker_sell_volume']
        df['cvd'] = df['volume_delta'].cumsum()
        
        # Delta as percentage of total volume
        total_vol = df['taker_buy_volume'] + df['taker_sell_volume'] + 1e-8
        df['delta_pct'] = df['volume_delta'] / total_vol
        df['buying_pressure'] = df['delta_pct'].rolling(10).mean()
        
        roller_mean = df['volume_delta'].rolling(50).mean().shift(1)
        roller_std = df['volume_delta'].rolling(50).std().shift(1) + 1e-8
        df['cvd_zscore'] = (df['volume_delta'] - roller_mean) / roller_std
        df['cvd_slope_4'] = df['cvd'].diff(4) / 4
        
        # GT #12: Taker Buy Ratio — market buyers vs total (real urgency signal)
        df['taker_buy_ratio'] = df['taker_buy_volume'] / total_vol
        tbr_mean = df['taker_buy_ratio'].rolling(50).mean().shift(1)
        tbr_std  = df['taker_buy_ratio'].rolling(50).std().shift(1) + 1e-8
        df['taker_ratio_zscore'] = (df['taker_buy_ratio'] - tbr_mean) / tbr_std
        
        # GT #12: Taker Surge — 2σ above average AND ratio increasing 3 bars in a row
        # This fires 1-3 bars BEFORE price moves because market orders = real urgency
        df['taker_surge'] = (
            (df['taker_ratio_zscore'] > 2.0) &
            (df['taker_buy_ratio'] > df['taker_buy_ratio'].shift(1)) &
            (df['taker_buy_ratio'].shift(1) > df['taker_buy_ratio'].shift(2))
        )
        
        # Orderbook Vacuum: CVD accelerating while volatility shrinks
        if 'volatility_velocity' in df.columns:
            df['orderbook_vacuum'] = (
                (df['cvd_slope_4'] > 0) &
                (df['cvd_slope_4'] > df['cvd_slope_4'].shift(1)) &
                (df['volatility_velocity'] < -0.05)
            )
        else:
            df['orderbook_vacuum'] = False
            
    else:
        df['volume_delta'] = 0.0
        df['cvd'] = 0.0
        df['cvd_zscore'] = 0.0
        df['cvd_slope_4'] = 0.0
        df['buying_pressure'] = 0.0
        df['taker_buy_ratio'] = 0.5
        df['taker_ratio_zscore'] = 0.0
        df['taker_surge'] = False
        df['orderbook_vacuum'] = False
            
    if 'open_interest' in df.columns:
        # Avoid 0 / 0 == NaN by mapping to 0 if original is 0
        h1_bars = TimeframeHelper.get_bars('1h', timeframe)
        h4_bars = TimeframeHelper.get_bars('4h', timeframe)
        df['oi_delta_1'] = np.where(df['open_interest'].shift(h1_bars) != 0, 
                                    (df['open_interest'] / df['open_interest'].shift(h1_bars)) - 1, 
                                    0.0)
        df['oi_delta_4'] = np.where(df['open_interest'].shift(h4_bars) != 0, 
                                    (df['open_interest'] / df['open_interest'].shift(h4_bars)) - 1, 
                                    0.0)
        
        # GT-NEW-2: OI Divergence — price flat + OI rising = smart money entering
        z_bars = TimeframeHelper.get_bars('12h', timeframe)
        oi_mean = df['open_interest'].rolling(z_bars).mean().shift(1)
        oi_std  = df['open_interest'].rolling(z_bars).std().shift(1) + 1e-8
        df['oi_zscore'] = (df['open_interest'] - oi_mean) / oi_std
        
        # Price is flat (< 1% move in 4h) but OI is elevated → accumulation
        price_flat = df['close'].pct_change(h4_bars).abs() < 0.01
        oi_rising  = df['oi_zscore'] > 1.5
        df['oi_divergence_bull'] = price_flat & oi_rising
        
        # GT-NEW-2: Short squeeze setup — OI dropping + negative funding
        oi_dropping = df['oi_zscore'] < -1.0
        if 'funding_rate' in df.columns:
            neg_funding = df['funding_rate'] < -0.001
            df['short_squeeze_setup'] = (oi_dropping & neg_funding).astype(bool)
        else:
            df['short_squeeze_setup'] = oi_dropping.astype(bool)
    else:
        df['oi_zscore'] = 0.0
        df['oi_divergence_bull'] = False
        df['short_squeeze_setup'] = False
        
    if 'funding_rate' in df.columns:
        # Simple rolling average of funding to smooth spikes
        df['funding_rate_ma_10'] = df['funding_rate'].rolling(10).mean()
        # Funding Velocity: Is the crowd getting aggressively long/short?
        df['funding_velocity'] = df['funding_rate'].diff(3)
        
        # GT-NEW-5 prep: extreme negative funding flag (feeds into rules.py)
        df['funding_extreme_short'] = (df['funding_rate'] < -0.0005).astype(bool)
        
    # Volume Efficiency (Price movement relative to volume intensity)
    if 'quote_volume_zscore' in df.columns:
        # High value = Price moving a lot with relatively little volume (Easy movement)
        # Low value = Heavy volume but price not moving (Absorption)
        df['volume_efficiency'] = df['close'].pct_change(1).abs() / (df['quote_volume_zscore'].abs() + 0.1)
        
    return df

def compute_pre_pump_profile(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Golden Trick #1: Computes pre-pump accumulation fingerprints.
    Key insight: Before a pump, price moves sideways while volume gradually increases.
    This is 'smart money' entering without moving price.
    """
    df = df.copy()
    
    h1_bars = TimeframeHelper.get_bars('1h', timeframe)
    
    # 1h backward lookbacks (no lookahead)
    price_change_h1 = df['close'].pct_change(h1_bars).abs()
    volume_change_h1 = df['volume'].pct_change(h1_bars)
    
    # Price quiet but volume building → accumulation zone
    df['accumulation_zone'] = (price_change_h1 < 0.02) & (volume_change_h1 > 0.3)
    
    # Volume acceleration: volume increasing bar-over-bar
    df['volume_acceleration'] = df['volume'].diff(1) / (df['volume'].shift(1) + 1e-8)
    df['volume_accel_h1_sum'] = df['volume_acceleration'].rolling(h1_bars).sum()
    
    # "Quiet Expansion" — ATR shrinking while volume growing (strongest pre-pump signal)
    if 'atr' in df.columns:
        diff_bars = TimeframeHelper.get_bars('45m', timeframe) # ~3 bars on 15m
        atr_shrinking = df['atr'].diff(diff_bars).shift(1) < 0
        df['quiet_expansion'] = atr_shrinking & (df['volume_accel_h1_sum'].shift(1) > 0.5)
    else:
        df['quiet_expansion'] = False
    
    # Volume Spike now (for sweep confirmation - GT #4)
    if 'volume_zscore' in df.columns:
        df['volume_spike_now'] = df['volume_zscore'] > 2.0
    else:
        df['volume_spike_now'] = False
        
    return df

def compute_squeeze_features(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    GT #13: Bollinger Band Squeeze + Keltner Channel.
    When BB is inside KC, volatility is compressed (squeeze state).
    When squeeze RELEASES → strongest momentum breakout signal (~70% hit rate historically).
    """
    df = df.copy()
    
    # Bollinger Bands (5h period, 2σ)
    bb_bars = TimeframeHelper.get_bars('5h', timeframe)
    bb_mid = df['close'].rolling(bb_bars).mean()
    bb_std = df['close'].rolling(bb_bars).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-8)).shift(1)  # shift: no lookahead
    
    # Keltner Channel (5h period, 1.5x ATR)
    if 'atr' in df.columns:
        df['kc_upper'] = bb_mid + 1.5 * df['atr']
        df['kc_lower'] = bb_mid - 1.5 * df['atr']
        
        # Squeeze = BB inside KC (extreme volatility compression)
        df['in_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        
        # Squeeze Release = was in squeeze last bar, now not → momentum incoming!
        df['squeeze_release'] = (~df['in_squeeze']) & (df['in_squeeze'].shift(1).fillna(False))
        
        # Duration in squeeze (bars)
        in_sq_int = df['in_squeeze'].astype(int)
        df['squeeze_duration'] = in_sq_int.groupby(
            (in_sq_int != in_sq_int.shift()).cumsum()
        ).cumsum() * in_sq_int
        
        # BB width percentile rank (how tight relative to last 1d?)
        rank_bars = TimeframeHelper.get_bars('24h', timeframe)
        df['bb_width_pct'] = df['bb_width'].rolling(rank_bars, min_periods=min(20, rank_bars)).rank(pct=True)
    else:
        df['in_squeeze'] = False
        df['squeeze_release'] = False
        df['squeeze_duration'] = 0
        df['bb_width_pct'] = 0.5
    
    return df

def compute_breakout_features(df: pd.DataFrame, timeframe: str = '15m', lookback_h=5, atr_threshold=0.5) -> pd.DataFrame:
    """
    Computes breakout levels, distances, retest features, and sweep logic.
    BUG #1 FIX: All tightness/threshold calculations now use .shift(1) to prevent lookahead.
    GT #4 FIX: sweep_reclaim_confirmed now requires volume spike confirmation.
    """
    df = df.copy()
    lookback = TimeframeHelper.get_bars(f'{lookback_h}h', timeframe)
    
    # Define resistance and support over lookback (shift(1) ensures no lookahead)
    df['rolling_high'] = df['high'].rolling(window=lookback).max().shift(1)
    df['rolling_low'] = df['low'].rolling(window=lookback).min().shift(1)
    
    # Breakout condition
    df['is_breakout_bar'] = (df['close'] > df['rolling_high'])
    
    # Distance in ATRs
    if 'atr' in df.columns:
        df['breakout_distance_atr'] = (df['close'] - df['rolling_high']) / df['atr']
        
        # Advanced Retest Logic
        # Was there a breakout in the last 2.5h?
        recent_bars = TimeframeHelper.get_bars('2.5h', timeframe)
        df['recent_breakout'] = df['is_breakout_bar'].rolling(window=recent_bars).max().fillna(0)
        
        # BUG #7 FIX: retest_hold logic — price pulls back to prior breakout level
        # but holds above it (close > rolling_high - 0.2*atr instead of tautological > rolling_high)
        df['retest_touch'] = (
            (df['recent_breakout'] == 1) & 
            (df['low'] <= df['rolling_high'] + (df['atr'] * atr_threshold)) & 
            (df['low'] >= df['rolling_high'] - (df['atr'] * atr_threshold))
        )
        # BUG #7 FIX: genuinely test if price is holding the level (not falling back through)
        df['retest_hold'] = df['retest_touch'] & (df['close'] > df['rolling_high'] - df['atr'] * 0.2)
        
        # Reclaim logic (fell back below, now surging above)
        df['fakeout_down'] = (df['recent_breakout'] == 1) & (df['close'] < df['rolling_high'])
        df['reclaim_ok'] = df['fakeout_down'].shift(1).fillna(0).astype(bool) & (df['close'] > df['rolling_high'])
    else:
        df['retest_touch'] = False
        df['retest_hold'] = False
        df['reclaim_ok'] = False
        
    # Sweep logic (undercut low and close above it)
    df['sweep_low'] = (df['low'] < df['rolling_low']) & (df['close'] > df['rolling_low'])
    
    # GT #4 FIX: sweep_reclaim_confirmed now requires volume spike confirmation
    volume_spike = df.get('volume_spike_now', pd.Series(True, index=df.index))
    df['sweep_reclaim_confirmed'] = (
        df['sweep_low'] & 
        (df['close'] > df['open']) & 
        (df['close_to_high'] < 0.4) &
        volume_spike  # Volume must confirm the sweep reclaim
    )
    
    # Base expansion logic (tight consolidation before breakout)
    # BUG #1 FIX: tightness now uses shift(1) to prevent lookahead
    df['range_pct'] = df['range'] / df['close']
    df['tightness'] = df['range_pct'].rolling(20).std().shift(1)
    
    # BUG #1 FIX: tight_threshold uses shift(1) so the current bar doesn't influence the threshold
    tight_threshold = df['tightness'].rolling(100).quantile(0.3).shift(1)
    df['is_base_expansion'] = df['is_breakout_bar'] & (df['tightness'] < tight_threshold)
    
    # Silent Accumulation (Golden Trick #1 - base version)
    # Price tight (sideways), CVD Z-score high (bids absorbing sells), above trend
    # BUG #1 FIX: uses shifted tightness/threshold, no lookahead
    if 'cvd_zscore' in df.columns and 'ema_100' in df.columns:
        df['silent_accumulation'] = (
            (df['tightness'] < tight_threshold) & 
            (df['cvd_zscore'] > 1.5) & 
            (df['close'] > df['ema_100'])
        )
    elif 'quiet_expansion' in df.columns:
        # Fallback to GT #1 quiet_expansion if no CVD data
        df['silent_accumulation'] = df['quiet_expansion'] & (df['close'] > df.get('ema_100', df['close'] * 0.95))
    else:
        df['silent_accumulation'] = False
        
    return df

def compute_persistence_features(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Computes quality of trend and exhaustion risk.
    """
    df = df.copy()
    
    # 1. Trend Persistence: Bars above EMA
    if 'ema_55' in df.columns:
        above_ema = (df['close'] > df['ema_55']).astype(int)
        df['trend_persistence_score'] = above_ema.groupby((above_ema != above_ema.shift()).cumsum()).cumsum() * above_ema
    else:
        df['trend_persistence_score'] = 0

    # 2. Close to High Persistence (Quality of buying)
    if 'close_to_high' in df.columns:
        df['close_to_high_persistence'] = df['close_to_high'].rolling(10).mean()
    else:
        df['close_to_high_persistence'] = 0.5

    # 3. Exhaustion Risk
    # RSI > 70 + Large distance from EMA + many green bars
    if 'rsi_14' in df.columns and 'ema_55' in df.columns:
        ema_dist = (df['close'] - df['ema_55']) / df['ema_55']
        green_bars = (df['close'] > df['open']).astype(int)
        green_streak = green_bars.groupby((green_bars != green_bars.shift()).cumsum()).cumsum() * green_bars
        
        df['exhaustion_risk_score'] = (
            (df['rsi_14'] / 100.0 * 0.4) + 
            (np.clip(ema_dist * 10, 0, 1) * 0.4) + 
            (np.clip(green_streak / 10, 0, 1) * 0.2)
        ) * 100.0
    else:
        df['exhaustion_risk_score'] = 0.0

    # 4. Volume Persistence
    if 'volume_ma_20' in df.columns:
        high_vol = (df['volume'] > df['volume_ma_20']).astype(int)
        df['volume_persistence_score'] = high_vol.groupby((high_vol != high_vol.shift()).cumsum()).cumsum() * high_vol
    else:
        df['volume_persistence_score'] = 0

    return df

def compute_exit_decay_features(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Computes features specifically useful for detecting trend death and leadership decay.
    """
    df = df.copy()
    
    # 1. Volume Climax Flag (Massive volume spike at local top)
    if 'volume_zscore' in df.columns and 'close_to_high_persistence' in df.columns:
        df['volume_climax_flag'] = (df['volume_zscore'] > 3.0) & (df['close_to_high_persistence'] < 0.3)
    else:
        df['volume_climax_flag'] = False
        
    # 2. Wick Exhaustion Flag (Multiple long upper wicks in a row)
    if 'upper_wick_ratio' in df.columns:
        long_upper = (df['upper_wick_ratio'] > 0.5).astype(int)
        wick_streak = long_upper.groupby((long_upper != long_upper.shift()).cumsum()).cumsum() * long_upper
        df['wick_exhaustion_flag'] = (wick_streak >= 2)
    else:
        df['wick_exhaustion_flag'] = False
        
    return df

def compute_family_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-sectional features for family members.
    Expects a merged DataFrame with 'symbol' and 'family' columns.
    """
    if df_merged.empty or 'family' not in df_merged.columns:
        return df_merged
        
    df = df_merged.copy()
    
    # 1. Peer Momentum & Rank
    if 'return_24h' in df.columns:
        df['peer_momentum'] = df.groupby('family')['return_24h'].transform(lambda x: x - x.mean())
        df['peer_rank'] = df.groupby('family')['return_24h'].rank(pct=True)
    
    # 2. Family Heat Score (Avg return of top 3)
    def compute_heat(group):
        if len(group) < 1: return 0.0
        return group.nlargest(3).mean()
        
    if 'return_4h' in df.columns:
        heat_map = df.groupby('family')['return_4h'].apply(compute_heat).to_dict()
        df['family_heat_score'] = df['family'].map(heat_map)
        
    # 3. Family Breadth Score (% of members showing strength)
    if 'total_score' in df.columns:
        breadth_map = df.groupby('family')['total_score'].apply(lambda x: (x > 50).mean()).to_dict()
        df['family_breadth_score'] = df['family'].map(breadth_map)
        
    return df

def build_feature_pipeline(raw_data: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Main pipeline joining all feature engineers per timestamp per symbol.
    GT-REDESIGN: Propagates timeframe to all sub-components.
    """
    if raw_data.empty or 'timestamp' not in raw_data.columns:
        return raw_data

    df = raw_data.sort_values('timestamp').reset_index(drop=True)

    df = compute_ohlcv_features(df, timeframe)
    df = compute_atr_and_emas(df, timeframe)
    df = compute_zscores(df, timeframe)
    df = compute_returns(df, timeframe)
    df = compute_time_features(df)
    df = compute_participation_features(df, timeframe)
    df = compute_pre_pump_profile(df, timeframe)
    df = compute_squeeze_features(df, timeframe)
    df = compute_breakout_features(df, timeframe)
    
    # Phase 5 Persistence
    df = compute_persistence_features(df, timeframe)
    
    # GT-PRO Features
    df = compute_gt_pro_features(df, timeframe)
    
    # Exit Decay Features
    df = compute_exit_decay_features(df, timeframe)
    
    # Final Cleanup
    df.fillna(0, inplace=True)

    return df
def compute_market_structure_break(df: pd.DataFrame, timeframe: str = '15m', lookback_h=5) -> pd.DataFrame:
    """
    GT-PRO #3: Market Structure Break (MSB).
    Detects if price breaks above the high of the last 'window' bars with volume expansion.
    """
    df = df.copy()
    window = TimeframeHelper.get_bars(f'{lookback_h}h', timeframe)
    df['prev_high'] = df['high'].shift(1).rolling(window=window).max()
    df['prev_low'] = df['low'].shift(1).rolling(window=window).min()
    
    # MSB Bullish: Close > Previous High AND Volume > SMA Volume
    vol_sma = df['volume'].rolling(window=window).mean()
    df['msb_bull'] = (df['close'] > df['prev_high']) & (df['volume'] > vol_sma * 1.5)
    df['msb_bear'] = (df['close'] < df['prev_low']) & (df['volume'] > vol_sma * 1.5)
    
    return df

def compute_funding_capitulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    GT-PRO #3: Funding Rate Capitulation.
    Detects when funding is deeply negative but price starts recovering.
    """
    df = df.copy()
    if 'funding_rate' not in df.columns:
        df['funding_capitulation'] = False
        return df
        
    negative_funding = df['funding_rate'] < -0.0002 # -0.2% or lower (adjusted for 8h)
    price_reversal = (df['close'] > df['open']) & (df['close'] > df['close'].shift(1))
    
    df['funding_capitulation'] = negative_funding & price_reversal
    return df

def compute_cvd_acceleration(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    GT-PRO #4: Velocity of Delta (V-Delta).
    Acceleration of CVD compared to price.
    """
    df = df.copy()
    if 'cvd' not in df.columns:
        return df
        
    df['cvd_velocity'] = df['cvd'].diff(1)
    df['cvd_acceleration'] = df['cvd_velocity'].diff(1)
    
    # Normalize by volume for cross-coin comparison (5h MA)
    ma_bars = TimeframeHelper.get_bars('5h', timeframe)
    df['v_delta_score'] = df['cvd_acceleration'] / (df['volume'].rolling(ma_bars).mean() + 1e-8)
    return df

def compute_gt_pro_features(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Orchestrates all GT-PRO feature calculations.
    """
    df = compute_market_structure_break(df, timeframe)
    df = compute_funding_capitulation(df)
    df = compute_cvd_acceleration(df, timeframe)
    return df
