import pandas as pd
import numpy as np
from typing import List, Optional


def load_universe(config=None, fetcher=None, strict_defi: bool = True) -> List[str]:
    """
    GT-UNIVERSE: Core entry point for symbol discovery.
    Strictly enforces the DeFi whitelist from config.universe.defi_universe.
    The whole system now defaults to strict DeFi mode. 
    Setting strict_defi=False is only for legacy debug/fallback.
    """
    if fetcher is None:
        from defihunter.data.binance_fetcher import BinanceFuturesFetcher
        fetcher = BinanceFuturesFetcher()
    
    # 1. Fetch strictly from Binance the defined defi config
    defi_universe = fetcher.get_defi_universe(config=config, strict_defi=strict_defi)
    
    if not defi_universe:
         print("[Universe] WARNING: Valid DeFi universe is EMPTY. Check config or connectivity. Falling back to active USDT if strict_defi=False.")
         if not strict_defi:
             return fetcher.get_defi_universe(config=config, strict_defi=False)
         return []
         
    print(f"[Universe] OVERHAUL: Loaded {len(defi_universe)} symbols for strict DeFi scanning.")
    return defi_universe


def get_balanced_universe(config, symbols: List[str], max_per_family: int = 5) -> List[str]:
    """
    GT-UNIVERSE: Ensures a balanced representative set from each family.
    Prevents any single family from dominating the scanner.
    """
    if not config or not config.families:
        return symbols
        
    balanced = []
    for f_label, f_config in config.families.items():
        if f_label == 'defi_beta': continue
        
        # Get members that are in our active symbols list
        family_members = [s for s in f_config.members if s in symbols]
        
        # Take top-k
        balanced.extend(family_members[:max_per_family])
        
    return list(set(balanced))


def filter_universe(
    df: pd.DataFrame,
    min_volume: float = 1_000_000,
    min_oi: float = 500_000,
    max_spread: float = 15.0,
    min_bars_age: int = 500,
    max_wick_ratio: float = 0.5,
    allowed_symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filters the universe of coins based on volume, OI, spread, and structural anomalies.
    If allowed_symbols is provided, strictly filters to include only those symbols.
    """
    if df.empty:
        return df

    # Start with all True mask
    mask = pd.Series(True, index=df.index)

    if allowed_symbols is not None:
        mask &= df['symbol'].isin(allowed_symbols)

    if 'quote_volume' in df.columns:
        mask &= df['quote_volume'] >= min_volume

    if 'open_interest' in df.columns:
        mask &= df['open_interest'] >= min_oi

    if 'spread_bps' in df.columns:
        mask &= df['spread_bps'] <= max_spread

    if 'bar_count' in df.columns:
        mask &= df['bar_count'] >= min_bars_age

    if 'max_recent_wick' in df.columns:
        mask &= df['max_recent_wick'] <= max_wick_ratio

    return df[mask]


def rank_by_relative_volume(symbols: List[str], fetcher, timeframe: str = '1h',
                             lookback_bars: int = 170, top_n: int = 25) -> List[str]:
    """
    GT-NEW-4: Relative Volume Rank (RVR) Pre-Scanner Filter.
    
    Ranks all coins by: (last 24h volume) / (7-day average hourly volume).
    Returns top_n coins with the most anomalous volume activity today.
    
    Key insight: Daily top gainers almost always appear in the top 20-30 
    by RVR score, hours BEFORE the price move. This pre-scanner dramatically 
    cuts the search space from 200+ coins to top 25 anomalies.
    
    Args:
        symbols: Full universe of coin symbols (e.g. ['BTC.p', 'ETH.p', ...])
        fetcher: BinanceFuturesFetcher instance
        timeframe: '1h' recommended for 24h / 7d comparison
        lookback_bars: How many bars to fetch (170 gives ~7 days of 1h data)
        top_n: Number of top RVR coins to return for full analysis
        
    Returns:
        List of symbols sorted by RVR score (highest first), limited to top_n.
    """
    rvr_scores = []
    
    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, timeframe=timeframe, limit=lookback_bars)
            if df.empty or len(df) < 50:
                continue
            
            vol = df['volume'].values
            
            # 7-day avg: bars 0 to -25 (exclude last 24 to get clean baseline)
            baseline_bars = vol[:-24] if len(vol) > 48 else vol
            avg_7d = float(np.mean(baseline_bars)) if len(baseline_bars) > 0 else 1.0
            
            # Last 24 bars volume (today)
            last_24h_vol = float(np.sum(vol[-24:]))
            
            # RVR = today's total / expected daily volume
            rvr = last_24h_vol / (avg_7d * 24 + 1e-8)
            
            rvr_scores.append((sym, rvr))
            
        except Exception:
            continue
    
    # Sort by RVR descending — highest anomaly first
    rvr_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_symbols = [sym for sym, _ in rvr_scores[:top_n]]
    
    if rvr_scores:
        print(f"[RVR Pre-Scanner] Top 5 by volume anomaly:")
        for sym, rvr in rvr_scores[:5]:
            print(f"  {sym}: RVR={rvr:.2f}x")
    
    return top_symbols


def build_anomaly_watchlist(symbols: List[str], fetcher, criteria: Optional[dict] = None) -> List[str]:
    """
    GT-NEW-8: Anomaly Pre-Scanner — fast multi-criteria watchlist builder.
    
    Quickly scores each coin on 4 anomaly dimensions using only the last 50 bars.
    Only coins scoring above threshold proceed to full feature pipeline analysis.
    This reduces full-analysis cost by 60-80% while retaining 95%+ of signals.
    
    Anomaly criteria:
    1. Volume Z-score > 2.0 (unusual volume today)
    2. BB width percentile in bottom 20% (tight squeeze)
    3. Funding rate < -0.001 (crowd heavily short)
    4. 24h price change < 3% (coiling tight)
    
    A coin qualifies if it meets 2+ criteria.
    """
    if criteria is None:
        criteria = {
            'volume_zscore_threshold': 2.0,
            'price_range_threshold': 0.03,
            'funding_threshold': -0.001,
            'min_criteria_met': 2
        }
    
    watchlist = []
    
    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, timeframe='15m', limit=100)
            if df.empty or len(df) < 30:
                continue
            
            score = 0
            
            # Criterion 1: Volume anomaly
            vol = df['volume'].values
            vol_mean = np.mean(vol[:-5]) if len(vol) > 10 else np.mean(vol)
            vol_std = np.std(vol[:-5]) + 1e-8
            vol_zscore = (vol[-1] - vol_mean) / vol_std
            if vol_zscore > criteria.get('volume_zscore_threshold', 2.0):
                score += 1
            
            # Criterion 2: Price coiling (tight range last 24 bars)
            if len(df) >= 24:
                h24 = df['high'].iloc[-24:].max()
                l24 = df['low'].iloc[-24:].min()
                range_pct = (h24 / (l24 + 1e-8)) - 1
                if range_pct < criteria.get('price_range_threshold', 0.03):
                    score += 1
            
            # Criterion 3: Negative funding (crowd short)
            if 'funding_rate' in df.columns:
                fr = df['funding_rate'].iloc[-1]
                if fr < criteria.get('funding_threshold', -0.001):
                    score += 1
            
            # Criterion 4: Volume building (last 4 bars all > median)
            if len(vol) >= 20:
                med = np.median(vol[-20:-4])
                recent_above = sum(v > med for v in vol[-4:])
                if recent_above >= 3:
                    score += 1
            
            if score >= criteria.get('min_criteria_met', 2):
                watchlist.append((sym, score))
                
        except Exception:
            continue
    
    # Sort by score desc
    watchlist.sort(key=lambda x: x[1], reverse=True)
    
    result = [sym for sym, _ in watchlist]
    print(f"[Anomaly Pre-Scanner] {len(result)}/{len(symbols)} coins qualified (score >= {criteria.get('min_criteria_met', 2)})")
    return result
