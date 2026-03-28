import pandas as pd
import numpy as np
from typing import Dict, Any

class MarketRegimeEngine:
    def detect_regime(self, btc_mtf: Dict[str, pd.DataFrame], eth_mtf: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Detects global market regime using TRUE Multi-Timeframe data.
        btc_mtf/eth_mtf: Dict containing '15m', '1h', '4h' DataFrames.
        """
        if '15m' not in btc_mtf or btc_mtf['15m'].empty:
            return {"label": "unknown", "confidence": 0.0, "metadata": {}}
            
        # Latest rows from 15m
        btc_15m = btc_mtf['15m'].iloc[-1]
        eth_15m = eth_mtf['15m'].iloc[-1]
        
        # 1. MTF Trend Alignment — BTC
        def is_uptrend(df):
            if df.empty or 'ema_55' not in df.columns: return False
            return df['close'].iloc[-1] > df['ema_55'].iloc[-1]

        btc_s = is_uptrend(btc_mtf.get('15m', pd.DataFrame()))
        btc_m = is_uptrend(btc_mtf.get('1h', pd.DataFrame()))
        btc_l = is_uptrend(btc_mtf.get('4h', pd.DataFrame()))
        
        # BUG-F FIX: ETH trend alignment also evaluated (not just for ROC)
        eth_s = is_uptrend(eth_mtf.get('15m', pd.DataFrame()))
        eth_m = is_uptrend(eth_mtf.get('1h', pd.DataFrame()))
        eth_l = is_uptrend(eth_mtf.get('4h', pd.DataFrame()))
        
        # 2. Volatility (Real ATR from 15m)
        vol_label = "normal"
        vol_pct = 0.0
        if 'atr' in btc_mtf['15m'].columns:
            vol_pct = btc_15m['atr'] / btc_15m['close']
            if vol_pct > 0.025: # Dynamic threshold can be added to config later
                vol_label = "high_vol"
            elif vol_pct < 0.006:
                vol_label = "low_vol"
                
        # 3. Risk On/Off & BTC Leadership (ROC over 24h roughly)
        # Using 1h data for ROC is more stable
        btc_1h = btc_mtf.get('1h', btc_mtf['15m'])
        eth_1h = eth_mtf.get('1h', eth_mtf['15m'])
        
        lookback = min(24, len(btc_1h)-1)
        btc_roc = (btc_1h['close'].iloc[-1] / btc_1h['close'].iloc[-lookback]) - 1
        eth_roc = (eth_1h['close'].iloc[-1] / eth_1h['close'].iloc[-lookback]) - 1
        
        alt_rotation = eth_roc > btc_roc + 0.015 
        btc_led = btc_roc > eth_roc + 0.015     
        
        # BUG-F FIX: Both ETH and BTC uptrend = strong alt rotation signal
        eth_trending_up = eth_s and eth_m
        
        # 4. Final Label Synthesis
        btc_ema_dist = (btc_15m['close'] - btc_15m['ema_55']) / btc_15m['ema_55']
        
        if abs(btc_ema_dist) < 0.003 and not (btc_s or btc_m):
            label = "chop"
        elif btc_s and btc_m:
            if alt_rotation and eth_trending_up:  # BUG-F FIX: ETH must also be trending for strong alt rotation
                label = "trend_alt_rotation"
            elif btc_led:
                label = "trend_btc_led"
            else:
                label = "trend_neutral"
        elif not btc_s and not btc_m:
            label = "downtrend"
        else:
            label = "unstable"
            
        # BUG-F FIX: Confidence now includes ETH MTF alignment (6 signals instead of 3)
        btc_alignment_score = sum([btc_s, btc_m, btc_l]) / 3.0
        eth_alignment_score = sum([eth_s, eth_m, eth_l]) / 3.0
        # Weighted: BTC drives macro, ETH drives DeFi sector — 60/40 weighting
        alignment_score = (btc_alignment_score * 0.6) + (eth_alignment_score * 0.4)
            
        return {
            "label": label,
            "volatility": vol_label,
            "risk_on": alt_rotation,
            "btc_led": btc_led,
            "eth_trending_up": eth_trending_up,
            "confidence": round(alignment_score, 2),
            "metadata": {
                "btc_ema_dist": round(btc_ema_dist, 4),
                "btc_roc_24h": round(btc_roc, 4),
                "eth_roc_24h": round(eth_roc, 4),
                "vol_pct": round(vol_pct, 4),
                "btc_mtf_score": round(btc_alignment_score, 2),
                "eth_mtf_score": round(eth_alignment_score, 2)
            }
        }
        
    def detect_historical_regimes(self, btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.Series:
        """
        Calculates a vectorized rolling regime for every row in a historical dataframe.
        BUG-4 FIX: ETH alignment now included to match live detect_regime() logic.
        Assumes btc_df and eth_df are aligned by timestamp (1h bars).
        """
        if btc_df.empty or eth_df.empty:
            return pd.Series("unknown", index=btc_df.index if not btc_df.empty else [])

        # Align eth_df length to btc_df
        if len(eth_df) > len(btc_df):
            eth_df = eth_df.tail(len(btc_df)).reset_index(drop=True)
        elif len(eth_df) < len(btc_df):
            eth_df = eth_df.reindex(range(len(btc_df))).ffill().bfill()

        btc_ema55 = btc_df['close'].ewm(span=55, adjust=False).mean()
        eth_ema55 = eth_df['close'].ewm(span=55, adjust=False).mean()

        # BTC and ETH short-term uptrend (vs EMA55)
        btc_s = btc_df['close'] > btc_ema55
        eth_s = eth_df['close'] > eth_ema55

        # 24h ROC (assuming 1h bars)
        btc_roc = btc_df['close'].pct_change(periods=24)
        eth_roc = eth_df['close'].pct_change(periods=24)

        alt_rotation = eth_roc > (btc_roc + 0.015)
        btc_led = btc_roc > (eth_roc + 0.015)

        # BUG-4 FIX: ETH must also be trending up for alt_rotation — matches live logic
        eth_trending_up = eth_s

        regimes = pd.Series("chop", index=btc_df.index)

        regimes.loc[btc_s & ~eth_s & ~btc_led] = "unstable"
        regimes.loc[btc_s & alt_rotation & eth_trending_up] = "trend_alt_rotation"
        regimes.loc[btc_s & btc_led] = "trend_btc_led"
        regimes.loc[btc_s & ~alt_rotation & ~btc_led & eth_trending_up] = "trend_neutral"
        regimes.loc[~btc_s & ~eth_s] = "downtrend"

        return regimes

    def detect_sector_momentum(self, defi_symbols_df: dict) -> dict:
        """
        GT-GOLD-10: Tüm DeFi coinlerinin kısa vadeli ROC ortalamasını hesapla.
        Sektör genelinde ısı artışını tespit: hot_sector=True ise tüm sinyallere bonus.
        Args:
            defi_symbols_df: {symbol: df} son 5 bar içeren DataFrame'ler
        """
        import numpy as np
        roc_list = []
        for sym, df in defi_symbols_df.items():
            if df is not None and len(df) >= 5 and 'close' in df.columns:
                roc = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
                roc_list.append(roc)

        if not roc_list:
            return {"sector_momentum": 0.0, "hot_sector": False, "sector_breadth": 0.0}

        mean_roc = float(np.mean(roc_list))
        std_roc  = float(np.std(roc_list)) + 1e-8
        pct_positive = sum(r > 0 for r in roc_list) / len(roc_list)

        return {
            "sector_momentum": round(mean_roc, 4),
            "hot_sector": mean_roc > 0.02 and pct_positive > 0.60,
            "sector_breadth": round(mean_roc / std_roc, 3),
            "pct_positive": round(pct_positive, 3),
        }


class SectorRegimeEngine:
    def get_sector_regime(self, eth_df: pd.DataFrame, aave_df: pd.DataFrame, uni_df: pd.DataFrame) -> dict:
        """
        Examine sector leadership breadth and internal strength.
        """
        def calculate_roc(df, lookback=24):
            if df is None or df.empty or len(df) < lookback: return 0.0
            return (df['close'].iloc[-1] / df['close'].iloc[-lookback]) - 1
            
        eth_roc = calculate_roc(eth_df)
        aave_roc = calculate_roc(aave_df)
        uni_roc = calculate_roc(uni_df)
        
        # Alpha Calculation
        lend_alpha = aave_roc - eth_roc
        dex_alpha = uni_roc - eth_roc
        
        # Breadth: Average outperformance of sector pillars
        sector_breadth = (lend_alpha + dex_alpha) / 2.0
        
        # Internal Strength: Qualitative alignment (Are both leading or is one dragging?)
        # 1.0 = both leading, 0.0 = both lagging, 0.5 = mixed
        lend_leading = 1 if lend_alpha > 0.01 else 0
        dex_leading = 1 if dex_alpha > 0.01 else 0
        internal_strength = (lend_leading + dex_leading) / 2.0
        
        label = "eth_led_beta"
        strongest = "eth"
        
        if lend_alpha > dex_alpha and lend_alpha > 0.015:
            strongest = "defi_lending"
            label = "defi_lend_led"
        elif dex_alpha > lend_alpha and dex_alpha > 0.015:
            strongest = "defi_dex"
            label = "defi_dex_led"
            
        # Bug #8 Fix: Keys must match FamilyEngine family labels (defi_lending, defi_dex_amm, etc.)
        # Previously "lending" != "defi_lending" caused sector_multiplier to always be 1.0
        lend_score = (1.0 + (lend_alpha * 10)) * (0.5 + internal_strength)
        dex_score  = (1.0 + (dex_alpha  * 10)) * (0.5 + internal_strength)
        
        return {
            "label": label,
            "strongest_family": strongest,
            "sector_breadth": round(sector_breadth, 4),
            "internal_strength": internal_strength,
            "lending_alpha": round(lend_alpha, 4),
            "dex_alpha": round(dex_alpha, 4),
            "sector_scores": {
                # Legacy aliases (optional)
                "lending": lend_score,
                "dex_amm": dex_score,
                # Canonical labels matching default.yaml and FamilyEngine
                "defi_lending": lend_score,
                "defi_dex":     dex_score,
                "defi_dex_amm": dex_score,
                "defi_perp":    dex_score,
                "defi_lsd":     lend_score,
                "defi_oracles": dex_score * 0.9,
                "defi_yield":   dex_score,
                "defi_infra":   lend_score * 0.8,
                "defi_beta":    1.0
            }
        }

