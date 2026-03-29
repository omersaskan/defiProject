import pandas as pd
from typing import Dict, Any, List

from defihunter.utils.structured_logger import s_logger
from defihunter.engines.setups_registry import setup_registry


class RuleEngine:
    def __init__(self, config: Any = None):
        self.config = config

        
    def evaluate(self, df: pd.DataFrame, regime: str, family: str, resolved_thresholds: dict, 
                 sector_data: dict = None, adaptive_weights: dict = None, primary_anchor: str = None) -> pd.DataFrame:
        """
        Deterministic signal engine: Hard filters -> Soft scores -> Veto.
        Integrates sector leadership multipliers and advanced setup detection.
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 0. Resolve Sector Multiplier
        sector_multiplier = 1.0
        if sector_data and 'sector_scores' in sector_data:
            sector_multiplier = sector_data['sector_scores'].get(family, 1.0)
        
        # 1. Hard filters
        df = self._apply_hard_filters(df, resolved_thresholds)
            
        # 2. Soft scores
        df = self._calculate_soft_scores(df, sector_multiplier, family, adaptive_weights, primary_anchor)
        
        # 3. Regime-specific entry threshold
        min_score = resolved_thresholds.get('min_score', 50)
        min_leadership = resolved_thresholds.get('min_relative_leadership', 0)
        
        # Vetoes
        df.loc[df['relative_leadership_score'] < min_leadership, 'veto_reason'] = "insufficient_leadership"
        
        # Final Signal
        df['entry_signal'] = (df['total_score'] >= min_score) & (df['veto_reason'].isna()) & (df['universe_ok'])
        
        # 4. Setup Class identification
        df = self._resolve_setup_class(df)

        # 5. Explainability & Targets
        df['explanation'] = df.apply(lambda row: self._build_explanation(row, regime, family, sector_multiplier), axis=1)

        # TP / SL based on ATR
        if 'atr' in df.columns:
            df['stop_price'] = df['close'] - (df['atr'] * 1.5)
            df['tp1_price'] = df['close'] + (df['atr'] * 2.0)
            df['tp2_price'] = df['close'] + (df['atr'] * 4.0)
        else:
            df['stop_price'] = df['close'] * 0.95
            df['tp1_price'] = df['close'] * 1.05
            df['tp2_price'] = df['close'] * 1.10
            
        # 6. Structured logging for signals
        signals = df[df['entry_signal']]
        for _, sig in signals.iterrows():
            s_logger.log(
                engine="RuleEngine",
                event="SIGNAL_GENERATED",
                symbol=sig['symbol'],
                data={
                    "total_score": round(sig['total_score'], 1),
                    "setup_class": sig['setup_class'],
                    "tp1": round(sig['tp1_price'], 4),
                    "stop": round(sig['stop_price'], 4)
                }
            )
            
        return df

    def _apply_hard_filters(self, df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
        min_vol = thresholds.get('min_volume', 10_000_000)
        min_bars = thresholds.get('min_bars', 200)
        
        df['universe_ok'] = True
        df['veto_reason'] = None
        
        if 'quote_volume' in df.columns:
            df['universe_ok'] = df['quote_volume'] > min_vol
        
        if 'cvd_zscore' in df.columns:
            mask_zero_cvd = (df['cvd_zscore'] == 0.0)
            df.loc[mask_zero_cvd, 'universe_ok'] = False
            df.loc[mask_zero_cvd, 'veto_reason'] = "Unknown Data Quality (CVD=0)"
            
        if 'bar_count' in df.columns:
            df.loc[df['bar_count'] < min_bars, 'universe_ok'] = False
            
        return df

    def _calculate_soft_scores(self, df: pd.DataFrame, sector_multiplier: float, family: str, 
                               adaptive_weights: dict, primary_anchor: str = None) -> pd.DataFrame:
        """
        GT-Institutional: Refactored Categorical Scoring.
        Each category is calculated independently to prevent 'God Function' complexity.
        """
        # A) Trend Score
        df = self._score_trend(df)
                
        # B) Expansion Score
        df = self._score_expansion(df)
            
        # C) Participation Score
        df = self._score_participation(df)
            
        # D) Relative Leadership Score
        df = self._calculate_relative_leadership_score(df, family, sector_multiplier, primary_anchor)
            
        # E) Funding & Squeeze Logic
        df = self._apply_funding_logic(df)
        
        # GT-WEIGHTING: Using centralized config weights
        cfg_s = getattr(self.config, "scoring", None)
        w = {
            "trend_score": getattr(cfg_s, "trend_weight", 0.5),
            "expansion_score": getattr(cfg_s, "expansion_weight", 0.5),
            "participation_score": getattr(cfg_s, "participation_weight", 0.5),
            "relative_leadership_score": getattr(cfg_s, "relative_leadership_weight", 2.0)
        }
        
        if adaptive_weights and isinstance(adaptive_weights, dict):
            for k in w:
                if k in adaptive_weights:
                    # Scale down adaptive weights for setups, keep/boost Leadership
                    w[k] = float(adaptive_weights[k]) * (0.5 if k != "relative_leadership_score" else 2.0)

        # Total Tally (Weighted Category Model)
        df['total_score'] = (
            (df['trend_score'] * w['trend_score'] +
             df['expansion_score'] * w['expansion_score'] +
             df['participation_score'] * w['participation_score']) * sector_multiplier +
            df['relative_leadership_score'] * w['relative_leadership_score'] +
            df['funding_penalty'] * getattr(cfg_s, "funding_penalty_weight", 1.0)
        )

        return df

    def _score_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['trend_score'] = 0.0
        if 'ema_55' in df.columns and 'ema_100' in df.columns:
            trend_cond = (df['close'] > df['ema_55']) & (df['ema_55'] > df['ema_100'])
            df.loc[trend_cond, 'trend_score'] += 20
            if 'ema_20' in df.columns:
                df.loc[df['close'] > df['ema_20'], 'trend_score'] += 10
        return df

    def _score_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        df['expansion_score'] = 0.0
        exp_features = [
            ('is_breakout_bar', 10), ('is_base_expansion', 5), ('sweep_low', 5), 
            ('sweep_reclaim_confirmed', 5), ('whale_absorption', 10), ('whale_absorption_strong', 15),
            ('high_quality_breakout', 10)
        ]
        for col, pts in exp_features:
            if col in df.columns:
                df.loc[df[col].astype(bool), 'expansion_score'] += pts

        if 'low_quality_breakout' in df.columns:
            df.loc[df['low_quality_breakout'].astype(bool), 'expansion_score'] *= 0.5

        df['expansion_score'] = df['expansion_score'].clip(upper=25)
        return df

    def _score_participation(self, df: pd.DataFrame) -> pd.DataFrame:
        df['participation_score'] = 0.0
        
        # 1. Volume Metrics
        if 'volume_zscore' in df.columns:
            df.loc[df['volume_zscore'] > 2.0, 'participation_score'] += 15
            df.loc[(df['volume_zscore'] > 1.0) & (df['volume_zscore'] <= 2.0), 'participation_score'] += 5

        # 2. Binary Participation Setups
        part_features = [
            ('orderbook_vacuum', 15), ('silent_accumulation', 20), ('quiet_expansion', 15), 
            ('accumulation_zone', 5), ('msb_bull', 10), ('funding_capitulation', 15),
            ('rs_strong_divergence', 10), ('taker_surge', 20), ('squeeze_release', 15),
            ('oi_divergence_bull', 20), ('short_squeeze_setup', 15), ('funding_extreme_short', 10),
            ('rsi_kink', 10), ('rsi_oversold_recovery', 5), ('launch_mode', 15),
            ('cvd_price_divergence', 15), ('near_liquidation_band', 20), ('is_short_squeeze_imminent', 25),
            ('coiling_breakout_alert', 20)
        ]
        for col, pts in part_features:
            if col in df.columns:
                df.loc[df[col].astype(bool), 'participation_score'] += pts
        
        # 3. Continuous Persistence Metrics
        if 'v_delta_score' in df.columns:
            df.loc[df['v_delta_score'] > 0.05, 'participation_score'] += 5
        if 'rs_divergence_persistence' in df.columns:
            df.loc[df['rs_divergence_persistence'] >= 3, 'participation_score'] += 10
        if 'squeeze_duration' in df.columns:
            df.loc[df['squeeze_duration'] >= 8, 'participation_score'] += 5
            df.loc[df['squeeze_duration'] >= 16, 'participation_score'] += 5
        
        # 4. Momentum Targets
        if 'pre_gainer_score' in df.columns:
            df['participation_score'] += df['pre_gainer_score'].clip(upper=15)
        if 'rvr_score' in df.columns:
            df['participation_score'] += df['rvr_score'].clip(upper=10)

        df['participation_score'] = df['participation_score'].clip(upper=35)
        return df

    def _apply_funding_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        df['funding_penalty'] = 0.0
        df['is_short_squeeze'] = False
        
        if 'funding_rate' in df.columns:
            mask_bad_funding = df['funding_rate'] < -0.002
            mask_squeeze = (df['funding_rate'] < -0.001)
            
            if 'sweep_reclaim_confirmed' in df.columns:
                mask_squeeze = mask_squeeze & df['sweep_reclaim_confirmed']
            if 'oi_delta_4' in df.columns:
                mask_squeeze = mask_squeeze & (df['oi_delta_4'] > 0.02)
                
            df.loc[mask_bad_funding & ~mask_squeeze, 'veto_reason'] = "funding_too_negative"
            df.loc[df['funding_rate'] > 0.001, 'funding_penalty'] = -15
            
            # Squeeze boost to expansion
            df.loc[mask_squeeze, 'expansion_score'] += 20
            df['is_short_squeeze'] = mask_squeeze
        return df

        return df

    def _calculate_relative_leadership_score(self, df: pd.DataFrame, family: str, sector_multiplier: float, primary_anchor: str = None) -> pd.DataFrame:
        df['relative_leadership_score'] = 0.0
        
        if sector_multiplier > 1.05 and 'return_4' in df.columns:
            mask_laggard = (df['return_4'] < 0.02) & (df['return_4'] > -0.05)
            df.loc[mask_laggard, 'relative_leadership_score'] += 15

        rel_cols = [c for c in df.columns if 'rel_spread_' in c and all(k not in c for k in ['slope', 'persistence', 'z', 'acceleration'])]
        for rc in rel_cols:
            base_anchor_name = rc.split('_')[2] # e.g. BTC from rel_spread_BTC_p
            
            # Anchor weighting fix: compare anchor name with family's primary_anchor (e.g. AAVE in AAVE.p)
            is_primary = False
            if primary_anchor and base_anchor_name.lower() in primary_anchor.lower():
                is_primary = True
                
            mult = 1.5 if is_primary else 1.0
            
            df.loc[df[rc] > 0.005, 'relative_leadership_score'] += (5 * mult)
            if f"{rc}_slope_4" in df.columns:
                df.loc[df[f"{rc}_slope_4"] > 0, 'relative_leadership_score'] += (3 * mult)
            if f"{rc}_persistence" in df.columns:
                df.loc[df[f"{rc}_persistence"] >= 3, 'relative_leadership_score'] += (5 * mult)
                
        df['relative_leadership_score'] = df['relative_leadership_score'].clip(upper=35)
        return df

    def _resolve_setup_class(self, df: pd.DataFrame) -> pd.DataFrame:
        df['setup_class'] = "unknown"
        df['_setup_priority'] = 0

        # Now using the modular Registry instead of a hardcoded list
        for setup in setup_registry.setups:
            col = setup.condition_col
            if col in df.columns:
                mask = df[col].astype(bool) & (df['_setup_priority'] < setup.priority)
                df.loc[mask, 'setup_class'] = setup.label
                df.loc[mask, '_setup_priority'] = setup.priority


        if 'rs_divergence_persistence' in df.columns:
            mask = (df['rs_divergence_persistence'] >= 3) & (df['_setup_priority'] < 2)
            df.loc[mask, 'setup_class'] = 'rs_divergence_bull'
            df.loc[mask, '_setup_priority'] = 2

        df.loc[(df['setup_class'] == 'unknown') & (df['total_score'] > 35), 'setup_class'] = 'potential_momentum'
        df.loc[(df['setup_class'] == 'unknown') & (df['total_score'] <= 35), 'setup_class'] = 'low_conviction'
        df.drop(columns=['_setup_priority'], inplace=True)
        return df

    def _build_explanation(self, row: pd.Series, regime: str, family: str, sector_multiplier: float) -> dict:
        reasons = []
        if row['trend_score'] > 20: reasons.append("Strong Trend")
        if row['expansion_score'] >= 15: reasons.append("Breakout/Sweep")
        if row['participation_score'] >= 15: reasons.append("High Volume")

        sc = str(row.get('setup_class', ''))
        sc_map = {
            'explosive_short_squeeze': "💥 EXPLOSIVE SHORT SQUEEZE",
            'short_squeeze_auto_trigger': "🚨 SHORT SQUEEZE",
            'silent_accumulation': "🎯 SILENT ACCUMULATION",
            'orderbook_vacuum': "🕳️ ORDERBOOK VACUUM",
            'quiet_expansion': "🔇 QUIET EXPANSION",
            'rs_divergence': "📡 RS DIVERGENCE",
            'taker_surge': "⚡ TAKER SURGE",
            'squeeze_release': "💥 SQUEEZE RELEASE",
            'oi_divergence': "📊 OI DIVERGENCE",
            'oi_short_squeeze': "🌀 OI+FUNDING SQUEEZE",
            'extreme_funding': "💸 FUNDING EXTREME",
            'rsi_kink': "📈 RSI KINK",
            'whale_absorption_strong': "🐋 WHALE ABSORPTION (STRONG)",
            'whale_absorption': "🐋 WHALE ABSORPTION",
            'momentum_launch_mode': "🚀 LAUNCH MODE",
            'cvd_smartmoney': "🧠 SMART MONEY CVD",
            'liquidation_squeeze_magnet': "🧲 LIQUIDATION MAGNET",
            'high_quality_breakout': "💎 HQ BREAKOUT",
            'coiling_catalyst': "⏰ COILING CATALYST"
        }
        for k, v in sc_map.items():
            if k in sc: reasons.append(v)
            
        if row.get('is_coiling', False): reasons.append("🔴 COILING")
        if sector_multiplier > 1.05: reasons.append(f"Sector Strength ({family})")

        rel_cols = [c for c in row.index if 'rel_spread_' in c and all(k not in c for k in ['slope', 'persistence', 'z', 'acceleration'])]
        passed_anchors = [c.split('_')[2].upper() for c in rel_cols if row[c] > 0.005]

        return {
            "passed_anchors": passed_anchors,
            "regime": regime,
            "family": family,
            "setup_type": row['setup_class'],
            "dominant_factors": reasons,
            "sector_alpha": round(sector_multiplier - 1.0, 3),
            "veto": row['veto_reason'],
            "orderbook_vacuum": bool(row.get('orderbook_vacuum', False)),
            "silent_accumulation": bool(row.get('silent_accumulation', False)),
            "quiet_expansion": bool(row.get('quiet_expansion', False)),
            "rs_strong_divergence": bool(row.get('rs_strong_divergence', False))
        }
