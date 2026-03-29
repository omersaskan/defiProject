import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from defihunter.utils.logger import logger
from defihunter.utils.structured_logger import s_logger
from defihunter.engines.ml.repository import ModelRepository

class MLPredictor:
    """
    GT-Institutional: Inference & Ranking Engine.
    Handles candidate scoring using regime-aware models and heuristic fallbacks.
    """
    def __init__(self, repository: ModelRepository):
        self.repository = repository

    def rank_candidates(self, candidates: pd.DataFrame, top_n: int = 5, use_family_ranker: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """Scores and ranks candidates using either regime-specific or family-ranker models."""
        if candidates.empty:
            return candidates, []
            
        candidates = candidates.copy()
        family_suite = {}
        if use_family_ranker:
            family_suite = self.repository.load_family_ranker()
            if not family_suite:
                use_family_ranker = False
            
        scored_groups = []
        for sym, group in candidates.groupby('symbol'):
            group = group.copy()
            
            if use_family_ranker:
                group = self._predict_family_ranker(group, family_suite)
            else:
                group = self._predict_standard(group, sym)
            
            scored_groups.append(group)
        
        candidates = pd.concat(scored_groups)
        candidates = self.ensure_canonical_columns(candidates)
        candidates = candidates.sort_values(by='ml_rank_score', ascending=False)
        
        top_candidates = candidates.head(top_n)['symbol'].tolist()
        
        s_logger.log(
            engine="MLPredictor",
            event="RANKING_COMPLETED",
            data={
                "top_n": top_n,
                "symbols_ranked": len(candidates),
                "top_symbol": top_candidates[0] if top_candidates else None
            }
        )
        
        return candidates, top_candidates

    def _predict_family_ranker(self, group: pd.DataFrame, suite: Dict[str, Any]) -> pd.DataFrame:
        try:
            feats = suite["features_used"]
            missing = [f for f in feats if f not in group.columns]
            for f in missing: group[f] = 0.0
            
            X_pred = group[feats].fillna(0.0)
            
            group['leader_prob'] = self._safe_predict_proba(suite["model_leader"], X_pred)
            group['setup_conversion_prob'] = self._safe_predict_proba(suite["model_setup"], X_pred)
            
            # Predicted Rank Pct (0.0 is best) -> 0-100 Holdability
            if suite["model_hold"]:
                rank_pct = suite["model_hold"].predict(X_pred)
                group['holdability_score'] = (1.0 - np.clip(rank_pct, 0, 1)) * 100
            else:
                group['holdability_score'] = 50.0
            
            group['ml_rank_score'] = (group['leader_prob'] * 0.5 + group['setup_conversion_prob'] * 0.3 + (group['holdability_score']/100) * 0.2) * 100
            group['ml_explanation'] = group.apply(
                lambda r: f"[FAMILY_RANKER] Prob: {r['leader_prob']:.1%} | Setup: {r['setup_conversion_prob']:.1%} | Hold: {r['holdability_score']:.1f}", axis=1
            )
        except Exception as e:
            logger.error(f"[ML-Predictor] Family-Ranker logic failed: {e}")
            group['ml_rank_score'] = 50.0
        return group

    def _predict_standard(self, group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # 1. Try Asset model, then Regime model, then Global
        models = self.repository.load_models(symbol)
        label = symbol
        
        if not models:
            regime = group.iloc[0].get('regime', 'chop')
            reg_key = self.repository.normalize_regime_key(regime)
            label = f"GLOBAL_{reg_key}"
            models = self.repository.load_models(label)
            
        if not models:
            label = "GLOBAL"
            models = self.repository.load_models(label)
            
        if models and models.get("long_clf") is not None:
            try:
                feats = models["features_used"]
                missing = [f for f in feats if f not in group.columns]
                for f in missing: group[f] = 0.0
                
                X_pred = group[feats].fillna(0.0)
                group['probability_long'] = self._safe_predict_proba(models["long_clf"], X_pred)
                group['future_rank_pct'] = models["reg_model"].predict(X_pred) if models["reg_model"] else 0.5
                
                group['ml_rank_score'] = (group['future_rank_pct'] * 0.6 + group['probability_long'] * 0.4) * 100
                group['ml_explanation'] = group.apply(
                    lambda r: f"[{label}] RankProb: {r['ml_rank_score']:.1f} | PredRank: {r['future_rank_pct']:.1%}", axis=1
                )
            except Exception as e:
                logger.error(f"[ML-Predictor] Standard logic failed for {symbol}: {e}")
                group['ml_rank_score'] = 50.0
        else:
            group['ml_rank_score'] = 50.0
            group['ml_explanation'] = f"No Model ({label})"
            
        return group

    def ensure_canonical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantees canonical ML columns exist, applying heuristic fallback if needed."""
        required = ["leader_prob", "setup_conversion_prob", "holdability_score", "ml_rank_score", "ml_explanation"]
        if all(c in df.columns and not df[c].isna().all() for c in required):
            return df
        return self.heuristic_fallback(df)

    def heuristic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule-based consensus scoring when ML is unavailable."""
        df = df.copy()
        
        def _safe_series(col: str, default: float = 0.5) -> pd.Series:
            return pd.to_numeric(df[col], errors="coerce").fillna(default) if col in df.columns else pd.Series(default, index=df.index)

        def _norm01(s: pd.Series) -> pd.Series:
            if s.isna().all(): return pd.Series(0.5, index=s.index)
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo) if abs(hi - lo) > 1e-12 else pd.Series(0.5, index=s.index)

        h_score = (
            0.22 * _norm01(_safe_series("ml_rank_score")) +
            0.22 * _norm01(_safe_series("entry_readiness")) +
            0.18 * (1.0 - _norm01(_safe_series("fakeout_risk"))) +
            0.12 * _norm01(_safe_series("peer_rank")) +
            0.10 * _norm01(_safe_series("peer_momentum")) +
            0.08 * _norm01(_safe_series("rs_divergence_persistence")) +
            0.08 * _safe_series("quiet_expansion", 0.0).clip(0, 1)
        ).clip(0.0, 1.0)

        df["leader_prob"] = df.get("leader_prob", (0.35 + 0.55 * h_score).clip(0.05, 0.95))
        df["setup_conversion_prob"] = df.get("setup_conversion_prob", (0.30 + 0.60 * h_score).clip(0.05, 0.95))
        df["holdability_score"] = df.get("holdability_score", (100.0 * (0.45 * h_score + 0.3 * _norm01(_safe_series("entry_readiness")))).clip(0, 100))
        df["ml_rank_score"] = df.get("ml_rank_score", h_score * 100.0)
        df["ml_explanation"] = df.get("ml_explanation", "[HEURISTIC_FALLBACK] Rule-consensus")
        
        # Aliases
        df["setup_quality_prob"] = df.get("setup_quality_prob", df["setup_conversion_prob"])
        df["hold_quality"] = df.get("hold_quality", df["holdability_score"])
        
        return df

    def _safe_predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if model is None: return np.zeros(len(X))
        try:
            probs = model.predict_proba(X)
            return probs[:, 1] if probs.shape[1] > 1 else (np.ones(len(X)) if model.classes_[0] == 1 else np.zeros(len(X)))
        except:
            return np.zeros(len(X))
