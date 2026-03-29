import os
import joblib
import json
from typing import Dict, Any, Optional, List
from defihunter.utils.logger import logger

class ModelRepository:
    """
    GT-Institutional: Model Storage & Lifecycle Management.
    Handles atomic disk I/O, 'healing' corrupted metadata, and memory caching.
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._active_symbol: Optional[str] = None

    @staticmethod
    def normalize_regime_key(regime_label: str) -> str:
        """BUG-B + BUG-C FIX: Normalizes regime engine output strings to TREND/CHOP keys."""
        r = str(regime_label).lower()
        if r.startswith('trend'):
            return 'TREND'
        elif r in ('chop', 'unstable', 'unknown', 'downtrend'):
            return 'CHOP'
        return 'CHOP'

    def save_models(self, symbol: str, long_clf, short_clf, reg_model, features_used, metrics=None, train_meta=None):
        """Atomic save for a suite of models associated with a symbol or regime."""
        os.makedirs(self.model_dir, exist_ok=True)
        
        def _atomic_dump(obj, filename):
            target_path = os.path.join(self.model_dir, filename)
            tmp_path = target_path + ".tmp"
            try:
                joblib.dump(obj, tmp_path)
                os.replace(tmp_path, target_path)
            except Exception as e:
                logger.error(f"[ML-Repo] Atomic save failed for {filename}: {e}")

        if long_clf: 
            _atomic_dump(long_clf, f'lgb_classifier_long_{symbol}.pkl')
        if short_clf: 
            _atomic_dump(short_clf, f'lgb_classifier_short_{symbol}.pkl')
        if reg_model: 
            _atomic_dump(reg_model, f'lgb_regressor_{symbol}.pkl')
        
        if features_used:
            _atomic_dump(features_used, f'features_used_{symbol}.pkl')
            if metrics:
                _atomic_dump(metrics, f'metrics_{symbol}.pkl')

        if train_meta:
            meta_path = os.path.join(self.model_dir, f'metadata_{symbol}.json')
            tmp_meta = meta_path + ".tmp"
            try:
                with open(tmp_meta, 'w') as f:
                    json.dump(train_meta, f, indent=2)
                os.replace(tmp_meta, meta_path)
            except Exception as e:
                logger.error(f"[ML-Repo] Metadata save failed for {symbol}: {e}")
        
        # Invalidate cache for this symbol after save
        if symbol in self._cache:
            del self._cache[symbol]
        logger.info(f"[ML-Repo] Models saved successfully for {symbol}")

    def load_models(self, symbol: str) -> Dict[str, Any]:
        """Loads models for a specific symbol or regime with memory caching."""
        if symbol in self._cache:
            return self._cache[symbol]

        paths = {
            "long_clf": os.path.join(self.model_dir, f'lgb_classifier_long_{symbol}.pkl'),
            "short_clf": os.path.join(self.model_dir, f'lgb_classifier_short_{symbol}.pkl'),
            "reg_model": os.path.join(self.model_dir, f'lgb_regressor_{symbol}.pkl'),
            "features_used": os.path.join(self.model_dir, f'features_used_{symbol}.pkl'),
            "importance": os.path.join(self.model_dir, f'feature_importance_{symbol}.pkl'),
            "legacy_clf": os.path.join(self.model_dir, f'lgb_classifier_{symbol}.pkl')
        }

        # Legacy fallback if new long/short convention isn't there yet
        if not os.path.exists(paths["long_clf"]) and os.path.exists(paths["legacy_clf"]):
            paths["long_clf"] = paths["legacy_clf"]

        if not os.path.exists(paths["long_clf"]) or not os.path.exists(paths["features_used"]):
            return {}

        try:
            res = {
                "long_clf": joblib.load(paths["long_clf"]),
                "short_clf": joblib.load(paths["short_clf"]) if os.path.exists(paths["short_clf"]) else None,
                "reg_model": joblib.load(paths["reg_model"]) if os.path.exists(paths["reg_model"]) else None,
                "features_used": joblib.load(paths["features_used"])
            }

            # BUG-FIX HEALING
            if len(res["features_used"]) < 100:
                res["features_used"] = self._heal_features(symbol, res["features_used"], paths["features_used"])

            # Metadata/Importance
            if os.path.exists(paths["importance"]):
                try:
                    imp_dict = joblib.load(paths["importance"])
                    res["top_features"] = sorted(
                        [k for k in imp_dict if k in res["features_used"]], 
                        key=lambda x: imp_dict[x], reverse=True
                    )[:15]
                except: pass

            self._cache[symbol] = res
            return res
        except Exception as e:
            logger.error(f"[ML-Repo] Load failed for {symbol}: {e}")
            return {}

    def _heal_features(self, symbol: str, current_feats: List[str], feat_path: str) -> List[str]:
        """Restores corrupted feature lists from known good 127-feature pools."""
        candidates = ['features_used_ALL.p.pkl', 'features_used_ALL.pkl', 'features_used_GLOBAL_TREND.pkl', 'features_used_BTC.p.pkl']
        for cand in candidates:
            cand_path = os.path.join(self.model_dir, cand)
            if os.path.exists(cand_path):
                try:
                    cand_features = joblib.load(cand_path)
                    if len(cand_features) == 127:
                        logger.info(f"[ML-Repo] HEALING: {symbol} restored from {cand} (127 features).")
                        joblib.dump(cand_features, feat_path)
                        return cand_features
                except: continue
        return current_feats

    def load_family_ranker(self) -> Dict[str, Any]:
        """Specialized loader for the Phase 4 Tri-Score family-ranker suite."""
        try:
            leader_path = os.path.join(self.model_dir, 'lgb_classifier_long_FAMILY_LEADER.pkl')
            feat_path   = os.path.join(self.model_dir, 'features_used_FAMILY_LEADER.pkl')
            
            if not os.path.exists(leader_path):
                return {}

            res = {
                "model_leader": joblib.load(leader_path),
                "features_used": joblib.load(feat_path)
            }

            # Optional Setup Classifier (fallback to leader)
            setup_path = os.path.join(self.model_dir, 'lgb_classifier_long_FAMILY_SETUP.pkl')
            if os.path.exists(setup_path):
                res["model_setup"] = joblib.load(setup_path)
            else:
                res["model_setup"] = res["model_leader"]

            # Optional Holdability Regressor (flexible fallbacks)
            hold_path = os.path.join(self.model_dir, 'lgb_regressor_FAMILY_HOLD.pkl')
            leader_reg_path = os.path.join(self.model_dir, 'lgb_regressor_FAMILY_LEADER.pkl')
            setup_reg_path = os.path.join(self.model_dir, 'lgb_regressor_FAMILY_SETUP.pkl')

            if os.path.exists(hold_path):
                res["model_hold"] = joblib.load(hold_path)
            elif os.path.exists(leader_reg_path):
                res["model_hold"] = joblib.load(leader_reg_path)
            elif os.path.exists(setup_reg_path):
                res["model_hold"] = joblib.load(setup_reg_path)
            else:
                res["model_hold"] = None

            return res
        except Exception as e:
            logger.error(f"[ML-Repo] Family-Ranker load failed: {e}")
            return {}
