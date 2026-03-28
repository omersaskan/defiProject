import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from typing import List, Tuple, Dict
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import warnings
from defihunter.utils.logger import logger

# Suppress sklearn 1.6+ cv='prefit' deprecation warnings to keep Streamlit / terminal logs clean
warnings.filterwarnings('ignore', message=".*cv='prefit'.*", category=UserWarning)

class MLRankingEngine:
    def __init__(self, model_type: str = "lightgbm", model_dir: str = "models"):
        self.model_type = model_type
        self.model_dir = model_dir
        self.long_clf_model = None
        self.short_clf_model = None
        self.reg_model = None
        self.features_used = []
        self.active_symbol = None
        
    @staticmethod
    def _normalize_regime_key(regime_label: str) -> str:
        """
        BUG-B + BUG-C FIX: Normalizes regime engine output strings
        (e.g. 'trend_alt_rotation', 'trend_btc_led', 'trend_neutral')
        to the 'TREND' or 'CHOP' keys used by saved model filenames.
        """
        r = str(regime_label).lower()
        if r.startswith('trend'):
            return 'TREND'
        elif r in ('chop', 'unstable', 'unknown'):
            return 'CHOP'
        elif r == 'downtrend':
            return 'CHOP'  # trade cautiously, use same model
        return 'CHOP'

    def train_global(self, combined_df: pd.DataFrame, target_clf_col: str = 'target_hit', target_short_col: str = 'short_target_hit', target_reg_col: str = 'mfe_r'):
        """
        Phase 2 & 3: Multi-Asset & Regime-Aware Training.
        Takes a combined dataset of multiple coins, splits by regime, and calls train() for each.
        GT-NEW-9: Prefers target_4h_2pct > prepump_target > target_hit as primary classifier.
        BUG-C FIX: Normalizes regime strings before splitting (trend_* → TREND, chop/unstable → CHOP).
        """
        if combined_df.empty or 'regime' not in combined_df.columns:
            logger.warning("Cannot perform global regime-aware training without 'regime' column.")
            return False
        
        # GT-RANKING: Highest priority is cross-sectional ranking (Leader Prediction)
        if 'is_top3_family_next_24h' in combined_df.columns:
            logger.info("[ML] GT-RANKING: Using is_top3_family_next_24h as primary classifier (Family Leader)")
            target_clf_col = 'is_top3_family_next_24h'
        elif 'is_top_decile_family_next_12h' in combined_df.columns:
            logger.info("[ML] GT-RANKING: Using is_top_decile_family_next_12h as primary classifier (Family Setup)")
            target_clf_col = 'is_top_decile_family_next_12h'
        elif 'is_top3_overall' in combined_df.columns:
            logger.info("[ML] GT-RANKING: Using is_top3_overall as primary classifier (Leader Discovery)")
            target_clf_col = 'is_top3_overall'
        elif 'target_early_big' in combined_df.columns:
            logger.info("[ML] GT-GOLD-2: Using target_early_big as primary (1h/1.5% + 24h/10% gainer)")
            target_clf_col = 'target_early_big'
        elif 'target_4h_2pct' in combined_df.columns:
            logger.info("[ML] GT-NEW-9: Using target_4h_2pct as primary classifier (fastest pre-pump signal)")
            target_clf_col = 'target_4h_2pct'
        elif 'prepump_target' in combined_df.columns:
            logger.info("[ML] GT #7: Using prepump_target as primary classifier (5%/24bar pre-pump signal)")
            target_clf_col = 'prepump_target'
            
        logger.info(f"Starting Global Multi-Asset Training on {len(combined_df)} total samples.")
        
        # BUG-C FIX: Normalize regime labels before splitting
        combined_df = combined_df.copy()
        combined_df['_regime_key'] = combined_df['regime'].apply(self._normalize_regime_key)
        
        df_trend = combined_df[combined_df['_regime_key'] == 'TREND'].sort_values('timestamp').reset_index(drop=True)
        df_chop  = combined_df[combined_df['_regime_key'] == 'CHOP'].sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"  Regime split → TREND: {len(df_trend)} rows | CHOP: {len(df_chop)} rows")
        
        success_trend = False
        success_chop = False
        
        if len(df_trend) > 1000:
            logger.info("\n" + "="*50 + "\nEquipping GLOBAL_TREND Model\n" + "="*50)
            success_trend = self.train(df_trend, target_clf_col, target_short_col, target_reg_col, symbol="GLOBAL_TREND")
            
        if len(df_chop) > 1000:
            logger.info("\n" + "="*50 + "\nEquipping GLOBAL_CHOP Model\n" + "="*50)
            success_chop = self.train(df_chop, target_clf_col, target_short_col, target_reg_col, symbol="GLOBAL_CHOP")
            
        return success_trend or success_chop

    def train(self, df_history: pd.DataFrame, target_clf_col: str = 'target_hit', target_short_col: str = 'short_target_hit', target_reg_col: str = 'mfe_r', feature_cols: List[str] = None, symbol: str = "GLOBAL", save: bool = True):
        """
        Train the ranking models (Long Clf, Short Clf, Regressor, and Ranker) using Walk-Forward Validation.
        GT-REDESIGN: Now explicitly supports ranking targets for Leader Discovery.
        """
        if df_history.empty:
            logger.warning("Insufficient data for ML training.")
            return False
            
        # Determine targets
        if target_clf_col not in df_history.columns:
            target_clf_col = 'target_hit' if 'target_hit' in df_history.columns else df_history.columns[-1]

        # GT-REDESIGN: Ranking Target Selection
        target_rank_col = target_reg_col if target_reg_col in df_history.columns else None
        if not target_rank_col:
            for cand in ['future_24h_rank_in_family_pct', 'future_24h_rank_pct', 'is_top3_overall']:
                if cand in df_history.columns:
                    target_rank_col = cand
                    break
        
        self._train_meta = {
            "symbol": symbol,
            "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "n_samples_total": len(df_history),
        }
        
        if feature_cols is None:
            exclude = [
                'timestamp', 'symbol', 'regime', 'family', 'open_time', 'close_time',
                'target_hit', 'short_target_hit', 'is_hyper_gainer', 'target_early_big',
                'future_6h_return', 'future_12h_return', 'future_24h_return',
                'pnl_r', 'entry_price', 'tp1', 'tp2', 'stop', 'exit_type', 
                'mae_r', 'mfe_r', 'quote_volume', 
                'future_24h_rank_pct', 'future_24h_rank_in_family_pct',
                'is_family_leader', 'family_relative_return', 'is_top3_overall',
                'future_6h_rank_in_family', 'future_12h_rank_in_family', 'future_24h_rank_in_family',
                'future_6h_rank_in_family_pct', 'future_12h_rank_in_family_pct', 'future_24h_rank_in_family_pct',
                'is_top3_family_next_24h', 'is_top_decile_family_next_12h',
                'is_top3_in_family_return_6h', 'is_top3_in_family_return_12h', 'is_top3_in_family_return_24h',
                'is_top_decile_in_family_return_6h', 'is_top_decile_in_family_return_12h', 'is_top_decile_in_family_return_24h',
                target_clf_col, target_short_col, target_reg_col # Dynamic targets
            ]
            self.features_used = [c for c in df_history.columns if pd.api.types.is_numeric_dtype(df_history[c]) and c not in exclude]
        else:
            self.features_used = feature_cols
            
        X = df_history[self.features_used]
        y_long = df_history[target_clf_col]
        y_rank = df_history[target_rank_col] if target_rank_col else y_long
        
        # ── Walk-Forward Validation (4 Folds) ──────────────────────────────────
        n_splits = 4
        if len(X) < 500: n_splits = 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        long_auc_scores, mse_scores = [], []
        best_long_auc_seen = 0.0
        best_mse_seen = float('inf')
        final_long_clf = None
        final_reg = None
        
        logger.info(f"[{symbol}] WF-CV ({n_splits} folds) | {len(X)} samples | {len(self.features_used)} features")
        
        for train_index, test_index in tscv.split(X):
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
            y_train_long, y_valid_long = y_long.iloc[train_index], y_long.iloc[test_index]
            
            # 1. Train LONG Classifier
            if len(np.unique(y_train_long)) >= 2 and len(np.unique(y_valid_long)) >= 2:
                num_neg = np.sum(y_train_long == 0)
                num_pos = np.sum(y_train_long == 1)
                spw_long = float(num_neg / num_pos) if num_pos > 0 else 1.0
                
                # Use params from config if available (via a hacky way since train() doesn't take config yet)
                # In a real refactor, we'd pass config to __init__ and use it here.
                # For now, let's keep it simple or assume defaults.
                params = {
                    "n_estimators": 200, "learning_rate": 0.03, "max_depth": 6,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "scale_pos_weight": min(spw_long, 30.0),
                    "min_child_samples": 20, "n_jobs": 4, "random_state": 42, "verbose": -1
                }
                
                clf_long = lgb.LGBMClassifier(**params)
                clf_long.fit(X_train, y_train_long, eval_set=[(X_valid, y_valid_long)], 
                             callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
                
                cal_long = CalibratedClassifierCV(clf_long, cv='prefit', method='isotonic')
                cal_long.fit(X_valid, y_valid_long)
                preds = cal_long.predict_proba(X_valid)[:, 1] if cal_long.predict_proba(X_valid).shape[1] > 1 else np.zeros(len(X_valid))
                
                if np.any(preds):
                    fold_long_auc = roc_auc_score(y_valid_long, preds)
                    long_auc_scores.append(fold_long_auc)
                    if fold_long_auc > best_long_auc_seen:
                        best_long_auc_seen = fold_long_auc
                        final_long_clf = cal_long

            # 2. Train RANKER (The 'Who' Leader Prediction)
            if target_rank_col and len(X_train) > 100:
                ranker = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=200, learning_rate=0.03, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                    random_state=45, verbose=-1
                )
                ranker.fit(X_train, y_rank.iloc[train_index], eval_set=[(X_valid, y_rank.iloc[test_index])], 
                           callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
                
                fold_rank_mse = mean_squared_error(y_rank.iloc[test_index], ranker.predict(X_valid))
                if fold_rank_mse < best_mse_seen:
                     best_mse_seen = fold_rank_mse
                     final_reg = ranker
            
        if final_long_clf is None and final_reg is None:
            logger.warning(f"Skipping {symbol}: Failed to complete valid CV folds.")
            return False
            
        cv_long_auc = np.mean(long_auc_scores) if long_auc_scores else 0.5
        
        logger.info(f"[{symbol}] WF-CV Result -> Long AUC: {cv_long_auc:.4f} | Best Rank MSE: {best_mse_seen:.4f}")
        
        self.long_clf_model = final_long_clf
        self.reg_model = final_reg
        self.last_metrics = {"long_auc": round(cv_long_auc, 4), "rank_mse": round(best_mse_seen, 4), "folds_completed": n_splits}
        
        self._train_meta["n_samples_train"] = len(X_train) if 'X_train' in locals() else len(X)
        self._train_meta["n_features"] = len(self.features_used)
        self._train_meta["auc"] = round(cv_long_auc, 4)
        self._train_meta["wf_folds"] = n_splits
        
        if save:
            self.save_models(symbol)
        
        # --- STAGE 2 & 3: Permutation Importance & Zero-Drop ---
        # Calculate permutation importance on the last validation fold
        # to get a true measure of feature value (resistant to cardinality bias)
        if final_long_clf is not None and 'X_valid' in locals() and 'y_valid_long' in locals():
            logger.info(f"[{symbol}] Stage 2 & 3: Calculating Permutation Importance...")
            try:
                pi = permutation_importance(
                    final_long_clf,
                    X_valid, y_valid_long,
                    n_repeats=5,  # 5 is enough for stability, 10 is too slow
                    n_jobs=1,     # CRITICAL: Prevent CPU locking / Streamlit event loop crash
                    random_state=42,
                    scoring='roc_auc'
                )
                
                # Filter out zero/negative importance features
                valid_features = []
                importances = []
                for idx, imp_val in enumerate(pi.importances_mean):
                    if imp_val > 0.0001:  # Must have some positive impact
                        valid_features.append(self.features_used[idx])
                        importances.append(float(imp_val))
                
                if len(valid_features) > 5:
                    logger.info(f"[{symbol}] Identified {len(valid_features)} important features via Permutation Importance.")
                    # Save the clean permutation dictionary
                    imp_dict = dict(zip(valid_features, importances))
                    joblib.dump(imp_dict, os.path.join(self.model_dir, f'feature_importance_{symbol}.pkl'))
                else:
                    logger.info(f"[{symbol}] Permutation left too few features, skipping importance save.")
            except Exception as e:
                logger.error(f"[{symbol}] Permutation importance failed: {e}")

        return True

    def train_family_ranker(self, combined_df: pd.DataFrame):
        """
        Phase 4: Global Family-Ranker Training.
        Trains three specialized models for DeFi Leadership Discovery.
        1. Leader Classifier: is_top3_family_next_24h
        2. Setup Quality Classifier: is_top_decile_family_next_12h
        3. Holdability Regressor: future_24h_rank_in_family (or pct)
        """
        if combined_df.empty:
            logger.warning("[ML] Cannot train family-ranker on empty dataset.")
            return False
            
        logger.info(f"\n{'='*50}\nTRAINING GLOBAL FAMILY-RANKER (DeFi Edition)\n{'='*50}")
        logger.info(f"Data Source: {len(combined_df)} master rows across multiple DeFi families.")
        
        # Mapping targets from Patch 4
        t_leader = 'is_top3_family_next_24h'
        t_setup  = 'is_top_decile_family_next_12h'
        t_hold   = 'future_24h_rank_in_family_pct' # Pct is more stable for regressor
        
        # 1/3 Train Leader Classifier
        logger.info(f"\n[1/3] Training Leader Classifier ({t_leader})...")
        self.train(combined_df, target_clf_col=t_leader, symbol="FAMILY_LEADER")
        
        # 2/3 Train Setup Quality Classifier
        logger.info(f"\n[2/3] Training Setup Quality Classifier ({t_setup})...")
        self.train(combined_df, target_clf_col=t_setup, symbol="FAMILY_SETUP")
        
        # 3/3 Train Holdability Regressor
        logger.info(f"\n[3/3] Training Holdability Regressor ({t_hold})...")
        self.train(combined_df, target_reg_col=t_hold, symbol="FAMILY_HOLD")
        
        return True

    def save_models(self, symbol="GLOBAL"):
        os.makedirs(self.model_dir, exist_ok=True)
        
        def _atomic_dump(obj, filename):
            target_path = os.path.join(self.model_dir, filename)
            tmp_path = target_path + ".tmp"
            try:
                joblib.dump(obj, tmp_path)
                os.replace(tmp_path, target_path)
            except Exception as e:
                logger.error(f"[ML-ERROR] Atomic save failed for {filename}: {e}")

        if self.long_clf_model:
            _atomic_dump(self.long_clf_model, f'lgb_classifier_long_{symbol}.pkl')
        if self.short_clf_model:
            _atomic_dump(self.short_clf_model, f'lgb_classifier_short_{symbol}.pkl')
        if self.reg_model:
            _atomic_dump(self.reg_model, f'lgb_regressor_{symbol}.pkl')
        
        if self.features_used:
            _atomic_dump(self.features_used, f'features_used_{symbol}.pkl')
            if hasattr(self, 'last_metrics'):
                _atomic_dump(self.last_metrics, f'metrics_{symbol}.pkl')

        meta_path = os.path.join(self.model_dir, f'metadata_{symbol}.json')
        tmp_meta = meta_path + ".tmp"
        meta = getattr(self, '_train_meta', {})
        meta['symbol'] = symbol
        try:
            with open(tmp_meta, 'w') as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp_meta, meta_path)
        except Exception as e:
            logger.error(f"[ML-ERROR] Atomic save failed for metadata_{symbol}.json: {e}")
        logger.info(f"Models saved for {symbol}")
        self.active_symbol = symbol
        
    def load_family_ranker_models(self):
        """Phase 4: Specialized loader for family-ranker suite."""
        try:
            # 1. Load Leader (top3)
            self.model_leader = joblib.load(os.path.join(self.model_dir, 'lgb_classifier_long_FAMILY_LEADER.pkl'))
            # 2. Load Setup (decile/target)
            self.model_setup = joblib.load(os.path.join(self.model_dir, 'lgb_classifier_long_FAMILY_SETUP.pkl'))
            # 3. Load Hold (regressor)
            self.model_hold = joblib.load(os.path.join(self.model_dir, 'lgb_regressor_FAMILY_HOLD.pkl'))
            
            # Use features from leader (all three should use same pool)
            self.features_used = joblib.load(os.path.join(self.model_dir, 'features_used_FAMILY_LEADER.pkl'))
            logger.info("[ML] Successfully loaded family-ranker suite.")
            return True
        except Exception as e:
            logger.error(f"[ML] Error loading family-ranker suite: {e}")
            return False

    def load_models(self, symbol="GLOBAL"):
        """Loads specific regime models. Only reloads if symbol changed to save disk I/O."""
        if self.active_symbol == symbol:
            return True

        clf_long_path  = os.path.join(self.model_dir, f'lgb_classifier_long_{symbol}.pkl')
        clf_short_path = os.path.join(self.model_dir, f'lgb_classifier_short_{symbol}.pkl')
        reg_path       = os.path.join(self.model_dir, f'lgb_regressor_{symbol}.pkl')
        feat_path      = os.path.join(self.model_dir, f'features_used_{symbol}.pkl')
        imp_path       = os.path.join(self.model_dir, f'feature_importance_{symbol}.pkl')

        # Legacy fallback if new long/short convention isn't there yet
        legacy_clf_path = os.path.join(self.model_dir, f'lgb_classifier_{symbol}.pkl')
        if not os.path.exists(clf_long_path) and os.path.exists(legacy_clf_path):
            clf_long_path = legacy_clf_path

        if os.path.exists(clf_long_path) and os.path.exists(feat_path):
            try:
                self.long_clf_model  = joblib.load(clf_long_path)
                self.short_clf_model = joblib.load(clf_short_path) if os.path.exists(clf_short_path) else None
                self.reg_model       = joblib.load(reg_path) if os.path.exists(reg_path) else None
                all_features         = joblib.load(feat_path)
                
                # BUG-FIX HEALING: If this coin's feature list was corrupted (sliced during previous bug)
                # we try to restore it from known 127-feature pools (ALL.p, GLOBAL_TREND, BTC.p etc.)
                healing_source = None
                healing_candidates = ['features_used_ALL.p.pkl', 'features_used_ALL.pkl', 'features_used_GLOBAL_TREND.pkl', 'features_used_BTC.p.pkl']
                
                if len(all_features) < 100: # We know the full set is 127
                    for cand in healing_candidates:
                        cand_path = os.path.join(self.model_dir, cand)
                        if os.path.exists(cand_path):
                            cand_features = joblib.load(cand_path)
                            if len(cand_features) == 127:
                                healing_source = cand_features
                                break
                    
                    if healing_source:
                        logger.info(f"[ML] HEALING: {symbol} feature list was corrupted ({len(all_features)}). Restored from {cand} (127).")
                        all_features = healing_source
                        joblib.dump(all_features, feat_path)

                # GT-GOLD-9: We still identify important features for metadata/analysis
                # but we MUST NOT slice self.features_used for prediction if the model 
                # was trained on more. LightGBM requires exact dimension match.
                self.features_used = all_features
                if os.path.exists(imp_path):
                    try:
                        imp_dict = joblib.load(imp_path)
                        valid_imp = {k: v for k, v in imp_dict.items() if k in all_features}
                        if len(valid_imp) >= 10:
                            # We keep track of top features for display/logging if needed, 
                            # but we do NOT change self.features_used used for X_pred
                            self.top_features_metasy = sorted(valid_imp, key=valid_imp.get, reverse=True)[:15]
                        pass
                    except Exception:
                        pass

                self.active_symbol = symbol
                return True
            except Exception:
                pass
        return False
        
    def _safe_predict_proba(self, model, X):
        """Robustly extracts probability for class 1, even if only one class was trained."""
        if model is None: 
            return np.zeros(len(X))
        try:
            probs = model.predict_proba(X)
            if probs.shape[1] > 1:
                return probs[:, 1]
            else:
                # If only one class is present in the model
                if hasattr(model, 'classes_') and model.classes_[0] == 1:
                    return np.ones(len(X))
                return np.zeros(len(X))
        except Exception:
            return np.zeros(len(X))

    def ensure_canonical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantees that the DataFrame contains all canonical ML columns.
        Fills missing ones with 0.5 (Neutral) or calls internal heuristic fallback.
        """
        required = ["leader_prob", "setup_conversion_prob", "holdability_score", "ml_rank_score", "ml_explanation"]
        missing = [c for c in required if c not in df.columns or df[c].isna().all()]
        
        if not missing:
            return df
            
        logger.info(f"[ML-CONTRACT] Filling missing canonical columns: {missing}")
        return self.heuristic_fallback(df)

    def heuristic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internalized heuristic fallback logic to provide high-quality estimates
        when ML models (especially family-ranker suite) are unavailable.
        """
        df = df.copy()
        
        def _safe_series(col: str, default: float = 0.5) -> pd.Series:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        def _norm01(s: pd.Series, neutral: float = 0.5) -> pd.Series:
            if s.isna().all(): return pd.Series(neutral, index=s.index)
            lo, hi = s.min(), s.max()
            if abs(hi - lo) < 1e-12: return pd.Series(neutral, index=s.index)
            return (s - lo) / (hi - lo)

        # Base components for heuristic
        ml_rank         = _norm01(_safe_series("ml_rank_score", 0.5))
        entry_readiness = _norm01(_safe_series("entry_readiness", 0.5))
        fakeout_risk    = _norm01(_safe_series("fakeout_risk", 0.5))
        low_fakeout     = 1.0 - fakeout_risk
        
        # Peer metrics from Discovery/Leadership
        peer_rank       = _norm01(_safe_series("peer_rank", 0.5))
        peer_momentum   = _norm01(_safe_series("peer_momentum", 0.0))
        rs_div          = _norm01(_safe_series("rs_divergence_persistence", 0.0))

        # Core Heuristic Score (0.0 to 1.0)
        h_score = (
            0.22 * ml_rank +
            0.22 * entry_readiness +
            0.18 * low_fakeout +
            0.12 * peer_rank +
            0.10 * peer_momentum +
            0.08 * rs_div +
            0.08 * _safe_series("quiet_expansion", 0.0).clip(0, 1)
        ).clip(0.0, 1.0)

        # Map to canonical columns if missing
        if "leader_prob" not in df.columns or df["leader_prob"].isna().all():
            df["leader_prob"] = (0.35 + 0.55 * h_score).clip(0.05, 0.95)
            
        if "setup_conversion_prob" not in df.columns or df["setup_conversion_prob"].isna().all():
            df["setup_conversion_prob"] = (0.30 + 0.60 * h_score).clip(0.05, 0.95)
            
        if "holdability_score" not in df.columns or df["holdability_score"].isna().all():
            df["holdability_score"] = (100.0 * (0.45 * h_score + 0.30 * entry_readiness + 0.25 * low_fakeout)).clip(0, 100)
            
        if "ml_rank_score" not in df.columns or df["ml_rank_score"].isna().all():
            df["ml_rank_score"] = h_score * 100.0
            
        if "ml_explanation" not in df.columns or df["ml_explanation"].isna().all():
            df["ml_explanation"] = "[HEURISTIC_FALLBACK] Rule-based consensus"
            
        # Backward compatibility aliases
        if "setup_quality_prob" not in df.columns:
            df["setup_quality_prob"] = df["setup_conversion_prob"]
        if "hold_quality" not in df.columns:
            df["hold_quality"] = df["holdability_score"]

        return df

    def rank_candidates(self, candidates: pd.DataFrame, top_n: int = 5, use_family_ranker: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        Dynamically loads the correct Regime-Aware model (GLOBAL_TREND or GLOBAL_CHOP)
        and scores candidates for both Long and Short EV.
        GT-REDESIGN: now supports Phase 4 tri-score output.
        Guarantees canonical columns via ensure_canonical_columns.
        """
        if candidates.empty:
            return candidates, []
            
        candidates = candidates.copy()
        
        # Phase 4 Trial: Global Family Ranker override
        if use_family_ranker:
            if not hasattr(self, 'model_leader'):
                if not self.load_family_ranker_models():
                    use_family_ranker = False # Fallback to standard
            
        scored_groups = []
        for sym, group in candidates.groupby('symbol'):
            group = group.copy()
            
            if use_family_ranker and hasattr(self, 'model_leader'):
                # Phase 4: Tri-Score Output
                try:
                    # Robust feature mapping
                    missing = [f for f in self.features_used if f not in group.columns]
                    if missing:
                        for f in missing: group[f] = 0.0
                        
                    X_pred = group[self.features_used].fillna(0.0)
                    
                    group['leader_prob'] = self._safe_predict_proba(self.model_leader, X_pred)
                    group['setup_conversion_prob'] = self._safe_predict_proba(self.model_setup, X_pred)
                    
                    # Predicted Rank Pct (0.0 is best, 1.0 is worst) -> Convert to 0-100 Holdability Score
                    rank_pct = self.model_hold.predict(X_pred) if self.model_hold else np.ones(len(X_pred))
                    group['holdability_score'] = (1.0 - np.clip(rank_pct, 0, 1)) * 100
                    
                    # Combined Score for sorting
                    group['ml_rank_score'] = (group['leader_prob'] * 0.5 + group['setup_conversion_prob'] * 0.3 + (group['holdability_score']/100) * 0.2) * 100
                    group['ml_explanation'] = group.apply(
                        lambda r: f"[FAMILY_RANKER] Prob: {r['leader_prob']:.1%} | Setup: {r['setup_conversion_prob']:.1%} | Hold: {r['holdability_score']:.1f}", axis=1
                    )
                except Exception as e:
                    logger.error(f"[ML-ERROR] Family-Ranker failed for {sym}: {e}")
                    group['ml_rank_score'] = 50.0
                    use_family_ranker = False # Fallback
            
            if not use_family_ranker: # Standard or fallback logic
                # 1. Try per-asset model first
                loaded = self.load_models(sym)
                model_label = sym if loaded else "GLOBAL"
    
                # 2. If no asset model, try regime-aware global model
                if not loaded:
                    raw_regime = group.iloc[0].get('regime', 'chop')
                    regime_key = self._normalize_regime_key(raw_regime)
                    model_label = f"GLOBAL_{regime_key}"
                    loaded = self.load_models(model_label)
                
                # 3. Last fallback to generic GLOBAL
                if not loaded:
                    loaded = self.load_models("GLOBAL")
                    model_label = "GLOBAL"
                    
                if loaded and self.long_clf_model is not None:
                    # Check for missing features
                    missing = [f for f in self.features_used if f not in group.columns]
                    if missing:
                        for f in missing: group[f] = 0.0
                    
                    try:
                        X_pred = group[self.features_used].fillna(0.0)
                        group['probability_long'] = self.long_clf_model.predict_proba(X_pred)[:, 1] if hasattr(self.long_clf_model, 'predict_proba') else 0.5
                        group['future_rank_pct'] = self.reg_model.predict(X_pred) if self.reg_model else 0.5
                        group['ml_rank_score'] = (group['future_rank_pct'] * 0.6 + group['probability_long'] * 0.4) * 100
                        group['ml_explanation'] = group.apply(
                            lambda r: f"[{model_label}] RankProb: {r['ml_rank_score']:.1f} | PredRank: {r['future_rank_pct']:.1%}", axis=1
                        )
                    except Exception as e:
                        logger.error(f"[ML] Group Error ({sym}): {e}")
                        group['ml_rank_score'] = 50.0
                        group['ml_explanation'] = f"ML Error ({sym})"
                else:
                    group['ml_rank_score'] = 50.0
                    group['ml_explanation'] = f"No Model ({model_label})"
            
            scored_groups.append(group)
        
        candidates = pd.concat(scored_groups)
        
        # FINAL ENSURE CANONICAL
        candidates = self.ensure_canonical_columns(candidates)
        
        candidates = candidates.sort_values(by='ml_rank_score', ascending=False)
        top_candidates = candidates.head(top_n)['symbol'].tolist()
        
        return candidates, top_candidates
