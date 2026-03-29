import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from typing import List, Dict, Any, Optional
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from defihunter.utils.logger import logger
from defihunter.engines.ml.repository import ModelRepository

class MLTrainer:
    """
    GT-Institutional: Model Training & Walk-Forward Validation.
    Specialized for cross-sectional ranking and regime-aware training.
    """
    def __init__(self, repository: ModelRepository):
        self.repository = repository
        self.model_dir = repository.model_dir

    def train_global(self, combined_df: pd.DataFrame, 
                     target_clf_col: str = 'target_hit', 
                     target_short_col: str = 'short_target_hit', 
                     target_reg_col: str = 'mfe_r'):
        """Phase 2 & 3: Multi-Asset & Regime-Aware Training."""
        if combined_df.empty or 'regime' not in combined_df.columns:
            logger.warning("[ML-Trainer] Cannot perform global training without 'regime' column.")
            return False
        
        # Priority mapping for target columns
        targets = [
            'is_top3_family_next_24h', 'is_top_decile_family_next_12h',
            'is_top3_overall', 'target_early_big', 'target_4h_2pct', 'prepump_target'
        ]
        for t in targets:
            if t in combined_df.columns:
                logger.info(f"[ML-Trainer] Using {t} as primary classifier.")
                target_clf_col = t
                break
            
        logger.info(f"[ML-Trainer] Starting Global Training on {len(combined_df)} samples.")
        
        combined_df = combined_df.copy()
        combined_df['_regime_key'] = combined_df['regime'].apply(self.repository.normalize_regime_key)
        
        df_trend = combined_df[combined_df['_regime_key'] == 'TREND'].sort_values('timestamp').reset_index(drop=True)
        df_chop  = combined_df[combined_df['_regime_key'] == 'CHOP'].sort_values('timestamp').reset_index(drop=True)
        
        success_trend = False
        success_chop = False
        
        if len(df_trend) > 1000:
            logger.info("Equipping GLOBAL_TREND Model...")
            success_trend = self.train(df_trend, target_clf_col, target_short_col, target_reg_col, symbol="GLOBAL_TREND")
            
        if len(df_chop) > 1000:
            logger.info("Equipping GLOBAL_CHOP Model...")
            success_chop = self.train(df_chop, target_clf_col, target_short_col, target_reg_col, symbol="GLOBAL_CHOP")
            
        return success_trend or success_chop

    def train(self, df_history: pd.DataFrame, 
              target_clf_col: str = 'target_hit', 
              target_short_col: str = 'short_target_hit', 
              target_reg_col: str = 'mfe_r', 
              feature_cols: List[str] = None, 
              symbol: str = "GLOBAL", 
              save: bool = True):
        """Train models using Walk-Forward Validation."""
        if df_history.empty:
            logger.warning(f"[ML-Trainer] No data for {symbol}")
            return False
            
        if target_clf_col not in df_history.columns:
            target_clf_col = 'target_hit' if 'target_hit' in df_history.columns else df_history.columns[-1]

        target_rank_col = target_reg_col if target_reg_col in df_history.columns else None
        if not target_rank_col:
            for cand in ['future_24h_rank_in_family_pct', 'future_24h_rank_pct', 'is_top3_overall']:
                if cand in df_history.columns:
                    target_rank_col = cand
                    break
        
        train_meta = {
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
                target_clf_col, target_short_col, target_reg_col
            ]
            feature_cols = [c for c in df_history.columns if pd.api.types.is_numeric_dtype(df_history[c]) and c not in exclude]
            
        X = df_history[feature_cols]
        y_long = df_history[target_clf_col]
        y_rank = df_history[target_rank_col] if target_rank_col else y_long
        
        n_splits = 4 if len(X) >= 500 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        long_auc_scores, mse_scores = [], []
        best_long_auc, best_mse = 0.0, float('inf')
        final_long_clf, final_reg = None, None
        
        logger.info(f"[{symbol}] WF-CV ({n_splits} folds) | {len(X)} samples | {len(feature_cols)} features")
        
        X_valid_last, y_valid_long_last = None, None

        for train_idx, test_idx in tscv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[test_idx]
            y_train_long, y_valid_long = y_long.iloc[train_idx], y_long.iloc[test_idx]
            X_valid_last, y_valid_long_last = X_valid, y_valid_long

            # 1. Train Classifier
            if len(np.unique(y_train_long)) >= 2 and len(np.unique(y_valid_long)) >= 2:
                num_neg = np.sum(y_train_long == 0)
                num_pos = np.sum(y_train_long == 1)
                spw = float(num_neg / num_pos) if num_pos > 0 else 1.0
                
                clf = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.03, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(spw, 30.0),
                    min_child_samples=20, n_jobs=4, random_state=42, verbose=-1
                )
                clf.fit(X_train, y_train_long, eval_set=[(X_valid, y_valid_long)], 
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
                
                cal = CalibratedClassifierCV(clf, cv='prefit', method='isotonic')
                cal.fit(X_valid, y_valid_long)
                
                preds = cal.predict_proba(X_valid)[:, 1] if cal.predict_proba(X_valid).shape[1] > 1 else np.zeros(len(X_valid))
                if np.any(preds):
                    auc = roc_auc_score(y_valid_long, preds)
                    long_auc_scores.append(auc)
                    if auc > best_long_auc:
                        best_long_auc = auc
                        final_long_clf = cal

            # 2. Train Regressor (Ranker)
            if target_rank_col and len(X_train) > 100:
                ranker = lgb.LGBMRegressor(
                    objective='regression', n_estimators=200, learning_rate=0.03, max_depth=7,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=45, verbose=-1
                )
                ranker.fit(X_train, y_rank.iloc[train_idx], eval_set=[(X_valid, y_rank.iloc[test_idx])], 
                           callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
                
                mse = mean_squared_error(y_rank.iloc[test_idx], ranker.predict(X_valid))
                if mse < best_mse:
                    best_mse = mse
                    final_reg = ranker
        
        if final_long_clf is None and final_reg is None:
            logger.warning(f"[ML-Trainer] Failed CV for {symbol}")
            return False
            
        metrics = {
            "long_auc": round(np.mean(long_auc_scores) if long_auc_scores else 0.5, 4),
            "rank_mse": round(best_mse, 4),
            "folds": n_splits
        }
        
        train_meta.update({
            "n_samples_train": len(X_train),
            "n_features": len(feature_cols),
            "auc": metrics["long_auc"],
            "wf_folds": n_splits
        })
        
        if save:
            self.repository.save_models(symbol, final_long_clf, None, final_reg, feature_cols, metrics, train_meta)
            
        # Permutation Importance
        if final_long_clf and X_valid_last is not None:
            self._save_importance(symbol, final_long_clf, X_valid_last, y_valid_long_last, feature_cols)

        return True

    def train_family_ranker(self, combined_df: pd.DataFrame):
        """Phase 4: Global Family-Ranker Training."""
        if combined_df.empty: return False
        
        logger.info(f"TRAINING GLOBAL FAMILY-RANKER ({len(combined_df)} rows)")
        
        self.train(combined_df, target_clf_col='is_top3_family_next_24h', symbol="FAMILY_LEADER")
        self.train(combined_df, target_clf_col='is_top_decile_family_next_12h', symbol="FAMILY_SETUP")
        self.train(combined_df, target_reg_col='future_24h_rank_in_family_pct', symbol="FAMILY_HOLD")
        
        return True

    def _save_importance(self, symbol, model, X_val, y_val, feature_cols):
        try:
            pi = permutation_importance(model, X_val, y_val, n_repeats=5, n_jobs=1, random_state=42, scoring='roc_auc')
            imp_dict = {feature_cols[i]: float(pi.importances_mean[i]) 
                        for i in range(len(feature_cols)) if pi.importances_mean[i] > 0.0001}
            
            if len(imp_dict) > 5:
                joblib.dump(imp_dict, os.path.join(self.model_dir, f'feature_importance_{symbol}.pkl'))
        except Exception as e:
            logger.error(f"[ML-Trainer] Permutation Importance failed for {symbol}: {e}")
