import yaml
import os
import pandas as pd
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class AdaptiveWeightsEngine:
    def __init__(self, persistence_path: str = "configs/adaptive_weights.yaml"):
        # Base neutral weights
        self.persistence_path = persistence_path
        self.current_weights = {
            "trend_score": 1.0,
            "expansion_score": 1.0,
            "participation_score": 1.0,
            "relative_leadership_score": 1.0
        }
        self.current_thresholds = {
            "min_score": 50,
            "min_volume": 10_000_000
        }
        self.snapshots = [] # History of weight configurations
        self.learning_rate = 0.05 
        self.load_weights()
        
    def load_weights(self):
        if os.path.exists(self.persistence_path):
            with open(self.persistence_path, 'r') as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    if 'current_weights' in data:
                        self.current_weights = data['current_weights']
                        self.current_thresholds = data.get('current_thresholds', self.current_thresholds)
                        self.snapshots = data.get('snapshots', [])
                    else:
                        # Legacy format support
                        self.current_weights = data
                        self.snapshots = []
                print(f"Loaded adapted weights: {self.current_weights} and thresholds: {self.current_thresholds}")

    def save_weights(self):
        data = {
            "current_weights": self.current_weights,
            "current_thresholds": self.current_thresholds,
            "snapshots": self.snapshots
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, 'w') as f:
            yaml.dump(data, f)
            
    def snapshot_weights(self, metrics: dict):
        """
        Saves the current weights along with the performance metrics that triggered them.
        BUG-7 FIX: Always retains at least 2 'good' snapshots (expectancy > 0) so
        rollback always has a valid target, even after 10+ bad trades in a row.
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "weights": self.current_weights.copy(),
            "thresholds": self.current_thresholds.copy(),
            "metrics": metrics
        }
        self.snapshots.append(snapshot)

        # BUG-7 FIX: Prune strategy — keep at most 10 snapshots BUT always keep
        # the 2 most recent snapshots where expectancy > 0 (good snapshots).
        if len(self.snapshots) > 10:
            good_snapshots = [s for s in self.snapshots if s.get('metrics', {}).get('expectancy', 0) > 0]
            bad_snapshots  = [s for s in self.snapshots if s not in good_snapshots]

            # Keep last 2 good snapshots + recent bad ones to fill up to 10
            protected = good_snapshots[-2:] if len(good_snapshots) >= 2 else good_snapshots
            remaining_slots = 10 - len(protected)
            recent_others   = [s for s in self.snapshots if s not in protected][-remaining_slots:]
            self.snapshots  = sorted(protected + recent_others, key=lambda s: s['timestamp'])

        self.save_weights()
        print(f"Snapshot saved. Total snapshots: {len(self.snapshots)}")

    def evaluate_and_rollback(self, performance_history: pd.DataFrame) -> bool:
        """
        Evaluates recent system performance. If it drops below a critical threshold (e.g. Win Rate < 30% 
        or negative expectancy over N trades), it rolls back to the last known "good" snapshot.
        """
        if performance_history.empty or len(performance_history) < 20:
            return False
            
        recent_trades = performance_history.tail(20)
        win_rate = len(recent_trades[recent_trades['pnl_r'] > 0]) / len(recent_trades)
        expectancy = recent_trades['pnl_r'].mean()
        
        # Degradation criteria: very low win rate OR high negative expectancy
        if win_rate < 0.30 or expectancy < -0.3:
            print(f"Performance Degradation Detected! Win Rate: {win_rate:.2%}, Expectancy: {expectancy:.2f}R")
            
            if self.snapshots:
                # Find the most recent snapshot where expectancy was positive
                # If none exist, just fallback to the oldest available (base state)
                valid_snapshots = [s for s in self.snapshots if s.get('metrics', {}).get('expectancy', 0) > 0]
                
                if valid_snapshots:
                    best_snapshot = valid_snapshots[-1]
                    print(f"Rolling back to good snapshot from {best_snapshot['timestamp']}")
                else:
                    best_snapshot = self.snapshots[0]
                    print("No purely positive snapshots found. Rolling back to oldest available state.")
                    
                self.current_weights = best_snapshot['weights'].copy()
                self.current_thresholds = best_snapshot.get('thresholds', self.current_thresholds).copy()
                
                # Prune history to prevent ping-ponging into the bad state we just left
                idx = self.snapshots.index(best_snapshot)
                self.snapshots = self.snapshots[:idx+1]
                
                self.save_weights()
                return True
            else:
                print("Degradation detected but no snapshots available for rollback. Resetting to default.")
                self.current_weights = {
                    "trend_score": 1.0,
                    "expansion_score": 1.0,
                    "participation_score": 1.0,
                    "relative_leadership_score": 1.0
                }
                self.current_thresholds = {
                    "min_score": 50,
                    "min_volume": 10_000_000
                }
                self.save_weights()
                return True
                
        return False

    def update_weights(self, performance_history: pd.DataFrame, current_regime: str) -> Dict[str, float]:
        """
        Adapts feature weights based on real recent performance within specific regimes.
        """
        if performance_history.empty or len(performance_history) < 20:
            return self.current_weights
            
        # Separate winners and losers
        winners = performance_history[performance_history['pnl_r'] >= 1.0]
        losers = performance_history[performance_history['pnl_r'] <= -0.5]
        
        if winners.empty: return self.current_weights
            
        features = ["trend_score", "expansion_score", "participation_score", "relative_leadership_score"]
        
        for feature in features:
            if feature not in performance_history.columns:
                continue
                
            mean_win = winners[feature].mean()
            mean_loss = losers[feature].mean() if not losers.empty else 0.0
            
            # If a feature is high in winners but low in losers, increase its weight
            diff_ratio = (mean_win - mean_loss) / (mean_win + mean_loss + 1e-8)
            
            drift = np.clip(diff_ratio * self.learning_rate, -self.learning_rate, self.learning_rate)
            new_weight = np.clip(self.current_weights[feature] * (1 + drift), 0.7, 1.5)
            self.current_weights[feature] = round(new_weight, 3)
            
        # 2. ADAPT THRESHOLDS
        # Check Total Score vs Win/Loss
        score_col = 'total_score' if 'total_score' in performance_history.columns else ('Total_Score' if 'Total_Score' in performance_history.columns else None)
        if score_col:
            win_mean_score = winners[score_col].mean()
            loss_mean_score = losers[score_col].mean() if not losers.empty else self.current_thresholds["min_score"]
            
            # If losers have an average score very close to the minimum (e.g. they barely passed to enter),
            # while winners had much higher scores, push min_score up.
            if loss_mean_score < (self.current_thresholds["min_score"] + 5) and win_mean_score > loss_mean_score:
                new_min_score = min(self.current_thresholds["min_score"] + 1, 65) # Cap at 65
                self.current_thresholds["min_score"] = int(new_min_score)
            # If win_rate is extremely high, loosen the threshold slightly to take more trades
            elif (len(winners) / len(performance_history)) > 0.60:
                new_min_score = max(self.current_thresholds["min_score"] - 1, 40) # Floor at 40
                self.current_thresholds["min_score"] = int(new_min_score)

        # Check Volume
        vol_col = 'quote_volume' if 'quote_volume' in performance_history.columns else ('volume' if 'volume' in performance_history.columns else None)
        if vol_col:
            win_mean_vol = winners[vol_col].mean()
            loss_mean_vol = losers[vol_col].mean() if not losers.empty else self.current_thresholds["min_volume"]
            
            if loss_mean_vol < win_mean_vol * 0.5:
                # Losers have much lower volume, raise the bar by 5%
                new_vol = self.current_thresholds["min_volume"] * 1.05
                self.current_thresholds["min_volume"] = min(int(new_vol), 50_000_000)
                
        # Snapshot the new state and the metrics that got us here
        metrics = {
            "win_rate": float(len(winners) / len(performance_history)),
            "expectancy": float(performance_history['pnl_r'].mean()),
            "sample_size": int(len(performance_history))
        }
        self.snapshot_weights(metrics)
        
        return self.current_weights

