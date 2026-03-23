import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from defihunter.engines.adaptive import AdaptiveWeightsEngine

def test_rollback():
    engine = AdaptiveWeightsEngine(persistence_path="configs/test_adaptive_weights.yaml")
    
    # 1. Create a "good" snapshot
    good_metrics = {"win_rate": 0.50, "expectancy": 0.5, "sample_size": 20}
    engine.current_weights = {"trend_score": 1.2, "expansion_score": 0.9}
    engine.snapshot_weights(good_metrics)
    
    # 2. Simulate bad performance (expectancy < -0.3)
    bad_rows = []
    for _ in range(20):
        bad_rows.append({"pnl_r": -1.0})
    bad_df = pd.DataFrame(bad_rows)
    
    print("\nSimulating bad performance...")
    rolled_back = engine.evaluate_and_rollback(bad_df)
    
    print(f"Rollback Triggered: {rolled_back}")
    print(f"Restored Weights: {engine.current_weights}")
    
    # Assert
    assert rolled_back == True
    assert engine.current_weights["trend_score"] == 1.2
    
if __name__ == "__main__":
    test_rollback()
    import os
    if os.path.exists("configs/test_adaptive_weights.yaml"):
        os.remove("configs/test_adaptive_weights.yaml")
