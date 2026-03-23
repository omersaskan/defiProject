import sys
import os
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from defihunter.engines.adaptive import AdaptiveWeightsEngine
from defihunter.engines.thresholds import ThresholdResolutionEngine

def test_adaptive_thresholds():
    print("--- Testing Adaptive Thresholds ---")
    
    # 1. Initialize Adaptive Engine
    test_path = "configs/test_adaptive_thresholds.yaml"
    if os.path.exists(test_path):
        os.remove(test_path)
        
    adaptive_engine = AdaptiveWeightsEngine(persistence_path=test_path)
    
    # Initial states
    print(f"Initial min_score: {adaptive_engine.current_thresholds['min_score']}")
    print(f"Initial min_volume: {adaptive_engine.current_thresholds['min_volume']}")
    
    # 2. Simulate Performance History
    # Scenario: Losers have a mean score of 51 (very close to min 50 threshold)
    # Winners have a mean score of 60. Min score should adapt upward.
    # Losers volume is very low, Winners volume is high.
    
    rows = []
    # 10 Winners
    for _ in range(10):
        rows.append({
            'pnl_r': 1.5,
            'total_score': 60,
            'quote_volume': 50_000_000
        })
    # 10 Losers
    for _ in range(10):
        rows.append({
            'pnl_r': -1.0,
            'total_score': 51,
            'quote_volume': 10_000_000
        })
        
    perf_df = pd.DataFrame(rows)
    
    print("\nSimulating weight and threshold update...")
    adaptive_engine.update_weights(perf_df, current_regime="trend")
    
    print(f"Adapted min_score: {adaptive_engine.current_thresholds['min_score']}")
    print(f"Adapted min_volume: {adaptive_engine.current_thresholds['min_volume']}")
    
    assert adaptive_engine.current_thresholds['min_score'] == 51 # Should bump up by 1
    assert adaptive_engine.current_thresholds['min_volume'] == 10_500_000 # Should bump up by 5%
    
    # 3. Verify ThresholdResolutionEngine overriding base config
    print("\nReading with ThresholdResolutionEngine...")
    
    # Mock base thresholds from config
    mock_config = {
        "min_score": 50,
        "min_volume": 10_000_000
    }
    
    resolver = ThresholdResolutionEngine(thresholds_config=mock_config, adaptive_path=test_path)
    resolved = resolver.resolve_thresholds(regime="trend", family="beta")
    
    print(f"Resolved Engine min_score: {resolved['min_score']} (Expected 51)")
    print(f"Resolved Engine min_volume: {resolved['min_volume']} (Expected 10_500_000)")
    
    assert resolved['min_score'] == 51
    assert resolved['min_volume'] == 10_500_000
    print("Test Passed!")

if __name__ == "__main__":
    test_adaptive_thresholds()
    import os
    if os.path.exists("configs/test_adaptive_thresholds.yaml"):
        os.remove("configs/test_adaptive_thresholds.yaml")
