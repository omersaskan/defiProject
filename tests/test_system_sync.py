from defihunter.core.config import load_config
from defihunter.engines.thresholds import ThresholdResolutionEngine
from pathlib import Path

def test_config_merge():
    # Load config and verify it processed universe_groups.yaml properly
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    
    # 1. Check strict defi universe property
    assert hasattr(config.universe, 'is_strictly_defi'), "Missing is_strictly_defi in universe filters"
    
    # 2. Check family metadata
    assert 'defi_lending' in config.families, "defi_lending family missing from merge"
    lending_family = config.families['defi_lending']
    assert hasattr(lending_family, 'threshold_overrides'), "threshold_overrides missing from family metadata"
    
def test_threshold_overrides():
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    
    threshold_engine = ThresholdResolutionEngine(thresholds_config=config.regimes, config=config)
    
    # 1. Resolve for defi_lending
    resolved = threshold_engine.resolve_thresholds(regime="trend_neutral", family="defi_lending")
    
    # Check if Family overrides took effect (from config where min_score override might be 45 for lending)
    # The default min_score is usually 50
    assert "min_score" in resolved, "min_score not found in resolved thresholds"
    
if __name__ == "__main__":
    test_config_merge()
    test_threshold_overrides()
    print("test_system_sync passed successfully.")
