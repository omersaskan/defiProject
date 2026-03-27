import yaml
import os

class ThresholdResolutionEngine:
    def __init__(self, thresholds_config: dict = None, config=None, adaptive_path: str = "configs/adaptive_weights.yaml"):
        self.thresholds_config = thresholds_config
        self.config = config
        self.adaptive_path = adaptive_path
        
    def resolve_thresholds(self, regime: str, family: str) -> dict:
        """
        Resolves thresholds based on precedence:
        1. Base thresholds
        2. Config overrides
        3. Regime overrides
        4. Family overrides
        5. Adaptive Systems Data (Highest Priority)
        """
        # 1. Base Defaults
        resolved = {
            "min_score": 50,
            "min_relative_leadership": 0,
            "min_volume": 10_000_000,
            "max_spread_bps": 15.0,
            "breakout_buffer_atr": 0.5,
            "retest_tolerance_atr": 0.3,
            "time_stop_bars": 24
        }
        
        # Pull from config if it exists
        if hasattr(self.thresholds_config, "min_volume"):
            resolved["min_volume"] = self.thresholds_config.min_volume
        elif isinstance(self.thresholds_config, dict):
            resolved["min_volume"] = self.thresholds_config.get("min_volume", resolved["min_volume"])
            
        # 2. Regime Overrides
        if hasattr(self.thresholds_config, "overrides") and regime in self.thresholds_config.overrides:
            override = self.thresholds_config.overrides[regime]
            for key in ["min_score", "min_relative_leadership", "min_volume", "max_spread_bps"]:
                val = getattr(override, key, None)
                if val is not None:
                    resolved[key] = val
        elif isinstance(self.thresholds_config, dict) and "overrides" in self.thresholds_config:
            override = self.thresholds_config["overrides"].get(regime, {})
            for key, val in override.items():
                if val is not None:
                    resolved[key] = val
                    
        # 4. Family Overrides (Config-Driven Patch 4)
        families_dict = {}
        if self.config and hasattr(self.config, 'families'):
            families_dict = self.config.families
        elif hasattr(self.thresholds_config, "families"):
            families_dict = self.thresholds_config.families
            
        if family in families_dict:
            family_config = families_dict[family]
            overrides = getattr(family_config, 'threshold_overrides', {})
            if isinstance(overrides, dict):
                for key, val in overrides.items():
                    if val is not None:
                        resolved[key] = val
            
        # 5. Adaptive System Overrides (Highest Priority)
        if os.path.exists(self.adaptive_path):
            try:
                with open(self.adaptive_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and 'current_thresholds' in data:
                        adapted = data['current_thresholds']
                        for key in ["min_score", "min_volume"]:
                            if key in adapted:
                                resolved[key] = adapted[key]
            except Exception as e:
                print(f"Failed to load adaptive thresholds: {e}")
                
        return resolved
