import yaml
import os
from defihunter.utils.logger import logger

class ThresholdResolutionEngine:
    def __init__(self, thresholds_config: dict = None, config=None, adaptive_path: str = "configs/adaptive_weights.yaml"):
        self.thresholds_config = thresholds_config
        self.config = config
        self.adaptive_path = adaptive_path
        self._adaptive_cache = {}
        self._load_adaptive_config()

    def _load_adaptive_config(self):
        """Pre-load adaptive thresholds to avoid repeated disk I/O during scan loops."""
        if self.adaptive_path and os.path.exists(self.adaptive_path):
            try:
                with open(self.adaptive_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and 'current_thresholds' in data:
                        self._adaptive_cache = data['current_thresholds']
                        logger.info(f"[Thresholds] Adaptive config loaded from {self.adaptive_path}")
            except Exception as e:
                logger.error(f"Failed to load adaptive thresholds from {self.adaptive_path}: {e}")

    def resolve_thresholds(self, regime: str, family: str, volatility: str = "normal") -> dict:
        """
        Resolves thresholds based on precedence:
        1. Base thresholds
        2. Config overrides
        3. Regime overrides
        4. Family overrides
        5. Adaptive Systems Data (Highest Priority - Cached)
        """
        # 1. Base Defaults
        resolved = {
            "min_score": getattr(self.thresholds_config, 'min_score', 50),
            "min_relative_leadership": getattr(self.thresholds_config, 'min_relative_leadership', 0),
            "min_volume": getattr(self.thresholds_config, 'min_volume', 10_000_000),
            "max_spread_bps": getattr(self.thresholds_config, 'max_spread_bps', 15.0),
            "breakout_buffer_atr": getattr(self.thresholds_config, 'breakout_buffer_atr', 0.5),
            "retest_tolerance_atr": getattr(self.thresholds_config, 'retest_tolerance_atr', 0.3),
            "time_stop_bars": getattr(self.thresholds_config, 'time_stop_bars', 24)
        }
        
        # 2. Regime Overrides
        lookup_regime = regime
        
        # Determine the override dictionary
        if hasattr(self.thresholds_config, "overrides"):
            overrides_dict = self.thresholds_config.overrides
        elif isinstance(self.thresholds_config, dict) and "overrides" in self.thresholds_config:
            overrides_dict = self.thresholds_config["overrides"]
        else:
            overrides_dict = {}

        # Fallback mapping: 'trend_btc_led', 'trend_alt_rotation' -> 'trend'
        if lookup_regime not in overrides_dict and lookup_regime.startswith('trend_'):
            lookup_regime = 'trend'

        def apply_override(override_key: str):
            if override_key in overrides_dict:
                curr_override = overrides_dict[override_key]
                if isinstance(curr_override, dict):
                    for key, val in curr_override.items():
                        if val is not None:
                            resolved[key] = val
                else:
                    for key in ["min_score", "min_relative_leadership", "min_volume", "max_spread_bps"]:
                        val = getattr(curr_override, key, None)
                        if val is not None:
                            resolved[key] = val
                            
        # Apply Base Regime Overrides
        apply_override(lookup_regime)
        
        # Apply Volatility Overrides (stacks on top of regime if present e.g. 'high_vol')
        if volatility and volatility != "normal":
            apply_override(volatility)
            
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
            
        # 5. Adaptive System Overrides (Highest Priority - Cached)
        if self._adaptive_cache:
            for key in ["min_score", "min_volume"]:
                if key in self._adaptive_cache:
                    resolved[key] = self._adaptive_cache[key]
                
        return resolved
