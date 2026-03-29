from typing import Dict, List, Any, Callable, Optional
import pandas as pd

class SetupDefinition:
    def __init__(self, name: str, priority: int, condition_col: str, label: str):
        self.name = name
        self.priority = priority
        self.condition_col = condition_col
        self.label = label

class SetupRegistry:
    """
    GT-Institutional: Centralized Strategy Registry.
    Adheres to Open/Closed Principle by allowing setup registration 
    without modifying the core RuleEngine.
    """
    def __init__(self):
        self.setups: List[SetupDefinition] = []
        self._initialize_core_setups()

    def register(self, name: str, priority: int, condition_col: str, label: Optional[str] = None):
        self.setups.append(SetupDefinition(name, priority, condition_col, label or name))
        # Keep sorted by priority descending
        self.setups.sort(key=lambda x: x.priority, reverse=True)

    def _initialize_core_setups(self):
        """Initializes the standard institutional setups."""
        core_setups = [
            ('is_short_squeeze_imminent', 11, 'explosive_short_squeeze'),
            ('is_short_squeeze', 10, 'short_squeeze_auto_trigger'),
            ('whale_absorption_strong', 9, 'whale_absorption_strong'),
            ('short_squeeze_setup', 8, 'oi_short_squeeze_setup'),
            ('near_liquidation_band', 8, 'liquidation_squeeze_magnet'),
            ('silent_accumulation', 7, 'silent_accumulation'),
            ('oi_divergence_bull', 7, 'oi_divergence_accumulation'),
            ('whale_absorption', 6, 'whale_absorption'),
            ('orderbook_vacuum', 6, 'orderbook_vacuum'),
            ('funding_extreme_short', 5, 'extreme_funding_reversal'),
            ('taker_surge', 5, 'taker_surge_breakout'),
            ('squeeze_release', 5, 'squeeze_release_momentum'),
            ('cvd_price_divergence', 5, 'cvd_smartmoney_accumulation'),
            ('rsi_kink', 5, 'rsi_kink_breakout'),
            ('coiling_breakout_alert', 4, 'coiling_catalyst_breakout'),
            ('high_quality_breakout', 4, 'high_quality_breakout'),
            ('sweep_reclaim_confirmed', 4, 'confirmed_sweep_reclaim'),
            ('retest_hold', 3, 'retest_reclaim'),
            ('launch_mode', 3, 'momentum_launch_mode'),
            ('is_base_expansion', 3, 'base_expansion_breakout'),
            ('is_breakout_bar', 2, 'momentum_breakout'),
            ('quiet_expansion', 2, 'quiet_expansion_accumulation'),
            ('rsi_oversold_recovery', 1, 'rsi_oversold_recovery')
        ]
        for col, prio, label in core_setups:
            self.register(col, prio, col, label)

# Global Singleton Registry
setup_registry = SetupRegistry()
