import pandas as pd
from typing import Dict, Any, List
from defihunter.core.models import CoinProfile
from defihunter.core.config import AppConfig

class FamilyEngine:
    def __init__(self, config: AppConfig):
        self.families = config.families
        
    def get_family(self, symbol: str) -> str:
        """Helper to get the family label for a symbol."""
        for f_label, f_config in self.families.items():
            members = f_config.members if hasattr(f_config, 'members') else f_config.get('members', [])
            if symbol in members:
                return f_label
        return "defi_beta"
        
    def get_family_members(self, family_label: str) -> List[str]:
        """Helper to get all members of a family."""
        if family_label in self.families:
            f_config = self.families[family_label]
            return f_config.members if hasattr(f_config, 'members') else f_config.get('members', [])
        return []
        
    def profile_coin(self, symbol: str, historical_data: pd.DataFrame = None) -> CoinProfile:
        """
        Profiles a coin based on symbol mapping and (optional) historical analysis.
        """
        family_label = self.get_family(symbol)
        primary_anchor = "ETH.p"
        
        for f_label, f_config in self.families.items():
            if f_label == family_label:
                primary_anchor = f_config.primary_anchor if hasattr(f_config, 'primary_anchor') else f_config.get('primary_anchor', "ETH.p")
                break
        
        # Base behavior from family (GT-UNIVERSE: Updated categories)
        behavior = "breakout_continuation"
        if family_label == "dex_amm":
            behavior = "mean_reverting"
        elif family_label in ["lending", "lsd"]:
            # Lending and Staking are TVL-heavy, slow movers, retests are reliable
            behavior = "retest_friendly"
        elif family_label in ["oracle", "perp_dex", "infra"]:
            # Catalyst driven, momentum/breakout friendly
            behavior = "catalyst_sensitive"
        elif family_label in ["restaking", "yield"]:
            # High volatility, hype driven
            behavior = "volatile_momentum"
            
        # Data-driven behavior refinement (Section D requirement)
        if historical_data is not None and len(historical_data) > 20:
            # Check for high wick activity (fakeout prone)
            if 'upper_wick_ratio' in historical_data.columns:
                avg_wick = historical_data['upper_wick_ratio'].tail(20).mean()
                if avg_wick > 0.4:
                    behavior = "fake_breakout_prone"
            
            # Check for range-bound vs trending
            if 'atr' in historical_data.columns:
                volatility_trend = historical_data['atr'].tail(20).std()
                if volatility_trend < 0.01:
                    behavior = "mean_reverting"

        preferred_setups = ["trend_pullback", "breakout"]
        if behavior == "mean_reverting":
            preferred_setups = ["sweep_reclaim", "range_reclaim"]
        elif behavior == "retest_friendly":
            preferred_setups = ["breakout_retest"]
        elif behavior == "fake_breakout_prone":
            preferred_setups = ["low_wick_reclaim"]
            
        return CoinProfile(
            symbol=symbol,
            family_label=family_label,
            primary_anchor=primary_anchor,
            behavior_profile=behavior,
            preferred_setups=preferred_setups,
            confidence=1.0
        )
