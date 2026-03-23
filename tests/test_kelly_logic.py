import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from defihunter.engines.risk import RiskEngine

def test_kelly():
    print("--- Testing Fractional Kelly Sizing Logic ---")
    # 1/4 Kelly, Max cap at 2% risk per trade
    config = {
        "kelly_fraction": 0.25,
        "max_risk_per_trade_pct": 5.0
    }
    risk_engine = RiskEngine(config)
    
    # Scenario 1: High confidence, high reward
    # W = 0.65 (65% win rate), R = 2.0 (2:1 Reward:Risk)
    # Kelly = 0.65 - (1-0.65)/2.0 = 0.65 - 0.175 = 0.475 (47.5% of bankroll)
    # Fractional (0.25) = 11.875%
    # Capped at 2.0%
    size1 = risk_engine.calculate_kelly_size(0.65, 2.0)
    print(f"High Conf (W=65%, R=2.0) -> Risk %: {size1:.3f}%")
    
    # Scenario 2: Marginal edge
    # W = 0.45, R = 2.0
    # Kelly = 0.45 - 0.55/2 = 0.45 - 0.275 = 0.175
    # Fractional (0.25) = 0.04375 (4.375%)
    size2 = risk_engine.calculate_kelly_size(0.45, 2.0)
    print(f"Marginal Edge (W=45%, R=2.0) -> Risk %: {size2:.3f}%")
    
    # Scenario 3: Negative expectancy (No edge)
    # W = 0.30, R = 2.0
    # Kelly = 0.30 - 0.70/2 = -0.05
    # Fractional = 0
    size3 = risk_engine.calculate_kelly_size(0.30, 2.0)
    print(f"Negative Edge (W=30%, R=2.0) -> Risk %: {size3:.3f}% (Expected: 0.0)")

if __name__ == "__main__":
    test_kelly()
