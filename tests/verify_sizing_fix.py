import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from defihunter.engines.risk import RiskEngine

def verify_sizing():
    print("=== Verifying Position Sizing Fix (Double Penalty Removal) ===")
    
    config = {
        "max_risk_per_trade_pct": 5.0,
        "kelly_fraction": 0.25
    }
    risk_engine = RiskEngine(config)
    
    equity = 10000.0
    kelly_pct = 1.0  # 1% risk of equity = $100
    entry_price = 100.0
    
    # 1. Base Case: 2% Stop
    stop_price_1 = 98.0
    notional_1 = risk_engine.estimate_notional_from_stop(equity, kelly_pct, entry_price, stop_price_1)
    # Expected: 100 / 0.02 = 5000
    
    # 2. Widened Case: 4% Stop (e.g. width_mult = 2.0)
    stop_price_2 = 96.0
    notional_2 = risk_engine.estimate_notional_from_stop(equity, kelly_pct, entry_price, stop_price_2)
    # Expected: 100 / 0.04 = 2500
    
    print(f"Equity: ${equity}")
    print(f"Kelly Risk: {kelly_pct}% (${equity * kelly_pct / 100})")
    print(f"Case 1 (2% stop): Notional = ${notional_1:.2f}")
    print(f"Case 2 (4% stop): Notional = ${notional_2:.2f}")
    
    ratio = notional_1 / notional_2
    print(f"Ratio (Case1/Case2): {ratio:.2f}")
    
    if abs(ratio - 2.0) < 0.001:
        print("✅ SUCCESS: Notional is proportional to stop distance. No double-penalty.")
    else:
        print("❌ FAILURE: Sizing logic is still incorrect.")
        sys.exit(1)

if __name__ == "__main__":
    verify_sizing()
