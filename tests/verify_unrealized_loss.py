import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from defihunter.execution.paper_trade import PaperTradeEngine, PaperPosition

def verify_unrealized_loss():
    print("=== Verifying Unrealized Daily Loss Tracking ===")
    
    state_path = "logs/test_portfolio_fix.json"
    if os.path.exists(state_path): os.remove(state_path)
    
    engine = PaperTradeEngine(state_path=state_path)
    engine.portfolio.daily_start_balance = 10000.0
    engine.portfolio.balance_usd = 10000.0
    
    # Open a position: $5000 notional at price 100
    pos = PaperPosition(
        symbol="TEST.p",
        entry_price=100.0,
        stop_price=90.0,
        tp1_price=120.0,
        tp2_price=150.0,
        size_usd=5000.0,
        entry_time="2024-01-01T00:00:00",
        family="test"
    )
    engine.portfolio.open_positions.append(pos)
    
    # 1. Realized loss (none yet)
    loss_realized = engine.get_daily_loss_pct()
    print(f"Realized Daily Loss: {loss_realized:.2f}%")
    
    # 2. Simulate price drop to 90 (10% drop on $5000 = $500 loss = 5% of bankroll)
    current_prices = {"TEST.p": 90.0}
    loss_unrealized = engine.get_daily_loss_pct(current_prices)
    print(f"Unrealized-aware Daily Loss (at price 90): {loss_unrealized:.2f}%")
    
    if abs(loss_unrealized - (-5.0)) < 0.001:
        print("✅ SUCCESS: Unrealized loss is correctly captured.")
    else:
        print(f"❌ FAILURE: Expected -5.00%, got {loss_unrealized:.2f}%")
        sys.exit(1)
        
    if os.path.exists(state_path): os.remove(state_path)

if __name__ == "__main__":
    verify_unrealized_loss()
