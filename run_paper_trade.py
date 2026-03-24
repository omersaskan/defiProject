import json
import os
from defihunter.execution.paper_trade import PaperTradeEngine

class MockDecision:
    def __init__(self):
        self.symbol = "DYDX.p"
        self.entry_price = 10.0
        self.stop_price = 9.0
        self.tp1_price = 12.0
        self.tp2_price = 15.0
        self.explanation = {'family': 'defi_perp', 'kelly_risk_pct': 2.0}

def main():
    if os.path.exists("logs/paper_portfolio.json"):
        os.remove("logs/paper_portfolio.json")
        
    engine = PaperTradeEngine()
    engine.portfolio.balance_usd = 10000
    
    # 1. Open Position
    decision = MockDecision()
    engine.open_position(decision, risk_pct=2.0)
    print("--- AFTER OPEN ---")
    print(json.dumps(engine.portfolio.open_positions[0].model_dump(), indent=2))
    
    # 2. Hit TP1 -> Turns into runner
    prices = {"DYDX.p": 12.5}
    engine.update_positions(prices)
    print("--- AFTER TP1 HIT ---")
    print(json.dumps(engine.portfolio.open_positions[0].model_dump(), indent=2))
    
    # 3. Trailing Stop Activation & Peak Price updates
    prices = {"DYDX.p": 14.0}
    engine.update_positions(prices)
    print("--- AFTER PEAK 14.0 (TRAILING STOP) ---")
    print(json.dumps(engine.portfolio.open_positions[0].model_dump(), indent=2))
    
    # 4. Leadership Decay Exit
    decay_signals = {"DYDX.p": {"exit_signal": True, "exit_reason": "ML Rank Dropped"}}
    prices = {"DYDX.p": 13.5}
    engine.update_positions(prices, decay_signals=decay_signals)
    print("--- AFTER DECAY EXIT ---")
    print(json.dumps(engine.portfolio.trade_history[0].model_dump(), indent=2))

if __name__ == "__main__":
    main()
