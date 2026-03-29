import pandas as pd
import pytest
from datetime import datetime, timedelta
from defihunter.execution.paper_trade import PaperTradeEngine
from defihunter.execution.backtest import BacktestEngine
from defihunter.core.models import FinalDecision

@pytest.fixture
def mock_config():
    class Config:
        class EMA:
            fast = 20
            medium = 50
        class Decision:
            top_n = 5
        ema = EMA()
        decision = Decision()
        timeframe = "15m"
        anchors = ["BTC.p", "ETH.p"]
        families = {}
        regimes = {}
        backtest = type('BT', (), {'time_stop_bars': 24})() # Mock backtest config
    return Config()

def test_management_parity_tp_flow(mock_config, tmp_path):
    # Setup Paper Engine with temp file
    state_file = tmp_path / "paper_state.json"
    paper = PaperTradeEngine(state_path=str(state_file))
    
    # Setup Backtest Engine
    bt = BacktestEngine(config=mock_config)
    
    # 1. Define a Trade
    symbol = "TEST.p"
    decision = FinalDecision(
        symbol=symbol,
        timestamp=datetime.now(),
        final_trade_score=80.0,
        decision="trade",
        entry_price=100.0,
        stop_price=90.0,
        tp1_price=110.0,
        tp2_price=130.0,
        explanation={"family": "defi"}
    )
    
    # Open in Paper
    paper.open_position(decision, risk_pct=1.0)
    
    # 3. Prepare Data for Backtest Engine (Flattened Format)
    # The backtest engine expects a single DF with 'symbol', 'timestamp', 'entry_signal' etc.
    prices = [100.0, 105.0, 115.0, 120.0, 140.0]
    ts_base = datetime(2026, 3, 29, 0, 0)
    df_bt = pd.DataFrame([
        {
            "timestamp": ts_base + timedelta(minutes=15 * i), 
            "symbol": symbol, 
            "open": p, "high": p+1, "low": p-1, "close": p,
            "entry_signal": (i == 0), # Trigger entry on first bar
            "stop_price": 90.0,
            "tp1_price": 110.0,
            "tp2_price": 130.0,
            "r_dist": 10.0, # (100 - 90)
            "status": "open"
        }
        for i, p in enumerate(prices)
    ])
    
    # 2. Run Paper Trade update step by step
    for i, p in enumerate(prices):
        paper.update_positions({symbol: p})
    
    # 4. Run Backtest simulation
    results = bt.simulate(df_bt)
    
    # 5. Verify Paper Engine results
    assert len(paper.portfolio.trade_history) == 1
    final_trade = paper.portfolio.trade_history[0]
    assert final_trade.status == "closed_tp2"
    assert "TP2" in final_trade.exit_reason
    assert final_trade.partial_taken is True

def test_management_parity_trailing_stop(mock_config, tmp_path):
    state_file = tmp_path / "paper_state_trail.json"
    paper = PaperTradeEngine(state_path=str(state_file))
    
    symbol = "TEST.p"
    decision = FinalDecision(
        symbol=symbol,
        timestamp=datetime.now(),
        final_trade_score=80.0,
        decision="trade",
        entry_price=100.0,
        stop_price=90.0,
        tp1_price=110.0, # TP1 @ 110 (10 point reward)
        tp2_price=130.0, # TP2 @ 130 (30 point reward)
        explanation={"family": "defi"}
    )
    
    # Reward distance = 30
    # Trail activation = 100 + (30 * 0.3) = 109.  Wait, TP1 is already at 110.
    # So once TP1 is hit, SL moves to 100.
    # If peak hits 120: Reward Dist=30. Trail Distance=30*0.2 = 6. 
    # New Stop = 120 - 6 = 114.
    
    paper.open_position(decision, risk_pct=1.0)
    
    # Path: 100 -> 112 (Hit TP1, SL=100) -> 125 (Peak, New SL=125-6=119) -> 115 (Hit Trail)
    path = [100, 112, 125, 115]
    for p in path:
        paper.update_positions({symbol: p})
        
    assert len(paper.portfolio.trade_history) == 1
    final_trade = paper.portfolio.trade_history[0]
    assert final_trade.status == "closed_sl"
    assert final_trade.stop_price == 119.0
    assert final_trade.partial_taken is True

if __name__ == "__main__":
    pytest.main([__file__])
