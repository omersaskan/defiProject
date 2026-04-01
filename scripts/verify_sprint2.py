import pandas as pd
from defihunter.execution.backtest import BacktestEngine
from defihunter.engines.risk import RiskEngine
from defihunter.core.config import load_config
from defihunter.engines.portfolio import CorrelationEngine

def run_tests():
    # 1. Test In-Memory Correlation Path in Backtest
    print("Testing In-Memory Correlation Determinism...")
    
    # Create simple monotonic dummy data
    df_btc = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='15T'),
        'symbol': 'BTC.p',
        'close': range(100, 300),
        'entry_signal': False
    })
    
    # Highly correlated asset
    df_eth = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='15T'),
        'symbol': 'ETH.p',
        'close': range(10, 210),
        'entry_signal': False
    })
    
    df_eth.loc[199, 'entry_signal'] = True  # Try to enter on the last bar
    clean_df = pd.concat([df_btc, df_eth])
    
    # Create backtest engine and dummy config
    class DummyConfig:
        risk = type('obj', (object,), {'dict': lambda s: {'max_correlated_exposure': 10, 'max_avg_correlation': 0.5, 'max_risk_per_trade_pct': 1.0, 'default_leverage': 1}})()
        def get_family_execution(self, family):
            return type('obj', (object,), {'mode': 'trade_allowed', 'max_open_positions': 5})()
    
    bt = BacktestEngine(config=DummyConfig())
    
    # Mock RiskEngine inside the simulation by patching it on the instance dynamically?
    # Actually, we can just run simulate and check if it vetoes due to correlation
    
    # Wait, we need BTC to be in the open_positions.
    # We can inject it into the engine during the first ts or just test RiskEngine directly
    
    r_engine = RiskEngine(config=DummyConfig().risk.dict(), fetcher=None)
    symbol_data_map_history = {
        'BTC.p': df_btc,
        'ETH.p': df_eth
    }
    
    port = [{"symbol": "BTC.p", "family": "defi_beta", "status": "open", "kelly_risk": 1.0, "size_usd": 1000, "leverage": 1}]
    is_val, reason = r_engine.validate_trade(
        symbol='ETH.p',
        family='defi_beta',
        current_portfolio=port,
        equity_val=10000.0,
        daily_loss_pct=0.0,
        leader_prob=0.8,
        new_trade_notional=1000.0,
        leverage=1.0,
        symbol_data_map=symbol_data_map_history
    )
    
    # Perfect linear correlation should exist
    assert not is_val, "Trade should have been vetoed due to high correlation"
    assert "high_portfolio_correlation" in reason, f"Reason was: {reason}"
    
    print("[1] SUCCESS: RiskEngine correctly utilized provided in-memory DataFrame (determinstic correlation without MonkeyPatching).")

    print("\nALL SPRINT 2 VALIDATIONS PASSED.")

if __name__ == '__main__':
    run_tests()
