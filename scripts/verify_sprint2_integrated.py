import pandas as pd
from defihunter.execution.backtest import BacktestEngine

def run_tests():
    print("Testing Integrated Backtest Correlation Determinism...")
    
    dates = pd.date_range(start='2024-01-01', periods=200, freq='15T')
    df_btc = pd.DataFrame({'timestamp': dates, 'symbol': 'BTC.p', 'close': range(100, 300), 'entry_signal': False})
    df_eth = pd.DataFrame({'timestamp': dates, 'symbol': 'ETH.p', 'close': range(10, 210), 'entry_signal': False})
    df_btc.loc[198, 'entry_signal'] = True
    df_eth.loc[199, 'entry_signal'] = True
    clean_df = pd.concat([df_btc, df_eth])
    
    class DummyRisk:
        def dict(self): return {'max_correlated_exposure': 10, 'max_avg_correlation': 0.5, 'max_risk_per_trade_pct': 1.0, 'default_leverage': 1}
    class DummyExecution:
        def __init__(self): self.mode = 'trade_allowed'; self.max_open_positions = 5; self.min_entry_readiness = 0.0; self.min_leader_prob = 0.0
    class DummyConfig:
        def __init__(self):
            self.risk = DummyRisk()
            self.backtest = type('obj', (object,), {'initial_equity': 10000.0, 'max_concurrent_positions': 5})()
        def get_family_execution(self, family): return DummyExecution()

    bt = BacktestEngine(config=DummyConfig())
    res = bt.simulate(clean_df)
    
    details = res.get('execution_details', [])
    btc_exec = [d for d in details if d['symbol'] == 'BTC.p' and d['decision'] == 'execute']
    eth_skip = [d for d in details if d['symbol'] == 'ETH.p' and d['decision'] == 'skip' and 'high_portfolio_correlation' in d['reason']]
    
    assert len(btc_exec) > 0, "BTC should have executed"
    assert len(eth_skip) > 0, "ETH should have been skipped due to correlation"
    
    print("[INTEGRATION] SUCCESS: BacktestEngine executes correlation veto accurately using pre-sliced memory without monkey patches.")

    print("\n[BOOLEAN] Sprint 1 Boolean Validation (Sanitizer behavior on 'entry_signal')")
    df_raw = pd.DataFrame({'timestamp': [dates[0]]*3, 'symbol': ['A.p', 'B.p', 'C.p'], 'entry_signal': [1.0, float('nan'), True], 'close': [1,2,3]})
    clean_boolean = bt._sanitize_simulation_data(df_raw)
    assert clean_boolean['entry_signal'].tolist() == [True, False, True], "Boolean strictness failed."
    print("[BOOLEAN] SUCCESS: Boolean sanitization strictly enforces cast boolean gating, replacing unsafe is True.")

if __name__ == '__main__':
    run_tests()
