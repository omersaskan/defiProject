import pandas as pd
from datetime import datetime
from defihunter.execution.backtest import BacktestEngine
from defihunter.core.config import load_config

def verify_backtest_execution_details():
    """
    STRENGTHENED VERIFICATION: Inspects explicit execution decisions.
    Proves watch_only exclusion via the new 'execution_details' audit trail.
    """
    print("\n--- [AUDIT] Verifying Backtest Execution Decisions ---")
    config = load_config("configs/default.yaml")
    ts = datetime.now()
    
    # 1. Prepare candidates
    data = [
        {"timestamp": ts, "symbol": "LINK.p", "close": 20.0, "entry_signal": True, "family": "defi_oracles", "leader_prob": 0.8},
        {"timestamp": ts, "symbol": "AAVE.p", "close": 150.0, "entry_signal": True, "family": "defi_lending", "leader_prob": 0.8},
    ]
    df = pd.concat([pd.DataFrame(data), pd.DataFrame([{"timestamp": ts + pd.Timedelta(minutes=15), "symbol": "AAVE.p", "close": 130.0}])])
    
    # 2. Run simulation
    bt = BacktestEngine(config=config)
    results = bt.simulate(df)
    
    details = results.get('execution_details', [])
    
    # 3. VERIFY EXPLICIT DECISIONS
    link_decision = next((d for d in details if d['symbol'] == "LINK.p"), None)
    aave_decision = next((d for d in details if d['symbol'] == "AAVE.p"), None)

    print(f"LINK.p Decision: {link_decision}")
    print(f"AAVE.p Decision: {aave_decision}")

    assert link_decision['decision'] == 'skip' and link_decision['reason'] == 'watch_only_gate', "Wrong skip reason for LINK.p"
    assert aave_decision['decision'] == 'execute' and aave_decision['reason'] == 'passed_all_gates', "Wrong execute reason for AAVE.p"

    print("SUCCESS: Backtest gating is now provably deterministic via execution_details.")

if __name__ == "__main__":
    verify_backtest_execution_details()
    # verify_backtest_gating_logic() # legacy
