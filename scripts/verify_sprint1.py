import pandas as pd
from defihunter.data.features import compute_cvd_acceleration
from defihunter.execution.backtest import BacktestEngine
from defihunter.engines.risk import RiskEngine
from defihunter.core.config import load_config
import os

def run_tests():
    # 1. Feature Contract Test
    df_no_cvd = pd.DataFrame({'close': [1,2,3], 'volume': [100, 200, 300]})
    res = compute_cvd_acceleration(df_no_cvd)
    assert isinstance(res, dict), "compute_cvd_acceleration must return a dict"
    print("[1] SUCCESS: compute_cvd_acceleration respects dict contract on missing feature")

    # 2. Backtest Entry Sanitizer Test
    bt = BacktestEngine()
    df_raw = pd.DataFrame({
        'entry_signal': [1.0, float('nan'), True, 0.0],
        'close': [100, 200, 300, 400],
        'future_return': [0.1, 0.2, 0.3, 0.4]
    })
    
    clean_df = bt._sanitize_simulation_data(df_raw)
    assert 'future_return' not in clean_df.columns, "Leaky columns must be dropped"
    assert clean_df['entry_signal'].tolist() == [True, False, True, False], "NaN truthiness bypass failed"
    print("[2] SUCCESS: Backtest entry sanitization strictly enforces cast boolean gating")

    # 3. Rigorous Correlation Safety Test
    # Simulate the exact mechanism backtest uses
    risk_cfg = {'max_correlated_exposure': 1, 'max_avg_correlation': 0.5, 'max_risk_per_trade_pct': 1.0, 'default_leverage': 1}
    r_engine = RiskEngine(config=risk_cfg, fetcher=None)
    
    # Backtest Monkey-Patch safeguard
    r_engine.corr_engine.calculate_correlation = lambda *args, **kwargs: {"mean_corr": 0.0, "max_corr": 0.0, "matrix": {}}
    
    # Setup Open Portfolio
    port = [{"symbol": "BTC.p", "family": "defi_beta", "status": "open", "kelly_risk": 1.0}]
    
    # Trigger correlation check with a second asset. Should not fail/crash.
    is_valid, reason = r_engine.validate_trade(
        symbol="ETH.p",
        family="defi_beta",
        current_portfolio=port,
        equity_val=10000.0,
        daily_loss_pct=0.0,
        leader_prob=0.8,
        new_trade_notional=1000.0,
        leverage=1.0,
        family_max_pos=5
    )
    
    assert is_valid is True, f"Trade vetoed incorrectly: {reason}"
    print("[3] SUCCESS: Backtest Correlation Engine safely bypassed network fetch without crashing")

    # 4. Config Family Verification Guard Test
    try:
        cfg = load_config('configs/default.yaml')
        print("[4] SUCCESS: Config loaded without Family Overlaps.")
    except ValueError as e:
        print(f"[4] FAILED: Family overlap detected, test failed as intended if unfixed: {e}")
        raise e

    print("\nALL SPRINT 1 VALIDATIONS PASSED.")

if __name__ == '__main__':
    run_tests()
