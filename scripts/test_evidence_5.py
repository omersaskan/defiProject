import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from defihunter.execution.backtest import BacktestEngine

# Mock data for backtest evaluation
np_rng = pd.Series(pd.date_range('2024-01-01', periods=100))
symbols = ['AAVE.p', 'UNI.p', 'MKR.p', 'LINK.p', 'LDO.p', 'DYDX.p', 'SNX.p', 'GMX.p']

scored_dfs = {}
for s in symbols:
    df = pd.DataFrame({
        'timestamp': np_rng,
        'symbol': [s] * 100,
        'close': 100 + pd.Series(range(100)) * (0.01 if s == 'AAVE.p' else 0.005),
        'ml_rank_score': 80 if s == 'AAVE.p' else 50
    })
    scored_dfs[s] = df

bt = BacktestEngine()
print("--- Leader-Capture Backtest Report ---")
# Ensure the method is actually called on the instance
ranking_report = bt.evaluate_ranking_quality(scored_dfs, k=3)
for k, v in ranking_report.items():
    print(f"{k}: {v}")

print("\n--- Exit Reason Distribution ---")
# Mock some trade log
bt.trade_log = [
    {'outcome': 'WIN', 'pnl_r': 2.0, 'peak_pnl_r': 2.5},
    {'outcome': 'EXIT_DECAY', 'pnl_r': 1.5, 'peak_pnl_r': 3.0},
    {'outcome': 'LOSS', 'pnl_r': -1.0, 'peak_pnl_r': 0.2},
    {'outcome': 'EXIT_DECAY', 'pnl_r': 0.8, 'peak_pnl_r': 1.2},
    {'outcome': 'WIN_TRAILED', 'pnl_r': 0.5, 'peak_pnl_r': 1.5}
]
# Manually trigger metric calculation instead of calling simulate on empty data
if bt.trade_log:
    trades_df = pd.DataFrame(bt.trade_log)
    trades_df['giveback_ratio'] = (trades_df['peak_pnl_r'] - trades_df['pnl_r']).clip(lower=0) / trades_df['peak_pnl_r'].replace(0, np.nan)
    trades_df['hold_efficiency'] = trades_df['pnl_r'] / trades_df['peak_pnl_r'].replace(0, np.nan)
    
    print(f"Avg Giveback: {round(trades_df['giveback_ratio'].mean(), 2)}")
    print(f"Avg Hold Efficiency: {round(trades_df['hold_efficiency'].mean(), 2)}")

reasons = pd.Series([t['outcome'] for t in bt.trade_log]).value_counts()
print("\nExit Reason Counts:")
print(reasons)
print("--- SUCCESS ---")
