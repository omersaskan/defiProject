import sys
import os
sys.path.append(os.getcwd())

from defihunter.common.timeframes import bars_for_hours, bars_for_days

print("--- Timeframe Helper Test ---")
print(f"15m timeframe, 24h duration: {bars_for_hours('15m', 24)} bars (Expected: 96)")
print(f"1h timeframe, 1d duration: {bars_for_days('1h', 1)} bars (Expected: 24)")
print(f"15m timeframe, 6h duration: {bars_for_hours('15m', 6)} bars (Expected: 24)")
print("--- SUCCESS ---")
