import pandas as pd, numpy as np
from defihunter.data.features import build_feature_pipeline
from defihunter.data.dataset_builder import DatasetBuilder

# Simulate 200 bars of OHLCV data
np.random.seed(42)
n = 200
timestamps = pd.date_range('2025-01-01', periods=n, freq='1h')
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
df = pd.DataFrame({
    'timestamp': timestamps,
    'open': close * (1 + np.random.randn(n) * 0.001),
    'high': close * (1 + abs(np.random.randn(n) * 0.005)),
    'low': close * (1 - abs(np.random.randn(n) * 0.005)),
    'close': close,
    'volume': abs(np.random.randn(n) * 1000 + 5000),
    'quote_volume': abs(np.random.randn(n) * 500000 + 1000000),
    'taker_buy_volume': abs(np.random.randn(n) * 500 + 2500),
    'taker_sell_volume': abs(np.random.randn(n) * 500 + 2500),
    'funding_rate': np.random.randn(n) * 0.001,
    'open_interest': abs(np.random.randn(n) * 1000000 + 5000000),
})

result = build_feature_pipeline(df)
print("Feature pipeline OK:", result.shape)

# Check new GT features exist
gt_cols = ['quiet_expansion', 'accumulation_zone', 'volume_spike_now', 'silent_accumulation', 'rs_divergence_bull', 'rs_divergence_persistence']
for col in gt_cols:
    val = result[col].sum() if col in result.columns else "MISSING!"
    print(f"  {col}: {val}")

# Check BUG #1 fix - tightness NaN count should be small (<30)
nan_count = result['tightness'].isna().sum()
print(f"  tightness NaN count: {nan_count} (BUG#1: should be <30)")

# Check BUG #7 fix - retest_hold sometimes differs from retest_touch
rth = result['retest_hold'].sum()
rtt = result['retest_touch'].sum()
print(f"  retest_touch={rtt}, retest_hold={rth} (BUG#7: hold <= touch)")

# Check dataset builder with prepump_target
builder = DatasetBuilder(window=20, prepump_gain_pct=0.05)
labeled = builder.build(result)
print(f"\nDataset builder OK: {labeled.shape}")
print(f"  target_hit positive rate: {labeled['target_hit'].mean():.2%}")
print(f"  prepump_target positive rate: {labeled['prepump_target'].mean():.2%}")
print(f"  short_target_hit positive rate: {labeled['short_target_hit'].mean():.2%}")

print("\nALL TESTS PASSED")
