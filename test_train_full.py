import pandas as pd
from defihunter.data.storage import TSDBManager
from defihunter.data.features import build_feature_pipeline
from defihunter.engines.leadership import LeadershipEngine
from defihunter.data.dataset_builder import DatasetBuilder
from defihunter.core.config import load_config

config = load_config('configs/default.yaml')
tsdb = TSDBManager()
builder = DatasetBuilder(window=24)

anchor_data = {}
for anchor in config.anchors:
    adf = tsdb.load_dataframe(anchor, timeframe='15m') # NO START DATE
    if not adf.empty:
        anchor_data[anchor] = build_feature_pipeline(adf)
        print(f"Loaded {anchor}: {len(anchor_data[anchor])} rows")

leadership_engine = LeadershipEngine(anchors=config.anchors, ema_lengths=[config.ema.fast, config.ema.medium])

df = tsdb.load_dataframe('AAVE.p', timeframe='15m')
print(f"Loaded AAVE.p: {len(df)} rows")
df = build_feature_pipeline(df)
df = leadership_engine.add_leadership_features(df, anchor_data)

labeled = builder.generate_labels(df)
exclude = ['timestamp', 'symbol', 'target_hit', 'mfe_r', 'mae_r', 'exit_type', 'open_time', 'close', 'high', 'low', 'open', 'volume', 'quote_volume']
feature_cols = [c for c in labeled.columns if pd.api.types.is_numeric_dtype(labeled[c]) and c not in exclude]

target_df = labeled.iloc[:-24]
final_df = target_df.dropna(subset=feature_cols)
print(f"Target rows before dropna: {len(target_df)}")
print(f"Target rows after dropna: {len(final_df)}")

if len(final_df) == 0:
    na_counts = target_df[feature_cols].isna().sum()
    bad_cols = na_counts[na_counts > 0]
    print("Columns with NA:")
    print(bad_cols.sort_values(ascending=False).head(10))
