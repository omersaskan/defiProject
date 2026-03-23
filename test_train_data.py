import pandas as pd
from defihunter.data.storage import TSDBManager
from defihunter.data.features import build_feature_pipeline
from defihunter.engines.leadership import LeadershipEngine
from defihunter.data.dataset_builder import DatasetBuilder
from datetime import datetime, timedelta

tsdb = TSDBManager()
builder = DatasetBuilder(window=24)
start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
df = tsdb.load_dataframe('BTC.p', '15m', start_date=start_date)

if not df.empty:
    print(f"Loaded {len(df)} raw rows for BTC.p.")
    df = build_feature_pipeline(df)
    labeled = builder.generate_labels(df)
    
    # Check if we dropped NA and lost everything
    exclude = ['timestamp', 'symbol', 'target_hit', 'mfe_r', 'mae_r', 'exit_type', 'open_time', 'close', 'high', 'low', 'open', 'volume', 'quote_volume']
    feature_cols = [c for c in labeled.columns if pd.api.types.is_numeric_dtype(labeled[c]) and c not in exclude]
    
    # Just what the train function does:
    df_final = labeled.iloc[:-24].dropna(subset=feature_cols)
    print(f"Final usable rows after dropna: {len(df_final)}")
    
    # Check what columns are primarily causing NA
    if len(df_final) == 0:
        na_counts = labeled[feature_cols].isna().sum()
        print("NA Counts per feature:")
        print(na_counts[na_counts > 0])
else:
    print('DF empty')
