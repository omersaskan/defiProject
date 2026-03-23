import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from defihunter.core.config import load_config
from defihunter.data.dataset_builder import DatasetBuilder

# Mock data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=200, freq='15min')
symbols = ['AAVE.p', 'COMP.p', 'MKR.p', 'UNI.p', 'SUSHI.p']
data = []
for s in symbols:
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': [s] * len(dates),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 102 + np.random.randn(len(dates)).cumsum(),
        'low': 98 + np.random.randn(len(dates)).cumsum(),
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.rand(len(dates)) * 1000,
        'family': 'lending' if s in ['AAVE.p', 'COMP.p', 'MKR.p'] else 'dex_amm'
    })
    data.append(df)

multi_df = pd.concat(data)
config = load_config('configs/default.yaml')
builder = DatasetBuilder(config=config, timeframe='15m')

print("--- Label Generation Proof ---")
labeled = builder.generate_cross_sectional_labels(multi_df)

cols = ['timestamp', 'symbol', 'family', 'future_24h_return', 'future_24h_rank_in_family', 'is_top3_family_next_24h']
print(labeled[cols].dropna().head(10).to_string())

print("\n--- Same Timestamp + Same Family Rank Check ---")
ts = labeled['timestamp'].iloc[0]
sample = labeled[(labeled['timestamp'] == ts) & (labeled['family'] == 'lending')]
print(sample[['timestamp', 'symbol', 'family', 'future_24h_return', 'future_24h_rank_in_family']].sort_values('future_24h_rank_in_family'))
