from defihunter.common.timeframes import to_minutes, bars_for_hours
from defihunter.utils.timeframe import TimeframeHelper
from defihunter.data.dataset_builder import DatasetBuilder
from defihunter.core.config import AppConfig
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

def debug_timeframes():
    print("--- Timeframe Debug ---")
    print(f"to_minutes('15m'): {to_minutes('15m')}")
    print(f"to_minutes('1h'): {to_minutes('1h')}")
    print(f"bars_for_hours('15m', 24): {bars_for_hours('15m', 24)}")
    print(f"bars_for_hours('1h', 24): {bars_for_hours('1h', 24)}")
    
    print("\n--- TimeframeHelper Debug ---")
    print(f"TimeframeHelper.get_bars('24h', '15m'): {TimeframeHelper.get_bars('24h', '15m')}")
    print(f"TimeframeHelper.get_bars('24h', '1h'): {TimeframeHelper.get_bars('24h', '1h')}")

def debug_dataset_builder():
    print("\n--- DatasetBuilder Debug ---")
    config = MagicMock(spec=AppConfig)
    config.families = {}
    
    try:
        builder = DatasetBuilder(config, timeframe='15m')
        print(f"Builder window (15m): {builder.window}")
        
        builder_1h = DatasetBuilder(config, timeframe='1h')
        print(f"Builder window (1h): {builder_1h.window}")
        
        # Test label generation
        n_bars = 200
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=n_bars, freq='15min'),
            'symbol': ['BTC'] * n_bars,
            'close': np.linspace(20000, 21000, n_bars),
            'high': np.linspace(20050, 21050, n_bars),
            'low': np.linspace(19950, 20950, n_bars),
            'volume': np.random.rand(n_bars) * 100,
            'atr': [100] * n_bars
        })
        
        df_labeled = builder.generate_labels(df)
        print("Labels generated successfully.")
        print(f"Columns: {df_labeled.columns.tolist()}")
        
        df_ranked = builder.generate_cross_sectional_labels(df)
        print("Cross-sectional labels generated successfully.")
        print(f"Ranking Columns: {[c for c in df_ranked.columns if 'future' in c]}")
        
    except Exception as e:
        print(f"Error in DatasetBuilder: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_timeframes()
    debug_dataset_builder()
