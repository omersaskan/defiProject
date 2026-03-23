import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from defihunter.data.dataset_builder import DatasetBuilder
from defihunter.core.config import AppConfig

@pytest.fixture
def mock_config():
    config = MagicMock(spec=AppConfig)
    config.families = {}
    return config

def create_dummy_df(n_bars=200):
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='15min')
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['BTC'] * n_bars,
        'open': np.linspace(20000, 21000, n_bars),
        'high': np.linspace(20100, 21100, n_bars),
        'low': np.linspace(19900, 20900, n_bars),
        'close': np.linspace(20050, 21050, n_bars),
        'volume': np.random.rand(n_bars) * 100,
        'atr': [100] * n_bars
    })
    return df

def test_dataset_builder_horizons_15m(mock_config):
    # 15m timeframe: 24h = 96 bars
    builder = DatasetBuilder(mock_config, timeframe='15m')
    assert builder.window == 96
    
    df = create_dummy_df(200)
    df_labeled = builder.generate_labels(df)
    
    # Check if target_24h_10pct has last 96 rows as 0
    assert (df_labeled['target_24h_10pct'].tail(96) == 0).all()
    # Check if target_4h_2pct has last 16 rows as 0 (4h * 4 bars/h = 16)
    assert (df_labeled['target_4h_2pct'].tail(16) == 0).all()

def test_dataset_builder_horizons_1h(mock_config):
    # 1h timeframe: 24h = 24 bars
    builder = DatasetBuilder(mock_config, timeframe='1h')
    assert builder.window == 24
    
    # Adjust dummy df for 1h
    df = create_dummy_df(100)
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    
    df_labeled = builder.generate_labels(df)
    
    # Check if target_24h_10pct has last 24 rows as 0
    assert (df_labeled['target_24h_10pct'].tail(24) == 0).all()
    # Check if target_4h_2pct has last 4 rows as 0
    assert (df_labeled['target_4h_2pct'].tail(4) == 0).all()

def test_future_return_columns(mock_config):
    builder = DatasetBuilder(mock_config, timeframe='15m')
    df = create_dummy_df(200)
    df['family'] = 'DeFi'
    
    # generate_cross_sectional_labels expects multi-symbol or specific structure
    # but let's test the return column generation
    df_ranked = builder.generate_cross_sectional_labels(df)
    
    assert 'future_6h_return' in df_ranked.columns
    assert 'future_12h_return' in df_ranked.columns
    assert 'future_24h_return' in df_ranked.columns
    
    # 15m, 6h = 24 bars
    # future_6h_return for row i should be (close[i+24] / close[i]) - 1
    idx = 100
    expected_ret = (df['close'].iloc[idx + 24] / df['close'].iloc[idx]) - 1
    np.testing.assert_almost_equal(df_ranked['future_6h_return'].iloc[idx], expected_ret)

if __name__ == "__main__":
    pytest.main([__file__])
