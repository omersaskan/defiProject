import pandas as pd
import numpy as np
import pytest
from defihunter.execution.pipeline import SignalPipeline
from defihunter.core.config import load_config
from defihunter.data.features import build_feature_pipeline

@pytest.fixture
def mock_pipeline_data():
    from defihunter.data.features import build_feature_pipeline
    import numpy as np
    symbols = ["AAVE.p", "UNI.p"]
    df_raw = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=200, freq="15min", tz='UTC'),
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0,
        "quote_volume": 100000.0, "taker_buy_volume": 600.0, "taker_sell_volume": 400.0
    })
    
    # Pre-process
    df = build_feature_pipeline(df_raw, timeframe="15m")
    
    symbol_data_map = {s: df.copy() for s in symbols}
    anchor_context = {
        "BTC.p": {"15m": df.copy(), "1h": df.copy(), "4h": df.copy()},
        "ETH.p": {"15m": df.copy(), "1h": df.copy(), "4h": df.copy()},
        "AAVE.p": {"1h": df.copy()}, "UNI.p": {"1h": df.copy()}
    }
    return symbol_data_map, anchor_context

def test_pipeline_canonical_columns(mock_pipeline_data):
    config = load_config("configs/default.yaml")
    pipeline = SignalPipeline(config)
    
    symbol_data_map, anchor_context = mock_pipeline_data
    
    res = pipeline.run(
        symbol_data_map=symbol_data_map,
        anchor_context=anchor_context,
        mode="live"
    )
    
    # Assert master_df is not empty
    assert not res.master_df.empty
    
    # Check Required Columns
    required = ["leader_prob", "setup_conversion_prob", "holdability_score", "ml_rank_score", "ml_explanation"]
    for col in required:
        assert col in res.master_df.columns, f"Missing required column: {col}"
        assert not res.master_df[col].isna().any(), f"NaN values in required column: {col}"

if __name__ == "__main__":
    pytest.main([__file__])
