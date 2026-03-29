import pandas as pd
import numpy as np
import pytest
from defihunter.execution.pipeline import SignalPipeline
from defihunter.core.config import load_config
from defihunter.data.features import build_feature_pipeline
from datetime import datetime, timezone

@pytest.fixture
def mock_pipeline_data():
    symbols = ["AAVE.p", "UNI.p"]
    # Ensure enough bars for features
    df_raw = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=200, freq="15min", tz='UTC'),
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0,
        "quote_volume": 100000.0, "taker_buy_volume": 600.0, "taker_sell_volume": 400.0
    })
    
    # Add variety to avoid all-same-values causing NaNs in Z-scores
    df_raw['close'] = df_raw['close'] * (1 + np.random.randn(len(df_raw)) * 0.01)
    
    # Pre-process
    df = build_feature_pipeline(df_raw, timeframe="15m")
    
    symbol_data_map = {s: df.copy() for s in symbols}
    anchor_context = {
        "BTC.p": {"15m": df.copy(), "1h": df.copy(), "4h": df.copy()},
        "ETH.p": {"15m": df.copy(), "1h": df.copy(), "4h": df.copy()},
        "AAVE.p": {"1h": df.copy()}, "UNI.p": {"1h": df.copy()}
    }
    return symbol_data_map, anchor_context, df['timestamp'].iloc[-1]

import numpy as np

def test_pipeline_parity(mock_pipeline_data):
    """
    Verifies that calling the pipeline in 'live' vs 'historical' mode 
    produces identical signal decisions for the same data.
    """
    config = load_config("configs/default.yaml")
    pipeline = SignalPipeline(config)
    
    symbol_data_map, anchor_context, ts = mock_pipeline_data
    
    # 1. Live Mode
    res_live = pipeline.run(
        symbol_data_map=symbol_data_map,
        anchor_context=anchor_context,
        mode="live",
        scan_timestamp=ts
    )
    
    # 2. Historical Mode
    res_hist = pipeline.run(
        symbol_data_map=symbol_data_map,
        anchor_context=anchor_context,
        mode="historical",
        scan_timestamp=ts
    )
    
    # 3. Shadow Mode
    res_shadow = pipeline.run(
        symbol_data_map=symbol_data_map,
        anchor_context=anchor_context,
        mode="shadow",
        scan_timestamp=ts
    )
    
    # ASSERTIONS
    assert not res_live.master_df.empty
    assert len(res_live.final_decisions) == len(res_hist.final_decisions)
    assert len(res_live.final_decisions) == len(res_shadow.final_decisions)
    
    # Check ML Parity
    pd.testing.assert_frame_equal(
        res_live.master_df.sort_values("symbol").reset_index(drop=True),
        res_hist.master_df.sort_values("symbol").reset_index(drop=True),
        check_dtype=False,
        atol=1e-5
    )
    
    # Check Decision Parity
    live_syms = sorted([d.symbol for d in res_live.final_decisions])
    hist_syms = sorted([d.symbol for d in res_hist.final_decisions])
    assert live_syms == hist_syms, "Decision symbols do not match across modes"
    
    for d_l, d_h in zip(sorted(res_live.final_decisions, key=lambda x: x.symbol), 
                         sorted(res_hist.final_decisions, key=lambda x: x.symbol)):
        assert d_l.decision == d_h.decision
        assert d_l.entry_price == d_h.entry_price
        assert d_l.leader_prob == d_h.leader_prob
        assert d_l.composite_leader_score == d_h.composite_leader_score

if __name__ == "__main__":
    pytest.main([__file__])
