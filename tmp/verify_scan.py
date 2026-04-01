import asyncio
import time
import pandas as pd
import numpy as np
from defihunter.core.config import load_config
from defihunter.execution.scanner import ScanPipeline

async def verify():
    print("Initializing ScanPipeline...")
    config = load_config("configs/default.yaml") # Assuming this exists
    pipeline = ScanPipeline(config)
    
    # Mock some data for 2 symbols
    symbols = ["BTC.p", "ETH.p"]
    mock_data = {}
    for sym in symbols:
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-03-29", periods=100, freq="15min"),
            "open": np.random.randn(100).cumsum() + 50000,
            "high": np.random.randn(100).cumsum() + 50100,
            "low": np.random.randn(100).cumsum() + 49900,
            "close": np.random.randn(100).cumsum() + 50000,
            "volume": np.random.randn(100).abs() * 100,
            "quote_volume": np.random.randn(100).abs() * 10000,
            "taker_buy_volume": np.random.randn(100).abs() * 50,
            "taker_sell_volume": np.random.randn(100).abs() * 50,
            "open_interest": np.random.randn(100).abs() * 1000,
            "funding_rate": np.random.randn(100) * 0.0001
        })
        mock_data[sym] = df
    
    print(f"Running parallel feature processing for {len(symbols)} symbols...")
    start_t = time.time()
    await pipeline._process_features_parallel(mock_data)
    elapsed = time.time() - start_t
    print(f"Feature processing completed in {elapsed:.4f}s")
    
    for sym in symbols:
        df = pipeline.symbol_data_map[sym]
        print(f"Symbol: {sym}, Columns: {len(df.columns)}, Rows: {len(df)}")
        # Check for fragmentation (via attribute _is_copy or similar, though Pandas doesn't always expose it easily)
        # We can at least see if it works
        assert "atr" in df.columns
        assert "is_breakout_bar" in df.columns
        assert "hour_sin" in df.columns
        
    print("Verification Successful!")

if __name__ == "__main__":
    asyncio.run(verify())
