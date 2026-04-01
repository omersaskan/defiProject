import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import time

from defihunter.core.config import load_config
import defihunter.execution.scanner
from concurrent.futures import ThreadPoolExecutor

# Force ThreadPool in Windows to avoid spawn/pickle issues in standalone test scripts
defihunter.execution.scanner._executor = ThreadPoolExecutor(max_workers=4)
from defihunter.execution.scanner import run_scanner

class MockFetcher:
    def __init__(self):
        self.exchange = self
    
    def fetch_tickers(self):
        return {
            f"COIN{i}/USDT:USDT": {"percentage": np.random.uniform(-5, 15)}
            for i in range(150)
        }
    
    def get_defi_universe(self, config=None, strict_defi=True):
        return [f"COIN{i}.p" for i in range(150)] + ["BTC.p", "ETH.p", "SOL.p"]

    async def async_fetch_ohlcv(self, symbol, timeframe='15m', limit=200):
        await asyncio.sleep(0.01) # Mock small network delay
        
        # generate minimum bars
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(10, 100, limit),
            "high": np.random.uniform(100, 110, limit),
            "low": np.random.uniform(5, 10, limit),
            "close": np.random.uniform(10, 100, limit),
            "volume": np.random.uniform(1000, 100000, limit),
            "quote_volume": np.random.uniform(10000, 1000000, limit),
            "funding_rate": np.random.uniform(-0.01, 0.01, limit),
            "open_interest": np.random.uniform(1000, 20000, limit)
        })
        
        # Randomly apply degraded telemetry
        if np.random.random() > 0.8:
            df.attrs["degraded_funding"] = True
        if np.random.random() > 0.8:
            df.attrs["degraded_oi"] = True
            
        return df
        
    async def close(self):
        pass

async def run_profiling():
    print("--- SPRINT 4: FULL-UNIVERSE DEGRADED PARITY PROFILING ---")
    config = load_config("configs/default.yaml")
    
    from defihunter.execution.scanner import ScanPipeline
    pipeline = ScanPipeline(config, fetcher=MockFetcher())
    
    t0 = time.time()
    res = await pipeline.run()
    elapsed = time.time() - t0
    
    print("\n--- STAGE TIMINGS BREAKDOWN ---")
    for key, ms in pipeline.timings.items():
        print(f"[Profiling] {key:25} -> {ms:7.1f} ms")
        
    print("\n--- TELEMETRY METRICS ---")
    print(f"[Profiling] Total Universe Evaluated: {len(pipeline.symbol_data_map)}")
    print(f"[Profiling] Degraded Funding Objects: {pipeline.degraded_funding_count}")
    print(f"[Profiling] Degraded OI Objects:      {pipeline.degraded_oi_count}")
    print(f"[Profiling] Total Profiling Execution:{elapsed:.2f}s")
    
if __name__ == '__main__':
    asyncio.run(run_profiling())
