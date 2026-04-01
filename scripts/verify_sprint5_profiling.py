import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import time

from defihunter.core.config import load_config
import defihunter.execution.scanner
from concurrent.futures import ThreadPoolExecutor

# Bound native execution on windows to dodge spawn serialization parameters
defihunter.execution.scanner._executor = ThreadPoolExecutor(max_workers=4)
from defihunter.execution.scanner import run_scanner

class MockFetcher:
    def __init__(self):
        self.exchange = self
        self.degradation_registry = {}
        
    def fetch_tickers(self):
        return {f"COIN{i}/USDT:USDT": {"percentage": np.random.uniform(-5, 15)} for i in range(150)}
        
    def get_defi_universe(self, config=None, strict_defi=True):
        return [f"COIN{i}.p" for i in range(150)] + ["BTC.p", "ETH.p", "SOL.p"]

    async def async_fetch_ohlcv(self, symbol, timeframe='15m', limit=200):
        await asyncio.sleep(0.01) # Parallelism network simulation array parameters
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        df = pd.DataFrame({"timestamp": dates, "open": 10, "high": 12, "low": 8, "close": 11, "volume": 1000})
        
        self.degradation_registry[symbol] = {"funding": False, "oi": False}
        if np.random.random() > 0.8:
            self.degradation_registry[symbol]["funding"] = True
        if np.random.random() > 0.8:
            self.degradation_registry[symbol]["oi"] = True
            
        return df

    async def close(self):
        pass

async def run_profiling():
    print("--- SPRINT 5: EXECUTION SUB-STAGE DECOUPLED PROFILING ---")
    config = load_config("configs/default.yaml")
    
    from defihunter.execution.scanner import ScanPipeline
    pipeline = ScanPipeline(config, fetcher=MockFetcher())
    
    t0 = time.time()
    await pipeline.run()
    elapsed = time.time() - t0
    
    print("\n--- STAGE TIMINGS BREAKDOWN ---")
    for key, ms in pipeline.timings.items():
        print(f"[Profiling] {key:25} -> {ms:7.1f} ms")
        
    print("\n--- TELEMETRY METRICS ---")
    print(f"[Profiling] Total Universe Evaluated: {len(pipeline.symbol_data_map)}")
    print(f"[Profiling] Registry Funding Objects: {pipeline.degraded_funding_count}")
    print(f"[Profiling] Registry OI Objects:      {pipeline.degraded_oi_count}")
    print(f"[Profiling] Total Profiling Execution:{elapsed:.2f}s")
    
if __name__ == '__main__':
    asyncio.run(run_profiling())
