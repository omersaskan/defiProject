import asyncio
import pandas as pd
import time
from unittest.mock import AsyncMock, MagicMock
from defihunter.data.binance_fetcher import BinanceFuturesFetcher

async def prove_cache_effectiveness():
    """
    EXPLICIT CACHE PROOF.
    Verifies that the 2nd call does NOT trigger new API requests.
    """
    print("--- [PROOF] Verifying Cache Effectiveness ---")
    
    # 1. Reset class-level cache
    BinanceFuturesFetcher._history_cache = {}
    fetcher = BinanceFuturesFetcher(cache_ttl=600)
    ts_ms = 1700000000000 
    
    # Use AsyncMock for network calls, but MagicMock for synchronous helper methods
    mock_exch = AsyncMock()
    # milliseconds() in CCXT is a synchronous helper, not a coroutine
    mock_exch.milliseconds = MagicMock(return_value=ts_ms + 10000)
    
    mock_exch.fetch_ohlcv.return_value = [[ts_ms, 100.0, 105.0, 95.0, 102.0, 1000.0, ts_ms+60000, 100000.0, 100, 600.0, 60000.0, 0]]
    mock_exch.fetch_funding_rate_history.return_value = [{'timestamp': ts_ms - 1000, 'fundingRate': 0.0001}]
    mock_exch.fetch_open_interest_history.return_value = [{'timestamp': ts_ms - 1000, 'openInterestValue': 500000.0}]
    
    fetcher.a_exchange = mock_exch
    
    # 2. First fetch (Populates cache)
    print("Executing Fetch 1...")
    await fetcher.async_fetch_ohlcv("BTC.p", limit=1)
    pop_count = mock_exch.fetch_funding_rate_history.call_count
    print(f"API Calls after Fetch 1: {pop_count}")
    
    # 3. Second fetch (Should hit cache)
    print("Executing Fetch 2...")
    await fetcher.async_fetch_ohlcv("BTC.p", limit=1)
    hit_count = mock_exch.fetch_funding_rate_history.call_count
    print(f"API Calls after Fetch 2: {hit_count}")
    
    # PROOF: Counter did not increment
    if hit_count == pop_count and hit_count > 0:
        print(f"SUCCESS: Cache hit confirmed ({hit_count} calls total).")
    else:
        print(f"FAILED: Cache Logic Drift. Pop: {pop_count}, Hit: {hit_count}")
        raise AssertionError("Cache Proof Failed")

if __name__ == "__main__":
    asyncio.run(prove_cache_effectiveness())
