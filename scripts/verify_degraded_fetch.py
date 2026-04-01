import asyncio
from unittest.mock import AsyncMock
from defihunter.data.binance_fetcher import BinanceFuturesFetcher

async def verify_degraded_parity():
    """
    PROPER DEGRADED PARITY VERIFICATION.
    Mocks a scenario where history is UNAVAILABLE to prove fallback safety.
    """
    print("--- [DEGRADED] Verifying Fallback Path (Empty History) ---")
    fetcher = BinanceFuturesFetcher()
    
    # 1. Mock exchange to return EMPTY history
    mock_exch = AsyncMock()
    mock_exch.fetch_ohlcv.return_value = [[1000, 10, 11, 9, 10.5, 100, 1000, 1000, 1, 1, 1, 0]]
    # These return empty DFs in the fetcher logic
    mock_exch.fetch_funding_rate_history.return_value = []
    mock_exch.fetch_open_interest_history.return_value = []
    mock_exch.milliseconds.return_value = 2000
    fetcher.a_exchange = mock_exch

    # 2. RUN FETCH
    df = await fetcher.async_fetch_ohlcv("BTC.p", limit=1)
    
    # 3. VERIFY FALLBACK
    print(f"Funding Rate: {df.iloc[0].get('funding_rate')}")
    print(f"Open Interest: {df.iloc[0].get('open_interest')}")
    
    is_degraded_f = getattr(df, 'attrs', {}).get('degraded_parity_funding', False)
    is_degraded_o = getattr(df, 'attrs', {}).get('degraded_parity_oi', False)
    is_degraded_legacy = getattr(df, 'attrs', {}).get('degraded_parity', False)

    print(f"Degraded Flags: Funding={is_degraded_f}, OI={is_degraded_o}, Legacy={is_degraded_legacy}")
    
    assert df.iloc[0]['funding_rate'] == 0.0, "Fallback failed: funding not 0.0"
    assert df.iloc[0]['open_interest'] == 0.0, "Fallback failed: OI not 0.0"
    assert is_degraded_f and is_degraded_o, "Degraded flags not set"
    assert is_degraded_legacy, "Legacy degraded flag not set"

    print("SUCCESS: Degraded parity fallback is safe and detectable.")

if __name__ == "__main__":
    asyncio.run(verify_degraded_parity())
