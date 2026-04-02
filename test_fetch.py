import asyncio
import pandas as pd
from defihunter.data.binance_fetcher import BinanceFuturesFetcher

async def run():
    f = BinanceFuturesFetcher()
    df = await f.async_fetch_ohlcv('BTC.p', limit=50)
    print("Length:", len(df))
    if not df.empty:
        print("Columns:", df.columns.tolist())
    await f.close()

if __name__ == '__main__':
    asyncio.run(run())
