import os
import io
import sys
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from defihunter.data.storage import TSDBManager
from defihunter.data.binance_fetcher import BinanceFuturesFetcher

def download_zip(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # There should be exactly one CSV in this zip
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # columns: Open Time, Open, High, Low, Close, Volume, Close Time, Quote Volume, ...
                    df = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5, 7], 
                                     names=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    # Vision Portal only provides OHLCV — funding/OI will be merged separately
                    df['open_interest'] = 0.0
                    df['funding_rate'] = 0.0  # Will be overwritten by merge_funding_history
                    df['spread_bps'] = 5.0
                    return df
        return None
    except Exception as e:
        return None


def merge_funding_history(parquet_df: pd.DataFrame, fetcher: BinanceFuturesFetcher, symbol: str, days: int) -> pd.DataFrame:
    """
    GT #3 / BUG #2 FIX: Fetches real historical funding rates from Binance and merges
    them into the OHLCV dataframe loaded from parquet (which had funding_rate=0.0).
    Funding is updated every 8h by Binance, so we forward-fill between updates.
    """
    funding_df = fetcher.fetch_historical_funding(symbol, days=days)
    if funding_df.empty:
        print(f"  [{symbol}] No funding history available — keeping 0.0")
        return parquet_df
    
    # Ensure both indexes are datetime for proper merge
    parquet_df = parquet_df.copy()
    parquet_df = parquet_df.set_index('timestamp')
    
    # Reindex funding to OHLCV timestamps, forward-fill (funding valid until next update)
    funding_aligned = funding_df['funding_rate'].reindex(
        parquet_df.index, method='ffill'
    ).fillna(0.0)
    
    parquet_df['funding_rate'] = funding_aligned.values
    parquet_df = parquet_df.reset_index()
    
    filled = (parquet_df['funding_rate'] != 0.0).sum()
    print(f"  [{symbol}] Funding merged: {filled}/{len(parquet_df)} bars have real funding_rate")
    return parquet_df

# S3 Fix: dict-based timeframe → seconds mapping
TF_SECONDS = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600,
    '8h': 28800, '12h': 43200, '1d': 86400
}

def _naive_utc(ts) -> datetime:
    """Ensure a pandas Timestamp or datetime is naive UTC (strip tzinfo)."""
    if ts is None:
        return None
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)  # S1+S2 Fix: strip timezone for safe comparison
    return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)

def process_symbol(symbol: str, timeframe: str, days: int, tsdb: TSDBManager, fetcher: BinanceFuturesFetcher):
    base = symbol.upper().replace('.P', '').split('/')[0]
    bv_symbol = f"{base}USDT"
    
    # S1+S2 Fix: always use naive UTC for all comparisons
    end_date = datetime.utcnow().replace(tzinfo=None)
    start_date = end_date - timedelta(days=days)
    
    # 1. Check existing Parquet state — normalize to naive UTC
    latest_ts_raw = tsdb.get_latest_timestamp(symbol, timeframe)
    latest_ts = _naive_utc(latest_ts_raw)  # S1+S2 Fix: guaranteed naive datetime
    
    # If we already have recent data, we might not need to hit the Vision Portal at all
    if latest_ts and latest_ts > (end_date - timedelta(days=2)):
        print(f"[{symbol}] Data is fresh up to {latest_ts}. Skipping bulk Vision download.")
    else:
        # If we have SOME data but missing a big chunk, or no data at all
        base_url = "https://data.binance.vision/data/futures/um/daily/klines"
        urls = []
        
        # We start looking from the latest timestamp we have, or from `days` ago
        # S2 Fix: latest_ts is already naive UTC (from _naive_utc above), safe to .replace()
        current_date_iter = latest_ts.replace(hour=0, minute=0, second=0, microsecond=0) if latest_ts else start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        while current_date_iter <= end_date:
            date_str = current_date_iter.strftime("%Y-%m-%d")
            url = f"{base_url}/{bv_symbol}/{timeframe}/{bv_symbol}-{timeframe}-{date_str}.zip"
            urls.append(url)
            current_date_iter += timedelta(days=1)
            
        if urls:
            print(f"[{symbol}] Downloading {len(urls)} daily chunks from Vision Portal (Timeframe: {timeframe})...")
            dfs = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(download_zip, url): url for url in urls}
                for future in as_completed(future_to_url):
                    res = future.result()
                    if res is not None and not res.empty:
                        dfs.append(res)
                        
            if dfs:
                bulk_df = pd.concat(dfs, ignore_index=True)
                bulk_df = bulk_df.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
                print(f"[{symbol}] Writing {len(bulk_df)} Vision Portal rows to TSDB...")
                tsdb.save_dataframe(bulk_df, symbol, timeframe)
            else:
                print(f"[{symbol}] No new daily zips found on Vision Portal.")
                
    # 2. HYBRID PADDING WITH CCXT (Fills the gap from yesterday to RIGHT NOW)
    latest_ts = tsdb.get_latest_timestamp(symbol, timeframe)
    
    if latest_ts:
        since_ms = int(latest_ts.timestamp() * 1000)
        # Add 1ms so we don't re-download the exact last candle
        since_ms += 1 
        
        diff = (datetime.utcnow().replace(tzinfo=None) - latest_ts).total_seconds()  # S1 Fix: naive comparison
        # S3 Fix: dict-based lookup, safe fallback for unknown timeframes
        tf_secs = TF_SECONDS.get(timeframe, 900)  # default to 15m if unknown
        if diff > tf_secs:
            print(f"[{symbol}] Padding missing recent data from CCXT via active API...")
            recent_df = fetcher.fetch_historical_ohlcv(symbol, timeframe, since_ms=since_ms)
            
            if not recent_df.empty:
                print(f"[{symbol}] Added {len(recent_df)} active API rows.")
                tsdb.save_dataframe(recent_df, symbol, timeframe)
    else:
        # Pure ccxt fallback if symbol isn't listed on Vision portal yet
        print(f"[{symbol}] Vision Portal failed entirely. Fallback to full active API fetch...")
        fallback_df = fetcher.fetch_historical_ohlcv(symbol, timeframe, days=days)
        if not fallback_df.empty:
            tsdb.save_dataframe(fallback_df, symbol, timeframe)
    
    # GT #3 / BUG #2 FIX: Merge real funding history into parquet after download
    # Vision Portal zips only have OHLCV (funding_rate=0.0 by default)
    # Only merge for 1h timeframe to avoid redundant API calls (funding updates every 8h)
    if timeframe == '1h':
        print(f"[{symbol}] Merging real funding rate history (GT #3)...")
        try:
            existing = tsdb.load_dataframe(symbol, timeframe)
            if existing is not None and not existing.empty and 'timestamp' in existing.columns:
                merged = merge_funding_history(existing, fetcher, symbol, days=days)
                tsdb.save_dataframe(merged, symbol, timeframe)
        except Exception as e:
            print(f"  [{symbol}] Funding merge failed: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="Days of history to construct")
    parser.add_argument("--timeframes", type=str, default="15m,1h,4h", help="Comma-separated timeframes")
    args = parser.parse_args()
    
    timeframes = args.timeframes.split(",")
    fetcher = BinanceFuturesFetcher()
    tsdb = TSDBManager()
    
    universe = fetcher.get_defi_universe()
    if not universe:
        print("Failed to fetch universe.")
        return
        
    universe = universe[:200]
    print(f"Starting FAST TSDB Sync for {len(universe)} symbols over {args.days} days.")
    for tf in timeframes:
        for sym in universe: 
            process_symbol(sym, tf, args.days, tsdb, fetcher)
            
if __name__ == "__main__":
    main()
