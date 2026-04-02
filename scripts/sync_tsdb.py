import sys
import os
import argparse
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List

# Ensure project root is in path
sys.path.append(os.getcwd())

from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.binance_vision import BinanceVisionFetcher
from defihunter.data.storage import TSDBManager
from defihunter.data.normalizer import CanonicalNormalizer
from defihunter.data.merger import DataMerger
from defihunter.utils.logger import logger

async def sync_symbol(symbol: str, timeframe: str, days_history: int = 30, overlap_days: int = 3, force: bool = False):
    """
    Syncs a single symbol from multiple sources into the TSDB with logical idempotency.
    """
    tsdb = TSDBManager()
    fetcher_binance = BinanceFuturesFetcher()
    fetcher_portal = BinanceVisionFetcher()
    
    logger.info(f"--- Logical Sync Start: {symbol} ({timeframe}) ---")
    
    # 1. Load Local & Check State for Idempotency
    df_local = tsdb.load_dataframe(symbol, timeframe)
    
    # Logical check: if local data already covers requested range with Portal/Binance provenance, skip portal
    if not df_local.empty and not force:
        counts = df_local.groupby('source').size().to_dict()
        latest_ts = df_local['timestamp'].max()
        logger.info(f"[Sync] Existing records: {counts}. Latest: {latest_ts}")
        
        # If we have Portal data and we're fairly recent, we might skip bulk fetch
        if counts.get('BinanceVision', 0) > 0 and (datetime.now() - latest_ts) < timedelta(hours=1):
             logger.info(f"[Sync] Logical state satisfied. Skipping bulk portal fetch for {symbol}.")
             # We still fetch recent Binance data to be safe (Incremental)
             start_portal = None 
        else:
             start_portal = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')
    else:
        df_local = pd.DataFrame(columns=CanonicalNormalizer.CANONICAL_COLUMNS)
        start_portal = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')

    # 2. Fetch Portal (Historical)
    df_portal = pd.DataFrame()
    if start_portal:
        end_portal = (datetime.now() - timedelta(days=overlap_days)).strftime('%Y-%m-%d')
        df_portal = fetcher_portal.fetch_historical_ohlcv(symbol, timeframe, start_portal, end_portal)
        # Ensure df_portal is canonical if it returned data
        if not df_portal.empty and 'timestamp' not in df_portal.columns:
            logger.error(f"[Sync] {symbol} Portal data missing timestamp column.")
            df_portal = pd.DataFrame()
    
    # 3. Fetch Binance (Recent)
    limit = (overlap_days + 1) * 96 if timeframe == '15m' else (overlap_days + 1) * 24
    df_binance = fetcher_binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not df_binance.empty:
        df_binance = CanonicalNormalizer.normalize(df_binance, symbol, timeframe, "Binance", priority=90)
    
    # 4. Deterministic Merge
    # We pass df_local back into the merger to ensure we don't LOSE provenance of old data
    merged_df = DataMerger.merge_sources([df_local, df_portal, df_binance])
    
    # 5. Save and Report
    if not merged_df.empty:
        # Detect if anything actually changed (Logical equality)
        if not df_local.empty and len(merged_df) == len(df_local):
            # Potential check: compare specific columns or hashes if size is same
            logger.info(f"[Sync] No new logical data added for {symbol}. TSDB is current.")
        else:
            success = tsdb.save_dataframe(merged_df, symbol, timeframe)
            if success:
                new_counts = merged_df.groupby('source').size().to_dict()
                logger.info(f"[Sync] Completed {symbol}. New Distribution: {new_counts}")
            else:
                logger.error(f"[Sync] Failed to persist {symbol}")
    
    await fetcher_binance.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Source Data Synchronization (Idempotent)")
    parser.add_argument("--symbols", type=str, default="BTC.p,ETH.p", help="Comma separated symbols")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (15m, 1h, 4h)")
    parser.add_argument("--days", type=int, default=30, help="Bulk history days")
    parser.add_argument("--overlap", type=int, default=3, help="Binance overlap days")
    parser.add_argument("--force", action="store_true", help="Force refresh even if TSDB is state-satisfied")
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    loop = asyncio.get_event_loop()
    for sym in symbols:
        try:
            loop.run_until_complete(sync_symbol(sym, args.timeframe, args.days, args.overlap, args.force))
        except Exception as e:
            logger.error(f"[Sync] Unexpected error for {sym}: {e}")
