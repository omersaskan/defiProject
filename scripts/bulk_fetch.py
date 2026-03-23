import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.features import build_feature_pipeline
from defihunter.data.storage import TSDBManager

def main():
    parser = argparse.ArgumentParser(description="Bulk DeFi Data Fetcher for ML Training")
    parser.add_argument('--days', type=int, default=180, help='Number of days (default: 180)')
    parser.add_argument('--timeframes', type=str, default='15m,1h', help='Comma-separated timeframes (e.g. 5m,15m,30m,1h)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (optional)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of coins to fetch (e.g. 50)')
    
    args = parser.parse_args()
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    fetcher = BinanceFuturesFetcher()
    tsdb = TSDBManager()
    
    if args.symbols:
        universe = [s.strip() for s in args.symbols.split(',')]
    else:
        print("Fetching full DeFi universe list...")
        universe = fetcher.get_defi_universe()
        if args.limit > 0:
            print(f"Applying limit: Top {args.limit} coins.")
            universe = universe[:args.limit]
            
    print(f"Targeting {len(universe)} symbols across {timeframes} for {args.days} days.")
    
    summary_stats = []
    
    for tf in timeframes:
        print(f"\n{'='*20} TIMEFRAME: {tf} {'='*20}")
        raw_dir = f"data/raw/{tf}"
        proc_dir = f"data/processed/{tf}"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(proc_dir, exist_ok=True)
        
        for symbol in universe:
            try:
                print(f"[{symbol} | {tf}] Starting fetch sequence...")
                raw_path = f"{raw_dir}/{symbol.replace('.p', '_raw')}.csv"
                
                existing_df = pd.DataFrame()
                fetch_since_ms = None
                
                if os.path.exists(raw_path):
                    try:
                        existing_df = pd.read_csv(raw_path)
                        if not existing_df.empty and 'timestamp' in existing_df.columns:
                            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                            last_ts = existing_df['timestamp'].max()
                            fetch_since_ms = int(last_ts.timestamp() * 1000) + 1
                            print(f"[{symbol}] Resuming from {last_ts}")
                    except Exception as e:
                        print(f"[{symbol}] Error reading file: {e}")

                if fetch_since_ms:
                    last_known = existing_df['timestamp'].max()
                    # Timezone fix: strip tz if present before comparing with naive datetime.now()
                    if hasattr(last_known, 'tzinfo') and last_known.tzinfo is not None:
                        last_known = last_known.replace(tzinfo=None)
                    delta = datetime.now() - last_known
                    fetch_days = max(1, delta.days + 1)
                    df_new = fetcher.fetch_historical_ohlcv(symbol, timeframe=tf, days=fetch_days)
                    if not df_new.empty:
                        # Normalize both sides for safe comparison
                        existing_max = pd.to_datetime(existing_df['timestamp'].max())
                        if existing_max.tzinfo is not None:
                            existing_max = existing_max.replace(tzinfo=None)
                        new_ts = pd.to_datetime(df_new['timestamp'])
                        if new_ts.dt.tz is not None:
                            new_ts = new_ts.dt.tz_localize(None)
                        df_new = df_new[new_ts > existing_max]
                else:
                    df_new = fetcher.fetch_historical_ohlcv(symbol, timeframe=tf, days=args.days)
                
                if df_new.empty and existing_df.empty: continue
                
                df_final = pd.concat([existing_df, df_new]).drop_duplicates(subset=['timestamp']).sort_values('timestamp') if not existing_df.empty else df_new
                
                if df_final.empty: continue

                df_final.to_csv(raw_path, index=False)
                print(f"[{symbol}] Updated: {len(df_final)} total rows.")
                
                print(f"[{symbol}] Computing features...")
                df_processed = build_feature_pipeline(df_final)
                processed_path = f"{proc_dir}/{symbol.replace('.p', '_processed')}.csv"
                df_processed.to_csv(processed_path, index=False)
                
                # NEW: Save to TSDB Parquet for ML training
                tsdb.save_dataframe(df_processed, symbol, tf)
                print(f"[{symbol}] Synced to TSDB Parquet.")
                
                summary_stats.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "total_rows": len(df_final),
                    "new_rows": len(df_new),
                    "start": df_final['timestamp'].min(),
                    "end": df_final['timestamp'].max()
                })
                
            except Exception as e:
                print(f"[{symbol}] Critical error during {tf} fetch: {e}")
                
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv("data/collection_summary.csv", index=False)
        print("\n" + "="*40)
        print("COLLECTION COMPLETE")
        print(summary_df)
        print("="*40)

if __name__ == "__main__":
    main()
