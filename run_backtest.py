import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.features import build_feature_pipeline
from defihunter.execution.backtest import BacktestEngine
from defihunter.execution.pipeline import SignalPipeline
from defihunter.core.config import load_config
from defihunter.utils.logger import logger

def run_historical_backtest(config_path="configs/default.yaml", limit=1000, k=3):
    """
    End-to-end backtest pipeline for true multi-coin cross-sectional leader testing.
    Now uses the unified SignalPipeline to guarantee live-parity.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting cross-sectional leader backtest...")
    config = load_config(config_path)
    fetcher = BinanceFuturesFetcher()
    
    symbols = config.universe.defi_universe if config.universe.defi_universe else config.anchors
    if not symbols:
        print("No universe found in config!")
        return

    # To keep it fast for testing, let's take a subset
    symbols = symbols[:12]
        
    print(f"Fetching context for anchors... (Timeframe: {config.timeframe})")
    anchor_data_mtf = {}
    for anchor in config.anchors:
        anchor_data_mtf[anchor] = {}
        for tf in ["15m", "1h", "4h"]:
            adf = fetcher.fetch_ohlcv(anchor, timeframe=tf, limit=limit + 100) # Buffer for features
            if not adf.empty:
                adf = build_feature_pipeline(adf, timeframe=tf)
                anchor_data_mtf[anchor][tf] = adf
            
    # Initialize unified pipeline
    pipeline = SignalPipeline(config)
    
    # 1. PRE-PROCESS ALL SYMBOLS (PHASE 0)
    all_processed_dfs = {}
    print(f"Pre-calculating features for {len(symbols)} symbols...")
    for symbol in symbols:
        df = fetcher.fetch_ohlcv(symbol, timeframe=config.timeframe, limit=limit + 100)
        if df.empty: continue
        df = build_feature_pipeline(df, timeframe=config.timeframe)
        all_processed_dfs[symbol] = df

    # 2. RUN PIPELINE OVER TIME (PARITY REPLAY)
    # We iterate through unique timestamps to ensure cross-sectional logic (ML/Family) matches live exactly.
    common_timestamps = sorted(next(iter(all_processed_dfs.values()))['timestamp'].unique()) if all_processed_dfs else []
    relevant_timestamps = common_timestamps[-limit:]
    
    all_snapshot_rows = []
    
    print(f"Running canonical SignalPipeline replay across {len(relevant_timestamps)} bars...")
    for i, ts in enumerate(relevant_timestamps):
        if i % 100 == 0:
            print(f"  → bar {i}/{len(relevant_timestamps)}...")
            
        # Create symbol_data_map sliced at ts
        symbol_data_map = {}
        for symbol, df in all_processed_dfs.items():
            mask = df['timestamp'] <= ts
            sliced_df = df[mask].tail(150) # Buffer for rolling features in pipeline
            if not sliced_df.empty:
                symbol_data_map[symbol] = sliced_df
        
        # Create anchor_context sliced at ts
        anchor_context = {}
        for anchor, mtfs in anchor_data_mtf.items():
            anchor_context[anchor] = {}
            for tf, adf in mtfs.items():
                # Slice anchor data up to ts (handling different timeframes)
                mask = adf['timestamp'] <= ts
                anchor_context[anchor][tf] = adf[mask].tail(100)
        
        # Run Pipeline
        result = pipeline.run(
            symbol_data_map=symbol_data_map,
            anchor_context=anchor_context,
            mode="historical",
            scan_timestamp=ts
        )
        
        if not result.master_df.empty:
            batch_df = result.master_df.copy()
            batch_df['timestamp'] = ts
            
            # Map decisions to entry_signal for BacktestEngine compatibility
            trade_set = {d.symbol for d in result.final_decisions if d.decision == "trade"}
            batch_df['entry_signal'] = batch_df['symbol'].isin(trade_set)
            
            # Carry over critical columns for BacktestEngine simulation
            all_snapshot_rows.append(batch_df)

    if not all_snapshot_rows:
        print("No valid data processed.")
        return
        
    combined_df = pd.concat(all_snapshot_rows, ignore_index=True)
    combined_df = combined_df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)
    
    print(f"Evaluating simulation for {len(symbols)} coins...")
    bt_engine = BacktestEngine(config=config)
    
    results = bt_engine.simulate(combined_df)
    
    print("\n[BACKTEST RESULTS]")
    print(f"Total Trades: {results.get('total_trades')}")
    print(f"Win Rate: {results.get('win_rate')}%")
    print(f"Profit Factor: {results.get('profit_factor')}")
    print(f"Expectancy (R): {results.get('expectancy_r')}")

    bars_horizon = 96 if config.timeframe == '15m' else (24 if config.timeframe == '1h' else 6)
    metrics = bt_engine.evaluate_ranking_quality(combined_df, bars_horizon=bars_horizon, k=k)
    print("\n[LEADER CAPTURE METRICS]")
    for k_metric, v_metric in metrics.items():
        print(f"{k_metric}: {v_metric}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    run_historical_backtest(k=args.k, limit=args.limit)
