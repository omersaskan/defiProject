import sys
import os
import argparse
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.getcwd())

from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.binance_vision import BinanceVisionFetcher
from defihunter.data.normalizer import CanonicalNormalizer
from defihunter.utils.logger import logger

async def run_stitch_audit(symbol: str, timeframe: str, overlap_days: int = 3, output_path: str = "quality_report.md"):
    """
    Independent audit script. Fetches overlapping data from Binance Vision (Portal) and Binance API,
    normalizes them, and generates a detailed drift report.
    """
    fetcher_binance = BinanceFuturesFetcher()
    fetcher_portal = BinanceVisionFetcher()
    
    logger.info(f"--- Audit Start: {symbol} ({timeframe}) ---")
    
    # 1. Fetch Overlap
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=overlap_days)
    
    df_portal = fetcher_portal.fetch_historical_ohlcv(symbol, timeframe, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
    # Use sync fetch and pass limit equivalent to overlap_days (approximate)
    limit = (overlap_days + 1) * 24 * (60 // int(timeframe.replace('m','').replace('h','0'))) if 'h' not in timeframe else (overlap_days + 1) * 24 // int(timeframe.replace('h',''))
    if timeframe == '15m': limit = (overlap_days + 1) * 96
    df_binance = fetcher_binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    logger.info(f"[Debug] Portal: {len(df_portal)} rows. Range: {df_portal['timestamp'].min()} to {df_portal['timestamp'].max()}")
    if not df_binance.empty:
        logger.info(f"[Debug] Binance: {len(df_binance)} rows. Range: {df_binance['timestamp'].min()} to {df_binance['timestamp'].max()}")
        df_binance = CanonicalNormalizer.normalize(df_binance, symbol, timeframe, "Binance", priority=90)
    
    if df_portal.empty or df_binance.empty:
        logger.error("[Audit] Insufficient data for stitch audit.")
        return

    # 2. Alignment
    merged = df_portal.merge(df_binance, on='timestamp', suffixes=('_p', '_b'), how='inner')
    if merged.empty:
        logger.error("[Audit] No overlapping timestamps found.")
        return
        
    # 3. Comprehensive Metrics Calculation
    audit_cols = ['close', 'high', 'low', 'volume', 'funding_rate', 'open_interest']
    stats = []
    
    for col in audit_cols:
        p_vals = merged[f'{col}_p'].astype(float)
        b_vals = merged[f'{col}_b'].astype(float)
        
        abs_diff = (p_vals - b_vals).abs()
        pct_err = (abs_diff / (b_vals + 1e-8)) * 100
        
        stats.append({
            "Field": col,
            "Mean Error (%)": pct_err.mean() if col in ['close', 'high', 'low', 'volume'] else "N/A",
            "Max Error (%)": pct_err.max() if col in ['close', 'high', 'low', 'volume'] else "N/A",
            "Mean Abs Diff": abs_diff.mean(),
            "Max Abs Diff": abs_diff.max()
        })

    # 4. Generate Markdown Report
    report = [
        f"# Data Quality Report: {symbol} ({timeframe})",
        f"**Audit Period**: {start_dt.date()} to {end_dt.date()}",
        f"**Overlap Samples**: {len(merged)}",
        f"**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| Field | Mean Err % | Max Err % | Mean Abs Diff | Max Abs Diff |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for s in stats:
        m_err = f"{s['Mean Error (%)']:.6f}%" if isinstance(s['Mean Error (%)'], float) else s['Mean Error (%)']
        x_err = f"{s['Max Error (%)']:.6f}%" if isinstance(s['Max Error (%)'], float) else s['Max Error (%)']
        report.append(f"| {s['Field']} | {m_err} | {x_err} | {s['Mean Abs Diff']:.6f} | {s['Max Abs Diff']:.6f} |")
    
    # 5. Threshold Validation
    max_price_drift = max(s['Max Error (%)'] for s in stats if s['Field'] == 'close')
    if max_price_drift > 0.05:
        report.append(f"\n> [!CAUTION]\n> Critical price drift detected: {max_price_drift:.4f}% exceeds 0.05% threshold.")
    else:
        report.append("\n> [!TIP]\n> Data alignment satisfies the 0.05% price drift threshold.")

    with open(output_path, "w") as f:
        f.write("\n".join(report))
        
    logger.info(f"[Audit] Quality report saved to {output_path}")
    await fetcher_binance.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Source Data Quality Audit & Reporting")
    parser.add_argument("--symbol", type=str, default="BTC.p")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--output", type=str, default="quality_report.md")
    
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_stitch_audit(args.symbol, args.timeframe, args.overlap, args.output))
    except Exception as e:
        logger.error(f"[Audit] Execution failed: {e}")
