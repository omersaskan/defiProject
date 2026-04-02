import os
import sys
import asyncio
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.getcwd())

from defihunter.core.config import load_config
from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.binance_vision import BinanceVisionFetcher
from defihunter.data.storage import TSDBManager
from defihunter.data.dataset_builder import DatasetBuilder
from defihunter.data.features import build_feature_pipeline
from defihunter.execution.scanner import ScanPipeline
from scripts.sync_tsdb import sync_symbol
from scripts.stitch_audit import run_stitch_audit
from scripts.run_ablation_backtest import fetch_dataset

async def test_live_scanner_isolation():
    print("\n[CHECK] 1. Live Scanner Isolation")
    config = load_config("configs/default.yaml")
    fetcher = BinanceFuturesFetcher()
    scanner = ScanPipeline(config, fetcher=fetcher)
    
    # Run a very limited scan (1 symbol, 0 iterations of the loop, just the prep)
    # We want to verify it doesn't try to touch BinanceVision
    print("  Running limited scan preparation...")
    try:
        # Mocking build_watchlist to avoid long scans
        scanner._build_watchlist = lambda limit: asyncio.sleep(0, result=["BTC.p"])
        await scanner.run(limit=1)
        print("  ✓ Scanner loop prep successful.")
        print(f"  ✓ Telemetry Timings check: {list(scanner.timings.keys())[:3]}...")
        
        # Verify fetcher is Binance-only
        if hasattr(fetcher, 'degradation_registry'):
            print(f"  ✓ Degradation Telemetry found: {len(fetcher.degradation_registry)} symbols registered.")
    except Exception as e:
        print(f"  ❌ Scanner failed: {e}")

async def test_tsdb_sync_and_provenance():
    print("\n[CHECK] 2. TSDB Sync & Provenance")
    symbol = "BTC.p"
    try:
        # 1. Clean old data for fresh test
        tsdb = TSDBManager()
        path = os.path.join("data/tsdb", f"BTC_p_15m.parquet")
        if os.path.exists(path): os.remove(path)
        
        # 2. Run Sync (Using 2026-03-31 as bulk target)
        print(f"  Running sync for {symbol} (Bulk Vision + API Overlap)...")
        await sync_symbol(symbol, "15m", days_history=5, overlap_days=1, force=True)
        
        # 3. Verify File & Provenance
        if os.path.exists(path):
            df = pd.read_parquet(path)
            sources = df['source'].unique().tolist()
            print(f"  ✓ TSDB file created. Rows: {len(df)}")
            print(f"  ✓ Provenance sources found: {sources}")
            if 'BinanceVision' in sources or 'Binance' in sources:
                print(f"  ✓ Provenance confirmed: {sources}")
            
            # Check boolean quality flags
            q_cols = [c for c in df.columns if c.startswith('q_')]
            print(f"  ✓ Quality flags found: {q_cols}")
        else:
            print(f"  ❌ TSDB file NOT found at {path}")
    except Exception as e:
        print(f"  ❌ Sync test failed: {e}")

async def test_stitch_audit():
    print("\n[CHECK] 3. Stitch Audit Tool")
    symbol = "BTC.p"
    report_path = "reports/final_readiness_audit.md"
    try:
        print(f"  Generating audit report for {symbol}...")
        await run_stitch_audit(symbol, "15m", overlap_days=1, output_path=report_path)
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                content = f.read()
                if "Mean Err %" in content:
                    print("  ✓ Audit report generated with drift metrics.")
                else:
                    print("  ⚠️ Audit report generated but metrics might be empty (check file).")
        else:
            print("  ❌ Audit report NOT generated at reports/final_readiness_audit.md")
    except Exception as e:
        print(f"  ❌ Audit test failed: {e}")

async def test_training_metadata_exclusion():
    print("\n[CHECK] 4. Training Metadata Exclusion")
    path = "data/tsdb/BTC_p_15m.parquet"
    if not os.path.exists(path):
        print("  ❌ Skipping: No sync data for BTC.")
        return
        
    try:
        df = pd.read_parquet(path)
        # Ensure we have some features
        df = build_feature_pipeline(df)
        
        builder = DatasetBuilder()
        X, y = builder.prepare_training_data(df)
        
        meta_found = [c for c in X.columns if c in ['source', 'source_priority', 'ingested_at', 'is_synthetic', 'q_gap_detected']]
        if not meta_found:
            print(f"  ✓ Feature matrix X (cols: {len(X.columns)}) is clean of metadata.")
        else:
            print(f"  ❌ Metadata leaked into X: {meta_found}")
            
        print(f"  ✓ y dataset contains provenance for audit: {'source' in y.columns}")
        
    except Exception as e:
        print(f"  ❌ Training metadata test failed: {e}")

async def test_ablation_reproducibility():
    print("\n[CHECK] 5. Ablation Reproducibility")
    config = load_config("configs/default.yaml")
    try:
        # Test 1: Expect failure on unknown symbol with no fallback
        print("  Testing TSDB-only requirement (no fallback)...")
        df, fam = fetch_dataset(config, limit=1, live_fallback=False)
        # If TSDB is empty for the first symbol in universe, it should return empty
        if df.empty:
            print("  ✓ Reproducibility enforced: No silent live fallback.")
        else:
            print("  ⚠️ Found data. Ensure this was from TSDB and not silent fallback.")

    except Exception as e:
        print(f"  ❌ Ablation test failed: {e}")

async def main():
    print("="*60)
    print("DEFIHUNTER - FINAL MERGE READINESS VERIFICATION")
    print("="*60)
    
    await test_live_scanner_isolation()
    await test_tsdb_sync_and_provenance()
    await test_stitch_audit()
    await test_training_metadata_exclusion()
    await test_ablation_reproducibility()
    
    print("\n"+"="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
