"""
7-Day Shadow Validation Runner (Refactored)

Uses the unified DefiHunter SignalPipeline to guarantee perfect parity with the live scanner.
Fetches historical data, replays the pipeline, logs to ShadowLogger, and links outcomes.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.features import build_feature_pipeline
from defihunter.execution.pipeline import SignalPipeline
from defihunter.execution.backtest import BacktestEngine
from defihunter.validation.shadow_logger import ShadowLogger
from defihunter.validation.outcome_linker import OutcomeLinker
from defihunter.engines.adaptive_stop import AdaptiveStopEngine
from defihunter.core.config import load_config
from defihunter.utils.logger import logger

def _build_price_df_from_map(symbol_data_map: dict, res) -> pd.DataFrame:
    """Flatten symbol_data_map into a cross-sectional price DataFrame for OutcomeLinker."""
    rows = []
    for sym, df in symbol_data_map.items():
        # Optimization: OutcomeLinker needs historical window + future window
        for _, bar in df.iterrows():
            ts = bar.get("timestamp", bar.name)
            ctx = res.symbol_context_map.get(sym, {})
            rows.append({
                "symbol":    sym,
                "timestamp": pd.to_datetime(ts, utc=True),
                "close":     bar["close"],
                "high":      bar.get("high", bar["close"]),
                "low":       bar.get("low",  bar["close"]),
                "family":    ctx.get("family", "unknown"),
            })
    return pd.DataFrame(rows)

def run_shadow_validation(
    config_path: str = "configs/default.yaml",
    days: int = 7,
    symbol_limit: int = 80,
    log_path: str = "logs/shadow_log.csv",
):
    print(f"\n{'='*60}")
    print(f"  DefiHunter — Shadow Validation (Unified Pipeline)")
    print(f"{'='*60}")

    config = load_config(config_path)
    if not config:
        logger.error("Config not found. Exiting.")
        return

    fetcher = BinanceFuturesFetcher()
    pipeline = SignalPipeline(config)
    
    # ── [1/5] Fetch Universe & Data ──────────────────────────────────────
    print("\n[1/5] Fetching DeFi universe and OHLCV data...")
    universe = fetcher.get_defi_universe(config=config)[:symbol_limit]
    
    # Horizon: shadow scan at (now - days), outcomes tracked for 24h
    timeframe = "15m"
    bars_needed = (days + 2) * 24 * 4 # +2 days buffer
    sim_ts = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Pre-process anchors
    anchor_dfs = {}
    for anchor in config.anchors:
        df = fetcher.fetch_ohlcv(anchor, timeframe=timeframe, limit=bars_needed)
        if not df.empty:
            anchor_dfs[anchor] = {}
            for tf in ["15m", "1h", "4h"]:
                # For simplicity in this script, we fetch and build TF-specific frames
                adf = fetcher.fetch_ohlcv(anchor, timeframe=tf, limit=150) # Buffer for MTF
                anchor_dfs[anchor][tf] = build_feature_pipeline(adf, timeframe=tf)

    symbol_data_map = {}
    print(f"  Fetching {len(universe)} symbols...")
    
    for symbol in universe:
        try:
            df = fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=bars_needed)
            if not df.empty and len(df) > 55:
                # Bare minimum prep for the shared pipeline
                df = build_feature_pipeline(df, timeframe=timeframe)
                # Slice logic is handled inside pipeline if we provide specific scan_timestamp
                # but to be safe and efficient, we slice to scan_ts here.
                mask = df['timestamp'] <= sim_ts
                symbol_data_map[symbol] = df[mask].tail(150)
        except Exception as e:
            logger.warning(f"Fetch failed for {symbol}: {e}")

    # ── [2/5] Execute Unified Pipeline ───────────────────────────────────
    print(f"\n[2/5] Running SignalPipeline at {sim_ts}...")
    
    res = pipeline.run(
        symbol_data_map=symbol_data_map,
        anchor_context=anchor_dfs,
        mode="shadow",
        scan_timestamp=sim_ts
    )
    
    if res.master_df.empty:
        print("  ✗ No signals generated at this timestamp. Exiting.")
        return

    # ── [3/5] Adaptive Stop & Shadow Logging ──────────────────────────────
    print("\n[3/5] Logging decisions to shadow log...")
    shadow_logger = ShadowLogger(log_path=log_path)
    adaptive_stop_eng = AdaptiveStopEngine()
    
    adaptive_stop_map = {}
    setup_class_map = {}
    trade_decisions = {d.symbol for d in res.final_decisions if d.decision == "trade"}
    
    for d in res.final_decisions:
        sym_rows = res.master_df[res.master_df["symbol"] == d.symbol]
        if sym_rows.empty: continue
        
        atr = float(sym_rows["_atr"].iloc[-1])
        adaptive_stop_map[d.symbol] = adaptive_stop_eng.compute_stop(
            {"close": d.entry_price, "atr": atr},
            family=d.explanation.get("family", "defi_beta"),
            regime=res.regime_label,
            fakeout_risk=d.fakeout_risk
        )
        setup_class_map[d.symbol] = str(sym_rows.get("setup_class", pd.Series([""])).iloc[-1])

    shadow_logger.log_scan(
        decisions=res.final_decisions,
        regime=res.regime_label,
        universe_size=len(universe),
        timeframe=timeframe,
        adaptive_stop_map=adaptive_stop_map,
        paper_opened_symbols=trade_decisions,
        kelly_map={d.symbol: d.explanation.get("kelly_risk_pct", 0) for d in res.final_decisions},
        setup_class_map=setup_class_map,
        scan_timestamp=sim_ts
    )

    # ── [4/5] Generate Labels via Backtest Simulation ─────────────────────
    print("\n[4/5] Simulating trade outcomes for labels...")
    ts_rows = []
    for symbol, df in symbol_data_map.items():
        sym_df = df.copy()
        sym_df["symbol"] = symbol
        sym_df["family"] = res.symbol_context_map.get(symbol, {}).get("family", "unknown")
        sym_df["historical_regime"] = res.regime_label
        sym_df["entry_signal"] = symbol in trade_decisions
        
        # Inject stop/tp for simulator
        asr = adaptive_stop_map.get(symbol, {})
        for k in ["stop_price", "tp1_price", "tp2_price"]:
            if k in asr: sym_df[k] = asr[k]
        
        ts_rows.append(sym_df)

    bt_engine = BacktestEngine()
    bt_engine.simulate(pd.concat(ts_rows, ignore_index=True))
    
    # ── [5/5] Outcome Linking ──────────────────────────────────────────
    print("\n[5/5] Linking outcomes and finalizing shadow log...")
    price_df = _build_price_df_from_map(symbol_data_map, res)
    shadow_df = pd.read_csv(log_path)
    linker = OutcomeLinker(horizon_hours=24)
    linked_df = linker.link(shadow_df, price_df, backtest_trade_log=bt_engine.trade_log)
    linked_df.to_csv(log_path, index=False)
    
    print(f"\n✓ Shadow validation complete. Log updated: {log_path}")
    print(f"  → Capture rate: {linked_df['leader_captured'].mean()*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    run_shadow_validation(days=args.days, symbol_limit=args.limit)
