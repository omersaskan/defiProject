"""
7-Day Shadow Validation Runner

Fetches last N days of real OHLCV data from Binance, replays the full
DefiHunter pipeline at each historical timestamp (leakage-safe), logs
decisions via ShadowLogger, then links outcomes via OutcomeLinker.

Usage:
    python scripts/run_shadow_validation.py [--config config.yaml] [--days 7] [--limit 80]
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
from defihunter.data.features import build_feature_pipeline, compute_family_features
from defihunter.engines.leadership import LeadershipEngine
from defihunter.engines.regime import MarketRegimeEngine
from defihunter.engines.rules import RuleEngine
from defihunter.engines.thresholds import ThresholdResolutionEngine
from defihunter.engines.ml_ranking import MLRankingEngine
from defihunter.engines.family import FamilyEngine
from defihunter.engines.discovery import DiscoveryEngine
from defihunter.engines.family_aggregator import FamilyAggregator
from defihunter.engines.decision import DecisionEngine
from defihunter.engines.adaptive_stop import AdaptiveStopEngine
from defihunter.validation.shadow_logger import ShadowLogger
from defihunter.validation.outcome_linker import OutcomeLinker
from defihunter.execution.backtest import BacktestEngine
from defihunter.utils.logger import logger


def load_config(config_path: str):
    """Load config YAML — searches multiple paths before falling back to None."""
    search_paths = [config_path, "configs/default.yaml", "configs/config.yaml", "config.yaml"]
    from defihunter.core.config import load_config as _load
    for path in search_paths:
        if os.path.exists(path):
            try:
                cfg = _load(path)
                logger.info(f"Config loaded from: {path}")
                return cfg
            except Exception as e:
                logger.warning(f"Config load failed for {path}: {e}")
    logger.warning("No config found — using minimal defaults")
    return None



def _build_price_df_from_map(symbol_data_map: dict, family_map: dict) -> pd.DataFrame:
    """Flatten symbol_data_map into a cross-sectional price DataFrame."""
    rows = []
    for sym, df in symbol_data_map.items():
        for _, bar in df.iterrows():
            ts = bar.get("timestamp", bar.name)
            rows.append({
                "symbol":    sym,
                "timestamp": pd.to_datetime(ts, utc=True, format='ISO8601'),
                "close":     bar["close"],
                "high":      bar.get("high", bar["close"]),
                "low":       bar.get("low",  bar["close"]),
                "family":    family_map.get(sym, "unknown"),
            })
    return pd.DataFrame(rows)


def run_shadow_validation(
    config_path: str = "config.yaml",
    days: int = 7,
    symbol_limit: int = 80,
    log_path: str = "logs/shadow_log.csv",
):
    """
    Main entry point: fetch 7 days of OHLCV, replay pipeline, log + link outcomes.
    """
    print(f"\n{'='*60}")
    print(f"  DefiHunter — Shadow Validation ({days} days)")
    print(f"{'='*60}")

    config = load_config(config_path)

    # ── Fetch universe ───────────────────────────────────────────────────────
    fetcher = BinanceFuturesFetcher()
    print("\n[1/6] Fetching DeFi universe...")
    try:
        universe = fetcher.get_defi_universe(config=config)
    except Exception as e:
        logger.error(f"Universe fetch failed: {e}")
        return

    universe = universe[:symbol_limit]
    print(f"  → {len(universe)} symbols (limited to {symbol_limit})")

    # Compute bars needed: days × 24h × (bars_per_hour for 15m = 4) + 24h buffer
    bars_needed = (days + 1) * 24 * 4  # +1 day for outcome linking buffer
    timeframe   = "15m"

    # ── Fetch anchors ────────────────────────────────────────────────────────
    anchors = getattr(config, "anchors", ["BTC.p", "ETH.p"]) if config else ["BTC.p", "ETH.p"]
    anchor_dfs = {}
    print(f"\n[2/6] Fetching {len(anchors)} anchor series ({bars_needed} bars each)...")
    for anchor in anchors:
        try:
            df = fetcher.fetch_ohlcv(anchor, timeframe=timeframe, limit=bars_needed)
            if not df.empty:
                anchor_dfs[anchor] = build_feature_pipeline(df, timeframe=timeframe)
                print(f"  ✓ {anchor}: {len(df)} bars")
        except Exception as e:
            logger.warning(f"Anchor fetch failed for {anchor}: {e}")

    # ── Detect regime from BTC anchor ───────────────────────────────────────
    regime_engine = MarketRegimeEngine()
    btc_anchor = next((a for a in anchor_dfs if "BTC" in a), None)
    eth_anchor = next((a for a in anchor_dfs if "ETH" in a), None)
    if btc_anchor and eth_anchor:
        regime_data = regime_engine.detect_regime({timeframe: anchor_dfs[btc_anchor]}, {timeframe: anchor_dfs[eth_anchor]})
        regime_label = regime_data.get("label", "trend")
    else:
        regime_label = "trend"
    print(f"\n  Detected regime: {regime_label}")

    # ── Fetch coin OHLCV in parallel ─────────────────────────────────────────
    import threading
    from concurrent.futures import ThreadPoolExecutor

    symbol_data_map = {}
    family_map      = {}
    data_lock       = threading.Lock()

    family_engine_inst = FamilyEngine(config) if config else None
    ml_engine          = MLRankingEngine(model_dir="models")
    ema_lengths        = [20, 55]
    if config:
        try:
            ema_lengths = [config.ema.fast, config.ema.medium]
        except Exception:
            pass
    leadership_engine  = LeadershipEngine(
        anchors=anchors,
        ema_lengths=ema_lengths,
    )
    rule_engine        = RuleEngine()
    threshold_engine   = ThresholdResolutionEngine(
        thresholds_config=config.regimes if config else None,
        config=config,
    )
    family_aggregator  = FamilyAggregator(config.families if config else {})
    discovery_engine   = DiscoveryEngine(top_n=10)
    decision_engine    = DecisionEngine(top_n=5)
    adaptive_stop_eng  = AdaptiveStopEngine()

    def fetch_and_build(symbol):
        try:
            df = fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=bars_needed)
            if df.empty or len(df) < 55:
                return
            df = build_feature_pipeline(df, timeframe=timeframe)
            df = leadership_engine.add_leadership_features(df, anchor_dfs)
            fam_label = "defi_beta"
            if family_engine_inst:
                try:
                    profile = family_engine_inst.profile_coin(symbol, historical_data=df)
                    fam_label = profile.family_label
                except Exception:
                    pass
            with data_lock:
                symbol_data_map[symbol] = df
                family_map[symbol] = fam_label
        except Exception as e:
            logger.warning(f"Fetch failed for {symbol}: {e}")

    print(f"\n[3/6] Fetching {len(universe)} coin series ({bars_needed} bars × {timeframe})...")
    max_workers = min(os.cpu_count() * 2, 20)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        ex.map(fetch_and_build, universe)
    print(f"  → {len(symbol_data_map)} symbols fetched successfully")

    # ── Build price DataFrame for OutcomeLinker ──────────────────────────────
    print("\n[4/6] Building cross-sectional price DataFrame...")
    price_df = _build_price_df_from_map(symbol_data_map, family_map)

    # ── Phase 2: Family Aggregation ─────────────────────────────────────────
    print("\n[Phase 2] Computing Global Family Aggregates...")
    family_stats = family_aggregator.compute_family_stats(symbol_data_map, timeframe=timeframe)

    # ── Phase 3: Layered Pipeline ─────────────────────────────────────────
    print("\n[Phase 3] Running layered pipeline and logging decisions...")
    shadow_logger = ShadowLogger(log_path=log_path)

    all_rows = []
    for symbol in symbol_data_map:
        df  = symbol_data_map[symbol]
        fam = family_map.get(symbol, "unknown")
        try:
            if config:
                resolved_thresholds = threshold_engine.resolve_thresholds(
                    regime=regime_label, family=fam
                )
            else:
                resolved_thresholds = {"min_score": 50, "min_relative_leadership": 0, "min_volume": 0}

            # Parity Fix: Inject real family stats
            df = family_aggregator.inject_family_features(symbol, df, family_stats, timeframe=timeframe)
            
            # Anchor weighting fix parity: pass primary_anchor
            profile = family_engine_inst.profile_coin(symbol, historical_data=df) if family_engine_inst else None
            primary_anchor = profile.primary_anchor if profile else "ETH.p"
            
            df = rule_engine.evaluate(df, regime=regime_label, family=fam,
                                     resolved_thresholds=resolved_thresholds,
                                     primary_anchor=primary_anchor)
            last_bar = df.tail(1).to_dict("records")[0]
            last_bar.update({"symbol": symbol, "family": fam, "regime": regime_label,
                             "_atr": float(df.iloc[-1].get("atr", 0.0)),
                             "_close_raw": float(df.iloc[-1]["close"])})
            all_rows.append(last_bar)
        except Exception as e:
            logger.warning(f"Pipeline failed for {symbol}: {e}")

    if not all_rows:
        print("  ✗ No candidates after pipeline. Exiting.")
        return

    master_df = pd.DataFrame(all_rows)
    master_df = compute_family_features(master_df)
    master_df, _ = ml_engine.rank_candidates(master_df, use_family_ranker=True)
    master_df     = discovery_engine.compute_discovery_scores(master_df)

    # Add a synthetic uniform timestamp for the historical "now"
    sim_ts = datetime.now(timezone.utc) - timedelta(days=days)
    if "timestamp" not in master_df.columns:
        master_df["timestamp"] = pd.Timestamp(sim_ts)

    decisions = decision_engine.process_candidates(master_df)
    print(f"  → {len(decisions)} decisions produced")

    # Build adaptive stop map + setup_class map
    adaptive_stop_map = {}
    setup_class_map   = {}
    for d in decisions:
        sym_rows = master_df[master_df["symbol"] == d.symbol]
        if sym_rows.empty:
            continue
        row_dict = {
            "close": float(sym_rows["_close_raw"].iloc[-1]) if "_close_raw" in sym_rows.columns else d.entry_price,
            "atr":   float(sym_rows["_atr"].iloc[-1]) if "_atr" in sym_rows.columns else 0.0,
        }
        try:
            adaptive_stop_map[d.symbol] = adaptive_stop_eng.compute_stop(
                row_dict,
                family=d.explanation.get("family", "defi_beta"),
                regime=regime_label,
                fakeout_risk=d.fakeout_risk,
            )
        except Exception:
            pass
        setup_class_map[d.symbol] = str(sym_rows.get("setup_class", pd.Series([""])).iloc[-1])

    # Log to shadow CSV
    trade_decisions = {d.symbol for d in decisions if d.decision == "trade"}
    shadow_logger.log_scan(
        decisions=decisions,
        regime=regime_label,
        universe_size=len(universe),
        timeframe=timeframe,
        adaptive_stop_map=adaptive_stop_map,
        paper_opened_symbols=trade_decisions,
        kelly_map={d.symbol: d.explanation.get("kelly_risk_pct", "") for d in decisions},
        setup_class_map=setup_class_map,
        scan_timestamp=sim_ts,
    )
    print(f"  → {len(decisions)} decisions logged to {log_path}")

    # ── Backtest for pnl_r / mfe_r / exit_reason ────────────────────────────
    print("\n[6/6] Running BacktestEngine for pnl_r / mfe_r labels...")
    master_df["entry_signal"] = master_df["symbol"].isin(trade_decisions)
    master_df["historical_regime"] = regime_label
    master_df["fakeout_risk"] = master_df.get("fakeout_risk", pd.Series(0.0, index=master_df.index))

    # Build a time-series master_df spanning all bars for backtest
    ts_rows = []
    for symbol in symbol_data_map:
        sym_df = symbol_data_map[symbol].copy()
        sym_df["symbol"] = symbol
        sym_df["family"] = family_map.get(symbol, "unknown")
        sym_df["historical_regime"] = regime_label
        sym_df["entry_signal"] = symbol in trade_decisions
        if "fakeout_risk" not in sym_df.columns:
            sym_df["fakeout_risk"] = 0.0
        # Carry adaptive stop result
        asr = adaptive_stop_map.get(symbol, {})
        for k in ["stop_price", "tp1_price", "tp2_price"]:
            if k in asr:
                sym_df[k] = asr[k]
        ts_rows.append(sym_df)

    if ts_rows:
        backtest_df = pd.concat(ts_rows, ignore_index=True)
        if "timestamp" not in backtest_df.columns:
            backtest_df.reset_index(inplace=True)
            backtest_df.rename(columns={"index": "timestamp"}, inplace=True)

        bt_engine = BacktestEngine()
        bt_metrics = bt_engine.simulate(backtest_df)
        trade_log  = bt_engine.trade_log
        print(f"  → Backtest: {bt_metrics.get('total_trades', 0)} trades | "
              f"win_rate={bt_metrics.get('win_rate', 0):.1f}% | "
              f"expectancy_r={bt_metrics.get('expectancy_r', 0):.3f}")
    else:
        trade_log = []

    # ── Outcome linking ───────────────────────────────────────────────────────
    print("\n[Outcome] Linking future returns post-hoc (no leakage)...")
    shadow_df = pd.read_csv(log_path)
    linker    = OutcomeLinker(horizon_hours=24)
    linked_df = linker.link(shadow_df, price_df, backtest_trade_log=trade_log)
    linked_df.to_csv(log_path, index=False)
    print(f"  → Shadow log updated: {len(linked_df)} rows, {linked_df['leader_captured'].notna().sum()} linked")

    # ── Summary metrics ───────────────────────────────────────────────────────
    linked_df["leader_captured"] = linked_df["leader_captured"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    linked_df["future_24h_return"] = pd.to_numeric(linked_df["future_24h_return"], errors="coerce")
    linked_df["pnl_r"]             = pd.to_numeric(linked_df["pnl_r"], errors="coerce")

    lcr = linked_df["leader_captured"].mean() * 100 if linked_df["leader_captured"].notna().any() else 0
    exp = linked_df[linked_df["paper_trade_opened"] == True]["pnl_r"].mean() if "paper_trade_opened" in linked_df.columns else np.nan

    print(f"\n{'─'*60}")
    print(f"  leader_capture_rate : {lcr:.1f}%")
    print(f"  expectancy_r (trades): {exp:.3f}" if not np.isnan(exp) else "  expectancy_r : N/A")
    print(f"  shadow log : {log_path}")
    print(f"{'─'*60}")
    print("\n✓ Shadow validation complete. Run generate_shadow_report.py to produce the full report.")

    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DefiHunter 7-Day Shadow Validation Runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--days",   type=int, default=7, help="Historical window in days")
    parser.add_argument("--limit",  type=int, default=80, help="Max symbols to scan")
    parser.add_argument("--log",    default="logs/shadow_log.csv", help="Shadow log output path")
    args = parser.parse_args()

    run_shadow_validation(
        config_path=args.config,
        days=args.days,
        symbol_limit=args.limit,
        log_path=args.log,
    )
