import os
import time
import json
import csv
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.features import build_feature_pipeline
from defihunter.engines.leadership import LeadershipEngine
from defihunter.engines.regime import MarketRegimeEngine, SectorRegimeEngine
from defihunter.engines.rules import RuleEngine
from defihunter.engines.thresholds import ThresholdResolutionEngine
from defihunter.engines.ml_ranking import MLRankingEngine
from defihunter.engines.family import FamilyEngine
from defihunter.engines.discovery import DiscoveryEngine
from defihunter.engines.entry import EntryEngine
from defihunter.engines.family_aggregator import FamilyAggregator
from defihunter.engines.decision import DecisionEngine
from defihunter.engines.adaptive import AdaptiveWeightsEngine
from defihunter.engines.adaptive_stop import AdaptiveStopEngine
from defihunter.utils.logger import logger
from defihunter.engines.exit_decay import ExitDecayEngine
from defihunter.execution.paper_trade import PaperTradeEngine
from defihunter.engines.risk import RiskEngine
from defihunter.execution.broadcaster import SignalBroadcaster
from defihunter.data.features import compute_family_features
import threading
from concurrent.futures import ThreadPoolExecutor

def log_shadow(decisions, universe_size):
    """Shadow Verification Mode Logging — uses top-level FinalDecision fields."""
    log_file = "logs/shadow_log.csv"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.exists(log_file)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'evaluated_universe_size', 'symbol', 'family',
                'discovery_score', 'entry_readiness', 'fakeout_risk', 'hold_quality',
                'leader_prob', 'composite_leader_score', 'suggested_action',
            ])

        for d in decisions:
            writer.writerow([
                d.timestamp if hasattr(d, 'timestamp') and d.timestamp else datetime.now(pytz.utc).isoformat(),
                universe_size,
                d.symbol,
                d.explanation.get('family', 'unknown'),  # family lives in explanation
                d.discovery_score,
                d.entry_readiness,
                d.fakeout_risk,
                d.hold_quality,
                d.leader_prob,
                d.composite_leader_score,
                d.decision,
            ])

def detect_momentum_cluster(rows: list) -> dict:
    """
    GT #5: Multi-Coin Momentum Cluster Detection.
    If 3+ coins from the same family signal accumulation simultaneously,
    a sector-wide move is likely imminent. Returns dict of {family: count}.
    """
    family_signals = {}
    for r in rows:
        fam = r.get('Family', 'unknown')
        has_gt_signal = (
            r.get('Silent_Accumulation', False) or
            r.get('Orderbook_Vacuum', False) or
            r.get('quiet_expansion', False) or
            r.get('rs_divergence_persistence', 0) >= 3
        )
        if has_gt_signal:
            family_signals[fam] = family_signals.get(fam, 0) + 1
    
    # Only flag clusters where 3+ coins signal simultaneously
    clusters = {f: count for f, count in family_signals.items() if count >= 3}
    return clusters


def compute_mtf_prepump_score(coin_data_dict: dict) -> float:
    """
    GT #6 + GT-GOLD-1: MTF Pre-Pump Confluence Score.
    4h trend alignment + 1h setup formation + 15m entry point = highest quality signal.
    GT-GOLD-1: Also checks if BOTH 15m AND 1h entry_signal are True (hard confluence gate).
    Returns 0-100 score. 70+ is high-conviction pre-pump.
    """
    score = 0.0

    # 4h: Trend alignment (the foundation — is the big picture right?)
    df_4h = coin_data_dict.get('4h', pd.DataFrame())
    if not df_4h.empty:
        last_4h = df_4h.iloc[-1]
        if last_4h.get('close', 0) > last_4h.get('ema_55', 0):
            score += 30  # Price above 4h EMA55
        if last_4h.get('close', 0) > last_4h.get('ema_20', 0):
            score += 10  # Extra for strong momentum

    # 1h: Setup formation (the trigger — is accumulation happening?)
    df_1h = coin_data_dict.get('1h', pd.DataFrame())
    if not df_1h.empty:
        last_1h = df_1h.iloc[-1]
        for flag, pts in [
            ('silent_accumulation',    40),
            ('orderbook_vacuum',       35),
            ('quiet_expansion',        25),
            ('rs_strong_divergence',   20),
            # GT-GOLD new signals on 1h
            ('cvd_price_divergence',   30),  # GT-GOLD-5
            ('near_liquidation_band',  30),  # GT-GOLD-6
            ('launch_mode',            20),  # GT-GOLD-4
            ('coiling_breakout_alert', 20),  # GT-GOLD-8
            ('high_quality_breakout',  15),  # GT-GOLD-7
        ]:
            if last_1h.get(flag, False):
                score += pts

    # 15m: Entry point refinement (the precision — is right now the moment?)
    df_15m = coin_data_dict.get('15m', pd.DataFrame())
    if not df_15m.empty:
        last_15m = df_15m.iloc[-1]
        if last_15m.get('sweep_reclaim_confirmed', False):
            score += 30
        if last_15m.get('rs_divergence_persistence', 0) >= 2:
            score += 10
        # GT-GOLD-1: Hard confluence gate — both timeframes must signal
        if last_15m.get('entry_signal', False) and not df_1h.empty and df_1h.iloc[-1].get('entry_signal', False):
            score += 25  # GT-GOLD-1: MTF confirmed bonus

    return min(score, 100.0)


# GT #17: False Signal Memory
# Persisted across scanner runs to track per-coin failure history
MEM_FILE = "logs/false_signals.json"
_false_signal_memory = {}  # {symbol: {count, last_scan_ts, hit}}
FALSE_SIGNAL_COOLDOWN_BARS = 48
FALSE_SIGNAL_MAX_FAILS = 3

def load_false_signal_memory():
    """Load failure memory from disk."""
    global _false_signal_memory
    if os.path.exists(MEM_FILE):
        try:
            with open(MEM_FILE, 'r') as f:
                _false_signal_memory = json.load(f)
        except Exception as e:
            logger.error(f"[GT #17] Load memory failed: {e}", exc_info=True)

def save_false_signal_memory():
    """Save failure memory to disk."""
    os.makedirs(os.path.dirname(MEM_FILE), exist_ok=True)
    try:
        with open(MEM_FILE, 'w') as f:
            json.dump(_false_signal_memory, f, indent=2)
    except Exception as e:
        logger.error(f"[GT #17] Save memory failed: {e}", exc_info=True)


def get_top_movers(fetcher, top_n: int = 30) -> list:
    """
    GT #16: Fetch current 24h gainers from Binance.
    Prioritize these coins for scanning — already moving means
    we can catch the continuation EARLY before main crowd enters.
    Returns list of COIN.p formatted symbols.
    """
    try:
        tickers = fetcher.exchange.fetch_tickers()
        movers = []
        for sym, t in tickers.items():
            if '/USDT:USDT' in sym or sym.endswith('/USDT'):
                pct = t.get('percentage') or t.get('change') or 0
                base = sym.split('/')[0]
                movers.append((f"{base}.p", float(pct)))
        # Sort by 24h gain (ascending = catch coins still near bottom of their move)
        # We want coins gaining 1%-10%, not 20%+ (those already pumped)
        pre_pump = [(s, p) for s, p in movers if 1.0 < p < 10.0]
        pre_pump.sort(key=lambda x: -x[1])
        return [s for s, _ in pre_pump[:top_n]]
    except Exception as e:
        logger.error(f"[GT #16] get_top_movers failed: {e}")
        return []

# GT #14: Behavior profile to numeric code
BEHAVIOR_ENCODING = {
    "breakout_continuation": 0,
    "mean_reverting": 1,
    "retest_friendly": 2,
    "catalyst_sensitive": 3,
    "fake_breakout_prone": 4,
}


def should_skip_false_signal(symbol: str, memory: dict, cooldown_bars: int = FALSE_SIGNAL_COOLDOWN_BARS) -> bool:
    """
    GT #17: Skip coins with too many consecutive failed signals.
    Prevents repeatedly signaling the same coin that keeps failing.
    """
    rec = memory.get(symbol, {})
    if rec.get('count', 0) >= FALSE_SIGNAL_MAX_FAILS and not rec.get('hit', False):
        last_scan = rec.get('last_scan_bar', 0)
        current_bar = rec.get('current_bar', last_scan + 1)
        if (current_bar - last_scan) < cooldown_bars:
            return True  # Still in cooldown
    return False


def update_false_signal_memory(memory: dict, paper_positions: list) -> None:
    """
    GT #17: Update false signal memory based on paper trade outcomes.
    Called after paper position updates.
    """
    for pos in paper_positions:
        sym = pos.get('symbol', '')
        outcome = pos.get('outcome', 'open')
        if outcome == 'win':
            # Reset failures on a win
            if sym in memory:
                memory[sym]['hit'] = True
                memory[sym]['count'] = 0
        elif outcome in ('loss', 'time_exit'):
            if sym not in memory:
                memory[sym] = {'count': 0, 'hit': False, 'last_scan_bar': 0}
            memory[sym]['count'] = memory[sym].get('count', 0) + 1
            memory[sym]['hit'] = False


def add_sector_momentum_features(rows: list) -> list:
    """
    GT #15: Peer Momentum — how does a coin perform vs its sector average?
    A coin gaining 3% while its sector gains 0.5% is showing real alpha.
    """
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    if 'Family' not in df.columns or 'return_4' not in df.columns:
        return rows
    
    # Safe mean per sector
    sector_avg = df.groupby('Family')['return_4'].transform('mean')
    df['peer_momentum'] = df['return_4'] - sector_avg  # coin vs sector average
    df['peer_rank'] = df.groupby('Family')['return_4'].rank(pct=True)  # 1.0 = top of sector
    
    return df.to_dict('records')

def run_scanner(config, force_regime=None, limit=0):
    """
    Live scanner mode using Binance API and all underlying analytical engines.
    BUG #3/4/5 FIX: regime key case, double universe call, adaptive timing
    GT #5: Momentum cluster detection
    GT #6: MTF pre-pump score
    """
    load_false_signal_memory()
    fetcher = BinanceFuturesFetcher()
    logger.info("Polling active DeFi perp markets...")
    universe = fetcher.get_defi_universe(config=config)
    if not universe:
        logger.error("Failed to load universe.")
        return []
        
    logger.info(f"Loaded {len(universe)} USDT perpetual contracts.")
    
    # 1. Fetch TRUE MTF data for anchors
    anchor_mtf = {}
    logger.info("Fetching TRUE MTF context for anchors (BTC, ETH, AAVE, UNI)...")
    for anchor in config.anchors:
        anchor_mtf[anchor] = {}
        for tf in ['15m', '1h', '4h']:
            df = fetcher.fetch_ohlcv(anchor, timeframe=tf, limit=150)
            if not df.empty:
                df = build_feature_pipeline(df)
                anchor_mtf[anchor][tf] = df
            
    # Initialize Engines
    regime_engine = MarketRegimeEngine()
    sector_engine = SectorRegimeEngine()
    family_engine = FamilyEngine(config)
    family_aggregator = FamilyAggregator(config.families)
    discovery_engine = DiscoveryEngine(top_n=getattr(config.decision, 'discovery_top_n', 10))
    entry_trigger_engine = EntryEngine(min_readiness=getattr(config.decision, 'min_entry_readiness', 65))
    decision_engine = DecisionEngine(top_n=getattr(config.decision, 'top_n', 5))
    leadership_engine = LeadershipEngine(anchors=config.anchors, ema_lengths=[config.ema.fast, config.ema.medium])
    adaptive_engine = AdaptiveWeightsEngine()
    
    
    decay_engine = ExitDecayEngine(config=config)
    paper_engine = PaperTradeEngine()
    adaptive_stop_engine = AdaptiveStopEngine()
    risk_engine = RiskEngine(config.risk.dict(), fetcher=fetcher)
    broadcaster = SignalBroadcaster(config=config)
    
    threshold_engine = ThresholdResolutionEngine(thresholds_config=config.regimes, config=config)
    rule_engine = RuleEngine()
    
    # 2. Evaluate Global Market Regime (True MTF)
    btc_anchor = next((a for a in anchor_mtf if a.startswith('BTC')), None)
    eth_anchor = next((a for a in anchor_mtf if a.startswith('ETH')), None)
    
    if btc_anchor and eth_anchor:
        global_regime_data = regime_engine.detect_regime(anchor_mtf[btc_anchor], anchor_mtf[eth_anchor])
        regime_label = global_regime_data.get('label', 'unknown')
    else:
        regime_label = "unknown"

    # Fix #2: Respect force_regime override from UI
    force_override = None
    if hasattr(config.regimes, 'overrides') and isinstance(config.regimes.overrides, dict):
        force_override = config.regimes.overrides.get('force_regime')
    if force_override and force_override != "Otomatik İzin Ver":
        logger.info(f"[OVERRIDE] Regime forced from '{regime_label}' to '{force_override}' by UI.")
        regime_label = force_override

    # BUG #5 FIX: adaptive engine update AFTER regime_label is defined
    adaptive_weights = adaptive_engine.current_weights  # Start with loaded defaults
    log_path = "logs/trade_performance.csv"
    if os.path.exists(log_path):
        try:
            perf_history = pd.read_csv(log_path)
            did_rollback = adaptive_engine.evaluate_and_rollback(perf_history)
            if did_rollback:
                logger.info("[ADAPTIVE] Performance degradation detected — rolled back to best known weights.")
            else:
                # BUG #5 FIX: regime_label is now defined before this call
                adaptive_weights = adaptive_engine.update_weights(perf_history, current_regime=regime_label)
                logger.info(f"[ADAPTIVE] Weights updated from {len(perf_history)} trade history records: {adaptive_weights}")
        except Exception as e:
            logger.error(f"[ADAPTIVE] Failed to process perf history: {e}. Using default weights.")
    else:
        logger.info("[ADAPTIVE] No performance log found. Using default weights.")

    # 3. Evaluate Sector Regime
    sector_data = sector_engine.get_sector_regime(
        anchor_mtf.get('ETH.p', {}).get('1h'),
        anchor_mtf.get('AAVE.p', {}).get('1h'),
        anchor_mtf.get('UNI.p', {}).get('1h')
    )
    logger.info(f"Sector Alpha: {sector_data['label']} (Strongest: {sector_data['strongest_family']})")

    rows = []
    current_market_prices = {}
    decay_signals = {}
    timeframe = getattr(config, 'timeframe', '15m')
    
    # Load trained ML models
    ml_engine = MLRankingEngine(model_dir="models")
    model_dir_path = "models"
    trained_coins = []
    if os.path.exists(model_dir_path):
        trained_coins = [
            f.replace('lgb_classifier_long_', '').replace('lgb_classifier_', '').replace('.pkl', '') 
            for f in os.listdir(model_dir_path) 
            if ('lgb_classifier_' in f) and ('GLOBAL' not in f) and ('calibrated' not in f)
        ]
    
    # BUG #4 FIX: Only slice/extend the already-fetched universe (no second API call)
    # GT #16: Get current top movers and prioritize them at start of scan list
    top_movers = get_top_movers(fetcher, top_n=30)
    logger.info(f"[GT #16] {len(top_movers)} pre-pump movers identified (1%-10% 24h change), scanning them first")
    
    # GT-GOLD-3: RVR Pre-Scanner Filter + Anomaly Watchlist
    # Step 1: Get top-30 by volume anomaly (RVR — highest today vs 7d avg)
    from defihunter.data.universe import rank_by_relative_volume, build_anomaly_watchlist
    universe_limited = universe[:200]
    logger.info("[GT-GOLD-3] Running RVR pre-scan filter...")
    rvr_top = rank_by_relative_volume(universe_limited, fetcher, timeframe='1h', lookback_bars=170, top_n=30)
    # Step 2: Build anomaly watchlist (fast multi-criteria filter)
    anomaly_qualified = build_anomaly_watchlist(rvr_top + top_movers + config.anchors, fetcher)
    logger.info(f"[GT-GOLD-3] Anomaly qualified: {len(anomaly_qualified)} coins after RVR+anomaly filter")
    
    # Final scan list: anomaly_qualified first (most likely gainers), then rest of universe for completeness
    rest_of_universe = [s for s in universe_limited if s not in anomaly_qualified]
    priority_set = list(dict.fromkeys(anomaly_qualified + rest_of_universe))
    watch_list = priority_set if not limit else priority_set[:limit]
        
        
    logger.info(f"Scanning universe and computing layered logic for {timeframe}...")

    symbol_data_map = {}
    data_lock = threading.Lock()
    
    # PHASE 1: Parallel Data Fetching & Base Feature Generation
    def fetch_and_prep(symbol):
        try:
            if should_skip_false_signal(symbol, _false_signal_memory):
                return
                
            df = fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            if df.empty or len(df) < 55:
                return
            
            # Base Features (Phase 1 & 2)
            df = build_feature_pipeline(df, timeframe=timeframe)
            
            # MTF Leadership Features (Phase 3)
            anchors_tf = {k: v[timeframe] for k, v in anchor_mtf.items() if timeframe in v}
            df = leadership_engine.add_leadership_features(df, anchors_tf)
            
            with data_lock:
                symbol_data_map[symbol] = df
                current_market_prices[symbol] = df.iloc[-1]['close']
                
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")

    max_workers = min(os.cpu_count() * 2, 20)
    logger.info(f"[Phase 1] Parallelizing data fetch with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(fetch_and_prep, watch_list)

    # PHASE 2: Global Family Aggregation
    logger.info("[Phase 2] Computing Global Family Aggregates...")
    family_stats = family_aggregator.compute_family_stats(symbol_data_map, timeframe=timeframe)
    
    # PHASE 3: Parallel Logic Evaluation (Rules, Decay, Ranking)
    all_rows = []
    rows_lock = threading.Lock()
    
    def evaluate_logic(symbol):
        try:
            df = symbol_data_map[symbol]
            
            # Inject Family Context (Relative Spread, Peer Momentum, Leader Consistency)
            df = family_aggregator.inject_family_features(symbol, df, family_stats, timeframe=timeframe)
            
            # Evaluate Exit Decay (Uses injected family features)
            decay_res = decay_engine.evaluate_exit_signals(symbol, df)
            with data_lock:
                decay_signals[symbol] = decay_res
            
            # Resolve Thresholds & Evaluate Rules
            profile = family_engine.profile_coin(symbol, historical_data=df)
            resolved_thresholds = threshold_engine.resolve_thresholds(regime=regime_label, family=profile.family_label)
            df = rule_engine.evaluate(df, regime=regime_label, family=profile.family_label, resolved_thresholds=resolved_thresholds, sector_data=sector_data, adaptive_weights=adaptive_weights)
            
            if df.empty:
                return
                
            # Collect latest state for ML ranking
            last_bar = df.tail(1).to_dict('records')[0]
            last_bar.update({
                "symbol": symbol,
                "family": profile.family_label,
                "regime": regime_label,
                # Carry raw ATR and current close from last bar for adaptive stop
                "_atr": float(df.iloc[-1].get('atr', 0.0)),
                "_close_raw": float(df.iloc[-1].get('close', 0.0)),
            })
            with rows_lock:
                all_rows.append(last_bar)
                
        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {e}")

    print(f"[Phase 1] Evaluating logic with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(evaluate_logic, list(symbol_data_map.keys()))
        
    master_df = pd.DataFrame(all_rows)
            
    if master_df.empty:
        logger.info("No candidates found in current scan.")
        return []

    # --- LAYERED PIPELINE CONTINUES ---
    
    
    # 3.5 Cross-Sectional Family Features
    master_df = compute_family_features(master_df)
    
    # 4. ML Leader Components
    logger.info("\n[ML] Predicting Leader Prob, Setup Quality, and Holdability...")
    master_df, _ = ml_engine.rank_candidates(master_df, use_family_ranker=True) 
    
    # 5. Discovery Layer
    logger.info("[Discovery] Identifying leader candidates...")
    master_df = discovery_engine.compute_discovery_scores(master_df)
    
    # 6. Entry & Decision Layer
    # DecisionEngine now handles the CompositeLeaderScore and SuggestedAction
    final_decisions = decision_engine.process_candidates(master_df)
    
    # GT-GOLD: Save shadow log
    log_shadow(final_decisions, len(watch_list))
    
    # Update Paper Positions
    paper_engine.update_positions(current_market_prices, decay_signals=decay_signals)

    # 7. RISK FILTERING & Execution
    # 7. RISK FILTERING & Execution
executed_decisions = []

# GT #18: Daily Loss Killswitch Wiring
daily_loss_pct = paper_engine.get_daily_loss_pct()

for d in final_decisions:
    current_portfolio_list = [p.dict() for p in paper_engine.portfolio.open_positions]

    if d.decision == 'trade':
        family_label = d.explanation.get('family', 'defi_beta')

        # ── Family Execution Mode ──────────────────────────────────────
        exec_cfg   = config.get_family_execution(family_label) if hasattr(config, 'get_family_execution') else None
        exec_mode  = getattr(exec_cfg, 'mode', 'trade_allowed') if exec_cfg else 'trade_allowed'
        width_mult = getattr(exec_cfg, 'stop_width_mult', 1.0) if exec_cfg else 1.0

        # watch_only: log to shadow, skip trade
        if exec_mode == 'watch_only':
            logger.info(f"[WatchOnly] {d.symbol} ({family_label}) — candidate logged, no trade opened")
            d.explanation['participation_mode'] = 'watch_only'
            executed_decisions.append(d)
            continue

        # ── Kelly first ───────────────────────────────────────────────
        win_prob = d.leader_prob if d.leader_prob > 0 else 0.5
        kelly_pct = risk_engine.calculate_kelly_size(
            win_prob=win_prob,
            reward_risk=2.0,
            leader_prob=d.leader_prob
        )

        # reduced_risk: apply gates BEFORE risk validation
        if exec_cfg and exec_mode == 'reduced_risk':
            min_er  = getattr(exec_cfg, 'min_entry_readiness', 0.0)
            min_lp  = getattr(exec_cfg, 'min_leader_prob', 0.0)
            max_pos = getattr(exec_cfg, 'max_open_positions', 5)

            family_open = sum(
                1 for p in paper_engine.portfolio.open_positions
                if p.family == family_label
            )

            if d.entry_readiness < min_er:
                d.decision = 'reject'
                d.explanation['rejection_reason'] = (
                    f'reduced_risk: entry_readiness {d.entry_readiness:.1f} < {min_er}'
                )
                executed_decisions.append(d)
                continue

            if d.leader_prob < min_lp:
                d.decision = 'reject'
                d.explanation['rejection_reason'] = (
                    f'reduced_risk: leader_prob {d.leader_prob:.2f} < {min_lp}'
                )
                executed_decisions.append(d)
                continue

            if family_open >= max_pos:
                d.decision = 'reject'
                d.explanation['rejection_reason'] = (
                    f'reduced_risk: {family_label} already has {family_open}/{max_pos} positions'
                )
                executed_decisions.append(d)
                continue

            risk_pct_mult = getattr(exec_cfg, 'risk_pct_mult', 1.0)
            kelly_pct *= risk_pct_mult

        d.explanation['participation_mode'] = exec_mode

        # ── Adaptive Stop ─────────────────────────────────────────────
        adaptive_stop_result = None
        try:
            sym_rows  = master_df[master_df['symbol'] == d.symbol]
            atr_val   = float(sym_rows['_atr'].iloc[-1]) if '_atr' in sym_rows.columns and not sym_rows.empty else 0.0
            close_val = float(sym_rows['_close_raw'].iloc[-1]) if '_close_raw' in sym_rows.columns and not sym_rows.empty else d.entry_price

            stop_row = {
                'close':         close_val,
                'atr':           atr_val,
                'structure_low': float(sym_rows.get('structure_low', pd.Series([0.0])).iloc[-1]) if 'structure_low' in sym_rows.columns else 0.0,
                'swing_low':     float(sym_rows.get('swing_low', pd.Series([0.0])).iloc[-1]) if 'swing_low' in sym_rows.columns else 0.0,
                'support_level': float(sym_rows.get('support_level', pd.Series([0.0])).iloc[-1]) if 'support_level' in sym_rows.columns else 0.0,
            }

            adaptive_stop_result = adaptive_stop_engine.compute_stop(
                row=stop_row,
                family=family_label,
                regime=regime_label,
                fakeout_risk=d.fakeout_risk,
                stop_width_mult=width_mult,
            )

            # Wider stop -> smaller % risk to keep net dollar risk sane
            if width_mult > 1.0:
                kelly_pct /= width_mult

            logger.info(
                f"[AdaptiveStop-V2] {d.symbol} | "
                f"mode={exec_mode} | "
                f"stop_mode={adaptive_stop_result['stop_mode']} | "
                f"atr_mult={adaptive_stop_result['atr_mult']} | "
                f"family={family_label} | regime={regime_label} | "
                f"hard_stop={adaptive_stop_result['stop_price']:.6f} | "
                f"soft_stop={adaptive_stop_result['soft_invalidation_price']:.6f} | "
                f"tp1={adaptive_stop_result['tp1_price']:.6f} | "
                f"confidence={adaptive_stop_result.get('stop_confidence', '?')} | "
                f"noise_bars={adaptive_stop_result.get('noise_tolerance_bars', '?')} | "
                f"reduce_size_first={adaptive_stop_result.get('reduce_size_first', False)}"
            )
        except Exception as _stop_err:
            logger.warning(
                f"[AdaptiveStop-V2] compute_stop failed for {d.symbol}: {_stop_err} — using legacy fallback",
                exc_info=True
            )
            adaptive_stop_result = None

        # ── Determine stop price for sizing ──────────────────────────
        stop_price_for_sizing = (
            adaptive_stop_result["stop_price"]
            if adaptive_stop_result and adaptive_stop_result.get("stop_price")
            else d.stop_price
        )

        # ── Estimate REAL notional from risk + stop distance ────────
        actual_notional = risk_engine.estimate_notional_from_stop(
            equity_val=paper_engine.portfolio.balance_usd,
            risk_pct=kelly_pct,
            entry_price=d.entry_price,
            stop_price=stop_price_for_sizing,
        )

        if actual_notional <= 0:
            d.decision = 'reject'
            d.explanation['rejection_reason'] = 'invalid_sizing_notional'
            executed_decisions.append(d)
            continue

        d.explanation['kelly_risk_pct'] = kelly_pct
        d.explanation['estimated_notional'] = actual_notional

        # ── NOW run risk validation with REAL values ────────────────
        is_valid, reason = risk_engine.validate_trade(
            symbol=d.symbol,
            family=family_label,
            current_portfolio=current_portfolio_list,
            equity_val=paper_engine.portfolio.balance_usd,
            daily_loss_pct=daily_loss_pct,   # TODO: wire real daily loss
            leader_prob=d.leader_prob,
            new_trade_notional=actual_notional,
            leverage=getattr(config.risk, "default_leverage", None),
        )

        if not is_valid:
            d.decision = 'reject'
            d.explanation['rejection_reason'] = reason
            executed_decisions.append(d)
            continue

        # Optional: persist leverage info for future portfolio margin calc
        d.explanation['leverage'] = getattr(config.risk, "default_leverage", None)

        paper_engine.open_position(
            d,
            risk_pct=kelly_pct,
            adaptive_stop_result=adaptive_stop_result
        )

    executed_decisions.append(d)     
    # DISPLAY NEW LEADERBOARD
    display_rows = []
    for d in executed_decisions:
        display_rows.append({
            "Symbol":    d.symbol,
            "Family":    d.explanation.get('family', '—'),
            "Disc_S":    round(d.discovery_score, 1),
            "Ready_S":   round(d.entry_readiness, 1),
            "Risk_F":    round(d.fakeout_risk, 1),
            "Hold_Q":    round(d.hold_quality, 1),
            "L_Prob":    f"{d.leader_prob * 100:.0f}%",
            "Comp_S":    round(d.composite_leader_score, 1),
            "Action":    d.decision,
            "Entry":     round(d.entry_price, 4),
            "Timestamp": d.timestamp.strftime('%H:%M:%S') if d.timestamp else '—',
        })
        
    final_display_df = pd.DataFrame(display_rows)
    logger.info("\n[PHASE 5: LAYERED DECISION ENGINE ACTIVE]")
    if not final_display_df.empty:
        logger.info("\n" + final_display_df.sort_values('Comp_S', ascending=False).to_markdown(index=False))
    
    logger.info(f"\n[PAPER PORTFOLIO] Balance: ${paper_engine.portfolio.balance_usd:.2f} | Open: {len(paper_engine.portfolio.open_positions)}")
    
    # BROADCAST ALERTS
    try:
        broadcaster.broadcast(executed_decisions)
    except Exception as e:
        logger.error(f"[Broadcaster] Error during signal broadcast: {e}")
    
    save_false_signal_memory()
    return executed_decisions
