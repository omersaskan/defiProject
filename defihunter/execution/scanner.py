import os
import time
import json
import pandas as pd
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime
from typing import List, Dict, Any, Optional


from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.features import build_feature_pipeline
from defihunter.engines.adaptive import AdaptiveWeightsEngine
from defihunter.engines.adaptive_stop import AdaptiveStopEngine
from defihunter.engines.exit_decay import ExitDecayEngine
from defihunter.engines.risk import RiskEngine
from defihunter.execution.paper_trade import PaperTradeEngine
from defihunter.execution.broadcaster import SignalBroadcaster
from defihunter.execution.pipeline import SignalPipeline
from defihunter.validation.shadow_logger import ShadowLogger
from defihunter.utils.logger import logger
from defihunter.utils.db_manager import db_manager
from defihunter.utils.monitor import monitor



# Constants
MEM_FILE = "logs/false_signals.json"
FALSE_SIGNAL_COOLDOWN_BARS = 48
FALSE_SIGNAL_MAX_FAILS = 3
_false_signal_memory = {}
_executor = ProcessPoolExecutor(max_workers=min(8, multiprocessing.cpu_count()))

def load_false_signal_memory():
    global _false_signal_memory
    if os.path.exists(MEM_FILE):
        try:
            with open(MEM_FILE, "r") as f:
                _false_signal_memory = json.load(f)
        except Exception as e:
            logger.error(f"[Memory] Load failed: {e}")

def save_false_signal_memory():
    os.makedirs(os.path.dirname(MEM_FILE), exist_ok=True)
    try:
        with open(MEM_FILE, "w") as f:
            json.dump(_false_signal_memory, f, indent=2)
    except Exception as e:
        logger.error(f"[Memory] Save failed: {e}")

class ScanPipeline:
    """
    GT-Institutional: Modular Execution Pipeline (Scanner Layer).
    Responsible for I/O, Risk, and side effects. Delegates core signal logic to SignalPipeline.
    """
    def __init__(self, config: Any, fetcher: Optional[BinanceFuturesFetcher] = None):
        self.config = config
        self.fetcher = fetcher or BinanceFuturesFetcher()
        self.timeframe = getattr(config, "timeframe", "15m")
        
        # 1. CORE SIGNAL PIPELINE (The brain)
        self.signal_core = SignalPipeline(config)
        
        # 2. ORCHESTRATION ENGINES (The hands)
        self.adaptive_engine = AdaptiveWeightsEngine()
        self.decay_engine = ExitDecayEngine(config=config)
        self.paper_engine = PaperTradeEngine()
        self.adaptive_stop_engine = AdaptiveStopEngine()
        self.risk_engine = RiskEngine(config.risk.dict(), fetcher=self.fetcher)
        self.broadcaster = SignalBroadcaster(config=config)
        self.shadow_logger = ShadowLogger()
        self.loop = asyncio.get_event_loop()

        # State
        self.anchor_mtf = {}
        self.symbol_data_map = {}
        self.current_market_prices = {}
        self.current_market_highs = {}
        self.current_market_lows = {}
        self.decay_signals = {}
        
        # Metadata for logging/UI
        self.adaptive_stop_map = {}
        self.kelly_map = {}
        self.paper_opened = set()


    async def run(self, force_regime: Optional[str] = None, limit: int = 0):
        """Orchestrate the entire scan from fetch to execution (ASYNCHRONOUS)."""
        start_t = time.time()
        load_false_signal_memory()

        try:
            # 1. MTF Context for Anchors (Async)
            await self._prepare_anchors()

            # 2. Resolve Global Context
            regime_label = self._resolve_regimes(force_regime)
            adaptive_weights = self._update_adaptive_weights(regime_label)
            sector_data = self._resolve_sector_regime()

            # 3. Build Watchlist & Async Fetch RAW
            watch_list = self._build_watchlist(limit)
            raw_dfs = await self._fetch_raw_only_async(watch_list)
            
            # 4. PARALLEL FEATURE ENGINEERING
            await self._process_features_parallel(raw_dfs)

            # 5. EXECUTE SIGNAL CORE (Unified Pipeline)
            pipeline_result = self.signal_core.run(
                symbol_data_map=self.symbol_data_map,
                anchor_context=self.anchor_mtf,
                regime_label=regime_label,
                sector_data=sector_data,
                adaptive_weights=adaptive_weights,
                mode="live"
            )

            # 5. EXECUTION & SIDE EFFECTS
            self._evaluate_exits_parallel()
            executed_decisions = self._execute_decisions(pipeline_result)

            elapsed = time.time() - start_t
            logger.info(f"Scan completed in {elapsed:.1f}s. Portfolio: ${self.paper_engine.portfolio.balance_usd:.2f}")
            
            # 6. PERSISTENCE & MONITORING
            try:
                db_manager.log_scan(
                    timestamp=datetime.now(),
                    regime=regime_label,
                    universe_size=len(self.symbol_data_map),
                    duration_ms=elapsed * 1000,
                    balance=self.paper_engine.portfolio.balance_usd
                )
                monitor.report_scan(
                    duration_ms=elapsed * 1000,
                    universe_size=len(self.symbol_data_map),
                    fallbacks=0 # We'll need to bubble this up from ml_engine later
                )
            except Exception as e:
                logger.warning(f"[Scanner] DB/Monitor logging failed: {e}")


            save_false_signal_memory()
            return executed_decisions
        finally:
            await self.fetcher.close()

    async def _prepare_anchors(self):
        logger.info("Fetching Anchor MTF Context (Async)...")
        tasks = []
        for anchor in self.config.anchors:
            self.anchor_mtf[anchor] = {}
            for tf in ["15m", "1h", "4h"]:
                tasks.append(self._fetch_and_prep_anchor(anchor, tf))
        await asyncio.gather(*tasks)

    async def _fetch_and_prep_anchor(self, anchor, tf):
        df = await self.fetcher.async_fetch_ohlcv(anchor, timeframe=tf, limit=150)
        if not df.empty:
            df = build_feature_pipeline(df, timeframe=tf)
            self.anchor_mtf[anchor][tf] = df


    def _resolve_regimes(self, force_regime: Optional[str]) -> str:
        # Use SignalPipeline's helper but allow forced override
        resolved = self.signal_core._resolve_regime(self.anchor_mtf)
        
        force_from_cfg = None
        if hasattr(self.config.regimes, "overrides") and isinstance(self.config.regimes.overrides, dict):
            force_from_cfg = self.config.regimes.overrides.get("force_regime")
        
        final_regime = force_regime or force_from_cfg
        if final_regime and final_regime != "Otomatik İzin Ver":
            logger.info(f"[Regime] Overridden to {final_regime}")
            return final_regime
        return resolved

    def _update_adaptive_weights(self, regime_label: str) -> dict:
        try:
            perf_history = db_manager.get_trade_history(limit=500)
            if not perf_history.empty:
                if self.adaptive_engine.evaluate_and_rollback(perf_history):
                    return self.adaptive_engine.current_weights
                return self.adaptive_engine.update_weights(perf_history, current_regime=regime_label)
        except Exception as e:
            logger.error(f"[Adaptive] Error reading from DB: {e}")
            
        # Fallback to CSV if DB is empty or fails (optional during migration)
        # return self.adaptive_engine.current_weights
        return self.adaptive_engine.current_weights


    def _resolve_sector_regime(self) -> dict:
        return self.signal_core._resolve_sector_regime(self.anchor_mtf)

    def _build_watchlist(self, limit: int) -> list:
        from defihunter.data.universe import rank_by_relative_volume, build_anomaly_watchlist
        universe = self.fetcher.get_defi_universe(config=self.config)
        top_movers = self._get_top_movers(top_n=30)
        u_lim = universe[:200]
        rvr_top = rank_by_relative_volume(u_lim, self.fetcher, timeframe="1h", lookback_bars=170, top_n=30)
        anomaly_q = build_anomaly_watchlist(rvr_top + top_movers + self.config.anchors, self.fetcher)
        full_list = list(dict.fromkeys(anomaly_q + [s for s in u_lim if s not in anomaly_q]))
        return full_list if not limit else full_list[:limit]

    def _get_top_movers(self, top_n: int = 30) -> list:
        try:
            tickers = self.fetcher.exchange.fetch_tickers()
            movers = []
            for sym, t in tickers.items():
                if "/USDT:USDT" in sym or sym.endswith("/USDT"):
                    pct = t.get("percentage") or t.get("change") or 0
                    base = sym.split("/")[0]
                    movers.append((f"{base}.p", float(pct)))
            pre_pump = [(s, p) for s, p in movers if 1.0 < p < 10.0]
            pre_pump.sort(key=lambda x: -x[1])
            return [s for s, _ in pre_pump[:top_n]]
        except Exception as e:
            logger.error(f"[Movers] Error: {e}")
            return []

    async def _fetch_raw_only_async(self, watch_list: list) -> Dict[str, pd.DataFrame]:
        """Asynchronously fetch OHLCV for all symbols without processing features yet."""
        tasks = {symbol: self.fetcher.async_fetch_ohlcv(symbol, timeframe=self.timeframe, limit=200) 
                 for symbol in watch_list if not self._should_skip(symbol)}
        
        results = await asyncio.gather(*tasks.values())
        return {symbol: df for symbol, df in zip(tasks.keys(), results) if not df.empty and len(df) >= 55}

    async def _process_features_parallel(self, raw_dfs: Dict[str, pd.DataFrame]):
        """Runs build_feature_pipeline in parallel across multiple processes."""
        if not raw_dfs:
            return

        logger.info(f"Processing features for {len(raw_dfs)} symbols in parallel...")
        symbols = list(raw_dfs.keys())
        dfs = list(raw_dfs.values())
        
        # Parallel execution using ProcessPoolExecutor
        processed_dfs = await self.loop.run_in_executor(
            _executor, 
            self._batch_build_features, 
            dfs, 
            self.timeframe
        )
        
        for symbol, df in zip(symbols, processed_dfs):
            self.symbol_data_map[symbol] = df
            self.current_market_prices[symbol] = df.iloc[-1]["close"]
            self.current_market_highs[symbol] = df.iloc[-1]["high"]
            self.current_market_lows[symbol] = df.iloc[-1]["low"]

    @staticmethod
    def _batch_build_features(dfs: List[pd.DataFrame], timeframe: str) -> List[pd.DataFrame]:
        """Static helper to process a list of DataFrames (runs in worker process)."""
        from defihunter.data.features import build_feature_pipeline
        return [build_feature_pipeline(df, timeframe=timeframe) for df in dfs]


    def _should_skip(self, symbol: str) -> bool:
        rec = _false_signal_memory.get(symbol, {})
        if rec.get("count", 0) >= FALSE_SIGNAL_MAX_FAILS and not rec.get("hit", False):
            if (rec.get("current_bar", 0) - rec.get("last_scan_bar", 0)) < FALSE_SIGNAL_COOLDOWN_BARS:
                return True
        return False

    def _evaluate_exits_parallel(self):
        # Exit evaluation (not in SignalPipeline core)
        for symbol, df in self.symbol_data_map.items():
            self.decay_signals[symbol] = self.decay_engine.evaluate_exit_signals(symbol, df)

    def _execute_decisions(self, res) -> list:
        # Update existing positions with full High/Low awareness
        self.paper_engine.update_positions(
            self.current_market_prices, 
            decay_signals=self.decay_signals,
            current_highs=self.current_market_highs,
            current_lows=self.current_market_lows
        )
        
        executed = []
        daily_loss = self.paper_engine.get_daily_loss_pct(self.current_market_prices)
        
        for d in res.final_decisions:
            if d.decision == "trade":
                self._handle_trade_decision(d, res.regime_label, daily_loss, res.master_df)
            executed.append(d)
            
        self._finalize_scan(executed, res.regime_label, len(self.symbol_data_map))
        return executed

    def _handle_trade_decision(self, d, regime, daily_loss, master_df):
        fam = d.explanation.get("family", "defi_beta")
        exec_cfg = self.config.get_family_execution(fam) if hasattr(self.config, "get_family_execution") else None
        mode = getattr(exec_cfg, "mode", "trade_allowed") if exec_cfg else "trade_allowed"
        
        if mode == "watch_only":
            d.explanation["participation_mode"] = "watch_only"
            return

        # Risk & Sizing
        kelly = self.risk_engine.calculate_kelly_size(win_prob=d.leader_prob or 0.5, reward_risk=2.0, leader_prob=d.leader_prob)
        
        # Adaptive Stop
        sym_rows = master_df[master_df["symbol"] == d.symbol]
        atr = float(sym_rows["_atr"].iloc[-1]) if not sym_rows.empty else 0.0
        
        stop_result = self.adaptive_stop_engine.compute_stop(
            row={"close": d.entry_price, "atr": atr}, family=fam, regime=regime, 
            fakeout_risk=d.fakeout_risk, stop_width_mult=getattr(exec_cfg, "stop_width_mult", 1.0)
        )
        
        notional = self.risk_engine.estimate_notional_from_stop(
            equity_val=self.paper_engine.portfolio.balance_usd, risk_pct=kelly,
            entry_price=d.entry_price, stop_price=stop_result["stop_price"]
        )
        
        is_val, reason = self.risk_engine.validate_trade(
            symbol=d.symbol, family=fam, current_portfolio=[p.dict() for p in self.paper_engine.portfolio.open_positions],
            equity_val=self.paper_engine.portfolio.balance_usd, daily_loss_pct=daily_loss,
            leader_prob=d.leader_prob, new_trade_notional=notional
        )
        
        if is_val:
            self.paper_engine.open_position(d, risk_pct=kelly, adaptive_stop_result=stop_result)
            self.paper_opened.add(d.symbol)
            self.kelly_map[d.symbol] = kelly
            self.adaptive_stop_map[d.symbol] = stop_result
        else:
            d.decision, d.explanation["rejection_reason"] = "reject", reason

    def _finalize_scan(self, executed, regime, watch_size):
        self.shadow_logger.log_scan(
            decisions=executed,
            regime=regime,
            universe_size=watch_size,
            timeframe=self.timeframe,
            adaptive_stop_map=self.adaptive_stop_map,
            paper_opened_symbols=self.paper_opened,
            kelly_map=self.kelly_map,
            setup_class_map={d.symbol: getattr(d, 'setup_class', 'unknown') for d in executed}
        )
        
        display_rows = []
        for dec in executed:
            display_rows.append({
                "Symbol": dec.symbol,
                "Family": dec.explanation.get("family", "—"),
                "Disc_S": round(dec.discovery_score, 1),
                "Ready_S": round(dec.entry_readiness, 1),
                "Risk_F": round(dec.fakeout_risk, 1),
                "L_Prob": f"{dec.leader_prob * 100:.0f}%",
                "Comp_S": round(dec.composite_leader_score, 1),
                "Action": dec.decision,
                "Entry": round(dec.entry_price, 4),
            })
        
        df = pd.DataFrame(display_rows)
        if not df.empty:
            logger.info("\n[LEADERBOARD]\n" + df.sort_values("Comp_S", ascending=False).to_markdown(index=False))
            
        self.broadcaster.broadcast(executed)

async def run_scanner(config, force_regime=None, limit=0):
    pipeline = ScanPipeline(config)
    return await pipeline.run(force_regime=force_regime, limit=limit)