import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, Tuple
from dataclasses import dataclass

from defihunter.data.features import build_feature_pipeline, compute_family_features
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
from defihunter.core.models import FinalDecision
from defihunter.utils.logger import logger
from defihunter.utils.db_manager import db_manager


@dataclass
class PipelineResult:
    """Canonical structured result from the SignalPipeline."""
    master_df: pd.DataFrame
    final_decisions: List[FinalDecision]
    family_stats: Dict[str, pd.DataFrame]
    regime_label: str
    sector_data: Dict[str, Any]
    symbol_context_map: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]

class SignalPipeline:
    """
    GT-Institutional: Unified Execution Pipeline (ExecutionCore).
    The single source of truth for signal generation across live, backtest, and shadow modes.
    
    This class is intended to be stateless regarding portfolio and trade execution.
    It focuses entirely on transforming market data into ranked decisions.
    """
    def __init__(self, config: Any, model_dir: Optional[str] = None):
        self.config = config
        self.timeframe = getattr(config, "timeframe", "15m")
        self.model_dir = model_dir or f"models_{self.timeframe}"
        self.logs_dir = getattr(config, "logs_dir", "logs")

        # Initialize Engines (Dependency Registry)
        self.regime_engine = MarketRegimeEngine()
        self.sector_engine = SectorRegimeEngine()
        self.family_engine = FamilyEngine(config)
        self.family_aggregator = FamilyAggregator(config.families)
        self.decision_engine = DecisionEngine(top_n=getattr(config.decision, "top_n", 5))
        self.leadership_engine = LeadershipEngine(
            anchors=config.anchors, 
            ema_lengths=[config.ema.fast, config.ema.medium]
        )
        self.threshold_engine = ThresholdResolutionEngine(
            thresholds_config=config.regimes, 
            config=config
        )
        self.rule_engine = RuleEngine(config)
        self.ml_engine = MLRankingEngine(model_dir=self.model_dir)


    def run(
        self,
        symbol_data_map: Dict[str, pd.DataFrame],
        anchor_context: Dict[str, Dict[str, pd.DataFrame]],
        regime_label: Optional[str] = None,
        sector_data: Optional[Dict[str, Any]] = None,
        adaptive_weights: Optional[Dict[str, float]] = None,
        scan_timestamp: Optional[datetime] = None,
        mode: Literal["live", "historical", "shadow"] = "live",
        volatility_label: Optional[str] = None
    ) -> PipelineResult:
        """
        Executes the canonical processing chain (Stage-based Processor Pattern).
        """
        start_t = datetime.now()
        scan_timestamp = scan_timestamp or start_t
        
        # STAGE 1: Global Context
        t0 = datetime.now()
        regime_label, sector_data, adaptive_weights, volat_label = self._stage_context(
            anchor_context, regime_label, sector_data, adaptive_weights, volatility_label
        )
        t1 = datetime.now()

        # STAGE 2: Per-Symbol Feature Enrichment (Leadership, Profiles)
        symbol_context_map, processed_data_map = self._stage_symbol_features(
            symbol_data_map, anchor_context
        )
        t2 = datetime.now()

        # STAGE 3: Family Stats & Scoring (Aggregations, Rules, Thresholds)
        family_stats, all_last_rows = self._stage_scoring(
            processed_data_map, symbol_context_map, regime_label, sector_data, adaptive_weights, volat_label, mode
        )
        t3 = datetime.now()

        if not all_last_rows:
            return self._empty_result(regime_label, sector_data, family_stats, mode, scan_timestamp)

        # STAGE 4: Global Ranking (ML, Cross-sectional Features)
        master_df = self._stage_ranking(all_last_rows, mode)
        t4 = datetime.now()

        # STAGE 5: Decision Logic (Final Selection)
        final_decisions = self._stage_decide(master_df)
        t5 = datetime.now()

        # Compile Metadata & Return
        metadata = self._build_metadata(start_t, scan_timestamp, mode, adaptive_weights)
        metadata['timings'] = {
            "s_context_ms": (t1 - t0).total_seconds() * 1000,
            "s_sym_feats_ms": (t2 - t1).total_seconds() * 1000,
            "s_scoring_eval_ms": (t3 - t2).total_seconds() * 1000,
            "s_ranking_ml_ms": (t4 - t3).total_seconds() * 1000,
            "s_decision_ms": (t5 - t4).total_seconds() * 1000,
        }
        
        return PipelineResult(
            master_df=master_df,
            final_decisions=final_decisions,
            family_stats=family_stats,
            regime_label=regime_label,
            sector_data=sector_data,
            symbol_context_map=symbol_context_map,
            metadata=metadata
        )
    # --- Processing Stages ---

    def _stage_context(self, anchor_context, regime, sector, weights, volatility=None):
        """Stage 1: Resolve global market and sector state."""
        out_volatility = volatility or "normal"
        
        # Protective unpack in case upstream forgot to split the tuple
        if isinstance(regime, tuple):
            regime, out_volatility = regime

        if regime is None:
            regime, out_volatility = self._resolve_regime(anchor_context)
            
        if sector is None:
            sector = self._resolve_sector_regime(anchor_context)
        if weights is None:
            weights = self._get_default_weights()
            
        return regime, sector, weights, out_volatility

    def _stage_symbol_features(self, symbol_data_map, anchor_context):
        """Stage 2: Per-symbol enrichment (Leadership, Family Profile)."""
        symbol_context_map = {}
        processed_data_map = {}
        
        anchors_tf = {k: v[self.timeframe] for k, v in anchor_context.items() if self.timeframe in v}

        for symbol, df in symbol_data_map.items():
            if df.empty: continue
            
            # Enrich features
            df = self.leadership_engine.add_leadership_features(df, anchors_tf, timeframe=self.timeframe)
            profile = self.family_engine.profile_coin(symbol, historical_data=df)
            
            symbol_context_map[symbol] = {
                "family": profile.family_label,
                "primary_anchor": profile.primary_anchor,
                "profile_confidence": profile.confidence
            }
            processed_data_map[symbol] = df
            
        return symbol_context_map, processed_data_map

    def _stage_scoring(self, processed_data_map, symbol_context_map, regime, sector, weights, volatility, mode="live"):
        """Stage 3: Cross-sectional aggregation and rule-based scoring."""
        family_stats = self.family_aggregator.compute_family_stats(processed_data_map, timeframe=self.timeframe)
        all_last_rows = []

        for symbol, df in processed_data_map.items():
            df = self.family_aggregator.inject_family_features(symbol, df, family_stats, timeframe=self.timeframe)
            
            ctx = symbol_context_map[symbol]
            res_thresholds = self.threshold_engine.resolve_thresholds(regime=regime, family=ctx["family"], volatility=volatility)
            ctx["resolved_thresholds"] = res_thresholds
            
            if mode in ["live", "shadow"] and not df.empty:
                # Optimize by evaluating only the ultimate bar for rules and string explanations
                last_row_df = df.iloc[[-1]].copy()
                last_row_df = self.rule_engine.evaluate(
                    last_row_df, regime=regime, family=ctx["family"], 
                    resolved_thresholds=res_thresholds, sector_data=sector,
                    adaptive_weights=weights, primary_anchor=ctx["primary_anchor"]
                )
                
                # Re-integrate evaluated final row back into existing historical array
                df_hist = df.iloc[:-1].copy()
                # Initialize new rule columns with NaN for historical rows to allow concat
                for c in last_row_df.columns:
                    if c not in df_hist.columns:
                        df_hist[c] = float('nan') if last_row_df[c].dtype.kind in 'bcif' else None
                df = pd.concat([df_hist, last_row_df])
            else:
                df = self.rule_engine.evaluate(
                    df, regime=regime, family=ctx["family"], 
                    resolved_thresholds=res_thresholds, sector_data=sector,
                    adaptive_weights=weights, primary_anchor=ctx["primary_anchor"]
                )
            
            if not df.empty:
                last_bar = df.tail(1).to_dict("records")[0]
                last_bar.update({
                    "symbol": symbol, "family": ctx["family"], "regime": regime,
                    "_atr": float(df.iloc[-1].get("atr", 0.0)),
                    "_close_raw": float(df.iloc[-1].get("close", 0.0))
                })
                all_last_rows.append(last_bar)
            
            processed_data_map[symbol] = df
            
        return family_stats, all_last_rows

    def _stage_ranking(self, all_last_rows, mode):
        """Stage 4: Batch ranking (ML and Peer Momentum)."""
        master_df = pd.DataFrame(all_last_rows)
        master_df = compute_family_features(master_df)
        
        try:
            master_df, _ = self.ml_engine.rank_candidates(master_df, use_family_ranker=True)
        except Exception as e:
            logger.error(f"[Pipeline] ML Ranking failed: {e}")
            master_df = self.ml_engine.ensure_canonical_columns(master_df)

        if mode in ["live", "shadow"]:
            self._log_features(master_df)
            
        return master_df

    def _stage_decide(self, master_df):
        """Stage 5: Signal selection."""
        return self.decision_engine.process_candidates(master_df)

    def _build_metadata(self, start_t, scan_timestamp, mode, adaptive_weights):
        """Helper to build consistent result metadata."""
        return {
            "mode": mode,
            "timeframe": self.timeframe,
            "scan_timestamp": scan_timestamp.isoformat(),
            "model_dir": self.model_dir,
            "elapsed_ms": (datetime.now() - start_t).total_seconds() * 1000,
            "adaptive_weights": adaptive_weights
        }

    # --- Utility Methods ---

    def _log_features(self, df: pd.DataFrame):
        try:
            db_manager.log_features(df)
        except Exception as e:
            logger.warning(f"[Pipeline] DB Feature logging failed: {e}")

    def _resolve_regime(self, anchor_context: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[str, str]:
        btc_a = next((a for a in anchor_context if a.startswith("BTC")), None)
        eth_a = next((a for a in anchor_context if a.startswith("ETH")), None)
        if btc_a and eth_a:
            data = self.regime_engine.detect_regime(anchor_context[btc_a], anchor_context[eth_a])
            return data.get("label", "unknown"), data.get("volatility", "normal")
        return "unknown", "normal"

    def _resolve_sector_regime(self, anchor_context: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        return self.sector_engine.get_sector_regime(
            anchor_context.get("ETH.p", {}).get("1h"),
            anchor_context.get("AAVE.p", {}).get("1h"),
            anchor_context.get("UNI.p", {}).get("1h"),
        )

    def _get_default_weights(self) -> Dict[str, float]:
        try:
            adaptive = AdaptiveWeightsEngine()
            return adaptive.current_weights
        except:
            return {"trend_score": 1.0, "expansion_score": 1.0, "participation_score": 1.0, "relative_leadership_score": 1.0}

    def _empty_result(self, regime, sector, family_stats, mode, ts) -> PipelineResult:
        return PipelineResult(
            master_df=pd.DataFrame(),
            final_decisions=[],
            family_stats=family_stats,
            regime_label=regime,
            sector_data=sector,
            symbol_context_map={},
            metadata={"mode": mode, "scan_timestamp": ts.isoformat(), "empty": True}
        )
