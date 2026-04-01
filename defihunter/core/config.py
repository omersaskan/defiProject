import yaml
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
from defihunter.utils.logger import logger

class DataConfig(BaseModel):
    timeframes: List[str] = ["15m", "1h", "4h"]
    main_execution_timeframe: str = "15m"

class UniverseConfig(BaseModel):
    min_24h_quote_volume: float = 1_000_000.0
    min_oi_usd: float = 500_000.0
    max_spread_bps: float = 15.0
    min_listing_age_bars: int = 500
    max_wick_ratio: float = 0.5
    is_strictly_defi: bool = True
    defi_universe: List[str] = Field(default_factory=list)

class FeatureConfig(BaseModel):
    horizon_returns: List[int] = [1, 4, 12, 24]
    zscore_window: int = 50
    breakout_lookback: int = 20
    retest_tolerance: float = 0.5
    
class BacktestConfig(BaseModel):
    initial_equity: float = 10000.0
    fee_bps: float = 2.0
    slippage_bps: float = 1.0
    max_concurrent_positions: int = 5
    funding_costs_enabled: bool = False
    time_stop_bars: int = 24

class EmaConfig(BaseModel):
    fast: int = 20
    medium: int = 55
    slow: int = 200

class RegimeOverrides(BaseModel):
    min_score: Optional[int] = None
    min_relative_leadership: Optional[int] = None
    min_volume: Optional[float] = None
    max_spread_bps: Optional[float] = None

class RegimeConfig(BaseModel):
    min_volume: float = 10_000_000.0
    min_oi: float = 500_000.0
    max_spread_bps: float = 15.0
    min_score: int = 50
    min_relative_leadership: int = 0
    breakout_buffer_atr: float = 0.5
    retest_tolerance_atr: float = 0.3
    time_stop_bars: int = 24
    overrides: Dict[str, RegimeOverrides] = Field(default_factory=dict)
    
class FamilyConfigItem(BaseModel):
    primary_anchor: str
    members: List[str]
    behavior_profile: Optional[str] = "breakout_continuation"
    preferred_setups: Optional[List[str]] = Field(default_factory=lambda: ["trend_pullback", "breakout"])
    threshold_overrides: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 5.0
    max_open_positions: int = 5
    max_correlated_exposure: int = 2
    backtest_fee_bps: float = 2.0
    backtest_slippage_bps: float = 1.0
    liquidation_buffer: float = 0.20
    max_avg_correlation: float = 0.70
    kelly_fraction: float = 0.25
    default_leverage: float = 5.0


PARTICIPATION_MODE = Literal["trade_allowed", "reduced_risk", "watch_only"]


class FamilyExecutionConfig(BaseModel):
    """
    Per-family execution gate and risk parameters.

    stop_width_mult: multiplier applied to stop distance (e.g. 1.3 = 30% wider stop).
    IMPORTANT: when stop_width_mult > 1, position size is proportionally reduced so
    net dollar risk (= size_usd × stop_pct) stays constant.
    """
    mode:                  PARTICIPATION_MODE = "trade_allowed"
    risk_pct_mult:         float = 1.0    # multiply base kelly% by this
    stop_width_mult:       float = 1.0    # widen stop; size shrinks proportionally
    min_entry_readiness:   float = 0.0    # extra gate for reduced_risk
    min_leader_prob:       float = 0.0    # extra gate for reduced_risk
    max_open_positions:    int   = 5      # per-family position cap

class AlertConfig(BaseModel):
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

class LabelingConfig(BaseModel):
    primary_target: str = "is_top3_family_next_24h"
    horizon_hours: int = 24

class DecisionConfig(BaseModel):
    use_layered_logic: bool = True
    top_n: int = 5
    discovery_top_n: int = 10
    min_entry_readiness: float = 65.0

class TrainingConfig(BaseModel):
    primary_objective: str = "family_ranker"
    fallback_legacy: bool = False
    lgbm_params: Dict[str, Any] = Field(default_factory=lambda: {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })

class ScoringConfig(BaseModel):
    trend_weight: float = 0.5
    expansion_weight: float = 0.5
    participation_weight: float = 0.5
    relative_leadership_weight: float = 2.0
    funding_penalty_weight: float = 1.0

class ExitConfig(BaseModel):
    enable_leadership_decay: bool = True

class AppConfig(BaseModel):
    anchors: List[str] = ["BTC.p", "ETH.p", "AAVE.p", "UNI.p"]
    timeframe: str = "15m"
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    ema: EmaConfig = Field(default_factory=EmaConfig)
    regimes: RegimeConfig = Field(default_factory=RegimeConfig)
    families: Dict[str, FamilyConfigItem] = Field(default_factory=dict)
    family_execution: Dict[str, FamilyExecutionConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)


    def get_family_execution(self, family: str) -> FamilyExecutionConfig:
        """Return FamilyExecutionConfig for a family, defaulting to trade_allowed."""
        return self.family_execution.get(family, FamilyExecutionConfig())

def load_config(path: str | Path) -> AppConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # GT-REDESIGN: Auto-load universe groups if file exists
    groups_path = Path(path).parent / "universe_groups.yaml"
    if groups_path.exists():
        with open(groups_path, 'r') as f:
            groups_data = yaml.safe_load(f)

            # Fix #1: Defensive merge — sadece geçerli FamilyConfigItem yapısına sahip
            # girişleri al; malformed olanları sessizce geçirme, uyar.
            families_data = {}
            for k, v in groups_data.items():
                if k == 'universe_filters':
                    continue
                if isinstance(v, dict) and 'primary_anchor' in v and 'members' in v:
                    families_data[k] = v
                else:
                    logger.warning(f"[CONFIG WARN] universe_groups.yaml — malformed family entry skipped: '{k}' "
                                   f"(primary_anchor veya members eksik)")

            if families_data:
                data['families'] = {**data.get('families', {}), **families_data}

            if 'universe_filters' in groups_data:
                data['universe'] = {**data.get('universe', {}), **groups_data['universe_filters']}

    config = AppConfig(**data)
    
    # ── Sprint 1 Guard: Strict Family Overlap Validation ──
    seen_members = {}
    for family_name, family_cfg in config.families.items():
        for member in family_cfg.members:
            if member in seen_members:
                error_msg = f"[CONFIG VALIDATION ERROR] Symbol '{member}' exists in multiple families: '{seen_members[member]}' and '{family_name}'. Symbols must explicitly belong to only one family."
                logger.error(error_msg)
                raise ValueError(error_msg)
            seen_members[member] = family_name

    return config
