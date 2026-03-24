import yaml
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from pathlib import Path

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
    min_volume: float = 1_000_000.0
    min_oi: float = 500_000.0
    max_spread_bps: float = 15.0
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

class AlertConfig(BaseModel):
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

class LabelingConfig(BaseModel):
    primary_target: str = "is_top3_family_next_24h"
    horizon_hours: int = 24

class TrainingConfig(BaseModel):
    primary_objective: str = "family_ranker"
    fallback_legacy: bool = False

class DecisionConfig(BaseModel):
    use_layered_logic: bool = True

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
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
    
def load_config(path: str | Path) -> AppConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # GT-REDESIGN: Auto-load universe groups if file exists
    groups_path = Path(path).parent / "universe_groups.yaml"
    if groups_path.exists():
        with open(groups_path, 'r') as f:
            groups_data = yaml.safe_load(f)
            if 'defi_families' in groups_data:
                data['families'] = {**data.get('families', {}), **groups_data['defi_families']}
            if 'universe_filters' in groups_data:
                data['universe'] = {**data.get('universe', {}), **groups_data['universe_filters']}
                
    return AppConfig(**data)
