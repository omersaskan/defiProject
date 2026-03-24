from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal, Any, List
from datetime import datetime

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float

class FundingPoint(BaseModel):
    timestamp: datetime
    rate: float

class OpenInterestPoint(BaseModel):
    timestamp: datetime
    value: float

class SpreadPoint(BaseModel):
    timestamp: datetime
    spread_bps: float
    
class FeatureRow(BaseModel):
    symbol: str
    timestamp: datetime
    features: Dict[str, Any]

class ThresholdProfile(BaseModel):
    min_score: int
    min_relative_leadership: int
    min_volume: Optional[float] = None
    max_spread_bps: Optional[float] = None
    breakout_buffer_atr: float = 0.5
    retest_tolerance_atr: float = 0.3
    time_stop_bars: int = 24

class TradeCandidate(BaseModel):
    symbol: str
    timestamp: datetime   # only once — Bug #10 Fix
    entry_signal: bool
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    total_score: float
    relative_leadership_score: float
    explanation: Dict[str, Any]

class SectorRegime(BaseModel):
    timestamp: datetime
    label: str
    sector_strength_score: float
    alignment_flags: Dict[str, bool]

class CoinProfile(BaseModel):
    symbol: str
    family_label: str # e.g. "lending", "dex_amm", "defi_beta"
    primary_anchor: str
    behavior_profile: Literal["breakout_continuation", "retest_friendly", "sweep_reclaim", "mean_reverting", "fake_breakout_prone", "catalyst_sensitive"] = "breakout_continuation"
    preferred_setups: list[str] = []
    confidence: float = 1.0

class SignalOutput(BaseModel):
    symbol: str
    timestamp: datetime
    entry_signal: bool
    entry_type: str
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    risk_r: float
    setup_class: Optional[str] = None
    veto_reason: Optional[str] = None
    explanation: Dict[str, Any]

class FinalDecision(BaseModel):
    symbol: str
    timestamp: datetime
    final_trade_score: float
    decision: Literal["trade", "watch", "reject", "urgent_watch", "avoid_fakeout"]
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    discovery_score: float = 0.0
    entry_readiness: float = 0.0
    fakeout_risk: float = 0.0
    hold_quality: float = 0.0
    leader_prob: float = 0.0
    composite_leader_score: float = 0.0
    explanation: Dict[str, Any]
