from typing import Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel

class PositionStatus(str, Enum):
    OPEN = "open"
    RUNNER = "runner"
    CLOSED_SL = "closed_sl"
    CLOSED_TP1 = "closed_tp1"
    CLOSED_TP2 = "closed_tp2"
    CLOSED_DECAY = "closed_decay"
    CLOSED_TIME = "closed_time"

class ManagementAction(str, Enum):
    NO_ACTION = "no_action"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    UPDATE_STOP = "update_stop"

class ManagementResult(BaseModel):
    action: ManagementAction
    new_status: Optional[PositionStatus] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    new_stop_price: Optional[float] = None
    realized_pct: float = 0.0 # % of original position to realize now

class ManagementCore:
    """
    GT-Institutional: Unified Execution Manager (ManagementCore).
    The single source of truth for post-entry trade management.
    Ensures 100% parity between Backtest and Live/Paper exits.
    """
    def __init__(self, config: Any = None):
        self.config = config
        
        # 1. Config Injection (Defaults from AppConfig if available)
        bt_cfg = getattr(config, "backtest", None) if config else None
        risk_cfg = getattr(config, "risk", None) if config else None
        
        self.tp1_ratio = 50.0 # Default 50%
        self.trail_activation_mult = 0.3 
        self.trail_distance_mult = 0.2
        self.time_stop_bars = getattr(bt_cfg, "time_stop_bars", 24) if bt_cfg else 24

    def evaluate(
        self,
        symbol: str,
        current_price: float,
        position_state: Dict[str, Any],
        decay_signal: Optional[Dict[str, Any]] = None,
        bars_held: int = 0,
        current_high: Optional[float] = None,
        current_low: Optional[float] = None
    ) -> ManagementResult:
        """
        Evaluates a single position against current market state.
        Stateless logic used by both Backtest and Paper/Live engines.
        
        High/Low Awareness: If current_high/low are provided, we check SL/TP 
        against these instead of just close to ensure conservative parity.
        """
        status = position_state.get("status", "open")
        entry_p = position_state["entry_price"]
        stop_p = position_state["stop_price"]
        tp1_p = position_state["tp1_price"]
        tp2_p = position_state["tp2_price"]
        
        # Use High/Low if provided, otherwise fallback to current_price
        high_to_check = current_high if current_high is not None else current_price
        low_to_check = current_low if current_low is not None else current_price
        
        peak_p = max(position_state.get("peak_price_seen", 0.0), high_to_check)
        
        # 1. DECAY EXIT (Priority 1)
        if decay_signal and decay_signal.get("exit_signal", False):
            return ManagementResult(
                action=ManagementAction.FULL_EXIT,
                new_status=PositionStatus.CLOSED_DECAY,
                exit_price=current_price,
                exit_reason=decay_signal.get("exit_reason", "DECAY_SIGNAL")
            )

        # 2. STOP LOSS (Priority 2) - Check against LOW
        if low_to_check <= stop_p:
            return ManagementResult(
                action=ManagementAction.FULL_EXIT,
                new_status=PositionStatus.CLOSED_SL,
                exit_price=stop_p,
                exit_reason="STOP_LOSS"
            )

        # 3. PROFIT TARGETS & MANAGEMENT
        # TP1 (Partial Exit) - Check against HIGH
        if status == "open" and high_to_check >= tp1_p:
            return ManagementResult(
                action=ManagementAction.PARTIAL_EXIT,
                new_status=PositionStatus.RUNNER,
                exit_price=tp1_p,
                exit_reason="TP1_PARTIAL",
                new_stop_price=entry_p, # SL to Breakeven
                realized_pct=self.tp1_ratio
            )

        # TP2 (Final Exit) - Check against HIGH
        if status == "runner" and high_to_check >= tp2_p:
            return ManagementResult(
                action=ManagementAction.FULL_EXIT,
                new_status=PositionStatus.CLOSED_TP2,
                exit_price=tp2_p,
                exit_reason="TP2_FINAL"
            )

        # 4. TRAILING STOP (Only for runners)
        if status == "runner":
            reward_dist = tp2_p - entry_p
            if reward_dist > 0:
                activation_p = entry_p + (reward_dist * self.trail_activation_mult)
                if peak_p > activation_p:
                    trail_stop = peak_p - (reward_dist * self.trail_distance_mult)
                    if trail_stop > stop_p:
                        return ManagementResult(
                            action=ManagementAction.UPDATE_STOP,
                            new_stop_price=trail_stop
                        )

        # 5. TIME STOP
        if bars_held >= self.time_stop_bars:
            return ManagementResult(
                action=ManagementAction.FULL_EXIT,
                new_status=PositionStatus.CLOSED_TIME,
                exit_price=current_price,
                exit_reason="TIME_STOP"
            )

        return ManagementResult(action=ManagementAction.NO_ACTION)
