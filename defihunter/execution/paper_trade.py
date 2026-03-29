import json
import os
from defihunter.utils.logger import logger
from defihunter.utils.trade_utils import TradeUtils
from datetime import datetime
from typing import List, Dict, Any
from defihunter.utils.db_manager import db_manager
from pydantic import BaseModel, Field

class PaperPosition(BaseModel):
    symbol: str
    entry_price: float
    stop_price: float
    tp1_price: float
    tp2_price: float
    size_usd: float
    entry_time: str
    family: str
    status: str = "open"  # open, runner, closed_tp, closed_sl, closed_decay
    runner_size_pct: float = 50.0 # % of position to keep after TP1
    partial_taken: bool = False
    peak_price_seen: float = 0.0
    max_favorable_excursion: float = 0.0
    giveback: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    regime: str = "unknown"
    setup_class: str = "unknown"
    risk_pct: float = 0.0


class PaperPortfolio(BaseModel):
    balance_usd: float = 10000.0
    initial_balance: float = 10000.0
    daily_start_balance: float = 10000.0
    last_reset_day: str = ""
    open_positions: List[PaperPosition] = Field(default_factory=list)
    trade_history: List[PaperPosition] = Field(default_factory=list)
    last_update: str = ""

class PaperTradeEngine:
    def __init__(self, state_path: str = "logs/paper_portfolio.json"):
        self.state_path = state_path
        self.portfolio = self.load_state()

    def load_state(self) -> PaperPortfolio:
        if os.path.exists(self.state_path) and os.path.getsize(self.state_path) > 0:
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    return PaperPortfolio(**data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load paper portfolio state: {e}. Resetting state.")
        
        return PaperPortfolio(last_update=datetime.now().isoformat())

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        try:
            with open(self.state_path, 'w') as f:
                f.write(self.portfolio.model_dump_json(indent=4))
        except Exception as e:
            logger.error(f"[PaperRunner] Failed to save state: {e}")

    def _check_daily_reset(self):
        """UTC-based daily reset for risk management."""
        now_utc = datetime.utcnow()
        today_str = now_utc.strftime("%Y-%m-%d")
        
        if self.portfolio.last_reset_day != today_str:
            logger.info(f"[PaperRisk] New day detected ({today_str}). Resetting daily_start_balance to ${self.portfolio.balance_usd:.2f}")
            self.portfolio.daily_start_balance = self.portfolio.balance_usd
            self.portfolio.last_reset_day = today_str
            self.save_state()

    def get_daily_loss_pct(self, current_prices: Dict[str, float] = None) -> float:
        """
        Returns daily PnL as a percentage of daily start balance.
        If current_prices is provided, includes UNREALIZED PnL (Equity-based).
        If current_prices is None, returns only REALIZED PnL.
        Negative values indicate losses (e.g., -1.5 for 1.5% loss).
        """
        self._check_daily_reset()
        if self.portfolio.daily_start_balance <= 0:
            return 0.0
            
        unrealized_pnl = 0.0
        if current_prices:
            for pos in self.portfolio.open_positions:
                if pos.symbol in current_prices:
                    curr_p = current_prices[pos.symbol]
                    unrealized_pnl += (curr_p / pos.entry_price - 1.0) * pos.size_usd
        
        total_equity = self.portfolio.balance_usd + unrealized_pnl
        # (Current / Start) - 1.0 -> e.g., (9900/10000) - 1.0 = -0.01
        pct = (total_equity / self.portfolio.daily_start_balance) - 1.0
        return pct * 100.0

    def open_position(self, decision: Any, risk_pct: float = None, adaptive_stop_result: dict = None):
        """
        Opens a paper position based on a FinalDecision object.
        If adaptive_stop_result is provided (from AdaptiveStopEngine), its stop/TP
        values override the decision's own values for full parity with backtest.
        """
        if risk_pct is None:
            risk_pct = decision.explanation.get('kelly_risk_pct', 1.0)

        risk_amount = self.portfolio.balance_usd * (risk_pct / 100.0)

        # Use adaptive stop/TP if provided, otherwise fall back to decision fields
        if adaptive_stop_result:
            stop_p  = adaptive_stop_result.get('stop_price', decision.stop_price)
            tp1_p   = adaptive_stop_result.get('tp1_price', decision.tp1_price)
            tp2_p   = adaptive_stop_result.get('tp2_price', decision.tp2_price)
        else:
            stop_p = decision.stop_price
            tp1_p  = decision.tp1_price
            tp2_p  = decision.tp2_price

        stop_dist = abs(decision.entry_price - stop_p)
        if stop_dist == 0:
            return False

        size_usd = risk_amount / (stop_dist / decision.entry_price)

        # Read family from top-level if available, fall back to explanation
        family = getattr(decision, 'explanation', {}).get('family', 'unknown')

        new_pos = PaperPosition(
            symbol=decision.symbol,
            entry_price=decision.entry_price,
            stop_price=stop_p,
            tp1_price=tp1_p,
            tp2_price=tp2_p,
            size_usd=size_usd,
            entry_time=datetime.now().isoformat(),
            family=family,
            peak_price_seen=decision.entry_price,
            regime=getattr(decision, 'regime_label', 'unknown'),
            setup_class=getattr(decision, 'setup_class', 'unknown'),
            risk_pct=risk_pct
        )


        self.portfolio.open_positions.append(new_pos)
        self.portfolio.last_update = datetime.now().isoformat()
        self.save_state()
        logger.info(f"Paper Trade Opened: {decision.symbol} at {decision.entry_price} | stop={stop_p:.4f} | tp1={tp1_p:.4f}")
        return True

    def update_positions(
        self, 
        current_prices: Dict[str, float], 
        decay_signals: Dict[str, Any] = None,
        current_highs: Dict[str, float] = None,
        current_lows: Dict[str, float] = None
    ):
        """
        GT-REDESIGN: Unified Management logic via ManagementCore.
        Now supports High/Low awareness for intra-bar SL/TP parity.
        """
        from defihunter.execution.manager import ManagementCore, ManagementAction, PositionStatus
        
        core = ManagementCore(config=getattr(self, "config", None))
        decay_signals = decay_signals or {}
        current_highs = current_highs or {}
        current_lows = current_lows or {}
        
        state_changed = False
        remaining_positions = []
        
        for pos in self.portfolio.open_positions:
            if pos.symbol not in current_prices:
                remaining_positions.append(pos)
                continue
                
            curr_price = current_prices[pos.symbol]
            curr_high = current_highs.get(pos.symbol)
            curr_low = current_lows.get(pos.symbol)
            
            pos.bars_held += 1
            
            # Peak tracking (conservative: use high if available)
            high_to_peak = curr_high if curr_high is not None else curr_price
            if high_to_peak > pos.peak_price_seen:
                pos.peak_price_seen = high_to_peak
                pos.max_favorable_excursion = (pos.peak_price_seen - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
                state_changed = True
            
            # Delegate to Unified ManagementCore
            res = core.evaluate(
                symbol=pos.symbol,
                current_price=curr_price,
                current_high=curr_high,
                current_low=curr_low,
                position_state=pos.model_dump(),
                decay_signal=decay_signals.get(pos.symbol),
                bars_held=pos.bars_held
            )
            
            if res.action == ManagementAction.FULL_EXIT:
                exit_p = res.exit_price or curr_price
                pos.status = res.new_status or "closed"
                pos.exit_reason = res.exit_reason or "EXIT"
                
                # Use Unified TradeUtils
                pnl = TradeUtils.calculate_pnl_usd(
                    entry_price=pos.entry_price,
                    exit_price=exit_p,
                    size_usd=pos.size_usd
                )
                self.portfolio.balance_usd += pnl
                pos.giveback = (pos.peak_price_seen - exit_p) / pos.entry_price if pos.entry_price > 0 else 0
                
                self.portfolio.trade_history.append(pos)
                state_changed = True
                logger.info(f"Paper Exit: {pos.symbol} at {exit_p} (Reason: {pos.exit_reason}, PnL: ${pnl:.2f})")
                
                # DB Logging
                try:
                    pos_dict = pos.model_dump()
                    pos_dict['exit_price'] = exit_p
                    db_manager.log_trade(pos_dict)
                except Exception as e:
                    logger.error(f"[PaperTrade] DB logging failed: {e}")

                
            elif res.action == ManagementAction.PARTIAL_EXIT:
                # Realize PnL for the exited portion
                realized_ratio = res.realized_pct / 100.0
                exit_p = res.exit_price or curr_price
                
                # Use Unified TradeUtils
                pnl_realized = TradeUtils.calculate_pnl_usd(
                    entry_price=pos.entry_price,
                    exit_price=exit_p,
                    size_usd=pos.size_usd * realized_ratio
                )
                self.portfolio.balance_usd += pnl_realized
                
                # Update remaining runner
                pos.status = "runner"
                pos.partial_taken = True
                pos.size_usd *= (1.0 - realized_ratio)
                if res.new_stop_price:
                    pos.stop_price = res.new_stop_price
                
                remaining_positions.append(pos)
                state_changed = True
                logger.info(f"Paper TP1 Hit (Partial): {pos.symbol} at {exit_p}. Runner active.")
                
            elif res.action == ManagementAction.UPDATE_STOP:
                if res.new_stop_price:
                    pos.stop_price = res.new_stop_price
                    state_changed = True
                remaining_positions.append(pos)
                
            else: # NO_ACTION
                remaining_positions.append(pos)
                
        if state_changed:
            self.portfolio.open_positions = remaining_positions
            self.portfolio.last_update = datetime.now().isoformat()
            self.save_state()
