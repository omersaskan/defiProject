import json
import os
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel

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

class PaperPortfolio(BaseModel):
    balance_usd: float = 10000.0
    initial_balance: float = 10000.0
    open_positions: List[PaperPosition] = []
    trade_history: List[PaperPosition] = []
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
                print(f"Warning: Failed to load paper portfolio state: {e}. Resetting state.")
        
        return PaperPortfolio(last_update=datetime.now().isoformat())

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            f.write(self.portfolio.model_dump_json(indent=4))

    def open_position(self, decision: Any, risk_pct: float = None):
        """
        Attempts to open a paper position based on a FinalDecision object.
        """
        if risk_pct is None:
            risk_pct = decision.explanation.get('kelly_risk_pct', 1.0)
            
        risk_amount = self.portfolio.balance_usd * (risk_pct / 100.0)
        stop_dist = abs(decision.entry_price - decision.stop_price)
        
        if stop_dist == 0: return False
        
        size_usd = risk_amount / (stop_dist / decision.entry_price)
        
        new_pos = PaperPosition(
            symbol=decision.symbol,
            entry_price=decision.entry_price,
            stop_price=decision.stop_price,
            tp1_price=decision.tp1_price,
            tp2_price=decision.tp2_price,
            size_usd=size_usd,
            entry_time=datetime.now().isoformat(),
            family=decision.explanation.get('family', 'unknown'),
            peak_price_seen=decision.entry_price
        )
        
        self.portfolio.open_positions.append(new_pos)
        self.portfolio.last_update = datetime.now().isoformat()
        self.save_state()
        print(f"Paper Trade Opened: {decision.symbol} at {decision.entry_price}")
        return True

    def update_positions(self, current_prices: Dict[str, float], decay_signals: Dict[str, Any] = None):
        """
        GT-REDESIGN: Check open positions for SL/TP/Runner/Decay.
        Supports Partial Take Profit at TP1. Ensures robust state saving.
        """
        if decay_signals is None:
            decay_signals = {}
        state_changed = False
        remaining_positions = []
        
        for pos in self.portfolio.open_positions:
            if pos.symbol not in current_prices:
                remaining_positions.append(pos)
                continue
                
            curr_price = current_prices[pos.symbol]
            if curr_price > pos.peak_price_seen:
                pos.peak_price_seen = curr_price
                if pos.entry_price > 0:
                    pos.max_favorable_excursion = (pos.peak_price_seen - pos.entry_price) / pos.entry_price
                state_changed = True
                
            closed = False
            
            # 1. Decay Exit Check (Highest Priority Patch 2)
            if decay_signals.get(pos.symbol, {}).get("exit_signal", False):
                pos.status = "closed_decay"
                pos.exit_reason = decay_signals[pos.symbol].get("exit_reason", "Decay")
                closed = True
                pnl = (curr_price / pos.entry_price - 1) * pos.size_usd
                self.portfolio.balance_usd += pnl
                print(f"Paper Decay Exit: {pos.symbol} at {curr_price} (Reason: {pos.exit_reason})")

            # 2. Stop Loss Check
            elif curr_price <= pos.stop_price:
                pos.status = "closed_sl"
                pos.exit_reason = "Stop Loss Hit"
                closed = True
                pnl = (pos.stop_price / pos.entry_price - 1) * pos.size_usd
                self.portfolio.balance_usd += pnl
                print(f"Paper SL Hit: {pos.symbol} at {curr_price} (PnL: ${pnl:.2f})")
                
            # 3. TP1 Check (Partial Take Profit)
            elif pos.status == "open" and curr_price >= pos.tp1_price:
                # Sell half (or runner_size_pct complement)
                sell_ratio = (100.0 - pos.runner_size_pct) / 100.0
                pnl_realized = (pos.tp1_price / pos.entry_price - 1) * (pos.size_usd * sell_ratio)
                self.portfolio.balance_usd += pnl_realized
                
                # Update remaining position to "runner"
                pos.status = "runner"
                pos.partial_taken = True
                pos.size_usd *= (pos.runner_size_pct / 100.0)
                # Move SL to breakeven for the runner
                pos.stop_price = pos.entry_price
                state_changed = True
                print(f"Paper TP1 Hit (Partial): {pos.symbol} at {curr_price}. Runner active.")

            # 4. TP2 Check (Final TP for runner) & Trailing Stop
            elif pos.status == "runner":
                if curr_price >= pos.tp2_price:
                    pos.status = "closed_tp"
                    pos.exit_reason = "Final TP Hit"
                    closed = True
                    pnl = (pos.tp2_price / pos.entry_price - 1) * pos.size_usd
                    self.portfolio.balance_usd += pnl
                    print(f"Paper TP2 Hit: {pos.symbol} at {curr_price} (PnL: ${pnl:.2f})")
                else:
                    # Trailing Stop Logic (Patch 2)
                    reward_dist = pos.tp2_price - pos.entry_price
                    activation_price = pos.entry_price + (reward_dist * 0.3)
                    if pos.peak_price_seen > activation_price:
                        # Trail 20% of reward distance behind peak
                        new_stop = pos.peak_price_seen - (reward_dist * 0.2)
                        if new_stop > pos.stop_price:
                            pos.stop_price = new_stop
                            state_changed = True

            if closed:
                if pos.peak_price_seen > 0:
                    pos.giveback = (pos.peak_price_seen - curr_price) / pos.peak_price_seen
                self.portfolio.trade_history.append(pos)
                state_changed = True
            else:
                remaining_positions.append(pos)
                
        if state_changed:
            self.portfolio.open_positions = remaining_positions
            self.portfolio.last_update = datetime.now().isoformat()
            self.save_state()
