from typing import Optional, Any
from defihunter.utils.structured_logger import s_logger

class RiskEngine:
    def __init__(self, config: dict, fetcher: Optional[Any] = None):
        """
        GT-Institutional: Risk Engine with Dependency Injection.
        fetcher: The data fetcher used for correlation analysis. (Optional but recommended)
        """
        self.max_open_positions = config.get("max_open_positions", 5)
        self.max_correlated_exposure = config.get("max_correlated_exposure", 2)
        self.max_risk_per_trade_pct = config.get("max_risk_per_trade_pct", 1.0)
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 5.0)
        self.liquidation_buffer = config.get("liquidation_buffer", 0.20)
        self.max_avg_correlation = config.get("max_avg_correlation", 0.70)
        self.kelly_fraction = config.get("kelly_fraction", 0.25)
        self.default_leverage = config.get("default_leverage", 5.0)
        
        from defihunter.engines.portfolio import CorrelationEngine
        self.corr_engine = CorrelationEngine(fetcher=fetcher)

    def calculate_kelly_size(
        self,
        win_prob: float,
        reward_risk: float,
        leader_prob: float = 0.5
    ) -> float:
        """
        Leadership-aware fractional Kelly.
        Returns risk percentage of equity (0..max_risk_per_trade_pct).
        """
        if reward_risk <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0

        kelly_perc = win_prob - ((1.0 - win_prob) / reward_risk)
        leader_multiplier = 1.0 + (max(0.0, leader_prob - 0.5) * 1.5)
        fractional_kelly = kelly_perc * self.kelly_fraction * leader_multiplier

        final_risk_pct = min(
            max(fractional_kelly * 100.0, 0.0),
            self.max_risk_per_trade_pct
        )
        return final_risk_pct

    def estimate_notional_from_stop(
        self,
        equity_val: float,
        risk_pct: float,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Convert desired % equity risk into actual notional using stop distance.

        Example:
            equity = 10,000
            risk_pct = 1.0
            entry = 100
            stop = 95  -> 5% stop distance
            risk_usd = 100
            notional = 100 / 0.05 = 2000
        """
        if equity_val <= 0 or risk_pct <= 0 or entry_price <= 0 or stop_price <= 0:
            return 0.0

        stop_distance_pct = abs(entry_price - stop_price) / entry_price
        if stop_distance_pct <= 0:
            return 0.0

        risk_usd = equity_val * (risk_pct / 100.0)
        notional = risk_usd / stop_distance_pct
        return max(notional, 0.0)

    def validate_trade(
        self,
        symbol: str,
        family: str,
        current_portfolio: list[dict],
        equity_val: float,
        daily_loss_pct: float,
        leader_prob: float,
        new_trade_notional: Optional[float],
        leverage: Optional[float] = None,
        family_max_pos: Optional[int] = None,
        symbol_data_map: Optional[dict] = None
    ) -> tuple[bool, str]:
        """
        Dynamic Position Manager with leadership awareness.
        family_max_pos: If provided, overrides the family exposure cap logic.
        """
        from defihunter.utils.logger import logger

        if equity_val <= 0:
            return False, "invalid_equity"

        if new_trade_notional is None or new_trade_notional <= 0:
            return False, "invalid_trade_notional"

        effective_leverage = leverage or self.default_leverage
        if effective_leverage <= 0:
            return False, "invalid_leverage"

        # 1. Rule: Max open positions (Global Cap)
        if len(current_portfolio) >= self.max_open_positions:
            return False, "max_open_positions_reached"

        # 2. Rule: Family-specific exposure cap
        # If family_max_pos is provided, it replaces the dynamic conviction-based logic.
        if family_max_pos is not None:
            allowed_exposure = family_max_pos
        else:
            allowed_exposure = self.max_correlated_exposure
            if leader_prob > 0.80:
                allowed_exposure += 1

        same_family_exposure = sum(
            1 for pos in current_portfolio if pos.get("family") == family
        )
        if same_family_exposure >= allowed_exposure:
            return False, f"max_family_exposure_reached ({allowed_exposure})"

        # 3. Rule: Daily Loss Killswitch
        if daily_loss_pct <= -self.max_daily_loss_pct:
            return False, "max_daily_loss_exceeded"

        # 4. Rule: No averaging down / duplicated symbols
        existing_symbols = [pos.get("symbol") for pos in current_portfolio]
        if symbol in existing_symbols:
            return False, "already_in_position_no_averaging_down"

        # 5. Rule: Correlation check
        if existing_symbols:
            try:
                corr_data = self.corr_engine.calculate_correlation(symbol, existing_symbols, symbol_data_map=symbol_data_map)
                mean_corr = corr_data.get("mean_corr", 0.0)
                if mean_corr > self.max_avg_correlation:
                    return False, f"high_portfolio_correlation ({mean_corr:.2f})"
            except Exception as e:
                logger.error(
                    f"Correlation engine error for {symbol}: {e}. Vetoing for safety."
                )
                return False, "correlation_engine_error"

        # 6. Margin check
        current_margin_usd = 0.0
        for pos in current_portfolio:
            pos_size = float(pos.get("size_usd", 0.0) or 0.0)
            pos_leverage = float(pos.get("leverage", effective_leverage) or effective_leverage)
            if pos_leverage <= 0: pos_leverage = effective_leverage
            current_margin_usd += pos_size / pos_leverage

        new_margin_req = new_trade_notional / effective_leverage
        total_margin_req = current_margin_usd + new_margin_req

        max_margin = equity_val * (1.0 - self.liquidation_buffer)
        if total_margin_req > max_margin:
            reason = f"insufficient_margin_buffer ({total_margin_req:.1f}$ > {max_margin:.1f}$)"
            s_logger.log("RiskEngine", "TRADE_REJECTED", symbol=symbol, data={"reason": reason})
            return False, reason

        s_logger.log("RiskEngine", "TRADE_ACCEPTED", symbol=symbol, data={"family": family, "notional": new_trade_notional})
        return True, ""