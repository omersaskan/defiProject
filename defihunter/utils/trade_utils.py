from typing import Optional

class TradeUtils:
    """
    GT-Institutional: Unified Trade Utility (PnL, Fees, Slippage).
    Ensures 100% numerical consistency between Backtest, Paper, and Live engines.
    """
    
    @staticmethod
    def calculate_fee(notional_usd: float, fee_bps: float) -> float:
        """Returns fee in USD."""
        return notional_usd * (fee_bps / 10_000.0)
    
    @staticmethod
    def calculate_slippage_cost(notional_usd: float, slippage_bps: float) -> float:
        """Returns slippage cost in USD."""
        return notional_usd * (slippage_bps / 10_000.0)

    @staticmethod
    def calculate_pnl_usd(
        entry_price: float, 
        exit_price: float, 
        size_usd: float, 
        side: str = "long"
    ) -> float:
        """
        Returns absolute PnL in USD (excluding fees).
        PnL = ((Exit / Entry) - 1) * Size
        """
        if entry_price <= 0: return 0.0
        multiplier = 1.0 if side.lower() == "long" else -1.0
        return ((exit_price / entry_price) - 1.0) * size_usd * multiplier

    @staticmethod
    def calculate_pnl_r(
        entry_price: float,
        exit_price: float,
        stop_price: float,
        side: str = "long"
    ) -> float:
        """
        Returns PnL in units of 'R' (Risk).
        1.0 R = Profit equal to the initial risk distance.
        -1.0 R = Loss equal to the initial risk distance.
        """
        risk_dist = abs(entry_price - stop_price)
        if risk_dist <= 0: return 0.0
        
        multiplier = 1.0 if side.lower() == "long" else -1.0
        return ((exit_price - entry_price) / risk_dist) * multiplier

    @staticmethod
    def calculate_net_pnl_r(
        entry_price: float,
        exit_price: float,
        stop_price: float,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        side: str = "long"
    ) -> float:
        """
        Returns PnL in R units, SUBTRACTING fees and slippage.
        """
        raw_r = TradeUtils.calculate_pnl_r(entry_price, exit_price, stop_price, side)
        risk_dist = abs(entry_price - stop_price)
        if risk_dist <= 0: return raw_r
        
        # Costs in terms of Price
        total_bps = fee_bps + slippage_bps
        cost_price = entry_price * (total_bps / 10_000.0)
        
        # Convert Price cost to R cost
        cost_r = cost_price / risk_dist
        
        return raw_r - cost_r
