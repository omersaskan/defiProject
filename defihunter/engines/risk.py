class RiskEngine:
    def __init__(self, config: dict, fetcher=None):
        self.max_open_positions = config.get("max_open_positions", 5)
        self.max_correlated_exposure = config.get("max_correlated_exposure", 2)
        self.max_risk_per_trade_pct = config.get("max_risk_per_trade_pct", 1.0)
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 5.0)
        self.liquidation_buffer = config.get("liquidation_buffer", 0.20)
        self.max_avg_correlation = config.get("max_avg_correlation", 0.70)
        self.kelly_fraction = config.get("kelly_fraction", 0.25)
        
        from defihunter.engines.portfolio import CorrelationEngine
        # Bug #6 Fix: Accept shared fetcher via DI instead of always creating a new one
        if fetcher is None:
            from defihunter.data.binance_fetcher import BinanceFuturesFetcher
            fetcher = BinanceFuturesFetcher()
        self.corr_engine = CorrelationEngine(fetcher=fetcher)
        
    def calculate_kelly_size(self, win_prob: float, reward_risk: float, leader_prob: float = 0.5) -> float:
        """
        GT-REDESIGN: Leadership-Aware Kelly Sizing.
        Scales the base Kelly fraction up if the coin is a high-probability family leader.
        """
        if reward_risk <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
            
        kelly_perc = win_prob - ((1.0 - win_prob) / reward_risk)
        
        # Leader Multiplier: Max conviction on clear leaders (prob > 0.8)
        leader_multiplier = 1.0 + (max(0, leader_prob - 0.5) * 1.5)
        fractional_kelly = kelly_perc * self.kelly_fraction * leader_multiplier
        
        final_risk_pct = min(max(fractional_kelly * 100.0, 0.0), self.max_risk_per_trade_pct)
        return final_risk_pct

    def validate_trade(self,
                     symbol: str, 
                     family: str, 
                     current_portfolio: list[dict],
                     equity_val: float,
                     daily_loss_pct: float,
                     leader_prob: float = 0.5) -> tuple[bool, str]:
        """
        GT-REDESIGN: Dynamic Position Manager with Leadership awareness.
        """
        # Rule: Max open positions
        if len(current_portfolio) >= self.max_open_positions:
            return False, "max_open_positions_reached"
            
        # Rule: Dynamic same-family exposure based on Leadership conviction
        allowed_exposure = self.max_correlated_exposure
        if leader_prob > 0.80:
            allowed_exposure += 1  # Allow one more if it's a very clear leader
            
        same_family_exposure = sum(1 for pos in current_portfolio if pos.get('family') == family)
        if same_family_exposure >= allowed_exposure:
            return False, f"max_family_exposure_reached ({allowed_exposure})"
            
        # Rule: Daily Loss Killswitch
        if daily_loss_pct <= -self.max_daily_loss_pct:
            return False, "max_daily_loss_exceeded"
            
        # Rule: No averaging down / duplicated symbols
        existing_symbols = [pos.get('symbol') for pos in current_portfolio]
        if symbol in existing_symbols:
            return False, "already_in_position_no_averaging_down"
            
        # Rule: Correlation Multiplier (Advanced)
        if existing_symbols:
            try:
                corr_data = self.corr_engine.calculate_correlation(symbol, existing_symbols)
                if corr_data.get('mean_corr', 0) > self.max_avg_correlation:
                    return False, f"high_portfolio_correlation ({corr_data.get('mean_corr', 0):.2f})"
            except Exception as e:
                # If correlation engine fails or is missing historical data, fallback to strict family limits
                pass
        # Margin required = sum(notional_size / leverage)
        # We assume a default leverage of 5.0 for DeFi perps
        assumed_leverage = 5.0
        current_margin_usd = sum(pos.get('size_usd', 0) / assumed_leverage for pos in current_portfolio)
        
        # New trade potential notional (simplified $1000 size for demo)
        new_trade_notional = 1000.0
        new_margin_req = new_trade_notional / assumed_leverage
        
        total_margin_req = current_margin_usd + new_margin_req
        
        # Max margin utilization (80% of equity to leave 20% buffer)
        max_margin = equity_val * (1.0 - self.liquidation_buffer)
        
        if total_margin_req > max_margin:
            return False, f"insufficient_margin_buffer (using {total_margin_req:.1f}$ of max {max_margin:.1f}$)"
            
        return True, ""
