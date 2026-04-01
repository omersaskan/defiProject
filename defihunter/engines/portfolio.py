import pandas as pd
import numpy as np
from typing import List

class CorrelationEngine:
    """
    Analyzes Pearson correlation between assets to prevent over-concentration.
    """
    def __init__(self, fetcher=None):
        self.fetcher = fetcher
        
    def calculate_correlation(self, candidate_symbol: str, portfolio_symbols: List[str], timeframe: str = '1h', limit: int = 100, symbol_data_map: dict = None) -> dict:
        """
        Calculates correlation matrix and summary stats for a candidate vs existing portfolio.
        """
        if not self.fetcher and not symbol_data_map:
            return {"mean_corr": 0.0, "max_corr": 0.0, "matrix": {}}
            
        if not portfolio_symbols:
            return {"mean_corr": 0.0, "max_corr": 0.0, "matrix": {}}
            
        all_symbols = [candidate_symbol] + portfolio_symbols
        price_data = {}
        
        for symbol in all_symbols:
            if symbol_data_map and symbol in symbol_data_map:
                df = symbol_data_map[symbol]
                if not df.empty:
                    ts_series = df.set_index('timestamp')['close']
                    if timeframe != '15m':
                        try:
                            # Resample to specified timeframe to preserve precise semantics
                            ts_series = ts_series.resample(timeframe).last().dropna()
                        except Exception:
                            pass
                    price_data[symbol] = ts_series.tail(limit)
            elif self.fetcher:
                df = self.fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not df.empty:
                    price_data[symbol] = df.set_index('timestamp')['close']
                
        if len(price_data) < 2:
            return {"mean_corr": 0.0, "max_corr": 0.0, "matrix": {}}
            
        # Align data and calculate log returns
        df_combined = pd.DataFrame(price_data).ffill().dropna()
        if len(df_combined) < 10:
            return {"mean_corr": 0.0, "max_corr": 0.0, "matrix": {}}
            
        returns_df = np.log(df_combined / df_combined.shift(1)).dropna()
        corr_matrix = returns_df.corr()
        
        # Stats for candidate vs existing
        candidate_corrs = corr_matrix[candidate_symbol].drop(candidate_symbol)
        
        return {
            "mean_corr": candidate_corrs.mean(),
            "max_corr": candidate_corrs.max(),
            "matrix": corr_matrix.to_dict()
        }
