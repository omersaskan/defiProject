from defihunter.engines.portfolio import CorrelationEngine
from defihunter.data.binance_fetcher import BinanceFuturesFetcher
import pandas as pd

def test_correlation():
    fetcher = BinanceFuturesFetcher()
    engine = CorrelationEngine(fetcher=fetcher)
    
    candidate = "AAVE.p"
    portfolio = ["ETH.p", "BTC.p"]
    
    print(f"Calculating correlation for {candidate} vs {portfolio}...")
    results = engine.calculate_correlation(candidate, portfolio)
    
    print("\nResults:")
    print(f"Mean Correlation: {results['mean_corr']:.4f}")
    print(f"Max Correlation: {results['max_corr']:.4f}")
    
    if results['matrix']:
        df_matrix = pd.DataFrame(results['matrix'])
        print("\nCorrelation Matrix:")
        print(df_matrix)

if __name__ == "__main__":
    test_correlation()
