import pandas as pd
import numpy as np
from defihunter.core.config import load_config
from defihunter.data.dataset_builder import DatasetBuilder
from defihunter.engines.ml_ranking import MLRankingEngine
from defihunter.engines.discovery import DiscoveryEngine
from defihunter.engines.entry import EntryEngine
from defihunter.engines.decision import DecisionEngine
from defihunter.data.universe import load_universe

def test_pipeline_wiring():
    print("--- 1. Testing Config & Universe ---")
    config = load_config('configs/default.yaml')
    universe = load_universe(config, strict_defi=True)
    print(f"Strict DeFi Universe Size: {len(universe)}")
    print(f"Sample: {universe[:5]}")

    print("\n--- 2. Testing DatasetBuilder (Timeframe-Aware) ---")
    builder = DatasetBuilder(config=config, timeframe='15m')
    # Dummy data
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='15min'),
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 101,
        'low': np.random.randn(200).cumsum() + 99,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.rand(200) * 1000,
        'family': 'defi_lending',
        'symbol': 'AAVE.p'
    })
    
    labeled_df = builder.generate_labels(df)
    target_cols = [c for c in labeled_df.columns if 'future' in c or 'target' in c]
    print(f"Generated Label Columns: {target_cols}")
    
    print("\n--- 3. Testing Discovery & Decision Engines ---")
    discovery = DiscoveryEngine(top_n=5)
    entry = EntryEngine(min_readiness=50)
    decision = DecisionEngine(top_n=3)
    
    # Mock scores for testing
    df['leader_prob'] = 0.85
    df['setup_conversion_prob'] = 0.70
    df['holdability_score'] = 90.0
    df['family_heat_score'] = 0.08
    df['family_breadth_score'] = 0.6
    df['peer_momentum'] = 0.03
    df['peer_rank'] = 0.9
    df['msb_bull'] = True
    df['taker_surge'] = True
    
    disc_df = discovery.compute_discovery_scores(df.tail(1))
    print(f"Discovery Score: {round(disc_df.iloc[-1]['discovery_score'], 1)}")
    
    final_decisions = decision.process_candidates(disc_df)
    if final_decisions:
        d = final_decisions[0]
        print(f"Final Decision for {d.symbol}: {d.decision}")
        print(f"Composite Score: {d.final_trade_score}")
        print(f"Explanation Keys: {list(d.explanation.keys())}")

if __name__ == "__main__":
    test_pipeline_wiring()
