import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from defihunter.engines.ml_ranking import MLRankingEngine
from defihunter.data.dataset_builder import DatasetBuilder

def test_ml_multi_target():
    print("--- Testing ML Multi-Target Implementation ---")
    
    # Generate mock df
    rows = []
    for i in range(100):
        rows.append({
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=15 * i),
            'symbol': 'AAVE.p',
            'close': 100 + (np.sin(i) * 5),
            'high': 102 + (np.sin(i) * 5),
            'low': 98 + (np.sin(i) * 5),
            'feature_1': np.random.rand(),
            'feature_2': np.random.rand(),
            'atr': 2.0
        })
    df = pd.DataFrame(rows)
    
    # 1. Dataset Builder
    builder = DatasetBuilder(target_r=1.5, stop_r=-1.0, timeframe='15m')
    X, y_df = builder.prepare_training_data(df)
    
    print(f"Generated Label DataFrame Shape: {y_df.shape}")
    print(f"Expected targets present: {'target_hit' in y_df.columns and 'mfe_r' in y_df.columns}")
    
    # Merge for training structure
    train_df = pd.concat([X, y_df], axis=1)
    
    # 2. Train Engine (Requires at least enough rows to split)
    engine = MLRankingEngine()
    
    print("\nTraining Engine on multi-target dataframe...")
    success = engine.train(train_df, target_clf_col='target_hit', target_reg_col='mfe_r')
    
    print(f"Training Success: {success}")
    if success:
        print(f"Classifier Model: {type(engine.long_clf_model)}")
        print(f"Regressor Model: {type(engine.reg_model)}")
        
        # 3. Predict & Rank
        candidates = df.iloc[-5:].copy()
        candidates, top_n = engine.rank_candidates(candidates, top_n=2)
        
        print("\nRanked Candidates:")
        for idx, row in candidates.iterrows():
            print(f"Prob: {row.get('probability_long', 0):.2f} | Rank Score: {row.get('ml_rank_score', 0):.2f}")
            print(f"Reasoning: {row.get('ml_explanation', 'N/A')}")
        
if __name__ == "__main__":
    test_ml_multi_target()
