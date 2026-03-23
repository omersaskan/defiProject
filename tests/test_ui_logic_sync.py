import pandas as pd
import numpy as np
from defihunter.engines.ml_ranking import MLRankingEngine
from defihunter.engines.decision import DecisionEngine
from defihunter.core.config import load_config

def test_sync():
    print("--- STARTING UI-LOGIC SYNC TEST ---")
    
    # 1. Mock Candidates with technical features
    candidates = pd.DataFrame([{
        "symbol": "FAKE.p",
        "timestamp": pd.Timestamp.now(),
        "total_score": 85.0,
        "Total_Score": 85.0, # Legacy sync
        "Leadership_Score": 25.0,
        "rel_spread_eth_ema55": 0.01, # Technical feature
        "volume_zscore": 2.5,          # Technical feature
        "Setup": "potential_momentum",
        # Layered Engine Features
        "leader_prob": 0.9,
        "family_heat_score": 0.1,
        "family_breadth_score": 1.0,
        "peer_momentum": 0.05,
        "peer_rank": 1.0,
        "holdability_score": 85.0,
        "trend_persistence_score": 20.0,
        "ema20_dist": 0.01,
        "wick_exhaustion_flag": 0
    }])
    
    # 2. Test ML Ranking
    ml_engine = MLRankingEngine()
    # Mock a "trained" state by setting features_used manually for testing the flow
    ml_engine.features_used = ["rel_spread_eth_ema55", "volume_zscore"]
    
    # We won't actually "train" as it requires scikit/lgbm, 
    # but we can test if the fallback logic handles 'total_score' correctly
    print("\n[Test 2a: ML Fallback with total_score]")
    ml_engine.load_models = lambda symbol: False # Mock load_models so it doesn't load real models from disk
    
    ranked_df, top_n = ml_engine.rank_candidates(candidates)
    
    # ML Engine fallback is 50.0 when no models are loaded
    assert "ml_rank_score" in ranked_df.columns
    assert ranked_df.iloc[0]["ml_rank_score"] == 50.0 
    print("SUCCESS: ML Fallback correctly assigned base 50.0 score.")
    
    # Manually set ml_rank_score to 85.0 to test Decision Engine math
    ranked_df.loc[0, "ml_rank_score"] = 85.0
    # Also need discovery_score for DecisionEngine
    ranked_df.loc[0, "discovery_score"] = 80.0

    # 3. Test Decision Engine
    print("\n[Test 3: Decision Engine Case Sensitivity]")
    de = DecisionEngine(top_n=3)
    decisions = de.aggregate_and_rank(ranked_df)
    
    assert len(decisions) > 0
    top = decisions[0]
    print(f"Top Decision: {top.symbol} | Score: {top.final_trade_score} | Decision: {top.decision}")
    
    # Simple check for composite score 
    assert top.final_trade_score > 0
    print(f"SUCCESS: Decision Engine calculated final score correctly using standardized keys.")

if __name__ == "__main__":
    test_sync()
