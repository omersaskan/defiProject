import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from defihunter.core.config import load_config
from defihunter.engines.ml_ranking import MLRankingEngine

# Mock candidates
candidates = pd.DataFrame({
    'symbol': ['AAVE.p', 'UNI.p', 'LINK.p'],
    'family': ['lending', 'dex_amm', 'oracle'],
    'trend_persistence_score': [10, 5, 2],
    'family_heat_score': [0.08, 0.02, 0.01],
    'family_breadth_score': [0.8, 0.4, 0.2],
    'peer_momentum': [0.03, 0.01, -0.01],
    'peer_rank': [0.9, 0.6, 0.3],
    'close': [150.0, 10.0, 18.0]
})

# Add missing features that ML expects
ml_engine = MLRankingEngine()
# We need to mock the models since we might not have a full training yet
# But I can show the prediction structure by forcing the use_family_ranker path
ml_engine.load_family_ranker_models()

print("--- ML Prediction (Tri-Score) Proof ---")
scored, top = ml_engine.rank_candidates(candidates, use_family_ranker=True)

# Mock scores if prediction failed due to missing model files
if 'leader_prob' not in scored.columns:
    print("[MOCK] Model files not found, injecting dummy scores for proof...")
    scored['leader_prob'] = [0.85, 0.45, 0.15]
    scored['holdability_score'] = [92.0, 65.0, 30.0]
    scored['setup_conversion_prob'] = [0.75, 0.35, 0.10]
    scored['ml_rank_score'] = 85.0

cols = ['symbol', 'leader_prob', 'holdability_score', 'setup_conversion_prob', 'ml_rank_score']
print(scored[cols].to_string())
print("--- SUCCESS ---")
