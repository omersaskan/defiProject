import pandas as pd
import numpy as np
from datetime import datetime
from defihunter.engines.decision import DecisionEngine
from defihunter.core.models import FinalDecision

def test_ml_data_propagation():
    print("Starting ML Data Propagation Test...")
    
    # Simulate scanner output
    candidates = pd.DataFrame([{
        'symbol': 'BTC.p',
        'Timestamp': datetime.now(),
        'total_score': 85.0,
        'leadership_score': 30.0,
        'ml_rank_score': 90.0,
        'probability_long': 0.85,
        'expected_return_r': 2.5,
        'ml_explanation': '[TEST] High confidence gainer',
        'Setup': 'strong_breakout',
        'Family': 'defi_beta',
        'entry_price': 60000.0,
        'tp1_price': 63000.0,
        'stop_price': 59000.0
    }])
    
    engine = DecisionEngine(top_n=1)
    decisions = engine.aggregate_and_rank(candidates)
    
    assert len(decisions) == 1, "Should have 1 decision"
    d = decisions[0]
    
    print(f"Decision for {d.symbol}: {d.decision} (Score: {d.final_trade_score})")
    print(f"ML Metrics - Prob: {d.explanation.get('probability_of_success')}, Exp_R: {d.explanation.get('expected_return_r_r', d.explanation.get('expected_return_r'))}")
    
    # Check if ML data reached the explanation
    assert d.explanation.get('ml_score') == 90.0
    assert d.explanation.get('probability_of_success') == 0.85
    assert d.explanation.get('expected_return_r') == 2.5
    
    print("SUCCESS: ML data correctly propagated to final decision.")

if __name__ == "__main__":
    try:
        test_ml_data_propagation()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
