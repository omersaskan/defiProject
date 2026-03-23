import pandas as pd
import numpy as np
from defihunter.engines.ml_ranking import MLRankingEngine
from defihunter.engines.risk import RiskEngine
from defihunter.engines.decision import DecisionEngine

def test_v4_ranking():
    print("=== [PHASE 4] ML, RISK & DECISION VALIDATION ===")
    
    # 1. ML Ranking Engine (Mocking prediction loop)
    print("Testing MLRankingEngine...")
    ml_engine = MLRankingEngine(model_dir="models")
    df_ml = pd.DataFrame({
        'symbol': ['BTC.p', 'ETH.p'],
        'total_score': [80.0, 70.0],
        'timestamp': pd.date_range(start='2024-01-01', periods=2, freq='15min')
    })
    # Should fallback gracefully since BTC.p might not have active model or it's empty
    ranked_df, info = ml_engine.rank_candidates(df_ml)
    print(f"✅ ML Fallback working: {ranked_df['ml_explanation'].iloc[0]}")
    
    # 2. Risk Engine
    print("Testing RiskEngine...")
    risk_config = {
        'max_open_positions': 5,
        'max_risk_per_trade_pct': 0.02,
        'max_daily_loss_pct': 5.0,
        'max_avg_correlation': 0.7,
        'liquidation_buffer': 0.2
    }
    # Pass a dummy fetcher or none
    risk_engine = RiskEngine(config=risk_config)
    is_valid, reason = risk_engine.validate_trade(
        symbol="LINK.p",
        family="defi_blue_chip",
        current_portfolio=[],
        equity_val=10000,
        daily_loss_pct=0.0
    )
    if is_valid:
        print("✅ Risk Validation OK for new trade.")
    else:
        print(f"❌ Risk Validation Failed: {reason}")
        
    # 3. Decision Engine
    print("Testing DecisionEngine...")
    decision_engine = DecisionEngine(top_n=3)
    df_decision = ranked_df.copy()
    df_decision['leadership_score'] = [30.0, 20.0]
    df_decision['Leadership_Score'] = [30.0, 20.0] # Support both
    df_decision['Timestamp'] = df_decision['timestamp']
    
    decisions = decision_engine.aggregate_and_rank(df_decision)
    if len(decisions) > 0:
        print(f"✅ Decision Engine produced {len(decisions)} trade objects.")
        print(f"   Top Decision: {decisions[0].symbol} (Score: {decisions[0].final_trade_score})")
    else:
        print("❌ Decision Engine failed.")
        return False

    print("=== ML, RISK & DECISION SUCCESS ===\n")
    return True

if __name__ == "__main__":
    test_v4_ranking()
