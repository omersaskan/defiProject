import sys
import pandas as pd
import numpy as np

# Apply basic mocked registry dependency to prevent setup_registry loading issues in mock env
# Actually, load the real one
from defihunter.engines.rules import RuleEngine

def run_parity_test():
    print("--- SPRINT 5: RULE EVALUATION PARITY TEST ---")
    
    # 1. Create a representative 200-row DataFrame with mixed features
    dates = pd.date_range(end=pd.Timestamp('2024-05-01'), periods=200, freq='15min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'MOCK.p', # ADDED SYMBOL TRACE
        'open': np.linspace(10, 20, 200) + np.random.normal(0, 0.5, 200),
        'high': np.linspace(10.5, 21, 200),
        'low': np.linspace(9.5, 19, 200),
        'close': np.linspace(10.2, 20.5, 200) + np.random.normal(0, 0.5, 200),
        'volume': np.random.randint(1000, 100000, 200),
        'quote_volume': np.random.randint(10000000, 50000000, 200),
        
        # Trend features
        'ema_20': np.linspace(10.1, 19.5, 200),
        'ema_55': np.linspace(10.0, 18.5, 200),
        'ema_100': np.linspace(9.8, 17.0, 200),
        'atr': np.random.uniform(0.1, 0.5, 200),
        'bar_count': np.linspace(1, 200, 200),
        
        # Expansion / Participation
        'is_breakout_bar': np.random.choice([0, 1], 200, p=[0.9, 0.1]),
        'sweep_reclaim_confirmed': np.random.choice([0, 1], 200, p=[0.9, 0.1]),
        'volume_zscore': np.random.uniform(-1, 3, 200),
        'rs_divergence_persistence': np.random.randint(0, 5, 200),
        'v_delta_score': np.random.uniform(-0.1, 0.2, 200),
        
        # Rel Spread Features
        'rel_spread_btc_p_ema55': np.random.uniform(-0.01, 0.02, 200),
        'rel_spread_btc_p_ema55_slope_4': np.random.uniform(-0.05, 0.05, 200),
        'rel_spread_btc_p_ema55_persistence': np.random.randint(0, 6, 200),
        
        # Setup specific conditions
        'orderbook_vacuum': np.zeros(200),
        
        # Funding
        'funding_rate': np.random.uniform(-0.003, 0.003, 200),
    })

    # Ensure the last row forces some triggers
    idx_last = df.index[-1]
    df.loc[idx_last, 'is_breakout_bar'] = 1
    df.loc[idx_last, 'volume_zscore'] = 2.5
    df.loc[idx_last, 'rel_spread_btc_p_ema55'] = 0.01
    df.loc[idx_last, 'funding_rate'] = -0.001
    df.loc[idx_last, 'orderbook_vacuum'] = 1

    # Add mock method
    class MockConfig:
        class ScoringConf:
            trend_weight = 0.5
            expansion_weight = 0.5
            participation_weight = 0.5
            relative_leadership_weight = 2.0
            funding_penalty_weight = 1.0
        scoring = ScoringConf()
        
    engine = RuleEngine(config=MockConfig())
    
    # Context
    regime = "trend_bull"
    family = "defi_alpha"
    sector_data = {"sector_scores": {"defi_alpha": 1.15}}
    resolved_thresholds = {"min_score": 40, "min_relative_leadership": 5, "min_volume": 100}
    adaptive_weights = {"trend_score": 1.0, "expansion_score": 1.0, "participation_score": 1.0, "relative_leadership_score": 1.0}
    primary_anchor = "BTC.p"

    # --- PATH 1: OLD FULL DATAFRAME PATH ---
    df_full_path = df.copy()
    res_full = engine.evaluate(
        df_full_path, regime=regime, family=family,
        resolved_thresholds=resolved_thresholds, sector_data=sector_data,
        adaptive_weights=adaptive_weights, primary_anchor=primary_anchor
    )
    last_full = dict(res_full.iloc[-1])

    # --- PATH 2: NEW OPTIMIZED LAST ROW PATH ---
    df_new_path = df.copy()
    last_row_df = df_new_path.iloc[[-1]].copy()
    res_subset = engine.evaluate(
        last_row_df, regime=regime, family=family,
        resolved_thresholds=resolved_thresholds, sector_data=sector_data,
        adaptive_weights=adaptive_weights, primary_anchor=primary_anchor
    )
    last_subset = dict(res_subset.iloc[-1])

    # --- COMPARISON ---
    keys_to_compare = [
        'trend_score', 'expansion_score', 'participation_score', 'relative_leadership_score',
        'funding_penalty', 'total_score', 'entry_signal', 'veto_reason', 'setup_class'
    ]

    print("\n[Parity Check Results]")
    print(f"{'Field':<30} | {'Full Path':<20} | {'Optimized Path':<20} | Matches?")
    print("-" * 85)
    
    all_match = True
    for k in keys_to_compare:
        v_full = last_full.get(k)
        v_sub = last_subset.get(k)
        
        # Handle nan comparisons safely
        if pd.isna(v_full) and pd.isna(v_sub):
            match = True
        elif isinstance(v_full, float) and isinstance(v_sub, float):
            match = abs(v_full - v_sub) < 1e-6
        else:
            match = (v_full == v_sub)

        if not match: all_match = False
        print(f"{k:<30} | {str(v_full):<20} | {str(v_sub):<20} | {'True' if match else 'FALSE!'}")

    print("\n[Explanation Parity Check]")
    exp_f = last_full.get('explanation', {})
    exp_s = last_subset.get('explanation', {})
    print(f"Full Path Doms: {exp_f.get('dominant_factors', [])}")
    print(f"Opt Route Doms: {exp_s.get('dominant_factors', [])}")

    if all_match and exp_f == exp_s:
        print("\n=> SEMANTIC PRESERVATION VERIFIED: The optimized last-row execution path matches the full historical pipeline array mathematically.")
    else:
        print("\n=> FAILED: Parity drift detected across execution paths.")

if __name__ == '__main__':
    run_parity_test()
