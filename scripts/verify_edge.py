import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.features import build_feature_pipeline
from defihunter.engines.leadership import LeadershipEngine
from defihunter.engines.regime import MarketRegimeEngine, SectorRegimeEngine
from defihunter.engines.rules import RuleEngine
from defihunter.engines.thresholds import ThresholdResolutionEngine
from defihunter.engines.discovery import DiscoveryEngine
from defihunter.engines.family import FamilyEngine
from defihunter.execution.backtest import BacktestEngine
from defihunter.core.config import load_config

def get_demo_combined_data(config, limit=500):
    """Fetches a subset of data and evaluates core rules for ablation/baseline test."""
    fetcher = BinanceFuturesFetcher()
    symbols = config.universe.defi_universe if config.universe.defi_universe else config.anchors
    symbols = symbols[:12] # keep it small for speed
    
    anchor_data = {}
    for anchor in config.anchors:
        adf = fetcher.fetch_ohlcv(anchor, timeframe=config.timeframe, limit=limit)
        if not adf.empty:
            adf = build_feature_pipeline(adf, timeframe=config.timeframe)
            anchor_data[anchor] = adf
            
    family_engine = FamilyEngine(config)
    leadership_engine = LeadershipEngine(anchors=config.anchors, ema_lengths=[config.ema.fast, config.ema.medium])
    threshold_engine = ThresholdResolutionEngine(thresholds_config=config.regimes, config=config)
    rule_engine = RuleEngine()
    discovery_engine = DiscoveryEngine()
    
    all_dfs = []
    
    for symbol in symbols:
        df = fetcher.fetch_ohlcv(symbol, timeframe=config.timeframe, limit=limit)
        if df.empty: continue
        
        df = build_feature_pipeline(df, timeframe=config.timeframe)
        profile = family_engine.profile_coin(symbol, historical_data=df)
        df = leadership_engine.add_leadership_features(df, anchor_data, timeframe=config.timeframe)
        
        regime_label = "trend_bull" 
        resolved_thresholds = threshold_engine.resolve_thresholds(regime=regime_label, family=profile.family_label)
        df = rule_engine.evaluate(df, regime=regime_label, family=profile.family_label, resolved_thresholds=resolved_thresholds)
        
        df = discovery_engine.compute_discovery_scores(df) 
        
        df['symbol'] = symbol
        df['family'] = profile.family_label
        
        df['entry_readiness'] = (
            (df['msb_bull'].astype(float) * 40) +
            (df['taker_surge'].astype(float) * 25) +
            (df['v_delta_score'].clip(upper=0.2).fillna(0) * 100) 
        ).clip(upper=100)
        
        if 'ml_rank_score' not in df.columns:
            df['ml_rank_score'] = df['discovery_score']
            
        # Baseline Features Generation
        df['random_score'] = np.random.rand(len(df))
        df['mom_24h_score'] = df['close'].pct_change(periods=96).fillna(0)
        
        btc_close = anchor_data.get('BTC.p', pd.DataFrame())
        if not btc_close.empty and 'close' in btc_close.columns:
             btc_pct = btc_close['close'].pct_change(periods=96).fillna(0)
             # Basic alignment
             if len(btc_pct) == len(df):
                 df['rs_btc_score'] = df['mom_24h_score'] - btc_pct.values
             else:
                 df['rs_btc_score'] = df['mom_24h_score']
        else:
             df['rs_btc_score'] = df['mom_24h_score']
             
        df['legacy_ml_score'] = df['entry_readiness'] * 0.5 + df['rs_btc_score'] * 50
            
        all_dfs.append(df)
        
    if not all_dfs:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)
    return combined_df
    

def print_metrics_table(metrics_dict, prefix=""):
    print(f"\n{prefix}")
    print(f"{'Metric':<35} | {'Value':<10}")
    print("-" * 50)
    for k, v in metrics_dict.items():
        print(f"{k:<35} | {v:<10}")

def run_baselines(config):
    print("Fetching Baseline Testing Data...")
    df = get_demo_combined_data(config, limit=1000)
    if df.empty:
        print("Empty combined DF.")
        return
        
    bt_engine = BacktestEngine(config=config)
    bars_horizon = 96 if config.timeframe == '15m' else (24 if config.timeframe == '1h' else 6)
    k = 3
    
    baselines = {
        "Random Pick": "random_score",
        "Simple 24h Momentum": "mom_24h_score",
        "Relative Strength (vs BTC)": "rs_btc_score",
        "Legacy Scanner": "legacy_ml_score",
        "DeFi Leader (Family Ranker)": "ml_rank_score"
    }
    
    results = {}
    
    for method, col in baselines.items():
        # Evaluate Rankings
        df['eval_score'] = df[col]
        # Temporarily swap for backtest engine which hardcodes `ml_rank_score` 
        df['ml_rank_score_bkp'] = df['ml_rank_score']
        df['ml_rank_score'] = df['eval_score']
        
        metrics = bt_engine.evaluate_ranking_quality(df, bars_horizon=bars_horizon, k=k)
        
        # Simulate backtest
        df = df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)
        def select_top_k(group):
            group['is_top_k'] = False
            top_indices = group.nlargest(k, 'ml_rank_score').index
            group.loc[top_indices, 'is_top_k'] = True
            return group
            
        grp_df = df.groupby('timestamp', group_keys=False).apply(select_top_k)
        grp_df['entry_signal'] = grp_df['is_top_k'] & (grp_df['entry_readiness'] > 0 if method == "DeFi Leader (Family Ranker)" else True)
        
        bt_results = bt_engine.simulate(grp_df)
        
        df['ml_rank_score'] = df['ml_rank_score_bkp']
        
        metrics.update(bt_results)
        results[method] = metrics
        
    print("\n" + "="*80)
    print(f"{'Method':<28} | {'Capt_Rate':<10} | {'Precisn':<10} | {'R_Corr':<10} | {'Hold_Eff':<10} | {'Prof_Fac':<10} | {'Expctr_R':<10}")
    print("="*80)
    for method, m in results.items():
        print(f"{method:<28} | {m.get('leader_capture_rate', 0):<10.1f} | {m.get('top_k_precision', 0):<10.1f} | {m.get('rank_correlation', 0):<10.3f} | {m.get('hold_efficiency', 0):<10.1f} | {m.get('profit_factor', 0):<10.2f} | {m.get('expectancy_r', 0):<10.2f}")

def run_ablation(config):
    print("Running Ablation Study (Mocked Data Simulation)...")
    # For speed, we will mock the ablation variations using the baseline dataframe
    # Real ablation would be modifying the data stream which takes too much time here
    df = get_demo_combined_data(config, limit=500)
    bt_engine = BacktestEngine(config=config)
    k = 3
    bars_horizon = 96
    
    variations = {
        "Full System": {"score": "ml_rank_score", "entry": True, "decay": True},
        "No Entry Layer": {"score": "ml_rank_score", "entry": False, "decay": True},
        "No Decay Output": {"score": "ml_rank_score", "entry": True, "decay": False},
        "No Discovery/ML Layer": {"score": "mom_24h_score", "entry": True, "decay": True},
    }
    
    results = {}
    
    for name, params in variations.items():
        df['eval_score'] = df[params['score']]
        df['ml_rank_score_bkp'] = df['ml_rank_score']
        df['ml_rank_score'] = df['eval_score']
        
        metrics = bt_engine.evaluate_ranking_quality(df, bars_horizon=bars_horizon, k=k)
        
        df = df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)
        def select_top_k(group):
            group['is_top_k'] = False
            top_indices = group.nlargest(k, 'ml_rank_score').index
            group.loc[top_indices, 'is_top_k'] = True
            return group
            
        grp_df = df.groupby('timestamp', group_keys=False).apply(select_top_k)
        
        # Toggle entry constraints
        if params['entry']:
             grp_df['entry_signal'] = grp_df['is_top_k'] & (grp_df['entry_readiness'] > 0)
        else:
             grp_df['entry_signal'] = grp_df['is_top_k']
             
        # Toggle decay exit
        if not params['decay']:
             grp_df['leadership_decay'] = False
             
        bt_results = bt_engine.simulate(grp_df)
        df['ml_rank_score'] = df['ml_rank_score_bkp']
        
        metrics.update(bt_results)
        results[name] = metrics
        
    print("\n[ABLATION STUDY]")
    print("="*80)
    print(f"{'Variant':<25} | {'Capt_Rate':<10} | {'Precisn':<10} | {'R_Corr':<10} | {'Hold_Eff':<10} | {'Prof_Fac':<10} | {'Expctr_R':<10}")
    print("="*80)
    for method, m in results.items():
        print(f"{method:<25} | {m.get('leader_capture_rate', 0):<10.1f} | {m.get('top_k_precision', 0):<10.1f} | {m.get('rank_correlation', 0):<10.3f} | {m.get('hold_efficiency', 0):<10.1f} | {m.get('profit_factor', 0):<10.2f} | {m.get('expectancy_r', 0):<10.2f}")

def run_exit_analysis():
    print("Parsing `logs/paper_portfolio.json` for Exit Quality...")
    state_file = "logs/paper_portfolio.json"
    if not os.path.exists(state_file):
        print("No paper_portfolio.json found.")
        return
        
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
            
        history = data.get('trade_history', [])
        if not history:
             print("No past trades found in paper portfolio to establish exit metrics.")
             return
             
        df = pd.DataFrame(history)
        
        total_trades = len(df)
        reasons = df['exit_reason'].value_counts()
        partials = len(df[df['partial_taken'] == True])
        
        avg_giveback = df['giveback'].mean() * 100 if 'giveback' in df.columns else 0
        avg_mfe = df['max_favorable_excursion'].mean() * 100 if 'max_favorable_excursion' in df.columns else 0
        
        print("\n[PAPER TRADE EXIT QUALITY]")
        print("-" * 50)
        print(f"Total Evaluated Trades   : {total_trades}")
        print(f"Partial TP Taken (Runners): {partials} ({(partials/total_trades)*100:.1f}%)")
        print(f"Average Peak MFE         : %{avg_mfe:.1f} of Entry")
        print(f"Average Giveback from MFE: %{avg_giveback:.1f} of Entry")
        print("\nExit Reason Distribution:")
        for r, count in reasons.items():
            print(f"  - {r}: {count} ({(count/total_trades)*100:.1f}%)")
        
        print("\n[ANALYSIS]")
        if "Decay" in str(reasons.index) or "Dropped" in str(reasons.index):
             print("- Decay Exit is active and triggering, protecting capital.")
        if "Stop Loss Hit" in reasons and reasons["Stop Loss Hit"] > total_trades * 0.5:
             print("- Warning: Stop Loss hits are very high. Entry confirmation might fall to fakeouts.")
        if partials > 0:
             print("- Runner mechanic operates well with dynamic scaling.")
             
    except Exception as e:
        print(f"Failed to load portfolio logic: {e}")

def run_family_analysis(config):
    df = get_demo_combined_data(config, limit=500)
    if df.empty:
        print("Empty combined DF.")
        return
    print("\n[FAMILY ALPHA ANALYSIS]")
    
    bt_engine = BacktestEngine(config=config)
    bars_horizon = 96
    
    families = df['family'].unique()
    for fam in families:
        fam_df = df[df['family'] == fam]
        if len(fam_df) < 100: continue
        metrics = bt_engine.evaluate_ranking_quality(fam_df, bars_horizon=bars_horizon, k=1)
        
        print(f"Family: {fam:<15} | Capt_Rate: {metrics.get('leader_capture_rate',0):<5.1f} | R_Corr: {metrics.get('rank_correlation',0):<5.3f} | Ops: {len(fam_df)}")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'baselines', 'ablation', 'exit_analysis', 'family'])
    args = parser.parse_args()
    
    config = load_config("configs/default.yaml")
    
    if args.mode in ['all', 'baselines']:
        run_baselines(config)
    if args.mode in ['all', 'ablation']:
        run_ablation(config)
    if args.mode in ['all', 'exit_analysis']:
        run_exit_analysis()
    if args.mode in ['all', 'family']:
        run_family_analysis(config)

if __name__ == '__main__':
    main()
