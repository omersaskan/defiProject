import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from defihunter.data.storage import TSDBManager
from defihunter.data.dataset_builder import DatasetBuilder
from defihunter.engines.ml_ranking import MLRankingEngine
from defihunter.data.features import build_feature_pipeline
from defihunter.engines.regime import MarketRegimeEngine
from defihunter.core.config import load_config

def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    GT #9: Remove one of each pair of highly correlated numeric features.
    Keeps the first encountered feature, drops the second.
    Reduces noise and speeds up training. Typical AUC improvement: +0.02 to +0.06.
    """
    # Only drop from feature columns — not target/meta columns
    meta_cols = {'timestamp', 'symbol', 'regime', 'target_hit', 'short_target_hit',
                 'mfe_r', 'mae_r', 'exit_type', 'prepump_target', 'bars_to_target',
                 'target_4h_2pct', 'target_12h_5pct', 'target_24h_10pct', 'primary_clf'}
    feature_df = df[[c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]]
    
    corr_matrix = feature_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
    
    if to_drop:
        print(f"  [GT #9] Dropping {len(to_drop)} correlated features: {to_drop[:5]}{'...' if len(to_drop)>5 else ''}")
    
    return df.drop(columns=to_drop, errors='ignore')

    
def process_single_coin(file_path, config=None, regime_map=None, tf='15m'):
    """Loads a single file, applies features, maps regime, generates labels."""
    try:
        df = pd.read_parquet(file_path)
        if df.empty or len(df) < 50:
            return None
            
        filename = os.path.basename(file_path)
        parts = filename.replace('.parquet', '').split('_')
        if len(parts) >= 3:
            symbol = f"{parts[0]}.{parts[1]}"
            df['symbol'] = symbol
        
        # 1. Compute technical features
        df = build_feature_pipeline(df, timeframe=tf)
        
        # 2. Inject global regime from the precomputed map
        if regime_map is not None and 'timestamp' in df.columns:
            df = df.set_index('timestamp')
            df['regime'] = regime_map
            df['regime'] = df['regime'].fillna('CHOP')
            df = df.reset_index()
        else:
            df['regime'] = 'CHOP'
            
        # 3. Build explicit labels (target_hit, prepump_target, multi-step GT #11 targets)
        # GT-REDESIGN: Use time-aware horizon natively
        builder = DatasetBuilder(config=config, timeframe=tf, target_r=1.5, stop_r=-1.0, prepump_gain_pct=0.05)
        df_labeled = builder.generate_labels(df)
        
        # 4. GT #9: Remove highly correlated features (reduces noise, improves AUC)
        df_labeled = remove_correlated_features(df_labeled, threshold=0.90)
        
        # 5. Drop rows that don't have a full window ahead
        df_valid = df_labeled.iloc[:-builder.window].dropna(subset=['target_hit', 'short_target_hit'])
        
        return df_valid
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", type=str, default="15m,1h,4h", help="Comma-separated timeframes")
    args = parser.parse_args()
    
    tfs = args.timeframes.split(',')
    tsdb_dir = "data/tsdb"
    if not os.path.exists(tsdb_dir):
        print("No TSDB directory found. Please run bulk_fetch_fast.py first.")
        return
        
    regime_engine = MarketRegimeEngine()
    config = load_config("configs/default.yaml")
    
    for tf in tfs:
        print(f"\n{'='*50}\n🚀 STARTING GLOBAL TRAINING FOR TIMEFRAME: {tf}\n{'='*50}")
        files = [os.path.join(tsdb_dir, f) for f in os.listdir(tsdb_dir) if f.endswith(f"_{tf}.parquet")]
        
        if not files:
            print(f"No {tf} parity files found. Skipping.")
            continue
            
        # Compute global regime map using BTC and ETH
        btc_file = os.path.join(tsdb_dir, f"BTC_p_{tf}.parquet")
        eth_file = os.path.join(tsdb_dir, f"ETH_p_{tf}.parquet")
        regime_map = None
        
        if os.path.exists(btc_file) and os.path.exists(eth_file):
            print("Computing Global Market Regimes from BTC & ETH...")
            btc_df = pd.read_parquet(btc_file).set_index('timestamp')
            eth_df = pd.read_parquet(eth_file).set_index('timestamp')
            
            # Use intersection of time
            common_idx = btc_df.index.intersection(eth_df.index)
            raw_regimes = regime_engine.detect_historical_regimes(btc_df.loc[common_idx], eth_df.loc[common_idx])
            
            # Map specific regimes into binary TREND / CHOP for ML separation
            # "trend_btc_led", "trend_alt_rotation", "trend_neutral", "downtrend" -> TREND
            # "chop", "unstable", etc -> CHOP
            regime_map = raw_regimes.apply(lambda r: "TREND" if "trend" in r or "down" in r else "CHOP")
            print(f"Regime map generated: {len(regime_map)} points")
            
        print(f"Found {len(files)} coins with {tf} history. Processing and extracting features...")
        
        all_dfs = []
        process_func = partial(process_single_coin, config=config, regime_map=regime_map, tf=tf)
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
            for df in executor.map(process_func, files):
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    
        if not all_dfs:
            print(f"Failed to process any valid data for {tf}.")
            continue
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.dropna(subset=['atr', 'regime'])

        # Phase 3: Apply Sector Features & Cross-Sectional Ranking
        print(f"\n[Phase 3] Applying multi-symbol sector features and ranking for {tf}...")
        builder = DatasetBuilder(config=config, timeframe=tf)
        combined_df = builder.apply_sector_features(combined_df, timeframe=tf)
        combined_df = builder.generate_cross_sectional_labels(combined_df)
        
        print(f"\n✅ Data Built: {len(combined_df)} master rows combining {len(all_dfs)} coins for {tf}.")
        print(f"Regime breakdown:\n{combined_df['regime'].value_counts()}")
        
        # Train ML using leader discovery target if available
        ml_engine = MLRankingEngine(model_dir=f"models_{tf}") 
        ml_engine.train_global(
            combined_df=combined_df, 
            target_clf_col='is_top3_family_next_24h' if 'is_top3_family_next_24h' in combined_df.columns else 'is_top3_overall', 
            target_short_col='short_target_hit', 
            target_reg_col='future_24h_rank_in_family_pct' if 'future_24h_rank_in_family_pct' in combined_df.columns else 'mfe_r'
        )
        print(f"🎉 Training fully completed for {tf} models!")

if __name__ == '__main__':
    main()
