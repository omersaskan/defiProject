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
    Remove one of each pair of highly correlated numeric features.
    Reduces noise and speeds up training.
    """
    meta_cols = {'timestamp', 'symbol', 'regime', 'target_hit', 'short_target_hit',
                 'mfe_r', 'mae_r', 'exit_type', 'prepump_target', 'bars_to_target',
                 'target_4h_2pct', 'target_12h_5pct', 'target_24h_10pct', 'primary_clf',
                 'is_top3_family_next_24h', 'family'}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in meta_cols]
    
    if len(feature_cols) < 2: return df
    
    feature_df = df[feature_cols]
    corr_matrix = feature_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
    
    if to_drop:
        print(f"  Dropping {len(to_drop)} correlated features.")
    
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
            
        # 3. Build features & labels
        builder = DatasetBuilder(config=config, timeframe=tf)
        # We need future returns for cross-sectional ranking building later
        return df
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
        print(f"\nSTARTING GLOBAL TRAINING FOR TIMEFRAME: {tf}")
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
            common_idx = btc_df.index.intersection(eth_df.index)
            raw_regimes = regime_engine.detect_historical_regimes(btc_df.loc[common_idx], eth_df.loc[common_idx])
            regime_map = raw_regimes.apply(lambda r: "TREND" if "trend" in r or "down" in r else "CHOP")
            
        print(f"Found {len(files)} coins. Processing features...")
        all_dfs = []
        process_func = partial(process_single_coin, config=config, regime_map=regime_map, tf=tf)
        
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-1)) as executor:
            for df in executor.map(process_func, files):
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    
        if not all_dfs:
            print(f"Failed to process data for {tf}.")
            continue
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sector Features & Cross-Sectional Ranking
        print(f"Applying sector features and family ranking for {tf}...")
        builder = DatasetBuilder(config=config, timeframe=tf)
        combined_df = builder.apply_sector_features(combined_df, timeframe=tf)
        combined_df = builder.generate_cross_sectional_labels(combined_df)
        
        # Filter for strictly DeFi if config says so
        if config.universe.is_strictly_defi:
            # Assume DatasetBuilder or FamilyEngine maps 'family' column
            combined_df = combined_df[combined_df['family'].notna()]
        
        combined_df = remove_correlated_features(combined_df, threshold=0.90)
        combined_df = combined_df.dropna(subset=['atr', 'regime', 'is_top3_family_next_24h'])
        
        print(f"Data Build Complete: {len(combined_df)} master rows.")
        
        # Train ML using strictly leader discovery target
        target_clf = 'is_top3_family_next_24h'
        print(f"Objective established: {target_clf}")
        
        ml_engine = MLRankingEngine(model_dir=f"models_{tf}") 
        ml_engine.train_global(
            combined_df=combined_df, 
            target_clf_col=target_clf, 
            target_short_col='short_target_hit', 
            target_reg_col='future_24h_rank_in_family_pct' if 'future_24h_rank_in_family_pct' in combined_df.columns else 'mfe_r'
        )
        print(f"[{tf}] Primary objective log complete: {target_clf}")
        print(f"Training fully completed for {tf} models.")

if __name__ == '__main__':
    main()
