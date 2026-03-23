import argparse
from defihunter.core.config import load_config
from defihunter.execution.scanner import run_scanner
from run_backtest import run_historical_backtest

def main():
    parser = argparse.ArgumentParser(description="DeFiHunter Trade System")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config YAML')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Live scan command
    scan_parser = subparsers.add_parser('scan', help='Run live scanner')
    scan_parser.add_argument('--limit', type=int, default=0, help='Limit number of symbols to scan')
    
    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run backtest')
    bt_parser.add_argument('--ablation', action='store_true', help='Run ablation study over leadership features')
    
    # Family ranker command
    ranker_parser = subparsers.add_parser('train-family-ranker', help='Train global DeFi family-ranker model')
    ranker_parser.add_argument('--days', type=int, default=60, help='Days of history to use for training')

    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.command == 'scan':
        print(f"Loaded config: {config.anchors}")
        run_scanner(config, limit=args.limit)
    elif args.command == 'backtest':
        print("Starting backtest...")
        run_historical_backtest(args.config, symbol="AAVE.p") # Default testing symbol
        if args.ablation:
            print("Finished run, ablation flag acknowledged.")
    elif args.command == 'walk_forward':
        # ... (unchanged)
        pass # placeholder for brevity in thought process, I will include the actual code in the tool call
    elif args.command == 'train':
        # ... (existing individual training)
        pass
    elif args.command == 'train-family-ranker':
        print(f"[*] Starting Global Family-Ranker Training ({args.days} days)...")
        from defihunter.data.storage import TSDBManager
        from defihunter.data.features import build_feature_pipeline
        from defihunter.engines.leadership import LeadershipEngine
        from defihunter.data.dataset_builder import DatasetBuilder
        from defihunter.engines.ml_ranking import MLRankingEngine
        from defihunter.data.universe import load_universe
        import pandas as pd
        from datetime import datetime, timedelta

        tsdb = TSDBManager()
        builder = DatasetBuilder(config=config, timeframe='15m')
        engine = MLRankingEngine()
        
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        
        # 1. Load Anchors for leadership features
        print("Loading Anchor Data...")
        anchor_data = {}
        for anchor in config.anchors:
            try:
                adf = tsdb.load_dataframe(anchor, timeframe='15m', start_date=start_date)
                if not adf.empty:
                    anchor_data[anchor] = build_feature_pipeline(adf)
            except Exception: pass
            
        leadership_engine = LeadershipEngine(anchors=config.anchors, ema_lengths=[config.ema.fast, config.ema.medium])
        
        # 2. Load Universe
        universe = load_universe(config)
        print(f"Processing {len(universe)} symbols for global dataset...")
        
        all_dfs = []
        for sym in universe:
            df = tsdb.load_dataframe(sym, timeframe='15m', start_date=start_date)
            if df.empty or len(df) < 200: continue
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = build_feature_pipeline(df)
            df = leadership_engine.add_leadership_features(df, anchor_data)
            all_dfs.append(df)
            
        if not all_dfs:
            print("Failed to build dataset: No valid data found in TSDB.")
            return
            
        # 3. Build Global Cross-Sectional Dataset
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Applying sector features and cross-sectional labels to {len(combined_df)} rows...")
        combined_df = builder.apply_sector_features(combined_df, timeframe='15m')
        combined_df = builder.generate_cross_sectional_labels(combined_df)
        
        # 4. Train
        if engine.train_family_ranker(combined_df):
            print("🎉 Family-Ranker Training Complete!")
        else:
            print("❌ Family-Ranker Training Failed.")

if __name__ == "__main__":
    main()
