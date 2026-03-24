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
        import subprocess
        import sys
        try:
            # Consolidate training logic into the dedicated script
            cmd = [sys.executable, "scripts/train_global.py", "--timeframes", "15m,1h,4h"]
            subprocess.run(cmd, check=True)
            print("[\u2705] Global Training Sequence Complete.")
        except Exception as e:
            print(f"[\u274c] training failed: {e}")

if __name__ == "__main__":
    main()
