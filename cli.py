import argparse
import sys
from defihunter.core.config import load_config
from defihunter.execution.scanner import run_scanner
from run_backtest import run_historical_backtest


def main():
    parser = argparse.ArgumentParser(description="DeFiHunter Trade System")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config YAML',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # ── Live scan ───────────────────────────────────────────────────────────
    scan_parser = subparsers.add_parser('scan', help='Run live scanner')
    scan_parser.add_argument(
        '--limit', type=int, default=0,
        help='Limit number of symbols to scan (0 = all)',
    )

    # ── Cross-sectional backtest ─────────────────────────────────────────────
    bt_parser = subparsers.add_parser(
        'backtest',
        help='[PRIMARY] Run multi-coin cross-sectional leader backtest',
    )
    bt_parser.add_argument(
        '--limit', type=int, default=1000,
        help='Number of OHLCV bars to fetch per symbol',
    )
    bt_parser.add_argument(
        '--k', type=int, default=3,
        help='Top-K leaders to select per timestamp',
    )
    bt_parser.add_argument(
        '--ablation', action='store_true',
        help='Print ablation notes after run',
    )

    # ── Global family-ranker training ───────────────────────────────────────
    ranker_parser = subparsers.add_parser(
        'train-family-ranker',
        help='[PRIMARY] Train global DeFi family-ranker model (is_top3_family_next_24h)',
    )
    ranker_parser.add_argument(
        '--days', type=int, default=60,
        help='Days of history to use for training',
    )
    ranker_parser.add_argument(
        '--timeframes', type=str, default='15m,1h,4h',
        help='Comma-separated timeframes to train (e.g. 15m,1h,4h)',
    )

    # ── Deprecated commands (kept for back-compat; do nothing) ──────────────
    # walk_forward and train are no longer routed — subparsers are intentionally
    # omitted so the CLI will 400 instead of silently no-op.

    args = parser.parse_args()
    config = load_config(args.config)

    # ── SCAN ────────────────────────────────────────────────────────────────
    if args.command == 'scan':
        print(f"[*] Config loaded: anchors={config.anchors}")
        run_scanner(config, limit=args.limit)

    # ── BACKTEST ─────────────────────────────────────────────────────────────
    elif args.command == 'backtest':
        print("[*] Starting cross-sectional leader backtest...")
        run_historical_backtest(
            config_path=args.config,
            limit=args.limit,
            k=args.k,
        )
        if args.ablation:
            print("[i] Ablation flag acknowledged. Check logs/backtest_*.log for per-feature breakdown.")

    # ── TRAIN-FAMILY-RANKER ──────────────────────────────────────────────────
    elif args.command == 'train-family-ranker':
        print(f"[*] Starting Global Family-Ranker Training (timeframes={args.timeframes})...")
        import subprocess
        try:
            cmd = [sys.executable, "scripts/train_global.py", "--timeframes", args.timeframes]
            subprocess.run(cmd, check=True)
            print("[✅] Global Training Sequence Complete.")
        except Exception as e:
            print(f"[❌] Training failed: {e}")


if __name__ == "__main__":
    main()
