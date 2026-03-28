import pandas as pd
from defihunter.data.binance_fetcher import BinanceFuturesFetcher
from defihunter.data.features import build_feature_pipeline
from defihunter.engines.leadership import LeadershipEngine
from defihunter.engines.regime import MarketRegimeEngine, SectorRegimeEngine
from defihunter.engines.rules import RuleEngine
from defihunter.engines.thresholds import ThresholdResolutionEngine
from defihunter.engines.discovery import DiscoveryEngine
from defihunter.engines.entry import EntryEngine
from defihunter.engines.family import FamilyEngine
from defihunter.execution.backtest import BacktestEngine
from defihunter.core.config import load_config
from datetime import datetime

def run_historical_backtest(config_path="configs/default.yaml", limit=1000, k=3):
    """
    End-to-end backtest pipeline for true multi-coin cross-sectional leader testing.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting cross-sectional leader backtest...")
    config = load_config(config_path)
    fetcher = BinanceFuturesFetcher()
    
    symbols = config.universe.defi_universe if config.universe.defi_universe else config.anchors
    if not symbols:
        print("No universe found in config!")
        return

    # To keep it fast for testing, let's take a subset
    symbols = symbols[:12]
        
    print(f"Fetching context for anchors... (Timeframe: {config.timeframe})")
    anchor_data = {}
    for anchor in config.anchors:
        adf = fetcher.fetch_ohlcv(anchor, timeframe=config.timeframe, limit=limit)
        if not adf.empty:
            adf = build_feature_pipeline(adf, timeframe=config.timeframe)
            anchor_data[anchor] = adf
            
    regime_engine = MarketRegimeEngine()
    family_engine = FamilyEngine(config)
    leadership_engine = LeadershipEngine(anchors=config.anchors, ema_lengths=[config.ema.fast, config.ema.medium])
    threshold_engine = ThresholdResolutionEngine(thresholds_config=config.regimes, config=config)
    rule_engine = RuleEngine()
    discovery_engine = DiscoveryEngine()
    
    all_dfs = []
    
    for symbol in symbols:
        print(f"Fetching and processing {symbol}...")
        df = fetcher.fetch_ohlcv(symbol, timeframe=config.timeframe, limit=limit)
        if df.empty: continue
        
        df = build_feature_pipeline(df, timeframe=config.timeframe)
        profile = family_engine.profile_coin(symbol, historical_data=df)
        df = leadership_engine.add_leadership_features(df, anchor_data, timeframe=config.timeframe)
        
        regime_label = "trend" 
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
            
        all_dfs.append(df)
        
    if not all_dfs:
        print("No valid data processed.")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)
    
    print(f"Evaluating ranking quality for {len(symbols)} coins...")
    bt_engine = BacktestEngine(config=config)
    
    bars_horizon = 96 if config.timeframe == '15m' else (24 if config.timeframe == '1h' else 6)

    # Fix #4: simulate() önce çalıştırılmalı ki trade_log dolsun;
    # evaluate_ranking_quality sonra çağrılırsa hold_efficiency / exit_efficiency sağlıklı döner.
    print(f"\nRunning Top-{k} cross-sectional event simulation first...")
    combined_df = combined_df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)

    def select_top_k(group):
        group['is_top_k'] = False
        top_indices = group.nlargest(k, 'ml_rank_score').index
        group.loc[top_indices, 'is_top_k'] = True
        return group

    combined_df = combined_df.groupby('timestamp', group_keys=False).apply(select_top_k)
    combined_df['entry_signal'] = combined_df['is_top_k'] & (combined_df['entry_readiness'] > 0)

    signals = combined_df[combined_df['entry_signal']]
    print(f"Generated {len(signals)} true cross-sectional entry signals.")

    results = bt_engine.simulate(combined_df)
    print("\n[BACKTEST RESULTS]")
    print(f"Total Trades: {results.get('total_trades')}")
    print(f"Win Rate: {results.get('win_rate')}%")
    print(f"Profit Factor: {results.get('profit_factor')}")
    print(f"Expectancy (R): {results.get('expectancy_r')}")

    # Now evaluate ranking quality — trade_log already populated from simulate()
    metrics = bt_engine.evaluate_ranking_quality(combined_df, bars_horizon=bars_horizon, k=k)
    print("\n[LEADER CAPTURE METRICS]")
    for k_metric, v_metric in metrics.items():
        print(f"{k_metric}: {v_metric}")

if __name__ == "__main__":
    run_historical_backtest(k=3, limit=1000)
