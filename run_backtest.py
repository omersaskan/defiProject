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

def run_historical_backtest(config_path="configs/default.yaml", symbol="LDO.p"):
    """
    End-to-end backtest pipeline for a single historical symbol.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting historical backtest for {symbol}...")
    config = load_config(config_path)
    fetcher = BinanceFuturesFetcher()
    
    # Needs much more history for a backtest (ideally 1000s of rows). Binance API max is usually 1500 per call, 
    # but ccxt handles pagination if configured. We use 1000 for a quick test.
    print(f"Fetching historical data for {symbol}...")
    df = fetcher.fetch_ohlcv(symbol, timeframe='15m', limit=1000)
    
    if df.empty:
        print("No data found.")
        return
        
    print("Fetching context for anchors...")
    anchor_data = {}
    for anchor in config.anchors:
        adf = fetcher.fetch_ohlcv(anchor, timeframe='15m', limit=1000)
        if not adf.empty:
            adf = build_feature_pipeline(adf, timeframe='15m')
            anchor_data[anchor] = adf
            
    # Initialize Engines
    regime_engine = MarketRegimeEngine()
    family_engine = FamilyEngine(config)
    leadership_engine = LeadershipEngine(anchors=config.anchors, ema_lengths=[config.ema.fast, config.ema.medium])
    threshold_engine = ThresholdResolutionEngine(thresholds_config=config.regimes)
    rule_engine = RuleEngine()
    discovery_engine = DiscoveryEngine()
    entry_trigger_engine = EntryEngine()
    
    # Process target coin
    print("Building features...")
    timeframe = '15m'
    df = build_feature_pipeline(df, timeframe=timeframe)
    
    # Profile coin
    profile = family_engine.profile_coin(symbol, historical_data=df)
    
    # Get Leadership
    df = leadership_engine.add_leadership_features(df, anchor_data, timeframe=timeframe)
    
    # In a true vectorized backtest, Regime is computed per row. 
    # For MVP speed, we assume static/latest context or compute a rolling regime label.
    # Here we mock a generic trend for the whole test.
    regime_label = "trend" 
    
    print("Evaluating rules...")
    resolved_thresholds = threshold_engine.resolve_thresholds(regime=regime_label, family=profile.family_label)
    df = rule_engine.evaluate(df, regime=regime_label, family=profile.family_label, resolved_thresholds=resolved_thresholds)
    
    # NEW: Phase 3 Discovery & Entry Flow
    print("Computing Discovery & Entry Trigger scores (Phase 3 logic)...")
    # For backtest, we compute scores row-by-row or vectorized if possible
    # DiscoveryEngine usually takes a cross-section, but here we run it series-wise
    df = discovery_engine.compute_discovery_scores(df) # Scoring one coin's history
    
    # Entry Trigger Readines (Vectorized approximation for backtest)
    # In live we use compute_entry_readiness(individual_df), here we can do a rolling apply or similar
    # For now, let's just make sure required columns for compute_entry_readiness exist and use a simplified version
    
    # Actually, EntryTriggerEngine has a compute_entry_readiness method that works on a single dataframe (last row focus).
    # To backtest it, we need to apply it at each index. 
    # For efficiency and consistency, let's simulate the gate:
    df['entry_readiness'] = (
        (df['msb_bull'].astype(float) * 40) +
        (df['taker_surge'].astype(float) * 25) +
        (df['v_delta_score'].clip(upper=0.2).fillna(0) * 100) # Simple linear CVD accel proxy
    ).clip(upper=100)

    # Final Phase 3 Signal Gate
    df['entry_signal'] = (df['discovery_score'] > 60) & (df['entry_readiness'] > 65)
    
    signals = df[df['entry_signal']]
    print(f"Generated {len(signals)} raw entry signals.")
    
    print("Running event simulation...")
    bt_engine = BacktestEngine(config=config)
    results = bt_engine.simulate(df)
    
    print("\n[BACKTEST RESULTS]")
    print(f"Total Trades: {results.get('total_trades')}")
    print(f"Win Rate: {results.get('win_rate')}%")
    print(f"Profit Factor: {results.get('profit_factor')}")
    print(f"Expectancy (R): {results.get('expectancy_r')}")

if __name__ == "__main__":
    run_historical_backtest(symbol="AAVE.p")
