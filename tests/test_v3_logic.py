import pandas as pd
import numpy as np
from defihunter.engines.rules import RuleEngine
from defihunter.engines.leadership import LeadershipEngine

def test_v3_logic():
    print("=== [PHASE 3] RULE & LEADERSHIP VALIDATION ===")
    
    # 1. Leadership Engine
    print("Testing LeadershipEngine...")
    leadership_engine = LeadershipEngine(anchors=['BTC.p'], ema_lengths=[20])
    
    df = pd.DataFrame({
        'close': [100, 105, 110, 115, 120],
        'open': [99, 101, 106, 111, 116],
        'high': [102, 107, 112, 117, 122],
        'low': [98, 100, 105, 110, 115],
        'ema_20': [90, 92, 94, 96, 98],
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='15min')
    })
    anchor_mtf = {
        'BTC.p': pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
            'high': [101, 102, 103, 104, 105],
            'low': [98, 99, 100, 101, 102],
            'ema_20': [99, 99.5, 100, 100.5, 101],
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='15min')
        })
    }
    
    df_with_lead = leadership_engine.add_leadership_features(df, anchor_mtf)
    spread_col = 'rel_spread_btc_ema20'
    if spread_col in df_with_lead.columns:
        print(f"✅ Leadership features added (BTC Spread: {df_with_lead[spread_col].iloc[-1]:.4f})")
    else:
        print(f"❌ Leadership feature calculation failed. Current columns: {df_with_lead.columns.tolist()}")
        return False

    # 2. Rule Engine
    print("Testing RuleEngine...")
    rule_engine = RuleEngine()
    # Mock data with MSB and Funding Capitulation to trigger GT-PRO bonuses
    df_test = pd.DataFrame({
        'close': [100, 110],
        'ema_20': [90, 95],
        'ema_55': [90, 90],
        'ema_100': [80, 80],
        'volume_zscore': [3.0, 3.0],
        'msb_bull': [False, True],
        'funding_capitulation': [False, True],
        'quote_volume': [20_000_000, 20_000_000],
        'bar_count': [300, 300]
    })
    
    resolved_thresholds = {'min_volume': 10_000_000, 'min_bars': 100, 'min_score': 50}
    df_evaluated = rule_engine.evaluate(df_test, regime='bull_volatile', family='defi_alpha', resolved_thresholds=resolved_thresholds)
    
    last_row = df_evaluated.iloc[-1]
    print(f"✅ Rule Engine Total Score: {last_row['total_score']}")
    print(f"✅ Setup Class: {last_row['setup_class']}")
    
    if last_row['total_score'] < 50:
        print("❌ Rule Engine failed to generate sufficient score for bullish setup.")
        return False

    print("=== RULE & LEADERSHIP SUCCESS ===\n")
    return True

if __name__ == "__main__":
    test_v3_logic()
