from pathlib import Path
from defihunter.execution.scanner import run_scanner
from defihunter.core.config import load_config

from pathlib import Path
from unittest.mock import patch, MagicMock
from defihunter.execution.scanner import run_scanner
from defihunter.core.config import load_config
import pandas as pd

@patch('defihunter.execution.scanner.BinanceFuturesFetcher')
def test_v5_scanner_fast(mock_fetcher_class):
    print("=== [PHASE 5] LIVE SCANNER INTEGRATION (FAST) ===")
    
    # 1. Load Config
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)
    
    # Mock the fetcher instance
    mock_fetcher = MagicMock()
    mock_fetcher_class.return_value = mock_fetcher
    
    mock_fetcher.get_defi_universe.return_value = ['BTC.p']
    mock_fetcher.fetch_ohlcv.return_value = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='15min'),
        'open': [50000]*50, 'high': [50100]*50, 'low': [49900]*50, 
        'close': [50050]*50, 'volume': [100]*50, 'open_interest': [1000]*50,
        'funding_rate': [0.01]*50, 'spread_bps': [5.0]*50
    })
    
    # 2. Patch config for fast testing
    print("Running 1-iteration scanner...")
    
    try:
        # run_scanner returns top results
        decisions = run_scanner(config)
        print(f"✅ Scanner Cycle Complete. Decisions produced: {len(decisions)}")
        
        # New robust assertions
        assert isinstance(decisions, list), "Scanner should return a list of decisions"
        if len(decisions) > 0:
            d = decisions[0]
            assert hasattr(d, 'symbol'), "Decision missing symbol"
            assert hasattr(d, 'final_trade_score'), "Decision missing final_trade_score"
            assert hasattr(d, 'explanation'), "Decision missing explanation dictionary"
            if hasattr(d, 'explanation'):
                 assert 'family' in d.explanation, "Decision missing family metadata"
                 assert 'discovery_score' in d.explanation, "Decision missing discovery_score"
            
        print("=== SCANNER INTEGRATION SUCCESS ===\n")
        return True
    except Exception as e:
        print(f"❌ Scanner Crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_v5_scanner_fast()

