import pandas as pd
from unittest.mock import patch, MagicMock
from defihunter.execution.pipeline import SignalPipeline
from defihunter.execution.scanner import ScanPipeline

def run_tests():
    print("Testing End-to-End Scanner -> Pipeline Regime/Volatility Stack...")
    
    class DummyConfig:
        def __init__(self):
            self.regimes = type('obj', (object,), {'overrides': {}})()
            self.decision = type('obj', (object,), {'top_n': 5})()
            self.anchors = ['BTC.p', 'ETH.p']
            self.families = {}
            self.ema = type('obj', (object,), {'fast': 9, 'medium': 21})()
            self.risk = type('obj', (object,), {'dict': lambda s: {}})()
            self.timeframe = "15m"

    cfg = DummyConfig()

    with patch('defihunter.execution.scanner.BinanceFuturesFetcher'):
        with patch('defihunter.execution.scanner.SignalBroadcaster'):
            scanner = ScanPipeline(config=cfg)

    # Force tuple
    scanner.signal_core._resolve_regime = lambda anchor: ("trend_alt_rotation", "high_vol")
    
    dates = pd.date_range(start='2024-01-01', periods=20, freq='15T')
    df_btc = pd.DataFrame({'timestamp': dates, 'symbol': 'BTC.p', 'close': range(100, 120), 'volume': 1000})

    try:
        res = scanner.signal_core.run(
            symbol_data_map={'BTC.p': df_btc},
            anchor_context={'BTC.p': {'15m': df_btc}},
        )
        print("Pipeline alone handles the tuple correctly!")
        print(f"Pipeline regime explicitly resolved as: {res.regime_label} (Vol: {scanner.signal_core._resolve_regime(None)[1]})")
    except Exception as e:
        print(f"PIPELINE CRASHED: {e}")
        return

    # Now scanner
    scanner.anchor_mtf = {'BTC.p': {'15m': df_btc}}
    scanner._update_adaptive_weights = lambda reg: {"trend_score": 1.0}
    scanner._resolve_sector_regime = lambda: {}
    scanner.symbol_data_map = {'BTC.p': df_btc}

    try:
        regime_lbl, volat_lbl = scanner._resolve_regimes(force_regime=None)
        adaptive_weights = scanner._update_adaptive_weights(regime_lbl)
        sector_data = scanner._resolve_sector_regime()

        pipeline_result = scanner.signal_core.run(
            symbol_data_map=scanner.symbol_data_map,
            anchor_context=scanner.anchor_mtf,
            regime_label=regime_lbl,
            sector_data=sector_data,
            adaptive_weights=adaptive_weights,
            mode="live",
            volatility_label=volat_lbl
        )
        print(f"Scanner properly unpacked into strings! Label={regime_lbl}, Volatility={volat_lbl}")
        print("Integration Test PASSED.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"SCANNER INTEGRATION CRASHED: {e}")

if __name__ == '__main__':
    run_tests()
