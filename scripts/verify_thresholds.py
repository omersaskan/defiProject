import pandas as pd
from defihunter.engines.thresholds import ThresholdResolutionEngine
from defihunter.core.config import AppConfig

def run_tests():
    print("Testing Threshold Stacking & Regime Overrides...")

    # Mock config that uses specific overrides
    class MockRegimes:
        overrides = {
            'trend': {'min_score': 55},
            'trend_alt_rotation': {'min_score': 60}, # specific
            'high_vol': {'breakout_buffer_atr': 0.8}, # stacked specific
            'chop': {'min_score': 70, 'min_relative_leadership': 0.5}
        }
        min_score = 50
        min_relative_leadership = 0.0
        min_volume = 10000000
        max_spread_bps = 15.0
        breakout_buffer_atr = 0.5
        retest_tolerance_atr = 0.3
        time_stop_bars = 24

    class MockConfig:
        regimes = MockRegimes()
        families = {}

    engine = ThresholdResolutionEngine(thresholds_config=MockRegimes(), config=MockConfig())

    # 1. trend_alt_rotation + high_vol
    # Expect: 
    # Base: min_score=50, breakout_buffer_atr=0.5
    # Regime ('trend_alt_rotation' maps? No, it's explicitly explicitly available! min_score=60)
    # Volatility ('high_vol'): breakout_buffer_atr=0.8
    res_1 = engine.resolve_thresholds(regime='trend_alt_rotation', family='defi_beta', volatility='high_vol')
    assert res_1['min_score'] == 60, f"Expected min_score=60, got {res_1['min_score']}"
    assert res_1['breakout_buffer_atr'] == 0.8, f"Expected buffer=0.8, got {res_1['breakout_buffer_atr']}"
    print("[1] SUCCESS: trend_alt_rotation + high_vol successfully stacks specific overrides.")

    # 2. trend_btc_led + normal 
    # Expect:
    # 'trend_btc_led' has NO explicit key. It falls back to 'trend' (min_score=55)
    # Volatility 'normal' means no stacking.
    res_2 = engine.resolve_thresholds(regime='trend_btc_led', family='defi_beta', volatility='normal')
    assert res_2['min_score'] == 55, f"Expected min_score=55 (fallback to trend), got {res_2['min_score']}"
    assert res_2['breakout_buffer_atr'] == 0.5, f"Expected buffer=0.5, got {res_2['breakout_buffer_atr']}"
    print("[2] SUCCESS: trend_btc_led + normal cleanly falls back to general 'trend' without dropping parameters.")

    # 3. plain chop + normal
    # Expect:
    # 'chop' explicit override (min_score=70, min_relative_leadership=0.5)
    res_3 = engine.resolve_thresholds(regime='chop', family='defi_beta', volatility='normal')
    assert res_3['min_score'] == 70, f"Expected min_score=70, got {res_3['min_score']}"
    assert res_3['min_relative_leadership'] == 0.5, f"Expected min_lead=0.5, got {res_3['min_relative_leadership']}"
    print("[3] SUCCESS: explicit 'chop' overrides work seamlessly.")

    print("\nALL REGIME / THRESHOLD RESOLUTION VERIFICATIONS PASSED.")

if __name__ == '__main__':
    run_tests()
