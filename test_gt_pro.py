import pandas as pd
import numpy as np
from defihunter.data.features import compute_gt_pro_features

def test_gt_pro_logic():
    print("Testing GT-PRO Logic...")
    
    # Create synthetic data for MSB
    df = pd.DataFrame({
        'high': [100]*20 + [110],
        'low': [90]*20 + [105],
        'close': [95]*20 + [108],
        'open': [94]*20 + [106],
        'volume': [1000]*20 + [2000],
        'funding_rate': [0.0]*20 + [-0.0003],
        'cvd': np.linspace(0, 100, 21)
    })
    
    df = compute_gt_pro_features(df)
    
    last = df.iloc[-1]
    print(f"MSB Bull: {last['msb_bull']}")
    print(f"Funding Capitulation: {last['funding_capitulation']}")
    print(f"CVD Acceleration: {last['cvd_acceleration']}")
    
    assert last['msb_bull'] == True, "MSB Bull should be True"
    assert last['funding_capitulation'] == True, "Funding Capitulation should be True"
    
    print("SUCCESS: GT-PRO logic verified.")

if __name__ == "__main__":
    test_gt_pro_logic()
