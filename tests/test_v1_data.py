import pandas as pd
from unittest.mock import MagicMock
from defihunter.data.binance_fetcher import BinanceFuturesFetcher

def test_data_layer():
    print("=== [PHASE 1] DATA LAYER VALIDATION ===")
    
    # Mock the fetcher to avoid real network calls
    fetcher = MagicMock(spec=BinanceFuturesFetcher)
    
    # Mock Futures Data
    futures_mock = pd.DataFrame({'close': [50000, 50100]}, index=[pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')])
    fetcher.fetch_ohlcv.return_value = futures_mock
    
    # Mock Spot Data
    spot_mock = pd.DataFrame({'close': [50050, 50150]}, index=[pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')])
    fetcher.fetch_spot_ohlcv.return_value = spot_mock
    
    # Mock Universe
    fetcher.get_defi_universe.return_value = ['BTC.p', 'ETH.p', 'AAVE.p']
    
    # 1. Futures Data Test
    print("Testing Futures OHLCV (BTC.p)...")
    futures_df = fetcher.fetch_ohlcv("BTC.p", timeframe="15m", limit=50)
    if not futures_df.empty:
        print(f"✅ Futures OK: {len(futures_df)} bars fetched.")
    else:
        print("❌ Futures Failed: Empty DataFrame.")
        assert False
        return False

    # 2. Spot Data Test
    print("Testing Spot OHLCV (BTCUSDT)...")
    spot_df = fetcher.fetch_spot_ohlcv("BTC.p", timeframe="15m", limit=20)
    if not spot_df.empty:
        print(f"✅ Spot OK: {len(spot_df)} bars fetched.")
        if 'close' in spot_df.columns:
            print(f"   Last Spot Price: {spot_df['close'].iloc[-1]}")
    else:
        print("❌ Spot Failed: Empty DataFrame.")
        assert False
        return False
        
    # 3. Universe Discovery
    print("Testing Universe Discovery...")
    uni = fetcher.get_defi_universe()
    if len(uni) > 0:
        print(f"✅ Universe OK: {len(uni)} coins found.")
    else:
        print("❌ Universe Failed.")
        assert False
        return False

    print("=== DATA LAYER SUCCESS ===\n")
    # Actually assert so pytest can run it correctly
    assert len(futures_df) > 0
    assert len(spot_df) > 0
    assert len(uni) > 0
    return True

if __name__ == "__main__":
    test_data_layer()
