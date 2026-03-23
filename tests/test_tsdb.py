import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from defihunter.data.storage import TSDBManager

def test_parquet():
    tsdb = TSDBManager()
    
    # Try reading the AAVE 1h file which should have been downloaded
    df = tsdb.load_dataframe('AAVE.p', '1h')
    
    if df.empty:
        print("Dataframe is empty! Parquet file might not have been created for AAVE.p")
        # Let's see what files exist
        parquet_files = list(Path("data/tsdb").glob("*.parquet"))
        print(f"Existing files: {parquet_files}")
        if parquet_files:
            # Pick the first one
            file_path = parquet_files[0]
            symbol = file_path.stem.split('_')[0] + '.p'
            print(f"Testing {symbol} instead...")
            df = pd.read_parquet(file_path)
        else:
            return
            
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Latest Timestamp in DB: {df['timestamp'].max()}")
    print(f"Earliest Timestamp in DB: {df['timestamp'].min()}")
    
    # Check nulls
    nulls = df.isnull().sum()
    print(f"Nulls:\n{nulls}")
    
    # Check file size
    path = tsdb._get_path('AAVE.p', '1h')
    if path.exists():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\nParquet File Size: {size_mb:.3f} MB")
        
    print("\nTest passed if shape > 0 and latest timestamp is very recent.")
    
if __name__ == "__main__":
    test_parquet()
