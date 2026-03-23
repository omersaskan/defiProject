import os
import pandas as pd
from pathlib import Path
from typing import Optional

class TSDBManager:
    """
    Manages historical Time-Series data using partitioned Parquet files.
    """
    def __init__(self, base_dir: str = "data/tsdb"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, symbol: str, timeframe: str) -> Path:
        """Helper to get standardized parquet path."""
        # Replace .p to keep directory safe
        safe_sym = symbol.replace('.p', '_p').replace('/', '_')
        return self.base_dir / f"{safe_sym}_{timeframe}.parquet"

    def save_dataframe(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Saves or appends a DataFrame to the Parquet TSDB.
        Expects 'timestamp' column to be an exact datetime object.
        """
        if df.empty or 'timestamp' not in df.columns:
            return False
            
        path = self._get_path(symbol, timeframe)
        
        # Ensure timestamp is datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        try:
            if path.exists():
                # Read existing, append, drop duplicates, re-save
                existing_df = pd.read_parquet(path)
                combined = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp'], keep='last')
                combined = combined.sort_values('timestamp').reset_index(drop=True)
                combined.to_parquet(path, compression='snappy', engine='pyarrow')
            else:
                # Save fresh
                df = df.reset_index(drop=True)
                df.to_parquet(path, compression='snappy', engine='pyarrow')
            return True
        except Exception as e:
            print(f"Error saving to TSDB Parquet: {e}")
            return False

    def load_dataframe(self, symbol: str, timeframe: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Loads data from Parquet, optionally using filters to minimize memory footprint.
        """
        path = self._get_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()
            
        try:
            filters = []
            if start_date:
                # Need to use pd.Timestamp to pass to PyArrow filters
                filters.append(('timestamp', '>=', pd.Timestamp(start_date)))
            if end_date:
                filters.append(('timestamp', '<=', pd.Timestamp(end_date)))
                
            # If filters are empty, don't pass them
            kwargs = {}
            if filters:
                kwargs['filters'] = filters
                
            df = pd.read_parquet(path, engine='pyarrow', **kwargs)
            return df
        except Exception as e:
            print(f"Error reading TSDB Parquet: {e}")
            return pd.DataFrame()

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
        """
        Quickly reads the last timestamp in the Parquet file to know where to resume fetching.
        """
        path = self._get_path(symbol, timeframe)
        if not path.exists():
            return None
            
        try:
            # We just read the timestamp column to keep memory very low, then get max
            df = pd.read_parquet(path, columns=['timestamp'], engine='pyarrow')
            if not df.empty:
                return df['timestamp'].max()
            return None
        except Exception as e:
            print(f"Error fetching latest TSDB timestamp: {e}")
            return None
