import os
import pandas as pd
from defihunter.utils.logger import logger
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
        GT-PORTAL FIX: Uses explicit (symbol, timeframe, timestamp) deduplication 
        and preserves provenance metadata.
        """
        if df.empty or 'timestamp' not in df.columns:
            return False
            
        path = self._get_path(symbol, timeframe)
        
        # Ensure timestamp is datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        try:
            if path.exists():
                # Read existing, append, drop duplicates, re-save
                existing_df = pd.read_parquet(path)
                
                # If existing is missing provenance columns, add them with lowest priority
                for col in ['source_priority', 'source', 'quality_flag', 'is_synthetic']:
                    if col not in existing_df.columns:
                        if col == 'source_priority': existing_df[col] = 0
                        else: existing_df[col] = "LEGACY"
                
                combined = pd.concat([existing_df, df])
                
                # Priority-based deduplication: keys (sym, tf, ts) + priority (asc) -> keep last
                dedup_keys = ['symbol', 'timeframe', 'timestamp']
                combined = combined.sort_values(by=dedup_keys + ['source_priority'], ascending=True)
                combined = combined.drop_duplicates(subset=dedup_keys, keep='last')
                
                combined = combined.sort_values('timestamp').reset_index(drop=True)
                combined.to_parquet(path, compression='snappy', engine='pyarrow')
            else:
                # Save fresh
                df = df.reset_index(drop=True)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df.to_parquet(path, compression='snappy', engine='pyarrow')
            return True
        except Exception as e:
            logger.error(f"Error saving to TSDB Parquet for {symbol}: {e}")
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
            logger.error(f"Error reading TSDB Parquet for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
        """
        Quickly reads the last timestamp in the Parquet file to know where to resume fetching.
        """
        path = self._get_path(symbol, timeframe)
        if not path or not path.exists():
            return None
            
        try:
            # We just read the timestamp column to keep memory footprint low
            df = pd.read_parquet(path, columns=['timestamp'], engine='pyarrow')
            if not df.empty:
                return df['timestamp'].max()
            return None
        except Exception as e:
            logger.error(f"Error fetching latest TSDB timestamp for {symbol}: {e}")
            return None
