import duckdb
import os
import pandas as pd
from datetime import datetime
from typing import Optional
from defihunter.utils.logger import logger

class DatabaseManager:
    """
    GT-Institutional: Centralized Persistence Layer (DuckDB).
    Handles analytical logging of features, trades, and scans.
    """
    _instance = None

    def __new__(cls, db_path: str = "logs/defihunter.db"):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._init_db(db_path)
        return cls._instance

    def _init_db(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = duckdb.connect(db_path)
        logger.info(f"[DB] Connected to {db_path}")
        self._create_schemas()

    def _create_schemas(self):
        """Initialize tables if they don't exist."""
        # 1. Trades Table (Performance History)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price DOUBLE,
                exit_price DOUBLE,
                status VARCHAR,
                pnl_usd DOUBLE,
                pnl_pct DOUBLE,
                risk_pct DOUBLE,
                setup_class VARCHAR,
                regime VARCHAR,
                family VARCHAR
            )
        """)

        # 3. Scans Table (Orchestration History)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                timestamp TIMESTAMP PRIMARY KEY,
                regime VARCHAR,
                universe_size INTEGER,
                duration_ms DOUBLE,
                total_balance DOUBLE
            )
        """)

    def log_features(self, df: pd.DataFrame):
        """Append features to the database. Converts extra columns to JSON if table exists."""
        if df.empty: return
        try:
            temp_df = df.copy()
            if 'timestamp' not in temp_df.columns:
                temp_df['timestamp'] = datetime.now()
                
            table_exists = self.conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'features'").fetchone()[0] > 0
            
            if not table_exists:
                self.conn.execute("CREATE TABLE features AS SELECT * FROM temp_df LIMIT 0")
                self.conn.execute("ALTER TABLE features ADD COLUMN log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            
            # Map columns to avoid mismatch
            cols = ", ".join(temp_df.columns)
            self.conn.execute(f"INSERT INTO features ({cols}) SELECT {cols} FROM temp_df")
            logger.debug(f"[DB] Logged {len(df)} feature rows.")
        except Exception as e:
            logger.error(f"[DB] Feature logging failed: {e}")



    def log_trade(self, pos_dict: dict):
        """Append a closed trade to the database."""
        try:
            trade_id = f"{pos_dict['symbol']}_{pos_dict['entry_time']}"
            entry_t = datetime.fromisoformat(pos_dict['entry_time'])
            exit_t = datetime.now()
            entry_p = pos_dict['entry_price']
            exit_p = pos_dict.get('exit_price', 0.0)
            size_usd = pos_dict['size_usd']
            pnl_usd = (exit_p / entry_p - 1.0) * size_usd
            pnl_pct = (exit_p / entry_p - 1.0) * 100.0
            
            self.conn.execute("""
                INSERT OR REPLACE INTO trades (
                    id, symbol, entry_time, exit_time, entry_price, exit_price, 
                    status, pnl_usd, pnl_pct, risk_pct, setup_class, regime, family
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, pos_dict['symbol'], entry_t, exit_t, entry_p, exit_p,
                pos_dict['status'], pnl_usd, pnl_pct, pos_dict.get('risk_pct', 0.0),
                pos_dict.get('setup_class', 'unknown'), pos_dict.get('regime', 'unknown'),
                pos_dict.get('family', 'unknown')
            ))
            logger.debug(f"[DB] Logged trade for {pos_dict['symbol']}")
        except Exception as e:
            logger.error(f"[DB] Trade logging failed: {e}")

    def log_scan(self, timestamp: datetime, regime: str, universe_size: int, duration_ms: float, balance: float):
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO scans (timestamp, regime, universe_size, duration_ms, total_balance)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, regime, universe_size, duration_ms, balance))
        except Exception as e:
            logger.error(f"[DB] Scan logging failed: {e}")

    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        return self.conn.execute(f"SELECT * FROM trades ORDER BY entry_time DESC LIMIT {limit}").df()

    def close(self):
        self.conn.close()

# Global Singleton
db_manager = DatabaseManager()
