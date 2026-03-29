import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional

class StructuredLogger:
    """
    GT-Institutional: Unified Structured JSON Logger.
    Logs each event as a single-line JSON (JSONL) to logs/trace_log.jsonl.
    Ensures observability for Risk, Rule, and ML decisions.
    """
    def __init__(self, log_path: str = "logs/trace_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
    def log(
        self, 
        engine: str, 
        event: str, 
        level: str = "INFO", 
        symbol: Optional[str] = None, 
        data: Optional[Dict[str, Any]] = None
    ):
        """Logs a structured event in JSON format."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "engine": engine,
            "event": event,
            "symbol": symbol,
            "data": data or {}
        }
        
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Fallback to standard logging if file write fails
            print(f"[StructuredLogger Error] {e}")

# Singleton for global use
s_logger = StructuredLogger()
