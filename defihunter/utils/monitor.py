from datetime import datetime
from typing import Dict, Any, List
from defihunter.utils.logger import logger
from defihunter.utils.db_manager import db_manager

class SystemMonitor:
    """
    GT-Institutional: System Observability & Health Monitoring.
    Tracks model fallback rates, execution latency, and data integrity.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemMonitor, cls).__new__(cls)
            cls._instance.metrics = {
                "scans_total": 0,
                "model_fallbacks": 0,
                "api_errors": 0,
                "avg_latency_ms": 0.0,
                "last_scan_time": None
            }
        return cls._instance

    def report_scan(self, duration_ms: float, universe_size: int, fallbacks: int = 0):
        self.metrics["scans_total"] += 1
        self.metrics["model_fallbacks"] += fallbacks
        self.metrics["last_scan_time"] = datetime.now()
        
        # Simple moving average for latency
        n = self.metrics["scans_total"]
        old_avg = self.metrics["avg_latency_ms"]
        self.metrics["avg_latency_ms"] = (old_avg * (n-1) + duration_ms) / n
        
        if fallbacks > 0:
            logger.warning(f"[Monitor] Scan with {fallbacks} model fallbacks detected!")

    def report_error(self, component: str):
        self.metrics["api_errors"] += 1
        logger.error(f"[Monitor] Critical error in {component}")

    def get_health_summary(self) -> Dict[str, Any]:
        """Returns a snapshot of system health."""
        fallback_rate = 0.0
        if self.metrics["scans_total"] > 0:
            fallback_rate = self.metrics["model_fallbacks"] / self.metrics["scans_total"]
            
        return {
            "status": "HEALTHY" if self.metrics["api_errors"] == 0 else "DEGRADED",
            "uptime_scans": self.metrics["scans_total"],
            "avg_latency_ms": round(self.metrics["avg_latency_ms"], 2),
            "fallback_rate_per_scan": round(fallback_rate, 2),
            "last_scan": self.metrics["last_scan_time"].isoformat() if self.metrics["last_scan_time"] else None
        }

# Global Singleton Monitor
monitor = SystemMonitor()
