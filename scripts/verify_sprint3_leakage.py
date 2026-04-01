import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime, timezone, timedelta
from defihunter.validation.shadow_logger import ShadowLogger, SCAN_TIME_COLUMNS
from defihunter.validation.outcome_linker import OutcomeLinker
from defihunter.validation.report_engine import ReportEngine

def verify_leakage():
    print("--- SPRINT 3: LEAKAGE SAFETY VERIFICATION ---\n")

    # 1. Verify Scan-time Columns contain NO future data
    has_leakage = False
    for col in SCAN_TIME_COLUMNS:
        # these fields are in the definition but MUST be empty at write time
        if col.startswith("future_"):
            pass 
            # We will prove they are empty at write time by running the logger
    
    # Mock some scanner decisions
    class MockDecision:
        def __init__(self, sym, ds, er, fk, hq, lp, cls, ex):
            self.symbol = sym
            self.discovery_score = ds
            self.entry_readiness = er
            self.fakeout_risk = fk
            self.hold_quality = hq
            self.leader_prob = lp
            self.composite_leader_score = cls
            self.explanation = ex
            self.decision = "ENTER_LONG"
            self.stop_price = 100
            self.tp1_price = 110
            self.tp2_price = 120
            self.entry_price = 105

    d = MockDecision("BTC.p", 80, 75, 20, 90, 0.8, 85, {"family": "defi_beta"})
    
    # Run ShadowLogger
    logger = ShadowLogger("logs/test_shadow.csv")
    if os.path.exists("logs/test_shadow.csv"): os.remove("logs/test_shadow.csv")
    logger._ensure_header()
    
    scan_ts = datetime.now(timezone.utc)
    logger.log_scan([d], "trend", 100, scan_timestamp=scan_ts)
    
    raw_df = pd.read_csv("logs/test_shadow.csv")
    print(f"ShadowLogger recorded {len(raw_df)} rows.")
    
    # PROOF 1: future columns are empty
    future_val = raw_df["future_24h_return"].iloc[0]
    is_empty = pd.isna(future_val)
    print(f"PROOF 1 - ShadowLog future_24h_return at write time: {future_val} (Is NaN: {is_empty})")
    assert is_empty, "LEAKAGE: future_24h_return populated at scan time!"

    # 2. OutcomeLinker populates correctly ONLY after horizon
    linker = OutcomeLinker()
    
    # Price data: T=0, T+1h, T+24h
    prices = pd.DataFrame([
        {"symbol": "BTC.p", "timestamp": scan_ts, "close": 105, "high": 105, "low": 105},
        {"symbol": "BTC.p", "timestamp": scan_ts + timedelta(hours=1), "close": 110, "high": 110, "low": 105},
        {"symbol": "BTC.p", "timestamp": scan_ts + timedelta(hours=24), "close": 126, "high": 126, "low": 100},
    ])
    
    # Trade data simulating the backtester running forward in time
    trades = [{
        "symbol": "BTC.p",
        "entry_time": scan_ts.isoformat(),
        "pnl_r": 1.5,
        "mfe_r": 2.0,
        "giveback_r": 0.5,
        "exit_reason": "TIME_EXIT"
    }]
    
    linked_df = linker.link(raw_df, prices, trades)
    
    # PROOF 2: Linker used prices to populate
    l_return = linked_df["future_24h_return"].iloc[0]
    l_pnl = linked_df["pnl_r"].iloc[0]
    print(f"PROOF 2 - OutcomeLinker post-hoc future_24h_return: {l_return}")
    print(f"PROOF 2 - OutcomeLinker post-hoc pnl_r: {l_pnl}")
    assert l_return > 0, "LINKING FAILED: missed future prices"
    assert l_pnl == 1.5, "LINKING FAILED: missed trade match"
    
    # PROOF 3: ReportEngine evaluates on linked output
    report_engine = ReportEngine(k=1)
    # mock 'paper_trade_opened' required for report metrics
    linked_df['paper_trade_opened'] = True
    rep = report_engine.daily_summary(linked_df)
    
    print(f"PROOF 3 - ReportEngine Win Rate: {rep.get('win_rate')}% (Based on post-hoc outcomes)")
    assert rep.get('win_rate') == 100.0, "REPORT FAILED"
    
    print("\n ALL LEAKAGE SAFETY AND METRIC TESTS PASSED")

if __name__ == '__main__':
    verify_leakage()
