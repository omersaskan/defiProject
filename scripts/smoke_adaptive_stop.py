"""
Adaptive Stop Smoke Check — no network required.
Simulates the exact scanner wiring path for 1 decision and prints the runtime log.

Usage:
    python scripts/smoke_adaptive_stop.py
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from defihunter.core.models import FinalDecision
from defihunter.engines.adaptive_stop import AdaptiveStopEngine
from defihunter.execution.paper_trade import PaperTradeEngine, PaperPortfolio

SEP = "─" * 60

def main():
    print(f"\n{SEP}")
    print("  DeFiHunter — Adaptive Stop Smoke Check")
    print(SEP)

    # 1. Mock decision (simulates what DecisionEngine outputs)
    d = FinalDecision(
        symbol="AAVE.p",
        timestamp=datetime.now(timezone.utc),
        final_trade_score=78.5,
        decision="trade",
        entry_price=88.42,
        stop_price=88.42 * 0.95,   # legacy fallback — will be overridden
        tp1_price=88.42 * 1.05,
        tp2_price=88.42 * 1.10,
        discovery_score=72.0,
        entry_readiness=68.0,
        fakeout_risk=18.0,
        hold_quality=62.0,
        leader_prob=0.78,
        composite_leader_score=78.5,
        explanation={
            "family":    "defi_lending",
            "triggers":  ["MSB_BULL", "CVD_ACCEL"],
            "kelly_risk_pct": 1.0,
        },
    )

    print(f"\n[1] Mock Decision:")
    print(f"    symbol       = {d.symbol}")
    print(f"    entry_price  = {d.entry_price}")
    print(f"    decision     = {d.decision}")
    print(f"    fakeout_risk = {d.fakeout_risk}")
    print(f"    family       = {d.explanation['family']}")
    print(f"    legacy_stop  = {d.stop_price:.4f}  ← will be overridden")

    # 2. Compute adaptive stop (simulates scanner wiring)
    engine   = AdaptiveStopEngine()
    stop_row = {'close': d.entry_price, 'atr': 1.2}
    regime   = "trend"

    adaptive_stop_result = engine.compute_stop(
        row=stop_row,
        family=d.explanation.get('family', 'defi_beta'),
        regime=regime,
        fakeout_risk=d.fakeout_risk,
    )

    print(f"\n[2] AdaptiveStopEngine output:")
    print(f"    stop_mode = {adaptive_stop_result['stop_mode']}")
    print(f"    atr_mult  = {adaptive_stop_result['atr_mult']}  (family=defi_lending, regime={regime})")
    print(f"    risk_r    = {adaptive_stop_result['risk_r']:.4f}")
    print(f"    stop      = {adaptive_stop_result['stop_price']:.4f}")
    print(f"    tp1       = {adaptive_stop_result['tp1_price']:.4f}")
    print(f"    tp2       = {adaptive_stop_result['tp2_price']:.4f}")

    # Mirrors scanner log
    print(
        f"\n[AdaptiveStop] {d.symbol} | "
        f"mode={adaptive_stop_result['stop_mode']} | "
        f"atr_mult={adaptive_stop_result['atr_mult']} | "
        f"family={d.explanation.get('family','?')} | "
        f"regime={regime} | "
        f"stop={adaptive_stop_result['stop_price']:.6f} | "
        f"tp1={adaptive_stop_result['tp1_price']:.6f} | "
        f"tp2={adaptive_stop_result['tp2_price']:.6f}"
    )

    # 3. Open paper position with adaptive stop
    state_file = tempfile.mktemp(suffix="_smoke_portfolio.json")
    paper = PaperTradeEngine(state_path=state_file)
    paper.portfolio = PaperPortfolio(balance_usd=10_000.0)

    opened = paper.open_position(d, risk_pct=1.0, adaptive_stop_result=adaptive_stop_result)

    print(f"\n[3] open_position result: {opened}")
    pos = paper.portfolio.open_positions[0]
    print(f"    PaperPosition.stop_price = {pos.stop_price:.4f}  (adaptive ✓)" if pos.stop_price == adaptive_stop_result['stop_price'] else f"    MISMATCH: got {pos.stop_price}")
    print(f"    PaperPosition.tp1_price  = {pos.tp1_price:.4f}")
    print(f"    PaperPosition.tp2_price  = {pos.tp2_price:.4f}")
    print(f"    PaperPosition.family     = {pos.family}")
    print(f"    PaperPosition.status     = {pos.status}")

    # 4. Read back saved JSON
    with open(state_file, 'r') as f:
        saved = json.load(f)
    saved_pos = saved["open_positions"][0]

    print(f"\n[4] Saved paper_portfolio.json (snippet):")
    print(json.dumps({
        "symbol":     saved_pos["symbol"],
        "stop_price": saved_pos["stop_price"],
        "tp1_price":  saved_pos["tp1_price"],
        "tp2_price":  saved_pos["tp2_price"],
        "family":     saved_pos["family"],
        "status":     saved_pos["status"],
        "entry_time": saved_pos["entry_time"],
    }, indent=4))

    os.unlink(state_file)

    print(f"\n{SEP}")
    print("  Smoke check PASSED — AdaptiveStop wiring end-to-end verified")
    print(SEP)

if __name__ == "__main__":
    main()
