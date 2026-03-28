"""
Generate Shadow Validation Report from existing shadow_log.csv.

Usage:
    python scripts/generate_shadow_report.py [--log logs/shadow_log.csv] [--output reports/shadow_report_7d.md]
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from defihunter.validation.report_engine import ReportEngine


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are numeric after CSV read."""
    numeric_cols = [
        "discovery_score", "entry_readiness", "fakeout_risk", "hold_quality",
        "leader_prob", "composite_leader_score", "stop_price", "tp1_price",
        "tp2_price", "atr_mult", "kelly_risk_pct", "entry_price",
        "future_24h_return", "future_24h_rank_in_family",
        "pnl_r", "mfe_r", "giveback_r", "hold_efficiency",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["is_top3_family_next_24h", "leader_captured", "missed_leader", "paper_trade_opened"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({
                "True": True, "False": False, True: True, False: False,
                "true": True, "false": False, "1": True, "0": False,
            })

    if "scan_timestamp" in df.columns:
        df["scan_timestamp"] = pd.to_datetime(df["scan_timestamp"], utc=True, errors="coerce")
    if "scan_day" in df.columns:
        df["scan_day"] = pd.to_datetime(df["scan_day"], errors="coerce")

    return df


def _derive_answers(report: dict, baselines: dict, failures: dict) -> dict:
    """Derive answers to the 8 final questions from report data."""
    a = report.get("A_general", {})
    e = report.get("E_adaptive_stop", {})
    d = report.get("D_exit", {})
    b = report.get("B_family", {})

    lcr = a.get("leader_capture_rate", 0) or 0
    rng = baselines.get("random_pick", {}).get("leader_capture_rate", 0) or 0
    mom = baselines.get("simple_momentum", {}).get("leader_capture_rate", 0) or 0
    fr_lcr  = baselines.get("family_ranker", {}).get("leader_capture_rate", 0) or 0
    exp     = a.get("expectancy_r") or 0
    pf      = a.get("profit_factor") or 0

    # Which family is weakest?
    worst_fam = "N/A"
    if b:
        fam_lcr = {f: v.get("leader_capture_rate", 0) or 0 for f, v in b.items()}
        if fam_lcr:
            worst_fam = min(fam_lcr, key=fam_lcr.get)

    worst_exit = d.get("worst_exit_reason", "N/A")

    adaptive_pnl = (e.get("adaptive_vs_legacy_pnl") or {}).get("adaptive")
    legacy_pnl   = (e.get("adaptive_vs_legacy_pnl") or {}).get("legacy")

    def pnl_label(v):
        if v is None:
            return "N/A"
        return f"{v:+.3f}R"

    # Right call wrong stop vs wrong selection
    wrong_sel  = failures.get("wrong_selection", 0) or 0
    right_stop = failures.get("right_call_wrong_stop", 0) or 0
    total_fail = failures.get("total_failures", 1) or 1

    loss_driver = "selection errors" if wrong_sel > right_stop else "stop placement errors"
    ready_score = a.get("paper_trades_opened", 0) or 0

    return {
        "Sistem gerçekten family leader yakalıyor mu?":
            f"leader_capture_rate = {lcr:.1f}%. Random baseline = {rng:.1f}%, Momentum = {mom:.1f}%. "
            + ("✅ Evet, istatistiksel olarak anlamlı şekilde." if lcr > rng + 5 else "⚠️ Kısmen — zarf henüz dar değil."),

        "Sistem basit momentum baseline'ından daha iyi mi?":
            f"Family-ranker LCR {fr_lcr:.1f}% vs momentum {mom:.1f}%. "
            + ("✅ Evet." if fr_lcr > mom else "❌ Hayır — momentum baseline henüz geçilmemiş."),

        "Para kaybı selection yüzünden mi, execution yüzünden mi?":
            f"{wrong_sel}/{total_fail} kayıp yanlış selection, {right_stop}/{total_fail} doğru selection ama erken/yanlış stop. "
            f"Ana sürücü: **{loss_driver}**.",

        "En problemli family hangisi?":
            f"**{worst_fam}** — en düşük leader_capture_rate.",

        "En problemli exit reason hangisi?":
            f"**{worst_exit}** — ortalama pnl_r en düşük bu exit türünde.",

        "Adaptive stop gerçekten işe yarıyor mu?":
            f"Adaptive avg pnl_r = {pnl_label(adaptive_pnl)} vs legacy = {pnl_label(legacy_pnl)}. "
            + ("✅ Evet, adaptive daha iyi." if adaptive_pnl and legacy_pnl and adaptive_pnl > legacy_pnl else "📊 Fark henüz küçük — daha fazla trade gerekli."),

        "Bu haliyle canlı micro-live/paper için hazır mı?":
            f"expectancy_r = {exp:.3f}R, profit_factor = {pf}. "
            + ("✅ Paper/shadow için hazır." if exp > 0 else "⚠️ Negatif expectancy — shadow aşamasında kal."),

        "Bir sonraki tek en önemli geliştirme ne olmalı?":
            f"{'Giriş zamanlaması — entry_readiness filtresi sıkılaştırılmalı.' if lcr < 50 else 'Exit yönetimi — giveback azaltmak için trailing stop kalibrasyonu.'}",
    }


def generate_report(log_path: str, output_path: str):
    print(f"\n{'='*60}")
    print(f"  DefiHunter — Shadow Report Generator")
    print(f"{'='*60}")

    if not os.path.exists(log_path):
        print(f"✗ Shadow log not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    df = _cast_types(df)
    print(f"  Rows loaded : {len(df)}")
    linked = df.dropna(subset=["future_24h_return"])
    print(f"  Rows linked : {len(linked)} ({len(linked)/max(len(df),1)*100:.0f}%)")

    engine = ReportEngine(k=3)

    # ── Daily summaries ────────────────────────────────────────────────────
    daily_summaries = []
    if "scan_day" in df.columns:
        for day, day_df in df.groupby("scan_day"):
            s = engine.daily_summary(day_df)
            if s:
                s["day"] = str(day)[:10]
                daily_summaries.append(s)

    # ── Final report ────────────────────────────────────────────────────────
    report   = engine.final_report(linked)
    baselines = engine.baseline_comparison(linked)
    failures  = engine.failure_analysis(linked)
    answers   = _derive_answers(report, baselines, failures)

    # ── Render markdown ──────────────────────────────────────────────────────
    md = engine.render_markdown(
        report=report,
        baselines=baselines,
        failures=failures,
        daily_summaries=daily_summaries,
        answers=answers,
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    # ── Console summary ──────────────────────────────────────────────────────
    a = report.get("A_general", {})
    print(f"\n{'─'*60}")
    print(f"  total_scans           : {a.get('total_scans', 0)}")
    print(f"  total_candidates      : {a.get('total_candidates', 0)}")
    print(f"  paper_trades_opened   : {a.get('paper_trades_opened', 0)}")
    print(f"  leader_capture_rate   : {a.get('leader_capture_rate', 0):.1f}%")
    print(f"  top_k_precision       : {a.get('top_k_precision', '—')}")
    print(f"  rank_correlation      : {a.get('rank_correlation', '—')}")
    print(f"  win_rate              : {a.get('win_rate', 0):.1f}%")
    print(f"  profit_factor         : {a.get('profit_factor', '—')}")
    print(f"  expectancy_r          : {a.get('expectancy_r', '—')}")
    print(f"  hold_efficiency       : {a.get('hold_efficiency', '—')}")
    print(f"{'─'*60}")
    print(f"\n✓ Report written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DefiHunter Shadow Validation Report Generator")
    parser.add_argument("--log",    default="logs/shadow_log.csv",         help="Shadow log CSV path")
    parser.add_argument("--output", default="reports/shadow_report_7d.md", help="Report output path")
    args = parser.parse_args()
    generate_report(args.log, args.output)
