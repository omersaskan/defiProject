"""
ReportEngine — generates daily summaries, 7-day final report, baselines, failure analysis.
All computations are post-hoc (reads from linked shadow log).
"""
import numpy as np
import pandas as pd
from typing import Optional


class ReportEngine:

    def __init__(self, k: int = 3):
        self.k = k  # top-k for precision/recall

    # ─────────────────────────────────────────────────────────────────────────
    # DAILY SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    def daily_summary(self, df: pd.DataFrame) -> dict:
        """Summarise one day of shadow log entries."""
        df = df.dropna(subset=["future_24h_return"])
        if df.empty:
            return {}

        trades   = df[df["paper_trade_opened"] == True]
        wins_idx = trades["pnl_r"].dropna() > 0
        wins     = trades[wins_idx.reindex(trades.index, fill_value=False)]
        losses   = trades[(trades["pnl_r"].dropna() <= 0).reindex(trades.index, fill_value=False)]

        pos_pnl = wins["pnl_r"].sum() if not wins.empty else 0
        neg_pnl = abs(losses["pnl_r"].sum()) if not losses.empty else 0

        lcr = df["leader_captured"].mean() * 100 if "leader_captured" in df.columns else np.nan
        avg_rank = df["future_24h_rank_in_family"].mean() if "future_24h_rank_in_family" in df.columns else np.nan
        exit_reason_mode = df["final_exit_reason"].mode().iloc[0] if not df["final_exit_reason"].dropna().empty else "—"

        family_lcr = df.groupby("family")["leader_captured"].mean() * 100
        strongest = family_lcr.idxmax() if not family_lcr.empty else "—"
        weakest   = family_lcr.idxmin() if not family_lcr.empty else "—"

        return {
            "scans":                  df["scan_id"].nunique(),
            "unique_coins_selected":  df["symbol"].nunique(),
            "trades_opened":          len(trades),
            "leader_capture_rate":    round(lcr, 1),
            "avg_selected_family_rank": round(avg_rank, 2) if not np.isnan(avg_rank) else None,
            "win_rate":               round(len(wins) / max(len(trades), 1) * 100, 1),
            "profit_factor":          round(pos_pnl / neg_pnl, 2) if neg_pnl else None,
            "expectancy_r":           round(trades["pnl_r"].mean(), 3) if not trades.empty else None,
            "avg_hold_efficiency":    round(df["hold_efficiency"].mean(), 3) if "hold_efficiency" in df.columns else None,
            "avg_giveback_r":         round(df["giveback_r"].mean(), 3) if "giveback_r" in df.columns else None,
            "top_exit_reason":        exit_reason_mode,
            "strongest_family":       strongest,
            "weakest_family":         weakest,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 7-DAY FINAL REPORT
    # ─────────────────────────────────────────────────────────────────────────
    def final_report(self, df: pd.DataFrame) -> dict:
        df = df.dropna(subset=["future_24h_return"])
        trades = df[df["paper_trade_opened"] == True].copy()

        # ── A: General performance ─────────────────────────────────────────
        wins   = trades[trades["pnl_r"] > 0]
        losses = trades[trades["pnl_r"] <= 0]
        pos_p  = wins["pnl_r"].sum() if not wins.empty else 0
        neg_p  = abs(losses["pnl_r"].sum()) if not losses.empty else 0

        # Rank correlation: composite_leader_score vs future_24h_return
        per_scan_corr = []
        for _, sg in df.groupby("scan_id"):
            if len(sg) >= 2:
                c = sg["composite_leader_score"].corr(sg["future_24h_return"], method="spearman")
                if not np.isnan(c):
                    per_scan_corr.append(c)

        # Top-k precision and recall
        per_scan_prec = []
        for _, sg in df.groupby("scan_id"):
            if len(sg) >= self.k:
                top_pred   = set(sg.nlargest(self.k, "composite_leader_score")["symbol"])
                top_actual = set(sg.nlargest(self.k, "future_24h_return")["symbol"])
                per_scan_prec.append(len(top_pred & top_actual) / self.k)

        section_a = {
            "total_scans":             df["scan_id"].nunique(),
            "total_candidates":        len(df),
            "paper_trades_opened":     len(trades),
            "leader_capture_rate":     round(df["leader_captured"].mean() * 100, 1),
            "top_k_precision":         round(np.nanmean(per_scan_prec) * 100, 1) if per_scan_prec else None,
            "top_k_recall":            round(np.nanmean(per_scan_prec) * 100, 1) if per_scan_prec else None,
            "rank_correlation":        round(np.nanmean(per_scan_corr), 3) if per_scan_corr else None,
            "avg_future_family_rank":  round(df["future_24h_rank_in_family"].mean(), 2),
            "profit_factor":           round(pos_p / neg_p, 2) if neg_p else None,
            "expectancy_r":            round(trades["pnl_r"].mean(), 3) if not trades.empty else None,
            "avg_pnl_r":               round(trades["pnl_r"].mean(), 3) if not trades.empty else None,
            "win_rate":                round(len(wins) / max(len(trades), 1) * 100, 1),
            "hold_efficiency":         round(df["hold_efficiency"].mean(), 3),
            "giveback_r":              round(df["giveback_r"].mean(), 3),
        }

        # ── B: Family performance ──────────────────────────────────────────
        section_b = {}
        for family, fdf in df.groupby("family"):
            ftrades = fdf[fdf["paper_trade_opened"] == True]
            exit_mode = ftrades["final_exit_reason"].mode().iloc[0] if not ftrades.empty and not ftrades["final_exit_reason"].dropna().empty else "—"
            section_b[family] = {
                "count":               len(fdf),
                "leader_capture_rate": round(fdf["leader_captured"].mean() * 100, 1),
                "avg_future_rank":     round(fdf["future_24h_rank_in_family"].mean(), 2),
                "avg_pnl_r":          round(ftrades["pnl_r"].mean(), 3) if not ftrades.empty else None,
                "hold_efficiency":     round(fdf["hold_efficiency"].mean(), 3) if not fdf["hold_efficiency"].dropna().empty else None,
                "top_exit_reason":     exit_mode,
            }

        # ── C: Regime performance ──────────────────────────────────────────
        section_c = {}
        for regime, rdf in df.groupby("regime"):
            section_c[regime] = {
                "count":               len(rdf),
                "leader_capture_rate": round(rdf["leader_captured"].mean() * 100, 1),
                "avg_pnl_r":          round(rdf[rdf["paper_trade_opened"] == True]["pnl_r"].mean(), 3) if not rdf.empty else None,
                "expectancy_r":        round(rdf[rdf["paper_trade_opened"] == True]["pnl_r"].mean(), 3) if not rdf.empty else None,
                "fakeout_frequency":   round((rdf["fakeout_risk"] > 60).mean() * 100, 1),
            }

        # ── D: Exit analysis ──────────────────────────────────────────────
        exit_counts = trades["final_exit_reason"].value_counts(normalize=True) * 100
        exit_pnl    = trades.groupby("final_exit_reason")["pnl_r"].mean()
        exit_gvb    = trades.groupby("final_exit_reason")["giveback_r"].mean()

        best_exit  = exit_pnl.idxmax() if not exit_pnl.empty else "—"
        worst_exit = exit_pnl.idxmin() if not exit_pnl.empty else "—"

        section_d = {
            "exit_rate_pct":       exit_counts.to_dict(),
            "avg_pnl_by_exit":     exit_pnl.round(3).to_dict(),
            "avg_giveback_by_exit":exit_gvb.round(3).to_dict(),
            "best_exit_reason":    best_exit,
            "worst_exit_reason":   worst_exit,
        }

        # ── E: Adaptive stop analysis ──────────────────────────────────────
        adapt_trades = trades[trades["stop_mode"].notna() & (trades["stop_mode"] != "none")]
        nonadapt     = trades[trades["stop_mode"].isna() | (trades["stop_mode"] == "none")]

        mode_pnl = adapt_trades.groupby("stop_mode")["pnl_r"].agg(["mean", "count"]).round(3)
        fakeout_high = adapt_trades[adapt_trades["fakeout_risk"] > 60]

        section_e = {
            "adaptive_stop_trades":  len(adapt_trades),
            "legacy_stop_trades":    len(nonadapt),
            "stop_mode_distribution":adapt_trades["stop_mode"].value_counts().to_dict(),
            "pnl_by_stop_mode":      mode_pnl["mean"].to_dict() if not mode_pnl.empty else {},
            "best_stop_mode":        mode_pnl["mean"].idxmax() if not mode_pnl.empty else "—",
            "fakeout_high_pnl":      round(fakeout_high["pnl_r"].mean(), 3) if not fakeout_high.empty else None,
            "adaptive_vs_legacy_pnl": {
                "adaptive": round(adapt_trades["pnl_r"].mean(), 3) if not adapt_trades.empty else None,
                "legacy":   round(nonadapt["pnl_r"].mean(), 3) if not nonadapt.empty else None,
            }
        }

        return {
            "A_general":         section_a,
            "B_family":          section_b,
            "C_regime":          section_c,
            "D_exit":            section_d,
            "E_adaptive_stop":   section_e,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # BASELINE COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    def baseline_comparison(self, df: pd.DataFrame) -> dict:
        """
        Compare current family-ranker against 3 baselines using the same price data.
        All baselines select top-k candidates per scan using a different ranking signal.
        No future data used for ranking (baselines are rank-lookalike signals available at scan-time).
        """
        df = df.dropna(subset=["future_24h_return"])
        results = {}

        # ── Current engine ─────────────────────────────────────────────────
        results["family_ranker"] = self._eval_ranking(df, "composite_leader_score")

        # ── Baseline 1: Random pick ────────────────────────────────────────
        import random
        random.seed(42)
        df["_random_score"] = [random.random() for _ in range(len(df))]
        results["random_pick"] = self._eval_ranking(df, "_random_score")

        # ── Baseline 2: 24h momentum (future_24h_return is NOT available;
        #    use return_4 bars as momentum proxy at scan time) ────────────
        if "entry_readiness" in df.columns:
            # entry_readiness is the closest thing to a momentum signal at scan-time
            results["simple_momentum"] = self._eval_ranking(df, "entry_readiness")
        else:
            results["simple_momentum"] = {}

        # ── Baseline 3: Simple relative strength (discovery_score) ─────────
        results["relative_strength"] = self._eval_ranking(df, "discovery_score")

        return results

    def _eval_ranking(self, df: pd.DataFrame, score_col: str) -> dict:
        if score_col not in df.columns:
            return {}
        if df[score_col].dropna().empty:
            return {}

        per_scan_prec, per_scan_corr, capture_rates, family_ranks = [], [], [], []

        for _, sg in df.groupby("scan_id"):
            if len(sg) < self.k:
                continue

            corr = sg[score_col].corr(sg["future_24h_return"], method="spearman")
            if not np.isnan(corr):
                per_scan_corr.append(corr)

            top_pred   = set(sg.nlargest(self.k, score_col)["symbol"])
            top_actual = set(sg.nlargest(self.k, "future_24h_return")["symbol"])
            per_scan_prec.append(len(top_pred & top_actual) / self.k)

            best_actual = sg.nlargest(1, "future_24h_return")["symbol"].iloc[0]
            capture_rates.append(1.0 if best_actual in top_pred else 0.0)

            sel_ranks = sg[sg["symbol"].isin(top_pred)]["future_24h_rank_in_family"].dropna()
            if not sel_ranks.empty:
                family_ranks.append(sel_ranks.mean())

        sel  = df[df["paper_trade_opened"] == True].copy() if "paper_trade_opened" in df.columns and score_col == "composite_leader_score" else df.groupby("scan_id").apply(lambda g: g.nlargest(self.k, score_col)).reset_index(drop=True)
        wins = sel[sel.get("pnl_r", pd.Series()) > 0] if "pnl_r" in sel.columns else pd.DataFrame()
        lss  = sel[sel.get("pnl_r", pd.Series()) <= 0] if "pnl_r" in sel.columns else pd.DataFrame()
        pos_p= wins["pnl_r"].sum() if not wins.empty else 0
        neg_p= abs(lss["pnl_r"].sum()) if not lss.empty else 0

        return {
            "leader_capture_rate": round(np.nanmean(capture_rates) * 100, 1) if capture_rates else None,
            "top_k_precision":     round(np.nanmean(per_scan_prec) * 100, 1) if per_scan_prec else None,
            "rank_correlation":    round(np.nanmean(per_scan_corr), 3) if per_scan_corr else None,
            "avg_future_rank":     round(np.nanmean(family_ranks), 2) if family_ranks else None,
            "profit_factor":       round(pos_p / neg_p, 2) if neg_p else None,
            "expectancy_r":        round(sel["pnl_r"].mean(), 3) if "pnl_r" in sel.columns and not sel.empty else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # FAILURE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    def failure_analysis(self, df: pd.DataFrame) -> dict:
        df = df.dropna(subset=["pnl_r"])
        failures = df[df["pnl_r"] < 0].copy()
        if failures.empty:
            return {"total_failures": 0}

        failures["failure_class"] = "unknown"

        # selection_error: wrong leader selected (not top3), high discovery_score
        failures.loc[
            (failures["is_top3_family_next_24h"] == False) &
            (failures["discovery_score"] > 60),
            "failure_class"
        ] = "selection_error"

        # entry_timing_error: low entry_readiness but trade was taken
        failures.loc[
            (failures["entry_readiness"] < 65) &
            (failures["final_exit_reason"].isin(["STOP_LOSS", "TIME_EXIT"])),
            "failure_class"
        ] = "entry_timing_error"

        # stop_placement_error: correct leader (top3) but stopped out
        failures.loc[
            (failures["is_top3_family_next_24h"] == True) &
            (failures["final_exit_reason"] == "STOP_LOSS"),
            "failure_class"
        ] = "stop_placement_error"

        # exit_timing_error: high mfe but poor pnl (gave back too much)
        failures.loc[
            (failures.get("mfe_r", 0) > 1.5) &
            (failures["pnl_r"] < 0),
            "failure_class"
        ] = "exit_timing_error"

        # fakeout_ignored: high fakeout_risk at entry
        failures.loc[
            (failures["fakeout_risk"] > 60) &
            (failures["failure_class"] == "unknown"),
            "failure_class"
        ] = "fakeout_ignored"

        class_dist   = failures["failure_class"].value_counts().to_dict()
        avg_pnl      = failures.groupby("failure_class")["pnl_r"].mean().round(3).to_dict()
        family_dist  = failures["family"].value_counts().head(5).to_dict()
        regime_dist  = failures["regime"].value_counts().to_dict()
        exit_dist    = failures["final_exit_reason"].value_counts().to_dict()

        # Were we at least right on the coin but wrong on stop?
        right_call_wrong_stop = int((
            (failures["is_top3_family_next_24h"] == True) &
            (failures["final_exit_reason"] == "STOP_LOSS")
        ).sum())

        wrong_call = int((failures["is_top3_family_next_24h"] == False).sum())
        entry_early = int((failures["entry_readiness"] < 65).sum())
        fakeout_ignored_count = int((failures["fakeout_risk"] > 60).sum())

        return {
            "total_failures":         len(failures),
            "failure_class_dist":     class_dist,
            "avg_pnl_by_class":       avg_pnl,
            "right_call_wrong_stop":  right_call_wrong_stop,
            "wrong_selection":        wrong_call,
            "low_entry_readiness":    entry_early,
            "fakeout_risk_ignored":   fakeout_ignored_count,
            "top_failure_families":   family_dist,
            "top_failure_regimes":    regime_dist,
            "top_exit_reasons":       exit_dist,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MARKDOWN RENDERER
    # ─────────────────────────────────────────────────────────────────────────
    def render_markdown(
        self,
        report: dict,
        baselines: dict,
        failures: dict,
        daily_summaries: list,
        answers: dict,
    ) -> str:
        lines = ["# DefiHunter — 7-Day Shadow Validation Report\n"]

        # ── Daily summaries ────────────────────────────────────────────────
        lines.append("## Daily Summaries\n")
        summary_rows = []
        for s in daily_summaries:
            if not s:
                continue
            summary_rows.append({k: v for k, v in s.items()})
        if summary_rows:
            lines.append(pd.DataFrame(summary_rows).to_markdown(index=False))
        lines.append("")

        # ── A: General ────────────────────────────────────────────────────
        a = report.get("A_general", {})
        lines.append("## A — General Performance\n")
        for k, v in a.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        # ── B: Family ─────────────────────────────────────────────────────
        lines.append("## B — Family Performance\n")
        b = report.get("B_family", {})
        if b:
            b_df = pd.DataFrame(b).T
            lines.append(b_df.to_markdown())
        lines.append("")

        # ── C: Regime ─────────────────────────────────────────────────────
        lines.append("## C — Regime Performance\n")
        c = report.get("C_regime", {})
        if c:
            lines.append(pd.DataFrame(c).T.to_markdown())
        lines.append("")

        # ── D: Exit analysis ──────────────────────────────────────────────
        lines.append("## D — Exit Analysis\n")
        d = report.get("D_exit", {})
        lines.append(f"- **Best exit reason** (highest avg pnl_r): {d.get('best_exit_reason', '—')}")
        lines.append(f"- **Worst exit reason** (kills most edge): {d.get('worst_exit_reason', '—')}")
        lines.append("\n**Exit rate distribution:**")
        for er, pct in d.get("exit_rate_pct", {}).items():
            avg_pnl = d.get("avg_pnl_by_exit", {}).get(er, "?")
            avg_gvb = d.get("avg_giveback_by_exit", {}).get(er, "?")
            lines.append(f"  - `{er}`: {pct:.1f}% | avg pnl_r={avg_pnl} | avg giveback_r={avg_gvb}")
        lines.append("")

        # ── E: Adaptive stop ──────────────────────────────────────────────
        lines.append("## E — Adaptive Stop Analysis\n")
        e = report.get("E_adaptive_stop", {})
        lines.append(f"- Adaptive stop trades: {e.get('adaptive_stop_trades', 0)}")
        lines.append(f"- Legacy stop trades: {e.get('legacy_stop_trades', 0)}")
        avl = e.get("adaptive_vs_legacy_pnl", {})
        lines.append(f"- Adaptive avg pnl_r: {avl.get('adaptive', '—')}")
        lines.append(f"- Legacy avg pnl_r: {avl.get('legacy', '—')}")
        lines.append(f"- Best stop mode: {e.get('best_stop_mode', '—')}")
        lines.append("")

        # ── Baseline comparison ────────────────────────────────────────────
        lines.append("## Baseline Comparison\n")
        if baselines:
            lines.append(pd.DataFrame(baselines).T.to_markdown())
        lines.append("")

        # ── Failure analysis ──────────────────────────────────────────────
        lines.append("## Failure Analysis\n")
        lines.append(f"- Total failures: {failures.get('total_failures', 0)}")
        lines.append(f"- Right coin, wrong stop: {failures.get('right_call_wrong_stop', '—')}")
        lines.append(f"- Wrong selection: {failures.get('wrong_selection', '—')}")
        lines.append(f"- Low entry readiness: {failures.get('low_entry_readiness', '—')}")
        lines.append(f"- Fakeout risk ignored: {failures.get('fakeout_risk_ignored', '—')}")
        lines.append("\n**Failure class distribution:**")
        for cls, cnt in failures.get("failure_class_dist", {}).items():
            lines.append(f"  - `{cls}`: {cnt}")
        lines.append("")

        # ── Leakage safety note ────────────────────────────────────────────
        lines.append("## Leakage Safety Note\n")
        lines.append("> All `future_*` columns in the shadow log are populated exclusively by")
        lines.append("> `OutcomeLinker` in a separate post-hoc pass, using only prices where")
        lines.append("> `price_timestamp >= scan_timestamp + 24h`.")
        lines.append("> The live scanner and decision engine never read or write these columns.")
        lines.append("> This can be verified: any scan within the last 24h will have `NaN` in all `future_*` fields.")
        lines.append("")

        # ── Final answers ──────────────────────────────────────────────────
        lines.append("## Final Verdict — 8 Questions\n")
        for i, (q, a_val) in enumerate(answers.items(), 1):
            lines.append(f"**{i}. {q}**")
            lines.append(f"> {a_val}\n")

        return "\n".join(lines)
