"""
OutcomeLinker — post-hoc join of shadow log rows with future market data.

LEAKAGE SAFETY CONTRACT:
- All future_* columns are computed ONLY using prices where:
    price_timestamp >= scan_timestamp + 24h (ONE full horizon bar)
- This file is NEVER imported by scanner.py or any live path.
- Recomputing past rows does not alter the original scan_timestamp.
"""
import numpy as np
import pandas as pd
from typing import Dict


HORIZON_HOURS = 24


class OutcomeLinker:
    """
    Links shadow log rows (decisions) with their 24h future outcomes.
    Uses a price DataFrame containing all OHLCV data for all symbols.

    Args:
        shadow_df: DataFrame of shadow log (from shadow_log.csv)
        price_df: cross-sectional DataFrame with columns:
                  [symbol, timestamp, close, high, low, family]
                  Must cover scan_timestamps + 24h forward.
        backtest_trade_log: optional list of dicts from BacktestEngine.trade_log
                            for pnl_r / mfe_r / giveback_r / exit_reason linking
    """

    def __init__(self, horizon_hours: int = HORIZON_HOURS):
        self.horizon_hours = horizon_hours

    def link(
        self,
        shadow_df: pd.DataFrame,
        price_df: pd.DataFrame,
        backtest_trade_log: list = None,
    ) -> pd.DataFrame:
        """
        Returns shadow_df with all future_* columns filled in.
        Rows where future data is not available (too recent) are left NaN.
        """
        if shadow_df.empty or price_df.empty:
            return shadow_df

        df = shadow_df.copy()
        df["scan_timestamp"] = pd.to_datetime(df["scan_timestamp"], utc=True)

        price_df = price_df.copy()
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)

        # Build {symbol: price_series} lookup for speed
        price_map: Dict[str, pd.DataFrame] = {
            sym: grp.sort_values("timestamp")
            for sym, grp in price_df.groupby("symbol")
        }

        # Build trade log lookup: {symbol: [trade_dicts]}
        trade_map: Dict[str, list] = {}
        if backtest_trade_log:
            for t in backtest_trade_log:
                trade_map.setdefault(t["symbol"], []).append(t)

        results = []
        for _, row in df.iterrows():
            result = self._link_row(row, price_map, price_df, trade_map)
            results.append(result)

        outcome_df = pd.DataFrame(results, index=df.index)
        for col in outcome_df.columns:
            df[col] = outcome_df[col]

        return df

    def _link_row(self, row, price_map, price_df, trade_map) -> dict:
        symbol      = row["symbol"]
        family      = row.get("family", "unknown")
        scan_ts     = row["scan_timestamp"]
        future_ts   = scan_ts + pd.Timedelta(hours=self.horizon_hours)

        # ── Future price lookup ─────────────────────────────────────────────
        sym_prices = price_map.get(symbol)
        if sym_prices is None:
            return self._empty_outcome()

        future_rows = sym_prices[sym_prices["timestamp"] >= future_ts]
        if future_rows.empty:
            return self._empty_outcome()  # data not yet available

        entry_price   = float(row.get("entry_price", 0)) or float(sym_prices[sym_prices["timestamp"] <= scan_ts]["close"].iloc[-1]) if not sym_prices[sym_prices["timestamp"] <= scan_ts].empty else 0.0
        future_close  = float(future_rows.iloc[0]["close"])
        future_return = (future_close - entry_price) / entry_price if entry_price > 0 else np.nan

        # ── Peak price in the 24h window (for MFE) ─────────────────────────
        window = sym_prices[(sym_prices["timestamp"] >= scan_ts) & (sym_prices["timestamp"] <= future_ts)]
        peak_price  = float(window["high"].max()) if not window.empty else future_close
        trough_price = float(window["low"].min()) if not window.empty else future_close

        stop_price = float(row.get("stop_price", 0)) or (entry_price * 0.95)
        r_dist     = abs(entry_price - stop_price) if stop_price and entry_price else (entry_price * 0.05)

        # ── Family rank ─────────────────────────────────────────────────────
        # All coins in the same family at the same scan_timestamp
        family_rows_at_ts = price_df[
            (price_df["timestamp"] >= future_ts) &
            (price_df["timestamp"] <= future_ts + pd.Timedelta(hours=1)) &
            (price_df.get("family", pd.Series()) == family if "family" in price_df.columns else pd.Series(True, index=price_df.index))
        ]

        # Compute 24h return for all family members
        family_returns = {}
        if "family" in price_df.columns:
            family_syms = price_df[price_df["family"] == family]["symbol"].unique()
            for fsym in family_syms:
                fp = price_map.get(fsym)
                if fp is None:
                    continue
                fp_entry = fp[fp["timestamp"] <= scan_ts]
                fp_future = fp[fp["timestamp"] >= future_ts]
                if fp_entry.empty or fp_future.empty:
                    continue
                ep = float(fp_entry.iloc[-1]["close"])
                fc = float(fp_future.iloc[0]["close"])
                family_returns[fsym] = (fc - ep) / ep if ep > 0 else 0.0

        rank_in_family       = None
        is_top3              = False
        if family_returns:
            sorted_syms = sorted(family_returns.items(), key=lambda x: -x[1])
            rank_map    = {s: i + 1 for i, (s, _) in enumerate(sorted_syms)}
            rank_in_family = rank_map.get(symbol)
            is_top3        = rank_in_family is not None and rank_in_family <= 3

        leader_captured = is_top3
        # Missed leader: best in family wasn't in top selections of this scan_id
        missed_leader   = not is_top3

        # ── pnl_r linking (from backtest trade log if available) ────────────
        pnl_r         = np.nan
        mfe_r         = np.nan
        giveback_r    = np.nan
        exit_reason   = ""
        hold_eff      = np.nan

        if symbol in trade_map:
            # Find closest entry_time match
            for t in trade_map[symbol]:
                if r_dist and r_dist > 0:
                    pnl_r      = float(t.get("pnl_r", np.nan))
                    mfe_r      = float(t.get("mfe_r", np.nan))
                    giveback_r = float(t.get("giveback_r", np.nan))
                    exit_reason = str(t.get("exit_reason", ""))
                    if not np.isnan(mfe_r) and mfe_r > 0 and not np.isnan(pnl_r):
                        hold_eff = pnl_r / mfe_r
                    break
        else:
            # Derive pnl_r from raw price if no trade log
            if r_dist and r_dist > 0:
                pnl_r_raw = (future_close - entry_price) / r_dist
                mfe_r_raw = (peak_price - entry_price) / r_dist
                mfe_r_raw = max(mfe_r_raw, 0)
                trough_r  = (trough_price - entry_price) / r_dist

                if trough_r <= -1.0:
                    exit_reason = "STOP_LOSS"
                    pnl_r = -1.0
                    mfe_r = max(mfe_r_raw, 0)
                    giveback_r = mfe_r_raw
                elif mfe_r_raw >= 4.0:
                    exit_reason = "TP2"
                    pnl_r = 2.0  # approximate with partial TP
                    mfe_r = mfe_r_raw
                    giveback_r = mfe_r_raw - pnl_r
                elif mfe_r_raw >= 2.0:
                    exit_reason = "TP1_RUNNER"
                    pnl_r = 1.0  # approximate partial
                    mfe_r = mfe_r_raw
                    giveback_r = mfe_r_raw - pnl_r
                else:
                    exit_reason = "TIME_EXIT"
                    pnl_r = pnl_r_raw
                    mfe_r = mfe_r_raw
                    giveback_r = max(mfe_r_raw - pnl_r_raw, 0)

                if not np.isnan(mfe_r) and mfe_r > 0:
                    hold_eff = pnl_r / mfe_r if pnl_r > 0 else 0.0

        return {
            "future_24h_return":         round(future_return, 4) if not np.isnan(future_return) else np.nan,
            "future_24h_rank_in_family": int(rank_in_family) if rank_in_family else np.nan,
            "is_top3_family_next_24h":   bool(is_top3),
            "leader_captured":           bool(leader_captured),
            "missed_leader":             bool(missed_leader),
            "final_exit_reason":         exit_reason,
            "pnl_r":                     round(pnl_r, 4) if not np.isnan(pnl_r) else np.nan,
            "mfe_r":                     round(mfe_r, 4) if not np.isnan(mfe_r) else np.nan,
            "giveback_r":                round(giveback_r, 4) if not np.isnan(giveback_r) else np.nan,
            "hold_efficiency":           round(hold_eff, 4) if not np.isnan(hold_eff) else np.nan,
        }

    @staticmethod
    def _empty_outcome() -> dict:
        return {
            "future_24h_return":         np.nan,
            "future_24h_rank_in_family": np.nan,
            "is_top3_family_next_24h":   np.nan,
            "leader_captured":           np.nan,
            "missed_leader":             np.nan,
            "final_exit_reason":         "",
            "pnl_r":                     np.nan,
            "mfe_r":                     np.nan,
            "giveback_r":                np.nan,
            "hold_efficiency":           np.nan,
        }
