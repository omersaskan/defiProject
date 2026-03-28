"""
AdaptiveStopEngine V2 — family + regime + fakeout_risk + participation_mode aware.

V2 key changes vs V1:
  1. Dual-layer stop: hard_stop_price + soft_invalidation_price
  2. noise_tolerance_bars — first N bars block trailing/breakeven
  3. fakeout_risk > 60 → reduce_size_first=True (stop WIDENS, not tightens)
     V1 bug: fakeout tight stop = more whipsaws. V2 fix: size down, stop stays wider.
  4. stop_width_mult parameter (from FamilyExecutionConfig); size responsibility
     is on the caller — caller MUST divide size_usd by stop_width_mult to maintain
     constant net dollar risk.
  5. defi_perp ATR mult: 1.2x → 1.6x (wider stop = less whipsaw; size halved externally)
  6. stop_confidence score returned (0–1): caller can log/use for diagnostics

Stop modes:
  atr_stop       — pure ATR-based stop
  structure_stop — structure low / pivot-based stop
  hybrid_stop    — blended (prefers structure if available and within range)

Regime modifier:
  chop / chop_*   → widen 20%  (wider stop in chop = fewer whipsaws) ← V2 change
  trend / trend_* → base
  high_vol*       → widen 30%

Family ATR multipliers (V2 vs V1):
  defi_perp:     1.6x (was 1.2x — stop widens, size halved externally)
  defi_restaking: 1.8x (was 1.6x)
  defi_rwa:      1.6x (was 1.4x)
"""

from typing import Any, Dict, Optional


# ── ATR multipliers (V2) ────────────────────────────────────────────────────
FAMILY_ATR_MULT: Dict[str, float] = {
    "defi_lending":    1.8,
    "defi_dex":        1.5,
    "defi_perp":       1.6,   # V2: was 1.2x — size halved by family_execution
    "defi_oracles":    1.5,
    "defi_lst":        1.5,
    "defi_restaking":  1.8,   # V2: was 1.6x
    "defi_yield":      1.5,
    "defi_infra":      1.4,
    "defi_rwa":        1.6,   # V2: was 1.4x
    "defi_beta":       1.5,
    "default":         1.5,
}

# ── Default stop mode per family ─────────────────────────────────────────────
FAMILY_STOP_MODE: Dict[str, str] = {
    "defi_lending":    "hybrid_stop",
    "defi_dex":        "hybrid_stop",
    "defi_perp":       "atr_stop",
    "defi_oracles":    "structure_stop",
    "defi_lst":        "hybrid_stop",
    "defi_restaking":  "hybrid_stop",
    "defi_yield":      "hybrid_stop",
    "defi_infra":      "atr_stop",
    "defi_rwa":        "structure_stop",
    "defi_beta":       "hybrid_stop",
    "default":         "hybrid_stop",
}

# ── TP multiples [tp1_r, tp2_r] — R multiples of risk distance ───────────────
FAMILY_TP_MULT: Dict[str, list] = {
    "defi_lending":    [2.0, 4.0],
    "defi_perp":       [1.5, 3.0],
    "defi_oracles":    [2.0, 3.5],
    "default":         [2.0, 4.0],
}

# ── Noise tolerance (bars) — no trailing/breakeven in first N bars ────────────
FAMILY_NOISE_BARS: Dict[str, int] = {
    "defi_perp":       8,    # faster-moving, give more breathing room
    "defi_oracles":    8,
    "defi_restaking":  6,
    "default":         4,    # ~1 hour at 15m bars
}

# ── Soft invalidation as fraction of hard stop distance ──────────────────────
SOFT_INVALIDATION_FRACTION = 0.5  # soft at 50% of hard stop distance from entry


class AdaptiveStopEngine:
    """
    V2: computes adaptive dual-layer stop/TP levels based on family, regime,
    fakeout_risk, ATR, and structural levels.

    Key contract about stop_width_mult:
        This engine accepts stop_width_mult as a parameter and applies it to
        the stop distance. The CALLER must divide position size_usd by
        stop_width_mult to ensure net dollar risk stays constant:
            effective_size_usd = base_size_usd / stop_width_mult
    """

    def compute_stop(
        self,
        row: Any,             # pandas Series or dict-like
        family: str,
        regime: str,
        fakeout_risk: float = 0.0,
        atr_col: str = "atr",
        stop_width_mult: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Returns:
            hard_stop_price      — full invalidation level (exit everything)
            stop_price           — alias for hard_stop_price (backward compat)
            soft_invalidation_price — early warning (~0.5× risk dist); triggers size reduce
            tp1_price            — partial take-profit target
            tp2_price            — full take-profit target
            stop_mode            — atr_stop / structure_stop / hybrid_stop
            atr_mult             — effective ATR multiplier after all modifiers
            risk_r               — absolute stop distance in price units
            noise_tolerance_bars — bars where trailing/breakeven is blocked
            reduce_size_first    — True when fakeout_risk > 60 (stop widens, size shrinks)
            stop_confidence      — 0–1 quality score of the stop placement
        """
        close = self._get(row, "close", 0.0)
        atr   = self._get(row, atr_col, 0.0)

        # Fallback ATR: 1.5% of close
        if atr <= 0.0 and close > 0.0:
            atr = close * 0.015

        # ── Family base multiplier + mode ────────────────────────────────────
        family_key = family if family in FAMILY_ATR_MULT else "default"
        atr_mult   = FAMILY_ATR_MULT.get(family_key, 1.5)
        stop_mode  = FAMILY_STOP_MODE.get(family_key, "hybrid_stop")

        # ── Regime modifier (V2 semantics) ───────────────────────────────────
        regime_lower = (regime or "").lower()
        if "chop" in regime_lower:
            # V2 CHANGE: chop → WIDEN stop 20% (V1 tightened — caused whipsaws)
            atr_mult *= 1.20
        elif "high_vol" in regime_lower or "volatile" in regime_lower:
            atr_mult *= 1.30  # widen 30% in high vol

        # ── Fakeout risk modifier (V2 CRITICAL FIX) ─────────────────────────
        # V1 bug: fakeout > 60 → tighten 15% = MORE whipsaws (stop too close)
        # V2 fix: fakeout > 60 → widen 10% + flag reduce_size_first
        reduce_size_first = False
        if fakeout_risk > 60:
            atr_mult *= 1.10       # wider stop
            reduce_size_first = True
        elif fakeout_risk > 40:
            atr_mult *= 1.05       # slight buffer

        # ── External stop_width_mult (from FamilyExecutionConfig) ────────────
        # Caller is responsible for shrinking size proportionally:
        #   size_usd /= stop_width_mult  →  net dollar risk unchanged
        if stop_width_mult != 1.0:
            atr_mult *= stop_width_mult

        # ── Compute risk distance ─────────────────────────────────────────────
        risk_dist = atr * atr_mult

        # ── Structure stop: prefer swing-low / support if available ──────────
        stop_confidence = 0.6  # default: ATR-only confidence
        if stop_mode in ("structure_stop", "hybrid_stop"):
            struct_low = None
            for col in ("structure_low", "swing_low", "support_level", "pivot_low"):
                val = self._get(row, col, None)
                if val and float(val) > 0:
                    struct_low = float(val)
                    break

            if struct_low and struct_low < close:
                structure_dist = close - struct_low
                if structure_dist > 0 and structure_dist < (risk_dist * 2.0):
                    if stop_mode == "structure_stop":
                        risk_dist = structure_dist  # pure structure stop
                        stop_confidence = 0.85
                    else:
                        # hybrid: blend 60% ATR / 40% structure
                        risk_dist = (risk_dist * 0.6) + (structure_dist * 0.4)
                        stop_confidence = 0.75

        # ── Hard stop ─────────────────────────────────────────────────────────
        hard_stop = max(close - risk_dist, close * 0.88)   # floor: -12%

        # ── Soft invalidation (50% of hard stop dist) ─────────────────────────
        soft_dist  = risk_dist * SOFT_INVALIDATION_FRACTION
        soft_price = close - soft_dist   # closer to entry than hard stop

        # ── TP levels ─────────────────────────────────────────────────────────
        tp_mults  = FAMILY_TP_MULT.get(family_key, FAMILY_TP_MULT["default"])
        tp1_price = close + (risk_dist * tp_mults[0])
        tp2_price = close + (risk_dist * tp_mults[1])

        # ── Noise tolerance ────────────────────────────────────────────────────
        noise_bars = FAMILY_NOISE_BARS.get(family_key, FAMILY_NOISE_BARS["default"])

        return {
            "hard_stop_price":          round(hard_stop, 6),
            "stop_price":               round(hard_stop, 6),    # backward compat
            "soft_invalidation_price":  round(soft_price, 6),
            "tp1_price":                round(tp1_price, 6),
            "tp2_price":                round(tp2_price, 6),
            "stop_mode":                stop_mode,
            "atr_mult":                 round(atr_mult, 3),
            "risk_r":                   round(risk_dist, 6),
            "noise_tolerance_bars":     noise_bars,
            "reduce_size_first":        reduce_size_first,
            "stop_confidence":          round(stop_confidence, 2),
        }

    def apply_width_mult(
        self, stop_result: Dict[str, Any], stop_width_mult: float
    ) -> Dict[str, Any]:
        """
        Post-hoc: re-apply a different stop_width_mult to an existing stop result.
        Used when the family_execution config changes after the initial compute.
        Return a new dict; do not mutate the input.
        """
        result = dict(stop_result)
        if stop_width_mult == 1.0 or stop_width_mult <= 0:
            return result

        close = result.get("tp1_price", 0)  # placeholder — use entry_price externally
        # Re-derive risk_dist and recompute. Easier to just call compute_stop again.
        # This method is intentionally minimal — callers should prefer passing
        # stop_width_mult directly to compute_stop.
        old_risk = result.get("risk_r", 0.0)
        new_risk  = old_risk * stop_width_mult
        delta     = new_risk - old_risk

        result["hard_stop_price"]        = round(result["hard_stop_price"] - delta, 6)
        result["stop_price"]             = result["hard_stop_price"]
        result["soft_invalidation_price"] = round(result["soft_invalidation_price"] - delta * SOFT_INVALIDATION_FRACTION, 6)
        result["tp1_price"]              = round(result["tp1_price"] + delta * 0.5, 6)
        result["tp2_price"]              = round(result["tp2_price"] + delta, 6)
        result["atr_mult"]               = round(result["atr_mult"] * stop_width_mult, 3)
        result["risk_r"]                 = round(new_risk, 6)
        return result

    @staticmethod
    def _get(row: Any, key: str, default: Any) -> Any:
        """Safe attribute/key access for both dict and pandas Series."""
        try:
            if hasattr(row, "get"):
                v = row.get(key, default)
            else:
                v = getattr(row, key, default)
            return float(v) if v is not None and v != "" else default
        except (TypeError, ValueError):
            return default
