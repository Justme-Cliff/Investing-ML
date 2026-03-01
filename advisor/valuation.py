# advisor/valuation.py — Multi-method intrinsic value engine
"""
Four independent valuation methods.  Their median = fair value.
Entry zone = [FV × 0.80, FV × 0.90]  (10-20% margin of safety)
Target      = max(FV × 1.20, analyst consensus)
Stop Loss   = entry_low × 0.92

Methods:
  1. DCF     — 2-stage discounted cash flow on FCF/share
  2. Graham  — sqrt(22.5 × EPS × Book Value per Share)
  3. EV/EBITDA — sector-median EV/EBITDA multiple → implied price
  4. FCF Yield — price at which FCF yield equals 4.5%

More methods agree = higher conviction (1-4 scale displayed).
"""

import math
import statistics
from typing import Dict, List, Optional

import pandas as pd

# Typical sector EV/EBITDA medians (updated periodically)
SECTOR_EV_EBITDA = {
    "Technology":  22,
    "Healthcare":  16,
    "Financials":  12,
    "Consumer":    16,
    "Energy":       8,
    "Industrials": 14,
    "Utilities":   12,
    "Real Estate": 20,
    "Materials":   10,
    "Unknown":     14,
}

# Signal labels
SIGNAL_LABELS = {
    "STRONG_BUY":        "STRONG BUY",
    "BUY":               "BUY",
    "HOLD_WATCH":        "HOLD/WATCH",
    "WAIT":              "WAIT",
    "AVOID_PEAK":        "AVOID PEAK",
    "INSUFFICIENT_DATA": "INSUF. DATA",
}


class ValuationEngine:
    """
    Computes intrinsic value via 4 independent methods then synthesises
    a buy zone, price target, stop loss, and risk/reward ratio.
    """

    def __init__(self, rf_rate: float = 0.045):
        self.rf_rate       = rf_rate
        self.discount_rate = rf_rate + 0.055   # rf + 5.5% equity risk premium (~10%)

    def analyze_all(self, top10, universe_data: dict) -> Dict[str, dict]:
        """Return valuation dict keyed by ticker for all stocks in top10."""
        results = {}
        for _, row in top10.iterrows():
            t    = row["ticker"]
            data = universe_data.get(t)
            if not data:
                continue
            results[t] = self.analyze(t, data["info"], data["history"], data["sector"])
        return results

    def analyze(self, ticker: str, info: dict,
                history: pd.DataFrame, sector: str) -> dict:
        close   = history["Close"].dropna()
        current = float(close.iloc[-1]) if len(close) > 0 else None
        if not current:
            return {"ticker": ticker, "signal": "INSUFFICIENT_DATA"}

        estimates: Dict[str, float] = {}

        dcf = self._dcf(info)
        if dcf and dcf > 0:
            estimates["dcf"] = dcf

        gn = self._graham_number(info)
        if gn and gn > 0:
            estimates["graham"] = gn

        ev = self._ev_ebitda_target(info, sector)
        if ev and ev > 0:
            estimates["ev_ebitda"] = ev

        fcf = self._fcf_yield_target(info)
        if fcf and fcf > 0:
            estimates["fcf_yield"] = fcf

        if not estimates:
            return {
                "ticker":        ticker,
                "current_price": round(current, 2),
                "estimates":     {},
                "fair_value":    None,
                "entry_low":     None,
                "entry_high":    None,
                "target_price":  None,
                "stop_loss":     None,
                "premium_pct":   None,
                "rr_ratio":      None,
                "signal":        "INSUFFICIENT_DATA",
                "methods_count": 0,
            }

        fair_value = statistics.median(list(estimates.values()))
        entry_low  = fair_value * 0.80   # 20% MoS
        entry_high = fair_value * 0.90   # 10% MoS
        target     = fair_value * 1.20   # 20% above fair value as base target

        # Use analyst target if it's more bullish than our estimate
        analyst_tgt = info.get("targetMeanPrice")
        if analyst_tgt and float(analyst_tgt) > target:
            target = float(analyst_tgt) * 0.95   # 5% haircut on consensus optimism

        stop_loss  = entry_low * 0.92           # 8% below entry
        premium    = (current - fair_value) / fair_value

        upside     = (target - current) / current
        downside   = (current - stop_loss) / current
        rr_ratio   = round(upside / downside, 1) if downside > 0 else None

        if current <= entry_low:
            signal = "STRONG_BUY"
        elif current <= entry_high:
            signal = "BUY"
        elif current <= fair_value:
            signal = "HOLD_WATCH"
        elif current <= fair_value * 1.10:
            signal = "WAIT"
        else:
            signal = "AVOID_PEAK"

        return {
            "ticker":          ticker,
            "current_price":   round(current, 2),
            "estimates":       {k: round(v, 2) for k, v in estimates.items()},
            "fair_value":      round(fair_value, 2),
            "entry_low":       round(entry_low, 2),
            "entry_high":      round(entry_high, 2),
            "target_price":    round(target, 2),
            "stop_loss":       round(stop_loss, 2),
            "premium_pct":     round(premium * 100, 1),
            "upside_pct":      round(upside * 100, 1),
            "downside_pct":    round(downside * 100, 1),
            "rr_ratio":        rr_ratio,
            "signal":          signal,
            "methods_count":   len(estimates),
            "sensitivity":     self.dcf_sensitivity(info),
        }

    # ── Method 1: 2-Stage DCF ─────────────────────────────────────────────────
    def _dcf(self, info: dict) -> Optional[float]:
        """
        2-stage discounted cash flow on free cash flow per share.
        Stage 1: 5 years at estimated growth rate
        Stage 2: perpetuity at terminal growth rate
        """
        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if not fcf or not shares:
            return None
        fcf, shares = float(fcf), float(shares)
        if fcf <= 0 or shares <= 0:
            return None

        fcf_ps = fcf / shares

        # Growth rate: blend revenue + earnings growth, cap at 25%, floor at -3%
        rev_g  = float(info.get("revenueGrowth")  or 0)
        earn_g = float(info.get("earningsGrowth") or 0)
        g      = min(0.25, max(-0.03, rev_g * 0.40 + earn_g * 0.60))

        # Terminal growth: conservative — min(3%, g × 0.30)
        tg = min(0.03, max(0.01, g * 0.30))
        dr = self.discount_rate

        # Stage 1: 5 years
        pv, cf = 0.0, fcf_ps
        for yr in range(1, 6):
            cf  = cf * (1 + g)
            pv += cf / (1 + dr) ** yr

        # Stage 2: terminal value (Gordon growth)
        tv_cf = cf * (1 + tg)
        if dr <= tg:
            dr = tg + 0.05      # safety floor to avoid division by zero
        tv     = tv_cf / (dr - tg)
        pv_tv  = tv / (1 + dr) ** 5

        intrinsic = pv + pv_tv

        # Sanity check: reject extreme values (> 50× current price or < $0.50)
        cur = info.get("currentPrice") or info.get("regularMarketPrice")
        if cur:
            cur = float(cur)
            if intrinsic > cur * 50 or intrinsic < 0.50:
                return None

        return round(intrinsic, 2)

    # ── Method 2: Graham Number ───────────────────────────────────────────────
    def _graham_number(self, info: dict) -> Optional[float]:
        """
        sqrt(22.5 × EPS × Book Value per Share)
        Benjamin Graham's classic formula for a stock's maximum intrinsic value.
        """
        eps  = info.get("trailingEps")
        bvps = info.get("bookValue")
        if not eps or not bvps:
            return None
        eps, bvps = float(eps), float(bvps)
        if eps <= 0 or bvps <= 0:
            return None
        return round(math.sqrt(22.5 * eps * bvps), 2)

    # ── Method 3: EV/EBITDA target ────────────────────────────────────────────
    def _ev_ebitda_target(self, info: dict, sector: str) -> Optional[float]:
        """
        Price implied by trading at the sector-median EV/EBITDA multiple.
        """
        ebitda = info.get("ebitda")
        shares = info.get("sharesOutstanding")
        if not ebitda or not shares:
            return None
        ebitda, shares = float(ebitda), float(shares)
        if ebitda <= 0 or shares <= 0:
            return None

        total_debt = float(info.get("totalDebt")  or 0)
        cash       = float(info.get("totalCash")  or 0)

        sector_mult = SECTOR_EV_EBITDA.get(sector, 14)
        target_ev   = ebitda * sector_mult
        target_mc   = target_ev - total_debt + cash

        if target_mc <= 0:
            return None

        return round(target_mc / shares, 2)

    # ── Method 4: FCF Yield target ────────────────────────────────────────────
    def _fcf_yield_target(self, info: dict) -> Optional[float]:
        """
        Price at which FCF yield = 4.5% (= 22.2× FCF per share).
        A 4.5% FCF yield is roughly fair for quality large-cap companies.
        """
        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if not fcf or not shares:
            return None
        fcf, shares = float(fcf), float(shares)
        if fcf <= 0 or shares <= 0:
            return None
        fcf_ps = fcf / shares
        return round(fcf_ps / 0.045, 2)    # price at 4.5% FCF yield

    # ── DCF Sensitivity (Bear / Base / Bull) ──────────────────────────────────
    def dcf_sensitivity(self, info: dict) -> dict:
        """
        Run DCF under three growth scenarios.
          Bear = base growth × 0.50
          Base = blended revenue + earnings growth (same as _dcf)
          Bull = base growth × 1.50  (capped at 30%)
        Returns: {"Bear": {fair_value, growth_rate, signal, premium_pct}, ...}
        """
        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if not fcf or not shares:
            return {}
        fcf, shares = float(fcf), float(shares)
        if fcf <= 0 or shares <= 0:
            return {}

        fcf_ps = fcf / shares
        rev_g  = float(info.get("revenueGrowth")  or 0)
        earn_g = float(info.get("earningsGrowth") or 0)
        base_g = min(0.25, max(-0.03, rev_g * 0.40 + earn_g * 0.60))
        cur    = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)

        scenarios = {
            "Bear": max(-0.03, base_g * 0.50),
            "Base": base_g,
            "Bull": min(0.30,  base_g * 1.50),
        }

        results = {}
        for name, g in scenarios.items():
            tg = min(0.03, max(0.01, g * 0.30))
            dr = self.discount_rate

            pv, cf = 0.0, fcf_ps
            for yr in range(1, 6):
                cf  = cf * (1 + g)
                pv += cf / (1 + dr) ** yr

            tv_cf = cf * (1 + tg)
            if dr <= tg:
                dr = tg + 0.05
            tv    = tv_cf / (dr - tg)
            fv    = pv + tv / (1 + dr) ** 5

            # Sanity check — same as _dcf
            if fv <= 0 or (cur and (fv > cur * 50 or fv < 0.50)):
                fv = None

            if fv:
                prem = ((cur / fv) - 1) * 100 if cur else None
                if   cur <= fv * 0.80: sig = "STRONG_BUY"
                elif cur <= fv * 0.90: sig = "BUY"
                elif cur <= fv:        sig = "HOLD_WATCH"
                elif cur <= fv * 1.10: sig = "WAIT"
                else:                  sig = "AVOID_PEAK"
            else:
                prem = None
                sig  = "INSUFFICIENT_DATA"

            results[name] = {
                "fair_value":  round(fv, 2) if fv else None,
                "growth_rate": round(g * 100, 1),
                "signal":      sig,
                "premium_pct": round(prem, 1) if prem is not None else None,
            }

        return results
