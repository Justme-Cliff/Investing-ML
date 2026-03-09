# advisor/valuation.py — Multi-method intrinsic value engine
"""
Five independent valuation methods.  Their median = fair value.
Entry zone = [FV × 0.80, FV × 0.90]  (10-20% margin of safety)
Target      = max(FV × 1.20, analyst consensus)
Stop Loss   = entry_low × 0.92

Methods:
  1. DCF     — 3-stage discounted cash flow on FCF/share
  2. Graham  — sqrt(22.5 × EPS × Book Value per Share)
  3. EV/EBITDA — sector-median EV/EBITDA multiple → implied price
  4. FCF Yield — price at which FCF yield equals 4.5%
  5. EPV     — Earnings Power Value (Greenwald): zero-growth perpetuity

More methods agree = higher conviction (1-5 scale displayed).
"""

import math
import statistics
from typing import Dict, List, Optional

import pandas as pd

from config import SECTOR_ERP, VALUATION_CONFIG, SECTOR_GROWTH_CAP

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
        self.discount_rate = rf_rate + VALUATION_CONFIG["fallback_discount_offset"]

    # ── Dynamic discount rate (sector ERP + size premium) ─────────────────────
    def _get_discount_rate(self, info: dict, sector: str) -> float:
        """
        rf_rate + sector_erp + size_premium

        Size premium: marketCap > $10B = 0%  |  $2–10B = +1%  |  < $2B = +2%
        """
        erp      = SECTOR_ERP.get(sector, 5.5) / 100
        mktcap   = float(info.get("marketCap") or 0)
        if mktcap > 10e9:
            size_prem = 0.0
        elif mktcap > 2e9:
            size_prem = 0.01
        else:
            size_prem = 0.02
        return self.rf_rate + erp + size_prem

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

        dcf = self._dcf(info, sector)
        if dcf and dcf > 0:
            estimates["dcf"] = dcf

        # Graham Number: only meaningful for asset-heavy sectors (not tech/SaaS/biotech)
        _GRAHAM_SECTORS = {
            "Financials", "Financial Services", "Industrials", "Consumer Defensive",
            "Consumer Cyclical", "Energy", "Materials", "Utilities", "Real Estate",
        }
        if sector in _GRAHAM_SECTORS:
            gn = self._graham_number(info)
            if gn and gn > 0:
                estimates["graham"] = gn

        ev = self._ev_ebitda_target(info, sector)
        if ev and ev > 0:
            estimates["ev_ebitda"] = ev

        fcf = self._fcf_yield_target(info)
        if fcf and fcf > 0:
            estimates["fcf_yield"] = fcf

        epv = self._epv(info, sector)
        if epv and epv > 0:
            estimates["epv"] = epv

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
                "reverse_dcf":   self.reverse_dcf(info, sector),
            }

        fair_value = statistics.median(list(estimates.values()))
        entry_low  = fair_value * (1 - VALUATION_CONFIG["margin_of_safety_low"])
        entry_high = fair_value * (1 - VALUATION_CONFIG["margin_of_safety_high"])
        target     = fair_value * (1 + VALUATION_CONFIG["price_target_upside"])

        # Use analyst target if it's more bullish than our estimate
        analyst_tgt = info.get("targetMeanPrice")
        if analyst_tgt and float(analyst_tgt) > target:
            target = float(analyst_tgt) * 0.95   # 5% haircut on consensus optimism

        stop_loss  = entry_low * (1 - VALUATION_CONFIG["stop_loss_pct"])
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
            "sensitivity":     self.dcf_sensitivity(info, sector),
            "reverse_dcf":     self.reverse_dcf(info, sector),
        }

    # ── Method 1: 3-Stage DCF ─────────────────────────────────────────────────
    def _dcf(self, info: dict, sector: str = "Unknown") -> Optional[float]:
        """
        3-stage discounted cash flow on free cash flow per share.
        Stage 1: years 1-5  — high growth at blended analyst estimate
        Stage 2: years 6-10 — fade linearly from g1 toward terminal (mean reversion)
        Stage 3: terminal   — Gordon Growth at conservative GDP rate (2.5%)

        Mean-reversion in Stage 2 is critical for growth stocks: a company growing
        30% today won't sustain 30% forever — competition and market saturation erode it.
        The 2-stage model snaps directly to terminal, mis-pricing this fade period.
        """
        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if not fcf or not shares:
            return None
        fcf, shares = float(fcf), float(shares)
        if fcf <= 0 or shares <= 0:
            return None

        fcf_ps = fcf / shares

        # Growth rate: blend revenue + earnings growth, sector-aware cap, floor at -3%
        rev_g      = float(info.get("revenueGrowth")  or 0)
        earn_g     = float(info.get("earningsGrowth") or 0)
        _gcap      = SECTOR_GROWTH_CAP.get(sector, 0.25)
        g1         = min(_gcap, max(-0.03, rev_g * 0.40 + earn_g * 0.60))

        # Accruals quality adjustment on growth rate
        ni  = info.get("netIncomeToCommon") or info.get("netIncome")
        ocf = info.get("operatingCashflow")
        ta  = info.get("totalAssets")
        if ni and ocf and ta and float(ta) > 0:
            accruals = (float(ni) - float(ocf)) / float(ta)
            if accruals < -0.05:
                g1 *= 1.08
            elif accruals > 0.05:
                g1 *= 0.92
            g1 = min(_gcap, max(-0.03, g1))

        tg = VALUATION_CONFIG["terminal_growth_rate"]
        dr = self._get_discount_rate(info, sector)

        # Stage 1: years 1–5 at g1
        pv, cf = 0.0, fcf_ps
        for yr in range(1, 6):
            cf   = cf * (1 + g1)
            pv  += cf / (1 + dr) ** yr

        # Stage 2: years 6–10, fade g1 → tg linearly (mean reversion)
        for idx, yr in enumerate(range(6, 11)):
            fade = g1 + (tg - g1) * (idx + 1) / 5.0
            cf   = cf * (1 + fade)
            pv  += cf / (1 + dr) ** yr

        # Stage 3: terminal value (Gordon Growth from year-10 cash flow)
        tv_cf = cf * (1 + tg)
        if dr <= tg:
            dr = tg + 0.05
        tv    = tv_cf / (dr - tg)
        pv_tv = tv / (1 + dr) ** 10

        intrinsic = pv + pv_tv

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
        Price at which FCF yield = rf_rate + 3.0% (dynamic).
        At rf=4.5% → 7.5% FCF yield target (vs old hard-coded 4.5%).
        Adjusts automatically with the rate environment.
        """
        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if not fcf or not shares:
            return None
        fcf, shares = float(fcf), float(shares)
        if fcf <= 0 or shares <= 0:
            return None
        fcf_ps       = fcf / shares
        target_yield = self.rf_rate + VALUATION_CONFIG["fcf_yield_premium"]
        return round(fcf_ps / target_yield, 2)

    # ── Method 5: Earnings Power Value (Greenwald) ────────────────────────────
    def _epv(self, info: dict, sector: str = "Unknown") -> Optional[float]:
        """
        Earnings Power Value — values the business assuming ZERO growth.
        EPV = Normalized NOPAT / WACC  (perpetuity, no growth premium)

        If EPV > current_price: growth is free → safest possible entry.
        If EPV < current_price: you are paying for growth → high risk if it disappoints.

        Uses EBITDA × (1 − tax) as NOPAT proxy (avoids D&A distortions).
        Adjusts EV → equity value via net debt before per-share calculation.
        """
        ebitda = info.get("ebitda")
        if not ebitda:
            rev = info.get("totalRevenue")
            om  = info.get("operatingMargins")
            if rev and om:
                ebitda = float(rev) * float(om)
            else:
                return None
        ebitda = float(ebitda)
        if ebitda <= 0:
            return None

        shares = float(info.get("sharesOutstanding") or 0)
        if shares <= 0:
            return None

        tax  = float(info.get("taxRateForCalcs") or 0.21)
        beta = float(info.get("beta")            or 1.0)
        erp  = SECTOR_ERP.get(sector, 5.5) / 100

        cost_equity = self.rf_rate + beta * erp

        total_debt = float(info.get("totalDebt")          or 0)
        bvps       = float(info.get("bookValue")          or 0)
        ie         = abs(float(info.get("interestExpense") or 0))
        E = bvps * shares
        D = total_debt
        V = D + E
        if V > 0 and D > 0:
            cost_debt = ie / D if D > 0 else 0.05
            wacc      = (E / V) * cost_equity + (D / V) * cost_debt * (1 - tax)
        else:
            wacc = cost_equity

        wacc = max(wacc, 0.04)   # floor to prevent division blow-up

        nopat       = ebitda * (1 - tax)
        epv_total   = nopat / wacc

        # EV → equity value (subtract net debt)
        cash        = float(info.get("totalCash") or 0)
        net_debt    = total_debt - cash
        equity_val  = epv_total - net_debt

        if equity_val <= 0:
            return None

        per_share = equity_val / shares

        cur = info.get("currentPrice") or info.get("regularMarketPrice")
        if cur:
            cur = float(cur)
            if per_share > cur * 50 or per_share < 0.50:
                return None

        return round(per_share, 2)

    # ── DCF Sensitivity (Bear / Base / Bull) ──────────────────────────────────
    def dcf_sensitivity(self, info: dict, sector: str = "Unknown") -> dict:
        """
        Run DCF under three growth scenarios.
          Bear = base growth × 0.50
          Base = blended revenue + earnings growth (same as _dcf)
          Bull = base growth × 1.50  (capped at 30%)
        Returns: {"Bear": {fair_value, growth_rate, signal, premium_pct}, ...}
        Uses the same dynamic discount rate as _dcf().
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
        _gcap  = SECTOR_GROWTH_CAP.get(sector, 0.25)
        base_g = min(_gcap, max(-0.03, rev_g * 0.40 + earn_g * 0.60))
        cur    = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        dr     = self._get_discount_rate(info, sector)

        scenarios = {
            "Bear": max(-0.03, base_g * 0.50),
            "Base": base_g,
            "Bull": min(_gcap * 1.2, base_g * 1.50),
        }

        results = {}
        for name, g in scenarios.items():
            tg    = VALUATION_CONFIG["terminal_growth_rate"]
            dr_sc = dr

            # Stage 1: years 1-5
            pv, cf = 0.0, fcf_ps
            for yr in range(1, 6):
                cf  = cf * (1 + g)
                pv += cf / (1 + dr_sc) ** yr

            # Stage 2: years 6-10 fade to terminal
            for idx, yr in enumerate(range(6, 11)):
                fade = g + (tg - g) * (idx + 1) / 5.0
                cf   = cf * (1 + fade)
                pv  += cf / (1 + dr_sc) ** yr

            tv_cf = cf * (1 + tg)
            if dr_sc <= tg:
                dr_sc = tg + 0.05
            tv = tv_cf / (dr_sc - tg)
            fv = pv + tv / (1 + dr_sc) ** 10

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

    # ── Reverse DCF ───────────────────────────────────────────────────────────
    def reverse_dcf(self, info: dict, sector: str = "Unknown") -> Optional[dict]:
        """
        Solve for the implied growth rate baked into the current stock price.

        Asks: 'What must this company grow at for the current price to be fair?'
        Compares implied growth to analyst estimates to assess over/under-valuation.

        Uses bisection on the 3-stage DCF model (same assumptions as _dcf).
        """
        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        cur    = info.get("currentPrice") or info.get("regularMarketPrice")
        if not fcf or not shares or not cur:
            return None
        fcf, shares, cur = float(fcf), float(shares), float(cur)
        if fcf <= 0 or shares <= 0 or cur <= 0:
            return None

        fcf_ps = fcf / shares
        dr     = self._get_discount_rate(info, sector)
        tg     = VALUATION_CONFIG["terminal_growth_rate"]

        def _price_at_g(g: float) -> float:
            """3-stage DCF price given growth rate g."""
            pv, cf = 0.0, fcf_ps
            for yr in range(1, 6):
                cf   = cf * (1 + g)
                pv  += cf / (1 + dr) ** yr
            for idx, yr in enumerate(range(6, 11)):
                fade = g + (tg - g) * (idx + 1) / 5.0
                cf   = cf * (1 + fade)
                pv  += cf / (1 + dr) ** yr
            dr_t  = max(dr, tg + 0.01)
            tv    = cf * (1 + tg) / (dr_t - tg)
            return pv + tv / (1 + dr) ** 10

        # Bisection: find g in [-5%, +60%] such that _price_at_g(g) == cur
        lo, hi = -0.05, 0.60
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if _price_at_g(mid) < cur:
                lo = mid
            else:
                hi = mid
        implied_g = (lo + hi) / 2.0

        # Analyst / historical growth estimate for comparison
        rev_g   = float(info.get("revenueGrowth")  or 0)
        earn_g  = float(info.get("earningsGrowth") or 0)
        real_g  = min(0.25, max(-0.03, rev_g * 0.40 + earn_g * 0.60))

        gap = implied_g - real_g
        if gap > 0.15:
            verdict = "OVERPRICED"
        elif gap > 0.05:
            verdict = "STRETCHED"
        elif gap > -0.03:
            verdict = "FAIR"
        elif gap > -0.10:
            verdict = "ATTRACTIVE"
        else:
            verdict = "DEEPLY_UNDERVALUED"

        return {
            "implied_growth":   round(implied_g * 100, 1),
            "realistic_growth": round(real_g   * 100, 1),
            "gap_pct":          round(gap       * 100, 1),
            "verdict":          verdict,
        }
