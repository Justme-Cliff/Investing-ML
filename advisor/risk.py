# advisor/risk.py — Quantitative risk and quality metrics engine
"""
Metrics computed here:

SAFETY
  Altman Z-Score  — bankruptcy probability (modified for service firms)
                    Safe > 2.6 | Gray 1.1-2.6 | Distress < 1.1

RISK-ADJUSTED PERFORMANCE
  Sharpe Ratio    — (return - rf) / volatility  (annualised)
  Sortino Ratio   — (return - rf) / downside-vol  (penalises losses only)
  Max Drawdown    — worst peak-to-trough loss over the period
  VaR 95%         — worst expected 1-month loss at 95% confidence

QUALITY
  ROIC / WACC     — is the business creating or destroying shareholder value?
  Accruals Ratio  — (Net Income - OCF) / Total Assets  (negative = clean earnings)
  Gross Profit/A  — Gross Profit / Total Assets  (Novy-Marx quality factor)
  Piotroski 9pt   — full 9-point F-Score (profitability + leverage + efficiency)
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import SECTOR_ERP


class RiskEngine:
    """Run all risk / quality metrics for the final top-10 stocks."""

    def analyze_all(self, top10, universe_data: dict,
                    rf_rate: float = 0.045) -> Dict[str, dict]:
        results = {}
        for _, row in top10.iterrows():
            t    = row["ticker"]
            data = universe_data.get(t)
            if not data:
                continue
            sector     = data.get("sector",     "Unknown")
            nearest_iv = data.get("nearest_iv")
            # Use enriched iv_rank if Tier 2 already computed it
            iv_rank_pre = data.get("iv_rank")
            results[t] = self.analyze(
                t, data["info"], data["history"], rf_rate,
                sector=sector, nearest_iv=nearest_iv, iv_rank_pre=iv_rank_pre,
            )
        return results

    def analyze(self, ticker: str, info: dict,
                history: pd.DataFrame, rf_rate: float = 0.045,
                sector: str = "Unknown", nearest_iv: Optional[float] = None,
                iv_rank_pre: Optional[float] = None) -> dict:
        close   = history["Close"].dropna()
        iv_rank = iv_rank_pre if iv_rank_pre is not None else self._compute_iv_rank(close, nearest_iv)
        return {
            "ticker":            ticker,
            "altman_z":          self.altman_z(info),
            "sharpe":            self.sharpe(close, rf_rate),
            "sortino":           self.sortino(close, rf_rate),
            "max_drawdown_pct":  self.max_drawdown(close),
            "var_95_pct":        self.var_95(close),
            "roic_wacc":         self.roic_wacc_spread(info, rf_rate, sector),
            "accruals":          self.accruals_ratio(info),
            "gross_prof":        self.gross_profitability(info),
            "piotroski":         self.piotroski_9pt(info),
            "iv_rank":           iv_rank,
        }

    # ── IV rank proxy ─────────────────────────────────────────────────────────
    def _compute_iv_rank(self, close: pd.Series,
                         nearest_iv: Optional[float]) -> Optional[float]:
        """
        Estimate IV rank (0–1) using nearest_iv vs historical vol as proxy.
        True IV rank requires 52w IV history which yfinance doesn't provide;
        this approximates it using historical vol as the baseline.

        > 0.70 = elevated fear (IV >> hist vol) → caution
        < 0.30 = complacency (IV ≈ or < hist vol) → potentially good entry
        """
        if nearest_iv is None or len(close) < 63:
            return None
        hist_vol = float(close.pct_change().dropna().std()) * math.sqrt(252)
        if hist_vol <= 0:
            return None
        # IV typically runs ~115% of hist vol for ATM options in normal markets
        iv_ratio = nearest_iv / (hist_vol * 1.15)
        iv_rank  = max(0.0, min(1.0, (iv_ratio - 0.5) * 1.25))
        return round(iv_rank, 3)

    # ── Altman Z-Score (modified for non-manufacturing) ───────────────────────
    def altman_z(self, info: dict) -> dict:
        """
        Z' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4

        X1 = Working Capital / Total Assets  (proxy: OCF/TA)
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets  (proxy: EBITDA/TA)
        X4 = Book Value of Equity / Total Liabilities  (proxy: BV equity / debt)
        """
        ta = float(info.get("totalAssets") or 0)
        if ta <= 0:
            return {"score": None, "zone": "UNKNOWN"}

        # X1 — working capital proxy via operating cash flow
        ocf = float(info.get("operatingCashflow") or 0)
        x1  = ocf / ta

        # X2 — retained earnings
        re = float(info.get("retainedEarnings") or 0)
        x2 = re / ta

        # X3 — EBITDA / total assets (EBIT proxy)
        ebitda = float(info.get("ebitda") or 0)
        x3 = ebitda / ta

        # X4 — book value of equity / total debt (total liabilities proxy)
        bvps   = float(info.get("bookValue") or 0)
        shares = float(info.get("sharesOutstanding") or 0)
        debt   = float(info.get("totalDebt") or 1)       # floor at 1 to avoid /0
        bv_eq  = bvps * shares
        x4     = bv_eq / debt

        z = 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4

        if z > 2.6:
            zone = "SAFE"
        elif z > 1.1:
            zone = "GRAY"
        else:
            zone = "DISTRESS"

        return {"score": round(z, 2), "zone": zone}

    # ── Risk-adjusted returns ─────────────────────────────────────────────────
    def sharpe(self, close: pd.Series, rf_rate: float = 0.045) -> Optional[float]:
        if len(close) < 63:
            return None
        daily_rf = rf_rate / 252
        ret      = close.pct_change().dropna()
        if float(ret.std()) == 0:
            return None
        return round(float((ret.mean() - daily_rf) / ret.std() * math.sqrt(252)), 2)

    def sortino(self, close: pd.Series, rf_rate: float = 0.045) -> Optional[float]:
        if len(close) < 63:
            return None
        daily_rf = rf_rate / 252
        ret      = close.pct_change().dropna()
        down     = ret[ret < 0]
        if len(down) < 5 or float(down.std()) == 0:
            return None
        return round(float((ret.mean() - daily_rf) / down.std() * math.sqrt(252)), 2)

    def max_drawdown(self, close: pd.Series) -> Optional[float]:
        """Returns the worst peak-to-trough loss as a negative percentage."""
        if len(close) < 20:
            return None
        cum = (1 + close.pct_change().dropna()).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        return round(float(dd.min()) * 100, 1)

    def var_95(self, close: pd.Series) -> Optional[float]:
        """
        Historical simulation 95% 1-month VaR.
        Returns a negative number (e.g. -8.3 means worst expected 1-month loss = 8.3%).
        """
        if len(close) < 63:
            return None
        monthly = close.pct_change(21).dropna()
        if len(monthly) < 5:
            return None
        return round(float(monthly.quantile(0.05)) * 100, 1)

    # ── Valuation quality metrics ─────────────────────────────────────────────
    def roic_wacc_spread(self, info: dict, rf_rate: float = 0.045,
                         sector: str = "Unknown") -> dict:
        """
        ROIC proxy = Return on Assets (yfinance)
        WACC = full Modigliani-Miller formula:
            cost_equity = rf + beta × sector_erp
            cost_debt   = interest_expense / total_debt
            WACC        = (E/V)×cost_equity + (D/V)×cost_debt×(1−tax_rate)
        Falls back to CAPM if balance-sheet fields are missing.
        Spread > 0 means the business is creating shareholder value.
        """
        roa  = info.get("returnOnAssets")
        beta = float(info.get("beta") or 1.0)
        erp  = SECTOR_ERP.get(sector, 5.5) / 100

        # Full WACC inputs
        interest_exp = abs(float(info.get("interestExpense") or 0))
        total_debt   = float(info.get("totalDebt")           or 0)
        bvps         = float(info.get("bookValue")           or 0)
        shares       = float(info.get("sharesOutstanding")   or 0)
        tax_rate     = float(info.get("taxRateForCalcs")     or 0.21)

        D = total_debt
        E = bvps * shares
        V = D + E

        cost_equity = rf_rate + beta * erp
        if V > 0 and total_debt > 0:
            cost_debt = interest_exp / total_debt
            wacc = (E / V * cost_equity + D / V * cost_debt * (1 - tax_rate)) * 100
        else:
            # CAPM fallback when debt structure is unknown
            wacc = (rf_rate + beta * 0.055) * 100

        if roa is None:
            return {"roic": None, "wacc": round(wacc, 1), "spread": None, "verdict": "N/A"}

        roic   = float(roa) * 100
        spread = roic - wacc

        if spread > 15:
            verdict = "EXCEPTIONAL"
        elif spread > 8:
            verdict = "STRONG"
        elif spread > 2:
            verdict = "POSITIVE"
        elif spread > -3:
            verdict = "NEUTRAL"
        else:
            verdict = "DESTROYING VALUE"

        return {
            "roic":    round(roic, 1),
            "wacc":    round(wacc, 1),
            "spread":  round(spread, 1),
            "verdict": verdict,
        }

    def accruals_ratio(self, info: dict) -> Optional[float]:
        """
        Accruals = (Net Income − Operating Cash Flow) / Total Assets

        Negative = cash earnings exceed accounting earnings → HIGH QUALITY
        Positive = accounting earnings exceed cash →  low quality (potential manipulation)
        """
        ni  = info.get("netIncomeToCommon") or info.get("netIncome")
        ocf = info.get("operatingCashflow")
        ta  = info.get("totalAssets")
        if not ni or not ocf or not ta or float(ta) == 0:
            return None
        return round((float(ni) - float(ocf)) / float(ta), 4)

    def gross_profitability(self, info: dict) -> Optional[float]:
        """
        Gross Profit / Total Assets  (Novy-Marx 2013 quality factor)
        Higher = better pricing power and asset efficiency.
        """
        gm  = info.get("grossMargins")
        rev = info.get("totalRevenue")
        ta  = info.get("totalAssets")
        if not gm or not rev or not ta or float(ta) == 0:
            return None
        gp = float(gm) * float(rev)
        return round(gp / float(ta), 3)

    def piotroski_9pt(self, info: dict) -> dict:
        """
        Full 9-point Piotroski F-Score.

        PROFITABILITY (4 pts):
          F1  ROA > 0
          F2  Operating Cash Flow > 0
          F3  Earnings growth positive  (YoY)
          F4  Accrual ratio < 0  (cash earnings quality)

        LEVERAGE / LIQUIDITY (3 pts):
          F5  Debt/Equity improved  (proxy: D/E < 0.8)
          F6  Current ratio > 1.5
          F7  No dilution signal  (revenue growth > 5%)

        EFFICIENCY (2 pts):
          F8  Gross margin > 25%
          F9  Asset turnover > 0.5
        """
        score   = 0
        passed  = []
        failed  = []

        def _check(name, condition):
            nonlocal score
            if condition:
                score += 1; passed.append(name)
            else:
                failed.append(name)

        # F1 — ROA > 0
        roa = info.get("returnOnAssets")
        _check("F1 ROA>0",   roa is not None and float(roa) > 0)

        # F2 — Operating Cash Flow > 0
        ocf = info.get("operatingCashflow")
        _check("F2 OCF>0",   ocf is not None and float(ocf) > 0)

        # F3 — Earnings growth positive
        eg = info.get("earningsGrowth")
        _check("F3 EPS↑",    eg  is not None and float(eg) > 0)

        # F4 — Accruals < 0 (cash earnings better than accounting)
        ni = info.get("netIncomeToCommon") or info.get("netIncome")
        ta = info.get("totalAssets")
        if ocf and ni and ta and float(ta) > 0:
            _check("F4 Accruals", float(ocf) / float(ta) > float(ni) / float(ta))
        else:
            failed.append("F4 Accruals")

        # F5 — Low leverage (D/E < 0.8)
        de = info.get("debtToEquity")
        if de is not None:
            v = float(de)
            if v > 10: v /= 100
            _check("F5 LowDebt", v < 0.8)
        else:
            failed.append("F5 LowDebt")

        # F6 — Current ratio > 1.5
        cr = info.get("currentRatio")
        _check("F6 CR>1.5",  cr is not None and float(cr) > 1.5)

        # F7 — Revenue growth > 5% (no dilution proxy)
        rg = info.get("revenueGrowth")
        _check("F7 RevGrow", rg is not None and float(rg) > 0.05)

        # F8 — Gross margin > 25%
        gm = info.get("grossMargins")
        _check("F8 GM>25%",  gm is not None and float(gm) > 0.25)

        # F9 — Asset turnover > 0.5
        rev = info.get("totalRevenue")
        if rev and ta and float(ta) > 0:
            _check("F9 ATO>0.5", float(rev) / float(ta) > 0.5)
        else:
            failed.append("F9 ATO>0.5")

        interp = "STRONG" if score >= 7 else "MODERATE" if score >= 5 else "WEAK"

        return {
            "score":          score,
            "out_of":         9,
            "passed":         passed,
            "failed":         failed,
            "interpretation": interp,
        }
