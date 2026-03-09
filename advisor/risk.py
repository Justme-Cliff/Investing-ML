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
                raw_data=data,
            )
        return results

    def analyze(self, ticker: str, info: dict,
                history: pd.DataFrame, rf_rate: float = 0.045,
                sector: str = "Unknown", nearest_iv: Optional[float] = None,
                iv_rank_pre: Optional[float] = None,
                raw_data: dict = None) -> dict:
        close   = history["Close"].dropna()
        iv_rank = iv_rank_pre if iv_rank_pre is not None else self._compute_iv_rank(close, nearest_iv)
        _raw = raw_data or {}
        return {
            "ticker":            ticker,
            "altman_z":          self.altman_z(info),
            "sharpe":            self.sharpe(close, rf_rate),
            "sortino":           self.sortino(close, rf_rate),
            "max_drawdown_pct":  self.max_drawdown(close),
            "var_95_pct":        self.var_95(close),
            "cvar_95_pct":       self.cvar_95(close),
            "roic_wacc":         self.roic_wacc_spread(info, rf_rate, sector),
            "roic_trend":        self.roic_trend(info),
            "accruals":          self.accruals_ratio(info),
            "gross_prof":        self.gross_profitability(info),
            "piotroski":         self.piotroski_9pt(info),
            "beneish":           self.beneish_m_score(info, _raw or {}),
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

    def cvar_95(self, close: pd.Series) -> Optional[float]:
        """
        Historical CVaR (Expected Shortfall) at 95% confidence.
        = mean loss across the worst 5% of 1-month return observations.
        Always <= VaR; captures tail severity, not just the threshold.
        Returns a negative number (e.g. -14.2 means expected monthly tail loss = 14.2%).
        """
        if len(close) < 63:
            return None
        monthly = close.pct_change(21).dropna()
        if len(monthly) < 10:
            return None
        cutoff = float(monthly.quantile(0.05))
        tail   = monthly[monthly <= cutoff]
        if len(tail) == 0:
            return None
        return round(float(tail.mean()) * 100, 1)

    # ── Beneish M-Score (full 8-variable) ─────────────────────────────────────
    def beneish_m_score(self, info: dict, raw_data: dict = None) -> dict:
        """
        Full 8-variable Beneish M-Score (Beneish 1999).
        Detects probability of earnings manipulation.
        M > -1.78: manipulator  |  M <= -1.78: non-manipulator

        Variables (year T vs year T-1):
          DSRI — Days Sales Receivable Index (receivables growing faster than sales?)
          GMI  — Gross Margin Index (margin compression = deteriorating competitive position)
          AQI  — Asset Quality Index (non-productive asset bloat)
          SGI  — Sales Growth Index
          DEPI — Depreciation Index (slowing depreciation inflates reported earnings)
          SGAI — SG&A Index (rising overhead signals loss of leverage)
          LVGI — Leverage Index (rising leverage = financial stress)
          TATA — Total Accruals / Total Assets (accruals = accounting > cash earnings)

        M = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI
                 + 0.115×DEPI  - 0.172×SGAI + 4.679×TATA - 0.327×LVGI

        For variables where prior-year data is unavailable, we use neutral means
        from Beneish's original sample (non-manipulator averages) per the paper.
        """
        raw = raw_data or {}

        # ── Variables computed from yfinance info ──────────────────────────────
        # TATA — most reliable: all inputs available
        ni  = info.get("netIncomeToCommon") or info.get("netIncome")
        ocf = info.get("operatingCashflow")
        ta  = info.get("totalAssets")
        if ni and ocf and ta and float(ta) > 0:
            tata = (float(ni) - float(ocf)) / float(ta)
        else:
            tata = 0.018   # Beneish sample mean (non-manipulator)

        # SGI — sales growth index (current / prior year revenue)
        rev_g = raw.get("revenue_trend") or float(info.get("revenueGrowth") or 0)
        sgi   = 1.0 + rev_g    # if rev grew 10%, SGI = 1.10

        # GMI — gross margin index (prior / current gross margin)
        # If margins are compressing year-over-year, GMI > 1 (red flag)
        # Proxy: use earningsGrowth relative to revenueGrowth
        # Divergence: revenue growing but earnings falling = margin compression
        earn_g = float(info.get("earningsGrowth") or 0)
        gm     = float(info.get("grossMargins")   or 0)
        if earn_g < rev_g - 0.05 and gm > 0:
            # Approximate: prior_gm = gm / (1 + earn_g); GMI = prior / current
            prior_gm_approx = min(1.0, gm * (1 + max(0, rev_g - earn_g)))
            gmi = prior_gm_approx / max(gm, 0.01)
        else:
            gmi = 1.0   # no detectable compression

        # AQI — asset quality index
        # Non-current / total assets ratio vs prior year (proxy via asset growth)
        asset_growth = raw.get("asset_growth") or 0.0
        aqi = 1.0 + max(0.0, float(asset_growth) - 0.05)  # penalise aggressive asset inflation

        # DEPI — depreciation index (proxy: capex intensity change)
        # Cannot compute without prior-year D&A data; omitted from partial model
        depi = None   # prior-year D&A unavailable from yfinance

        # DSRI — days sales receivable index
        # Cannot compute without prior-year receivables; omitted from partial model
        dsri = None   # prior-year receivables unavailable from yfinance

        # SGAI — SG&A index
        # Without prior year SG&A, proxy via margin spread
        om = float(info.get("operatingMargins") or 0)
        if gm > 0 and om >= 0:
            sga_ratio = gm - om   # implied SG&A / Revenue ratio
            sgai = 1.0 + max(0.0, sga_ratio - 0.20)   # penalise bloated SG&A
        else:
            sgai = 1.0   # neutral

        # LVGI — leverage index (current leverage vs prior year)
        de  = info.get("debtToEquity")
        if de is not None:
            de_v = float(de)
            if de_v > 10:
                de_v /= 100
            # Proxy: if earnings are falling but debt is high, leverage likely rising
            lvgi = 1.0 + max(0.0, (de_v - 0.80) * 0.10)
        else:
            lvgi = 1.0   # neutral

        # ── M-Score formula (partial 6-variable model when dsri/depi unavailable) ──
        # Full 8-variable Beneish: M = -4.84 + 0.920×DSRI + 0.528×GMI + 0.404×AQI
        #                              + 0.892×SGI + 0.115×DEPI - 0.172×SGAI
        #                              + 4.679×TATA - 0.327×LVGI
        # Partial model omits DSRI and DEPI terms (prior-year data unavailable),
        # adjusts intercept to -4.27 to preserve roughly correct scale.
        partial_model = dsri is None or depi is None
        if partial_model:
            m = (-4.27
                 + 0.528 * gmi
                 + 0.404 * aqi
                 + 0.892 * sgi
                 - 0.172 * sgai
                 + 4.679 * tata
                 - 0.327 * lvgi)
        else:
            m = (-4.84
                 + 0.920 * dsri
                 + 0.528 * gmi
                 + 0.404 * aqi
                 + 0.892 * sgi
                 + 0.115 * depi
                 - 0.172 * sgai
                 + 4.679 * tata
                 - 0.327 * lvgi)

        manipulator = m > -1.78

        if m > -1.00:
            risk_label = "HIGH MANIPULATION RISK"
        elif m > -1.78:
            risk_label = "POSSIBLE MANIPULATION"
        elif m > -2.22:
            risk_label = "CLEAN"
        else:
            risk_label = "VERY CLEAN"

        return {
            "m_score":      round(m, 3),
            "manipulator":  manipulator,
            "risk_label":   risk_label,
            "partial_model": partial_model,
            "components":   {
                "TATA": round(tata, 4),
                "SGI":  round(sgi,  3),
                "GMI":  round(gmi,  3),
                "AQI":  round(aqi,  3),
                "LVGI": round(lvgi, 3),
                "SGAI": round(sgai, 3),
            },
        }

    # ── ROIC Trend ────────────────────────────────────────────────────────────
    def roic_trend(self, info: dict) -> dict:
        """
        ROIC directional trend: is return on invested capital improving or declining?

        Uses margin trajectory as proxy:
          - Earnings growing faster than revenue = margin expansion = improving ROIC
          - Earnings growing slower than revenue = margin compression = declining ROIC

        Also uses ROA absolute level for context.
        Returns {trend, score_adj, detail}
        """
        roa    = info.get("returnOnAssets")
        rev_g  = float(info.get("revenueGrowth")  or 0)
        earn_g = float(info.get("earningsGrowth") or 0)
        om     = float(info.get("operatingMargins") or 0)
        gm     = float(info.get("grossMargins")     or 0)

        if roa is None:
            return {"trend": "UNKNOWN", "score_adj": 0.0, "detail": "Insufficient data"}

        roa_v     = float(roa)
        gap       = earn_g - rev_g   # positive = earnings outpacing revenue = margin expansion

        if roa_v > 0.12 and gap > 0.05:
            trend     = "EXPANDING"
            score_adj = 0.06
            detail    = f"ROIC expanding: ROA {roa_v:.1%}, EPS growing {gap:.1%} faster than revenue"
        elif roa_v > 0.06 and gap > 0.0:
            trend     = "IMPROVING"
            score_adj = 0.03
            detail    = f"ROIC improving: ROA {roa_v:.1%}, margins trending up"
        elif roa_v < 0:
            trend     = "NEGATIVE"
            score_adj = -0.08
            detail    = f"ROIC negative: ROA {roa_v:.1%} — destroying shareholder value"
        elif roa_v > 0.0 and abs(gap) < 0.05:
            trend     = "STABLE"
            score_adj = 0.0
            detail    = f"ROIC stable: ROA {roa_v:.1%}, revenue and earnings growing in line"
        elif gap < -0.10:
            trend     = "CONTRACTING"
            score_adj = -0.05
            detail    = f"ROIC contracting: EPS lagging revenue by {abs(gap):.1%} — margin compression"
        else:
            trend     = "DECLINING"
            score_adj = -0.03
            detail    = f"ROIC declining: ROA {roa_v:.1%}, earnings under pressure"

        return {"trend": trend, "score_adj": score_adj, "detail": detail}

    # ── Anti-Thesis Engine ────────────────────────────────────────────────────
    def anti_thesis(self, ticker: str, info: dict,
                    history: pd.DataFrame, risk_result: dict,
                    raw_data: dict = None) -> list:
        """
        Structural red flags that challenge a Buy signal.
        Returns list of {flag, severity, detail} sorted HIGH → MEDIUM → LOW.
        Severity: HIGH | MEDIUM | LOW
        """
        flags = []
        raw   = raw_data or {}

        # 1. Leverage risk
        de = info.get("debtToEquity")
        if de is not None:
            v = float(de)
            if v > 10:
                v /= 100      # yfinance sometimes returns 150 meaning 1.5×
            if v > 2.0:
                flags.append({"flag": "High Leverage", "severity": "HIGH",
                              "detail": f"D/E {v:.1f}× — debt load may strain FCF in rate-shock scenarios"})
            elif v > 1.2:
                flags.append({"flag": "Elevated Leverage", "severity": "MEDIUM",
                              "detail": f"D/E {v:.1f}× — above 1.2× threshold; monitor interest coverage"})

        # 2. Earnings quality (accruals)
        accruals = risk_result.get("accruals")
        if accruals is not None and accruals > 0.04:
            flags.append({"flag": "Low Earnings Quality", "severity": "HIGH",
                          "detail": f"Accruals {accruals:+.3f} — accounting earnings exceed cash flow; potential manipulation risk"})
        elif accruals is not None and accruals > 0.01:
            flags.append({"flag": "Accruals Warning", "severity": "MEDIUM",
                          "detail": f"Accruals {accruals:+.3f} — cash earnings lag reported earnings"})

        # 3. FCF vs Net Income divergence
        fcf = info.get("freeCashflow")
        ni  = info.get("netIncomeToCommon") or info.get("netIncome")
        if fcf is not None and ni is not None and float(ni) > 0:
            ratio = float(fcf) / float(ni)
            if ratio < 0.5:
                flags.append({"flag": "FCF/NI Divergence", "severity": "HIGH",
                              "detail": f"FCF is only {ratio:.0%} of net income — earnings may overstate cash generation"})
            elif ratio < 0.75:
                flags.append({"flag": "FCF Lag", "severity": "MEDIUM",
                              "detail": f"FCF is {ratio:.0%} of net income — elevated capex intensity; monitor conversion"})

        # 4. Negative FCF
        if fcf is not None and float(fcf) < 0:
            flags.append({"flag": "Negative Free Cash Flow", "severity": "HIGH",
                          "detail": f"FCF ${float(fcf)/1e9:.2f}B — company consumes more cash than it generates"})

        # 5. Revenue deceleration
        rev_trend = raw.get("revenue_trend")
        rg        = info.get("revenueGrowth")
        if rev_trend is not None and rev_trend < -0.05:
            flags.append({"flag": "Revenue Contraction", "severity": "HIGH",
                          "detail": f"QoQ revenue trend {rev_trend:+.1%} — sequential top-line decline compresses forward multiples"})
        elif rg is not None and float(rg) < 0:
            flags.append({"flag": "Revenue Declining", "severity": "MEDIUM",
                          "detail": f"YoY revenue growth {float(rg):+.1%} — negative growth narrows the bull-case valuation range"})

        # 6. Altman Z distress / gray zone
        az    = risk_result.get("altman_z", {})
        z_sc  = az.get("score")
        z_str = f"Z={z_sc:.2f} " if z_sc is not None else ""
        if az.get("zone") == "DISTRESS":
            flags.append({"flag": "Bankruptcy Risk", "severity": "HIGH",
                          "detail": f"Altman {z_str}— DISTRESS zone; elevated insolvency probability"})
        elif az.get("zone") == "GRAY":
            flags.append({"flag": "Financial Stress", "severity": "MEDIUM",
                          "detail": f"Altman {z_str}— GRAY zone; monitor balance sheet over next 2 quarters"})

        # 7. Short interest
        short_pct = raw.get("short_percent")
        if short_pct is not None:
            sp = float(short_pct)
            if sp > 0.15:
                flags.append({"flag": "High Short Interest", "severity": "HIGH",
                              "detail": f"{sp:.1%} of float sold short — sophisticated capital positioned against this thesis"})
            elif sp > 0.08:
                flags.append({"flag": "Elevated Short Interest", "severity": "MEDIUM",
                              "detail": f"{sp:.1%} of float sold short — notable bearish positioning warrants scrutiny"})

        # 8. Piotroski weak spots
        pf       = risk_result.get("piotroski", {})
        pf_score = pf.get("score")
        failed   = pf.get("failed", [])
        if pf_score is not None and pf_score <= 3:
            flags.append({"flag": "Weak Fundamentals", "severity": "HIGH",
                          "detail": f"Piotroski F={pf_score}/9 WEAK — failed: {', '.join(failed[:4])}"})
        elif failed:
            key_fails = [f for f in failed if any(k in f for k in ["ROA", "OCF", "LowDebt", "EPS"])]
            if key_fails:
                flags.append({"flag": "Key Piotroski Failures", "severity": "MEDIUM",
                              "detail": f"Failed critical checks: {', '.join(key_fails[:3])}"})

        # 9. SG&A bloat (gross - operating margin spread)
        gm = info.get("grossMargins")
        om = info.get("operatingMargins")
        if gm is not None and om is not None:
            gm_v, om_v = float(gm), float(om)
            if gm_v > 0 and (gm_v - om_v) > 0.35:
                flags.append({"flag": "High Overhead Structure", "severity": "MEDIUM",
                              "detail": f"Gross {gm_v:.0%} vs operating {om_v:.0%} — {(gm_v-om_v):.0%} gap signals bloated SG&A or R&D burn"})

        # 10. Earnings deterioration
        eg = info.get("earningsGrowth")
        if eg is not None and float(eg) < -0.15:
            flags.append({"flag": "Earnings Deterioration", "severity": "HIGH",
                          "detail": f"EPS growth {float(eg):+.1%} YoY — significant decline challenges the valuation bull case"})

        flags.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["severity"], 3))
        return flags

    # ── Portfolio Tail-Risk ───────────────────────────────────────────────────
    def portfolio_tail_risk(self, tickers: list, universe_data: dict,
                            rf_rate: float = 0.045) -> dict:
        """
        Portfolio-level CVaR, correlated worst-day returns, and 3 macro stress
        scenarios per stock (rate shock, recession, liquidity crunch).
        """
        ret_series: Dict[str, pd.Series] = {}
        for t in tickers:
            data  = universe_data.get(t, {})
            hist  = data.get("history")
            if hist is None or hist.empty:
                continue
            close = hist["Close"].dropna()
            if len(close) >= 63:
                ret_series[t] = close.pct_change().dropna()

        result: dict = {
            "portfolio_cvar": None,
            "portfolio_var":  None,
            "worst_day_avg":  {},
            "stress_scenarios": {},
        }

        if len(ret_series) >= 2:
            df        = pd.DataFrame(ret_series).dropna()
            port_rets = df.mean(axis=1)

            # Monthly portfolio returns (21-trading-day rolling product)
            port_m = (1 + port_rets).rolling(21).apply(np.prod, raw=True) - 1
            port_m = port_m.dropna()

            if len(port_m) >= 10:
                var_cut  = float(port_m.quantile(0.05))
                tail     = port_m[port_m <= var_cut]
                result["portfolio_cvar"] = round(float(tail.mean()) * 100, 1) if len(tail) > 0 else None
                result["portfolio_var"]  = round(var_cut * 100, 1)

            # Average per-stock return on the portfolio's worst 10% of days
            n_worst   = max(5, int(len(port_rets) * 0.10))
            worst_idx = port_rets.nsmallest(n_worst).index
            worst_avg = df.loc[worst_idx].mean() * 100
            result["worst_day_avg"] = {t: round(float(v), 2) for t, v in worst_avg.items()}

        # Per-stock stress scenarios (pure math, no external data needed)
        for t in tickers:
            data    = universe_data.get(t, {})
            info    = data.get("info", {})
            hist    = data.get("history")
            beta    = float(info.get("beta") or 1.0)
            monthly = pd.Series([], dtype=float)
            if hist is not None and not hist.empty:
                close = hist["Close"].dropna()
                if len(close) >= 63:
                    monthly = close.pct_change(21).dropna()

            # Scenario 1: Rate shock (+200bps) — beta-scaled drawdown estimate
            rate_shock = round(beta * -8.0, 1)

            # Scenario 2: Recession — worst 2% historical monthly return
            recession  = (round(float(monthly.quantile(0.02)) * 100, 1)
                          if len(monthly) >= 10 else None)

            # Scenario 3: Liquidity crunch — VaR × 1.5 stress multiplier
            liq_crunch = (round(float(monthly.quantile(0.05)) * 150, 1)
                          if len(monthly) >= 10 else None)

            result["stress_scenarios"][t] = {
                "rate_shock_pct": rate_shock,
                "recession_pct":  recession,
                "liquidity_pct":  liq_crunch,
                "beta":           round(beta, 2),
            }

        return result

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

    @staticmethod
    def piotroski_score(info: dict) -> float:
        """Return Piotroski F-Score normalized to 0-100. Convenience wrapper for fetcher."""
        result = RiskEngine().piotroski_9pt(info)
        score  = result.get("score", 0) if isinstance(result, dict) else 0
        return float(score) / 9.0 * 100.0
