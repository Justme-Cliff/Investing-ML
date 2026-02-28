# advisor/protocol.py — 7-gate Warren Buffett-style investment protocol
"""
Each stock must pass through 7 quality gates before being recommended.

Gate scores: 0–100
Status:  PASS (≥60)  |  WARN (35–59)  |  FAIL (<35)

Gate weights for overall score:
  Business Quality  0.20
  Competitive Moat  0.15
  Financial Health  0.15   ← includes Altman Z-Score
  Valuation         0.22   ← biggest weight — uses multi-method ValuationEngine
  Technical Entry   0.10
  News & Sentiment  0.08
  Trend Alignment   0.10

Conviction:
  HIGH   → ≤ 1 FAIL  and  overall ≥ 70  and  pass_count ≥ 6
  MEDIUM → ≤ 2 FAILs  and  pass_count ≥ 4
  LOW    → everything else (3+ FAILs)
"""

import statistics
from typing import Dict, List, Optional

from config import SECTOR_MEDIAN_PE
from advisor.risk import RiskEngine


GATE_NAMES = [
    "Business Quality",
    "Competitive Moat",
    "Financial Health",
    "Valuation",
    "Technical Entry",
    "News & Sentiment",
    "Trend Alignment",
]

GATE_SHORT = ["Quality", "Moat", "Health", "Value", "Entry", "News", "Trend"]

GATE_WEIGHTS = [0.20, 0.15, 0.15, 0.22, 0.10, 0.08, 0.10]   # must sum to 1.0

PASS_THRESHOLD = 60
WARN_THRESHOLD = 35

# Mapping from ValuationEngine signal → Gate 4 score
_SIGNAL_TO_GATE4 = {
    "STRONG_BUY":        95,
    "BUY":               78,
    "HOLD_WATCH":        62,
    "WAIT":              42,
    "AVOID_PEAK":        18,
    "INSUFFICIENT_DATA": None,   # fall back to own calculation
}

_risk_engine = RiskEngine()


class ProtocolAnalyzer:
    """Runs each stock through the 7-gate investment protocol."""

    def analyze_all(self, top10, universe_data: dict,
                    valuation_results: dict = None) -> List[dict]:
        """Run protocol on every row in the top10 DataFrame."""
        results = []
        for _, row in top10.iterrows():
            t    = row["ticker"]
            data = universe_data.get(t)
            if not data:
                continue
            val = (valuation_results or {}).get(t, {})
            results.append(self.analyze(t, data, row.to_dict(), val))
        return results

    def analyze(self, ticker: str, data: dict, row: dict,
                valuation: dict = None) -> dict:
        info    = data["info"]
        history = data["history"]
        close   = history["Close"].dropna()
        sector  = data["sector"]

        gate_scores = [
            self._gate_quality(info),
            self._gate_moat(info),
            self._gate_health(info),
            self._gate_valuation(info, sector, valuation),
            self._gate_entry(info, close),
            self._gate_news(data, row),
            self._gate_trend(close),
        ]

        statuses = []
        for sc in gate_scores:
            if sc >= PASS_THRESHOLD:
                statuses.append("pass")
            elif sc >= WARN_THRESHOLD:
                statuses.append("warn")
            else:
                statuses.append("fail")

        pass_c = statuses.count("pass")
        warn_c = statuses.count("warn")
        fail_c = statuses.count("fail")

        overall = sum(w * s for w, s in zip(GATE_WEIGHTS, gate_scores))

        if fail_c >= 3 or overall < 45:
            conviction = "LOW"
        elif fail_c == 0 and pass_c >= 6 and overall >= 70:
            conviction = "HIGH"
        elif fail_c <= 1 and pass_c >= 4:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        entry = self._entry_analysis(info, sector, close, valuation)

        return {
            "ticker":         ticker,
            "gates":          [round(s, 1) for s in gate_scores],
            "gate_statuses":  statuses,
            "pass_count":     pass_c,
            "warn_count":     warn_c,
            "fail_count":     fail_c,
            "overall_score":  round(overall, 1),
            "conviction":     conviction,
            "entry_analysis": entry,
        }

    # ── Gate 1: Business Quality ──────────────────────────────────────────────
    def _gate_quality(self, info: dict) -> float:
        s, w = [], []

        roa = info.get("returnOnAssets")
        if roa is not None:
            v = float(roa)
            s.append(100 if v > 0.08 else 70 if v > 0.04 else 40 if v > 0 else 5)
            w.append(1.5)

        roe = info.get("returnOnEquity")
        if roe is not None:
            v = float(roe)
            s.append(100 if v > 0.20 else 80 if v > 0.15 else 55 if v > 0.08 else 25 if v > 0 else 5)
            w.append(2.0)

        fcf = info.get("freeCashflow")
        if fcf is not None:
            s.append(80 if float(fcf) > 0 else 5)
            w.append(1.5)

        pm = info.get("profitMargins")
        if pm is not None:
            v = float(pm)
            s.append(100 if v > 0.20 else 80 if v > 0.15 else 55 if v > 0.08 else 25 if v > 0 else 5)
            w.append(1.5)

        rg = info.get("revenueGrowth")
        if rg is not None:
            v = float(rg)
            s.append(90 if v > 0.15 else 65 if v > 0.05 else 40 if v > 0 else 15)
            w.append(1.0)

        eg = info.get("earningsGrowth")
        if eg is not None:
            v = float(eg)
            s.append(90 if v > 0.15 else 65 if v > 0.05 else 40 if v > 0 else 10)
            w.append(1.0)

        return self._wavg(s, w)

    # ── Gate 2: Competitive Moat ──────────────────────────────────────────────
    def _gate_moat(self, info: dict) -> float:
        s, w = [], []

        mc = info.get("marketCap")
        if mc:
            v = float(mc)
            s.append(100 if v > 100e9 else 70 if v > 10e9 else 40 if v > 2e9 else 15)
            w.append(1.0)

        gm = info.get("grossMargins")
        if gm is not None:
            v = float(gm)
            s.append(100 if v > 0.60 else 75 if v > 0.40 else 50 if v > 0.25 else 20)
            w.append(2.0)

        om = info.get("operatingMargins")
        if om is not None:
            v = float(om)
            s.append(90 if v > 0.25 else 65 if v > 0.15 else 45 if v > 0.08 else 20 if v > 0 else 5)
            w.append(2.0)

        return self._wavg(s, w)

    # ── Gate 3: Financial Health (includes Altman Z-Score) ───────────────────
    def _gate_health(self, info: dict) -> float:
        s, w = [], []

        de = info.get("debtToEquity")
        if de is not None:
            v = float(de)
            if v > 10:
                v /= 100   # yfinance sometimes returns 150 meaning 1.5×
            s.append(100 if v < 0.3 else 75 if v < 0.8 else 50 if v < 1.5 else 25 if v < 3.0 else 5)
            w.append(2.5)

        cr = info.get("currentRatio")
        if cr is not None:
            v = float(cr)
            s.append(90 if v > 2.5 else 75 if v > 1.5 else 50 if v > 1.0 else 15)
            w.append(1.5)

        qr = info.get("quickRatio")
        if qr is not None:
            v = float(qr)
            s.append(90 if v > 1.5 else 70 if v > 1.0 else 45 if v > 0.7 else 15)
            w.append(1.0)

        ocf = info.get("operatingCashflow")
        if ocf is not None:
            s.append(80 if float(ocf) > 0 else 5)
            w.append(2.0)

        ebitda = info.get("ebitda")
        ie     = info.get("interestExpense")
        if ebitda and ie and abs(float(ie)) > 0:
            cov = abs(float(ebitda)) / abs(float(ie))
            s.append(100 if cov > 10 else 80 if cov > 5 else 55 if cov > 2.5 else 20)
            w.append(1.5)

        # Altman Z-Score — quantitative bankruptcy risk (academic model)
        az = _risk_engine.altman_z(info)
        az_score = az.get("score")
        if az_score is not None:
            zone = az.get("zone", "UNKNOWN")
            az_gate = 90 if zone == "SAFE" else 45 if zone == "GRAY" else 5
            s.append(az_gate)
            w.append(2.0)

        return self._wavg(s, w)

    # ── Gate 4: Valuation (uses ValuationEngine signal if available) ──────────
    def _gate_valuation(self, info: dict, sector: str,
                        valuation: dict = None) -> float:
        # If ValuationEngine computed a signal, use it (much more rigorous)
        if valuation:
            sig = valuation.get("signal")
            gate4 = _SIGNAL_TO_GATE4.get(sig)
            if gate4 is not None:
                # Blend with traditional metrics for robustness
                traditional = self._gate_valuation_traditional(info, sector)
                return gate4 * 0.65 + traditional * 0.35

        return self._gate_valuation_traditional(info, sector)

    def _gate_valuation_traditional(self, info: dict, sector: str) -> float:
        s, w = [], []
        sp = SECTOR_MEDIAN_PE.get(sector, 20)

        pe = info.get("trailingPE")
        if pe is not None and 0 < float(pe) <= 1000:
            r = float(pe) / sp
            s.append(100 if r < 0.7 else 80 if r < 0.9 else 65 if r < 1.1 else 45 if r < 1.3 else 25 if r < 1.6 else 5)
            w.append(2.5)

        fpe = info.get("forwardPE")
        if fpe is not None and 0 < float(fpe) < 500:
            r = float(fpe) / sp
            s.append(90 if r < 0.8 else 70 if r < 1.0 else 50 if r < 1.3 else 20)
            w.append(2.0)

        fcf = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        if fcf and mktcap and float(mktcap) > 0:
            y = float(fcf) / float(mktcap)
            s.append(95 if y > 0.06 else 70 if y > 0.03 else 45 if y > 0.01 else 20 if y > 0 else 5)
            w.append(2.0)

        peg = info.get("pegRatio")
        if peg is not None and 0 < float(peg) < 50:
            v = float(peg)
            s.append(100 if v < 0.5 else 80 if v < 1.0 else 55 if v < 1.5 else 30 if v < 2.0 else 5)
            w.append(1.5)

        return self._wavg(s, w)

    # ── Gate 5: Technical Entry (don't buy at peak) ───────────────────────────
    def _gate_entry(self, info: dict, close) -> float:
        s, w = [], []

        h52 = info.get("fiftyTwoWeekHigh")
        if h52 and len(close) > 0:
            cur = float(close.iloc[-1])
            h   = float(h52)
            pct = (h - cur) / h    # fraction below 52-week high
            s.append(90 if pct > 0.20 else 70 if pct > 0.10 else 45 if pct > 0.05 else 10)
            w.append(3.0)          # biggest weight in this gate

        tgt = info.get("targetMeanPrice")
        if tgt and len(close) > 0:
            cur    = float(close.iloc[-1])
            upside = (float(tgt) - cur) / cur
            s.append(90 if upside > 0.25 else 75 if upside > 0.15 else 55 if upside > 0.05 else 35 if upside > -0.05 else 10)
            w.append(2.0)

        fpe = info.get("forwardPE")
        if fpe and float(fpe) > 0:
            v = float(fpe)
            s.append(85 if v < 15 else 65 if v < 25 else 45 if v < 35 else 20 if v < 50 else 5)
            w.append(1.0)

        return self._wavg(s, w)

    # ── Gate 6: News & Sentiment ─────────────────────────────────────────────
    def _gate_news(self, data: dict, row: dict) -> float:
        base = float(data.get("sentiment", 50.0))
        info = data["info"]
        rec  = (info.get("recommendationKey") or "").lower()
        adj  = {"strong_buy": 15, "buy": 10, "hold": 5, "sell": -15, "strong_sell": -25}.get(rec, 0)
        base = max(0.0, min(100.0, base + adj))
        if len(data.get("news_titles", [])) >= 5:
            base = min(100.0, base + 5)
        return base

    # ── Gate 7: Trend Alignment ──────────────────────────────────────────────
    def _gate_trend(self, close) -> float:
        if len(close) < 50:
            return 50.0
        s, w = [], []
        cur = float(close.iloc[-1])

        if len(close) >= 200:
            sma200 = float(close.rolling(200).mean().dropna().iloc[-1])
            s.append(85 if cur > sma200 * 1.05 else 65 if cur > sma200 else 40 if cur > sma200 * 0.95 else 15)
            w.append(3.0)

        sma50 = float(close.rolling(50).mean().dropna().iloc[-1])
        s.append(75 if cur > sma50 else 25)
        w.append(2.0)

        if len(close) >= 63:
            r3m = float(close.iloc[-1] / close.iloc[-63] - 1)
            s.append(90 if r3m > 0.10 else 70 if r3m > 0.03 else 50 if r3m > -0.03 else 15)
            w.append(1.5)

        return self._wavg(s, w)

    # ── Intrinsic value + entry price analysis ────────────────────────────────
    def _entry_analysis(self, info: dict, sector: str, close,
                        valuation: dict = None) -> dict:
        """
        Use ValuationEngine results if available (DCF, Graham, EV/EBITDA, FCF yield).
        Falls back to simpler 4-method calculation if not.
        """
        if len(close) == 0:
            return {}

        current = float(close.iloc[-1])

        # Prefer deep ValuationEngine results
        if valuation and valuation.get("fair_value"):
            fv       = valuation["fair_value"]
            entry    = valuation.get("entry_low", fv * 0.85)
            signal   = valuation.get("signal", "INSUFFICIENT_DATA")
            premium  = valuation.get("premium_pct", (current - fv) / fv * 100)
            n_methods = valuation.get("methods_count", 0)
            return {
                "current_price": round(current, 2),
                "fair_value":    round(fv, 2),
                "entry_target":  round(entry, 2),
                "signal":        signal,
                "premium_pct":   round(premium, 1),
                "num_methods":   n_methods,
            }

        # Fallback: simple 4-method calculation
        sp      = SECTOR_MEDIAN_PE.get(sector, 20)
        fv_list = []

        eps = info.get("trailingEps")
        if eps and float(eps) > 0:
            fv_list.append(float(eps) * sp * 0.85)

        fwd = info.get("forwardEps")
        if fwd and float(fwd) > 0:
            fv_list.append(float(fwd) * min(sp * 0.90, 32))

        tgt = info.get("targetMeanPrice")
        if tgt and float(tgt) > 0:
            fv_list.append(float(tgt) * 0.88)

        fcf    = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if fcf and shares and float(shares) > 0 and float(fcf) > 0:
            fv_list.append(float(fcf) / float(shares) * 18)

        if not fv_list:
            return {
                "current_price": round(current, 2),
                "fair_value":    None,
                "entry_target":  None,
                "signal":        "INSUFFICIENT_DATA",
                "premium_pct":   None,
                "num_methods":   0,
            }

        fv      = statistics.median(fv_list)
        entry   = fv * 0.85
        premium = (current - fv) / fv * 100

        if current <= entry:
            signal = "STRONG_BUY"
        elif current <= fv:
            signal = "BUY"
        elif current <= fv * 1.08:
            signal = "HOLD_WATCH"
        elif current <= fv * 1.22:
            signal = "WAIT"
        else:
            signal = "AVOID_PEAK"

        return {
            "current_price": round(current, 2),
            "fair_value":    round(fv, 2),
            "entry_target":  round(entry, 2),
            "signal":        signal,
            "premium_pct":   round(premium, 1),
            "num_methods":   len(fv_list),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────
    @staticmethod
    def _wavg(scores: list, weights: list) -> float:
        if not scores:
            return 50.0
        tw = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / tw if tw else 50.0
