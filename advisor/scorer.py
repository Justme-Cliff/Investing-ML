# advisor/scorer.py — Enhanced 7-factor MultiFactorScorer
"""
Each of the 7 factors is significantly upgraded from v1:

  1. Momentum   — 12-1 skip-month (academic grade) + 3/6-month blend
  2. Volatility — annualised std dev (inverted: low vol = high score)
  3. Value      — P/E + EV/EBITDA + FCF yield composite (3 signals)
  4. Quality    — Piotroski + ROE/PM + ROIC proxy + accruals + gross profitability
  5. Technical  — RSI + MACD + MA + Bollinger %B + OBV (5 signals)
  6. Sentiment  — news keywords + analyst rec
  7. Dividend   — dividend yield (capped at 15%)

Cross-sectional normalisation to 0-100, then macro regime tilt applied.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    WEIGHT_MATRIX, FACTOR_NAMES, SECTOR_MEDIAN_PE, SECTOR_EV_EBITDA, MACRO_TILTS,
    HORIZON_LABELS,
)
from advisor.collector import UserProfile


class MultiFactorScorer:

    def __init__(self, profile: UserProfile, macro_data: dict,
                 learned_weights: Optional[List[float]] = None):
        self.profile         = profile
        self.macro_data      = macro_data
        self.learned_weights = learned_weights

    # ── Main entry ────────────────────────────────────────────────────────────
    def score_all(self, universe_data: Dict) -> pd.DataFrame:
        rows = [self._score_one(t, d) for t, d in universe_data.items()]
        rows = [r for r in rows if r is not None]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = self._filter(df, strict=True)
        if len(df) < 10:
            df_all = pd.DataFrame([r for r in rows if r is not None])
            df     = self._filter(df_all, strict=False)

        # Preferred sector thumb-on-scale (small boost before normalising)
        if self.profile.preferred_sectors:
            mask = df["sector"].isin(self.profile.preferred_sectors)
            df.loc[mask, "quality_raw"]  += 0.05
            df.loc[mask, "momentum_raw"] += 0.02

        # Normalise each raw column to 0–100
        raw_to_score = {
            "momentum_raw":   "momentum_score",
            "volatility_raw": "volatility_score",
            "value_raw":      "value_score",
            "quality_raw":    "quality_score",
            "technical_raw":  "technical_score",
            "sentiment_raw":  "sentiment_score",
            "dividend_raw":   "dividend_score",
        }
        for raw_col, score_col in raw_to_score.items():
            df[score_col] = self._normalise(df[raw_col])

        w = self._get_weights()
        df["composite_score"] = (
            w[0] * df["momentum_score"]   +
            w[1] * df["volatility_score"] +
            w[2] * df["value_score"]      +
            w[3] * df["quality_score"]    +
            w[4] * df["technical_score"]  +
            w[5] * df["sentiment_score"]  +
            w[6] * df["dividend_score"]
        )

        df = self._apply_macro_tilt(df)

        # ── Distress penalty + hard exclusion (survivorship bias mitigation) ────
        # Stocks showing multiple financial distress signals get penalised.
        # This prevents a high-momentum distressed company from sneaking into
        # the top-10 and later going bankrupt / delisting.
        if "distress_flags" in df.columns:
            df["composite_score"] = (
                df["composite_score"] - df["distress_flags"] * 5.0
            ).clip(lower=0)
            # Hard-exclude maximum distress (3 flags = near-certain financial distress).
            # Soft penalty of -15pts is insufficient when all 5 danger signals fire.
            df = df[df["distress_flags"] < 3].reset_index(drop=True)

        if self.profile.income_focused:
            df["composite_score"] += df["dividend_score"] * 0.08
            df["composite_score"]  = df["composite_score"].clip(upper=100)

        if self.profile.drawdown_ok < 0.20:
            df["composite_score"] -= (100 - df["volatility_score"]) * 0.05
            df["composite_score"]  = df["composite_score"].clip(lower=0)

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    # ── Per-ticker raw score computation ──────────────────────────────────────
    def _score_one(self, ticker: str, data: dict) -> Optional[dict]:
        info    = data["info"]
        history = data["history"]
        sector  = data["sector"]
        close   = history["Close"].dropna()

        if len(close) < 63:
            return None

        price = float(close.iloc[-1])
        if price < 5:
            return None

        # ── 1. Momentum (12-1 academic grade) ────────────────────────────────
        r1m = self._ret(close, 21)
        r3m = self._ret(close, 63)
        r6m = self._ret(close, 126)

        # 12-1 skip-month momentum (avoids 1-month reversal, documented factor)
        r12_1 = None
        if len(close) >= 252:
            r12_1 = float(close.iloc[-21] / close.iloc[-252] - 1)

        if r12_1 is not None:
            # Academic-grade blend: heavier on medium/long with skip
            momentum_raw = (
                0.10 * (r1m or 0.0) +
                0.25 * (r3m or 0.0) +
                0.35 * (r6m or 0.0) +
                0.30 * r12_1
            )
        else:
            momentum_raw = (
                0.20 * (r1m or 0.0) +
                0.35 * (r3m or 0.0) +
                0.45 * (r6m or 0.0)
            )

        # ── 2. Volatility (inverted: low vol = high score) ────────────────────
        daily_ret    = close.pct_change().dropna()
        vol          = float(daily_ret.std()) * math.sqrt(252)
        volatility_raw = -vol

        # ── 3. Value — 3-signal composite (P/E + EV/EBITDA + FCF yield) ──────
        pe = info.get("trailingPE")
        if pe and 0 < float(pe) <= 1000:
            sp       = SECTOR_MEDIAN_PE.get(sector, 20)
            pe_score = max(-3.0, -(float(pe) / sp))
        else:
            pe_score = float("nan")

        # EV/EBITDA vs sector median
        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda and 0 < float(ev_ebitda) < 500:
            sp_ev    = SECTOR_EV_EBITDA.get(sector, 14)
            ev_score = max(-2.5, -(float(ev_ebitda) / sp_ev))
        else:
            ev_score = None

        # FCF yield
        fcf    = info.get("freeCashflow")
        mktcap = info.get("marketCap") or 1
        if fcf and mktcap and float(mktcap) > 0:
            fcf_ratio = float(fcf) / float(mktcap)
        else:
            fcf_ratio = None

        # Blend value signals (weighted by availability)
        val_components = []
        if not math.isnan(pe_score):
            val_components.append((pe_score, 0.40))
        if ev_score is not None:
            val_components.append((ev_score, 0.35))
        if fcf_ratio is not None:
            val_components.append((fcf_ratio, 0.25))

        if val_components:
            total_w  = sum(wt for _, wt in val_components)
            value_raw = sum(v * wt for v, wt in val_components) / total_w
        else:
            value_raw = float("nan")

        # ── 4. Quality — Piotroski + ROE/PM + accruals + gross profitability ─
        quality_raw = data.get("piotroski", 50.0) / 100

        roe = self._clamp(info.get("returnOnEquity"), -1.0, 1.0)
        pm  = self._clamp(info.get("profitMargins"),  -1.0, 1.0)
        extras = [v for v in [roe, pm] if v is not None]
        if extras:
            quality_raw = quality_raw * 0.60 + float(np.mean(extras)) * 0.40

        # Accruals quality adjustment (negative accruals = clean earnings = bonus)
        ni  = info.get("netIncomeToCommon") or info.get("netIncome")
        ocf = info.get("operatingCashflow")
        ta  = info.get("totalAssets")
        if ni and ocf and ta and float(ta) > 0:
            accruals = (float(ni) - float(ocf)) / float(ta)
            # Good: OCF >> NI (real earnings).  Bad: NI >> OCF (accounting tricks)
            quality_raw += self._clamp(-accruals * 1.5, -0.20, 0.20)

        # Gross profitability (Novy-Marx 2013 anomaly factor)
        rev = info.get("totalRevenue")
        gm  = info.get("grossMargins")
        if rev and gm and ta and float(ta) > 0:
            gp_ratio     = min(float(gm) * float(rev) / float(ta), 2.0)
            quality_raw  = quality_raw * 0.80 + gp_ratio * 0.10

        # ── 5. Technical (pre-computed with Bollinger + OBV) ──────────────────
        technical_raw = data.get("technical", 50.0) / 100

        # ── 6. Sentiment (pre-computed) ───────────────────────────────────────
        sentiment_raw = data.get("sentiment", 50.0) / 100

        # Analyst recommendation nudge
        rec = (info.get("recommendationKey") or "").lower()
        rec_adj = {"strong_buy": 0.15, "buy": 0.10, "hold": 0.02,
                   "sell": -0.15, "strong_sell": -0.25}.get(rec, 0)
        sentiment_raw = max(0.0, min(1.0, sentiment_raw + rec_adj))

        # Insider trading signal (Finnhub): net buy pressure boosts sentiment
        # +1 = all insiders buying  |  -1 = all insiders selling
        insider_signal = data.get("insider_signal", 0.0) or 0.0
        if insider_signal != 0.0:
            insider_adj   = insider_signal * 0.12          # max ±0.12 nudge
            sentiment_raw = max(0.0, min(1.0, sentiment_raw + insider_adj))

        # Earnings surprise signal (Finnhub): recent positive surprises → momentum boost
        eps_surprise = data.get("earnings_surprise_avg")
        if eps_surprise is not None:
            # Clamp to ±50% surprise, map to ±0.06 momentum nudge
            eps_adj = max(-0.06, min(0.06, float(eps_surprise) * 0.12))
            momentum_raw = momentum_raw + eps_adj

        # ── 7. Dividend ───────────────────────────────────────────────────────
        div = float(info.get("dividendYield", 0) or 0)
        div = min(div, 0.15)
        dividend_raw = div

        # Filter fields
        beta       = float(info.get("beta",      1.0) or 1.0)
        market_cap = float(info.get("marketCap", 0)   or 0)

        # ── Distress flags (count of danger signals) ──────────────────────────
        # Each flag = -5pts on composite (see score_all). Max penalty = -15 pts.
        # Addresses survivorship bias: identifies potential future delistings.
        distress = 0
        ocf = info.get("operatingCashflow")
        if ocf is not None and float(ocf) < 0:
            distress += 1                                          # burning cash
        ni = info.get("netIncomeToCommon") or info.get("netIncome")
        if ni is not None and float(ni) < 0:
            distress += 1                                          # losing money
        de = info.get("debtToEquity")
        if de is not None:
            de_r = float(de) / 100 if float(de) > 10 else float(de)
            if de_r > 4.0:
                distress += 1                                      # dangerously levered
        cr = info.get("currentRatio")
        if cr is not None and float(cr) < 1.0:
            distress += 1                                          # can't cover short-term debt
        fcf = info.get("freeCashflow")
        if fcf is not None and float(fcf) < 0:
            distress += 1                                          # negative free cash flow
        distress = min(distress, 3)   # cap at 3 flags = max -15 pt penalty

        return {
            "ticker":         ticker,
            "sector":         sector,
            "current_price":  price,
            "beta":           beta,
            "vol":            vol,
            "market_cap":     market_cap,
            "momentum_raw":   momentum_raw,
            "volatility_raw": volatility_raw,
            "value_raw":      value_raw,
            "quality_raw":    quality_raw,
            "technical_raw":  technical_raw,
            "sentiment_raw":  sentiment_raw,
            "dividend_raw":   dividend_raw,
            "div_pct":        div * 100,
            "distress_flags": distress,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _ret(self, close: pd.Series, days: int) -> Optional[float]:
        if len(close) > days:
            return float(close.iloc[-1] / close.iloc[-days] - 1)
        return None

    def _clamp(self, v, lo: float, hi: float) -> Optional[float]:
        if v is None:
            return None
        return max(lo, min(hi, float(v)))

    def _normalise(self, s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) == 0 or valid.max() == valid.min():
            return pd.Series(50.0, index=s.index)
        normed = (s - valid.min()) / (valid.max() - valid.min()) * 100
        return normed.fillna(50.0)

    def _filter(self, df: pd.DataFrame, strict: bool) -> pd.DataFrame:
        level = self.profile.risk_level
        if self.profile.excluded_sectors:
            df = df[~df["sector"].isin(self.profile.excluded_sectors)]
        if strict:
            if level == 1:
                df = df[(df["beta"] <= 1.8) & (df["vol"] <= 0.45)]
            elif level == 2:
                df = df[(df["beta"] <= 2.5) & (df["vol"] <= 0.65)]
        else:
            if level == 1:
                df = df[(df["beta"] <= 2.3) & (df["vol"] <= 0.60)]
            elif level == 2:
                df = df[df["beta"] <= 3.0]
        return df.copy().reset_index(drop=True)

    def _apply_macro_tilt(self, df: pd.DataFrame) -> pd.DataFrame:
        regime = self.macro_data.get("regime", "neutral")
        tilts  = dict(MACRO_TILTS.get(regime, {}))

        etf_perf = self.macro_data.get("sector_etf", {})
        if etf_perf:
            ranked     = sorted(etf_perf.items(), key=lambda x: x[1], reverse=True)
            for sector, _ in ranked[:3]:
                tilts[sector] = tilts.get(sector, 0) + 3

        for sector, adj in tilts.items():
            mask = df["sector"] == sector
            df.loc[mask, "composite_score"] = (
                df.loc[mask, "composite_score"] + adj
            ).clip(0, 100)

        return df

    def _get_weights(self) -> List[float]:
        if self.learned_weights is not None:
            return self.learned_weights
        return WEIGHT_MATRIX[(self.profile.risk_level, self.profile.time_horizon)]
