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

        # Short interest nudge: >15% short float signals squeeze potential or
        # crowded short confirming weakness depending on momentum direction
        short_pct = float(info.get("shortPercentOfFloat") or 0)
        if short_pct > 0.15:
            if momentum_raw > 0:
                momentum_raw += 0.05   # squeeze fuel
            else:
                momentum_raw -= 0.05   # short interest confirming weakness

        # Relative sector strength: outperformance vs sector ETF → momentum boost
        sector_etf_perf = self.macro_data.get("sector_etf", {})
        sector_etf_3m   = sector_etf_perf.get(sector)
        if sector_etf_3m is not None and r3m is not None:
            relative      = r3m - sector_etf_3m / 100.0
            momentum_raw += max(-0.08, min(0.08, relative * 0.20))

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

        # ── Source 1a: Revenue trend (QoQ acceleration) ───────────────────────
        revenue_trend = data.get("revenue_trend")
        if revenue_trend is not None:
            rv = float(revenue_trend)
            # +10% QoQ → strong; -10% → penalty. Max ±0.10 on quality, ±0.04 on momentum
            rev_adj = max(-0.10, min(0.10, rv * 0.60))
            quality_raw  += rev_adj * 0.50
            momentum_raw += max(-0.04, min(0.04, rv * 0.18))

        # ── Source 1b: EPS beat rate (earnings momentum) ───────────────────────
        earnings_beat_rate = data.get("earnings_beat_rate")
        if earnings_beat_rate is not None:
            br = float(earnings_beat_rate)
            # 75%+ beat rate → consistent outperformer; <25% → recurring disappointments
            beat_adj = (br - 0.50) * 0.20   # range: −0.10 to +0.10
            quality_raw  += beat_adj * 0.60
            momentum_raw += beat_adj * 0.40

        # ── Source 1c: Institutional ownership (smart-money validation) ───────
        institutional_pct = data.get("institutional_pct")
        if institutional_pct is not None:
            ip = float(institutional_pct)
            # >70% = heavily validated; <20% = underfollowed. Range: −0.05 to +0.06
            inst_adj = max(-0.05, min(0.06, (ip - 0.45) * 0.12))
            quality_raw += inst_adj

        # ── 5. Technical (pre-computed with Bollinger + OBV) ──────────────────
        technical_raw = data.get("technical", 50.0) / 100

        # ── 6. Sentiment — 3-source composite (news + insider + analyst) ────────
        news_score    = float(data.get("sentiment",     50.0))
        insider_score = float(data.get("insider_score", 50.0))

        # Analyst score: rec key (40%) + target upside (40%) + coverage breadth (20%)
        rec_map = {"strong_buy": 90, "buy": 75, "hold": 50, "sell": 25, "strong_sell": 10}
        rec       = (info.get("recommendationKey") or "").lower()
        rec_score = rec_map.get(rec, 50)
        cur_p     = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        tgt_p     = float(info.get("targetMeanPrice") or 0)
        upside_sc = float(max(0, min(100, 50 + (tgt_p - cur_p) / max(cur_p, 1) * 200))) if cur_p and tgt_p else 50.0
        n_ana     = float(info.get("numberOfAnalystOpinions") or 0)
        cov_sc    = float(max(10, min(100, n_ana / 20 * 100)))
        analyst_score = 0.40 * rec_score + 0.40 * upside_sc + 0.20 * cov_sc

        # Tier 1 weighted composite (0–1): news 45%, insider 35%, analyst 20%
        sentiment_raw = (
            0.45 * news_score +
            0.35 * insider_score +
            0.20 * analyst_score
        ) / 100

        sentiment_raw = max(0.0, min(1.0, sentiment_raw))

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
