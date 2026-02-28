# advisor/scorer.py — 7-factor MultiFactorScorer with macro regime tilt

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    WEIGHT_MATRIX, FACTOR_NAMES, SECTOR_MEDIAN_PE, MACRO_TILTS,
    HORIZON_LABELS,
)
from advisor.collector import UserProfile


class MultiFactorScorer:
    """
    Computes 7 factor scores per stock, normalises cross-sectionally,
    applies a user-profile weighted combination, then adds a macro regime tilt.

    Factors:
      1. momentum   — 1m/3m/6m price returns
      2. volatility — inverse annualised vol (lower = better)
      3. value      — P/E vs sector median + P/FCF proxy
      4. quality    — Piotroski 8-pt score (pre-computed in fetcher)
      5. technical  — RSI + MACD + MA crossover (pre-computed in fetcher)
      6. sentiment  — news headline keyword score (pre-computed in fetcher)
      7. dividend   — dividend yield
    """

    def __init__(self, profile: UserProfile, macro_data: dict,
                 learned_weights: Optional[List[float]] = None):
        self.profile        = profile
        self.macro_data     = macro_data
        self.learned_weights = learned_weights

    # ── Main entry ────────────────────────────────────────────────────────────
    def score_all(self, universe_data: Dict) -> pd.DataFrame:
        rows = [self._score_one(t, d) for t, d in universe_data.items()]
        rows = [r for r in rows if r is not None]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Apply pre-filter (beta / vol / price)
        df = self._filter(df, strict=True)
        if len(df) < 10:
            df_all = pd.DataFrame([r for r in rows if r is not None])
            df = self._filter(df_all, strict=False)

        # Honour preferred sectors — give them a flat bonus before normalising
        if self.profile.preferred_sectors:
            mask = df["sector"].isin(self.profile.preferred_sectors)
            df.loc[mask, "quality_raw"]   += 0.05   # small thumb on the scale
            df.loc[mask, "momentum_raw"]  += 0.02

        # Normalise each raw component to 0–100 cross-sectionally
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

        # Weights (learned override or default)
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

        # Macro regime tilt (additive, applied after normalisation)
        df = self._apply_macro_tilt(df)

        # Income-focused: boost dividend score contribution
        if self.profile.income_focused:
            df["composite_score"] += df["dividend_score"] * 0.08
            df["composite_score"]  = df["composite_score"].clip(upper=100)

        # Drawdown-sensitive: extra volatility penalty if user has low tolerance
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

        # ── 1. Momentum ───────────────────────────────────────────────────────
        r1m = self._ret(close, 21)
        r3m = self._ret(close, 63)
        r6m = self._ret(close, 126)
        momentum_raw = (
            0.20 * (r1m or 0.0) +
            0.35 * (r3m or 0.0) +
            0.45 * (r6m or 0.0)
        )

        # ── 2. Volatility ─────────────────────────────────────────────────────
        daily_ret    = close.pct_change().dropna()
        vol          = float(daily_ret.std()) * math.sqrt(252)
        volatility_raw = -vol   # negated: low vol → high raw → high score

        # ── 3. Value (P/E vs sector + P/FCF proxy) ───────────────────────────
        pe = info.get("trailingPE")
        if pe and 0 < float(pe) <= 1000:
            sp  = SECTOR_MEDIAN_PE.get(sector, 20)
            pe_score = max(-(float(pe) / sp), -3.0)
        else:
            pe_score = float("nan")

        # P/FCF proxy: use earningsYield (1/PE) adjusted by FCF vs net income
        fcf_yield = info.get("freeCashflow")
        mktcap    = info.get("marketCap") or 1
        if fcf_yield and mktcap:
            fcf_ratio = float(fcf_yield) / float(mktcap)  # positive = good
            pe_score  = (pe_score if not math.isnan(pe_score) else 0) * 0.7 + fcf_ratio * 0.3

        value_raw = pe_score

        # ── 4. Quality (Piotroski pre-computed in fetcher, scaled 0–100) ─────
        quality_raw = data.get("piotroski", 50.0) / 100   # keep in same scale as others

        # ROE + profit margin blend to enrich quality
        roe = self._clamp(info.get("returnOnEquity"), -1.0, 1.0)
        pm  = self._clamp(info.get("profitMargins"),  -1.0, 1.0)
        extras = [v for v in [roe, pm] if v is not None]
        if extras:
            quality_raw = quality_raw * 0.60 + float(np.mean(extras)) * 0.40

        # ── 5. Technical (pre-computed) ───────────────────────────────────────
        technical_raw = data.get("technical", 50.0) / 100

        # ── 6. Sentiment (pre-computed) ───────────────────────────────────────
        sentiment_raw = data.get("sentiment", 50.0) / 100

        # ── 7. Dividend ───────────────────────────────────────────────────────
        div = float(info.get("dividendYield", 0) or 0)
        div = min(div, 0.15)
        dividend_raw = div

        # Filter fields
        beta       = float(info.get("beta", 1.0) or 1.0)
        market_cap = float(info.get("marketCap", 0) or 0)

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
        # Always exclude excluded sectors
        if self.profile.excluded_sectors:
            df = df[~df["sector"].isin(self.profile.excluded_sectors)]
        # Apply beta / vol filters
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
        tilts  = MACRO_TILTS.get(regime, {})

        # Add sector-ETF momentum bonus (top 3 ETF-performing sectors get +3)
        etf_perf = self.macro_data.get("sector_etf", {})
        if etf_perf:
            ranked = sorted(etf_perf.items(), key=lambda x: x[1], reverse=True)
            top_sectors = {s for s, _ in ranked[:3]}
            for sector in top_sectors:
                if sector not in tilts:
                    tilts[sector] = 0
                tilts[sector] = tilts.get(sector, 0) + 3

        for sector, adj in tilts.items():
            mask = df["sector"] == sector
            df.loc[mask, "composite_score"] = (df.loc[mask, "composite_score"] + adj).clip(0, 100)

        return df

    def _get_weights(self) -> List[float]:
        if self.learned_weights is not None:
            return self.learned_weights
        return WEIGHT_MATRIX[(self.profile.risk_level, self.profile.time_horizon)]
