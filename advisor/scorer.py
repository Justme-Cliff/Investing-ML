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
    def score_all(self, universe_data: Dict, sp500_hist=None) -> pd.DataFrame:
        # ── Precompute market returns for Jensen's alpha ───────────────────────
        # Allows _score_one() to reward stocks that beat the market after
        # accounting for their beta — true alpha, not just market riding.
        self._market_returns: dict = {}
        if sp500_hist is not None:
            try:
                if isinstance(sp500_hist.columns, pd.MultiIndex):
                    sp_close = sp500_hist.xs("Close", axis=1, level=0).squeeze().dropna()
                else:
                    sp_close = sp500_hist["Close"].dropna()
                if hasattr(sp_close.index, "tz") and sp_close.index.tz is not None:
                    sp_close.index = sp_close.index.tz_localize(None)
                for days in [21, 63, 126, 252]:
                    if len(sp_close) > days:
                        self._market_returns[days] = float(
                            sp_close.iloc[-1] / sp_close.iloc[-days] - 1
                        )
            except Exception:
                self._market_returns = {}

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

        # ── GARP interaction term (Asness 1997) ──────────────────────────────
        # Growth at a Reasonable Price: stocks that are simultaneously high-quality
        # AND high-momentum beat the market more than either factor alone.
        # The interaction captures non-linearity that linear factor models miss.
        # Bonus scales from 0 to +15 pts at the theoretical max (100/100 × 100/100).
        # Implemented AFTER normalization so scores are already cross-sectional.
        if "momentum_score" in df.columns and "quality_score" in df.columns:
            garp_bonus = (df["momentum_score"] / 100.0) * (df["quality_score"] / 100.0) * 15.0
            df["composite_score"] = (df["composite_score"] + garp_bonus).clip(upper=100)

        # ── Value × Quality interaction (Cheap + Quality = persistent alpha) ─
        # Asness, Frazzini & Pedersen (2019): cheap stocks with strong quality
        # metrics produce persistent excess returns beyond either factor alone.
        # GARP above captures momentum×quality; this captures the orthogonal
        # cheap×quality signal (different economic rationale: margin of safety).
        # Bonus 0–+10 pts, so it's additive with GARP but smaller in magnitude.
        if "value_score" in df.columns and "quality_score" in df.columns:
            vq_bonus = (df["value_score"] / 100.0) * (df["quality_score"] / 100.0) * 10.0
            df["composite_score"] = (df["composite_score"] + vq_bonus).clip(upper=100)

        # ── Factor crowding detection ─────────────────────────────────────────
        # When a single factor score correlates >0.92 with composite_score, the
        # top portfolio is effectively a pure single-factor bet — crowded and
        # arbitrage-prone. Stocks that score high on the crowded factor but
        # mediocre across all other factors get a small penalty to push the
        # selection toward genuinely multi-factor names.
        _score_cols = [
            "momentum_score", "volatility_score", "value_score",
            "quality_score",  "technical_score",  "sentiment_score", "dividend_score",
        ]
        _crowded_factor: Optional[str] = None
        for _col in _score_cols:
            if _col in df.columns and len(df) > 20:
                try:
                    if abs(df["composite_score"].corr(df[_col])) > 0.92:
                        _crowded_factor = _col
                        break
                except Exception:
                    pass

        if _crowded_factor is not None:
            _other_cols = [c for c in _score_cols if c != _crowded_factor and c in df.columns]
            if _other_cols:
                _other_mean   = df[_other_cols].mean(axis=1)
                _low_thresh   = _other_mean.quantile(0.33)
                _top_in_crowd = df[_crowded_factor] > df[_crowded_factor].quantile(0.67)
                _weak_else    = _other_mean < _low_thresh
                _crowded_mask = _top_in_crowd & _weak_else
                df.loc[_crowded_mask, "composite_score"] = (
                    df.loc[_crowded_mask, "composite_score"] - 4.0
                ).clip(lower=0)

        # ── Data quality penalty ──────────────────────────────────────────────
        # Stocks with < 60% of key fundamental fields populated are penalised.
        # Missing data inflates/deflates factor scores unpredictably; this
        # prevents data-sparse stocks from accidentally reaching the top-15.
        # Penalty: (60 − score) × 0.20 pts. At 0/100 → −12 pts; at 60+ → 0.
        if "data_quality_score" in df.columns:
            _dq_low = df["data_quality_score"] < 60
            if _dq_low.any():
                _dq_pen = (60 - df.loc[_dq_low, "data_quality_score"]) * 0.20
                df.loc[_dq_low, "composite_score"] = (
                    df.loc[_dq_low, "composite_score"] - _dq_pen
                ).clip(lower=0)

        df = self._apply_macro_tilt(df)

        # ── Regime-aware beta penalty ─────────────────────────────────────────
        # In defensive regimes, high-beta stocks are doubly dangerous: they
        # amplify market losses AND have wider idiosyncratic swings.
        # This penalty ensures the scorer selects stocks that can hold up in
        # downturns — not just momentum darlings tied to the index direction.
        #
        # Penalty scale (pts lost per unit of beta above 1.0):
        #   crisis / pre_crisis → −12 / −8   (market likely rolling over)
        #   risk_off / neutral  → −5 / −2    (cautious positioning)
        #   risk_on             →  0          (let winners run)
        _beta_coeffs = {
            "crisis":      12.0,
            "pre_crisis":   8.0,
            "risk_off":     5.0,
            "neutral":      2.0,
            "rising_rate":  2.0,
            "falling_rate": 0.0,
            "risk_on":      0.0,
        }
        _beta_coeff = _beta_coeffs.get(self.macro_data.get("regime", "neutral"), 2.0)
        if _beta_coeff > 0 and "beta" in df.columns:
            excess_beta = (df["beta"] - 1.0).clip(lower=0.0)
            df["composite_score"] = (
                df["composite_score"] - excess_beta * _beta_coeff
            ).clip(lower=0)

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

        # Extract beta early — needed for volatility factor and Jensen's alpha
        beta_raw = max(0.1, min(float(info.get("beta", 1.0) or 1.0), 4.0))

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

        # Jensen's alpha: excess return vs what the stock's beta predicts.
        # alpha = stock_return − beta × market_return
        # Positive alpha = stock beat the market after adjusting for market risk.
        # This rewards genuine outperformers rather than stocks that merely
        # ride the market wave with high beta.
        _mr = self._market_returns
        if _mr:
            if r3m is not None and 63 in _mr and _mr[63] != 0:
                alpha_3m      = r3m - beta_raw * _mr[63]
                momentum_raw += max(-0.05, min(0.05, alpha_3m * 0.20))
            if r6m is not None and 126 in _mr and _mr[126] != 0:
                alpha_6m      = r6m - beta_raw * _mr[126]
                momentum_raw += max(-0.05, min(0.05, alpha_6m * 0.15))

        # Forward fundamental alpha: analyst underestimation pattern.
        # Jensen's alpha above is backward-looking (past price vs past market).
        # This is forward-looking: when a company CONSISTENTLY beats estimates
        # by a WIDE margin, analyst models are structurally too pessimistic —
        # and structural biases persist. Future beats → future re-ratings → alpha.
        # Only fires when BOTH rate (chronic) AND magnitude (meaningful) align,
        # filtering out lucky one-off beats from persistent edge.
        _beat_rate = data.get("earnings_beat_rate")
        _eps_surp  = data.get("earnings_surprise_avg")
        if _beat_rate is not None and _eps_surp is not None:
            _br, _es = float(_beat_rate), float(_eps_surp)
            if _br > 0.65 and _es > 2.0:
                # Analysts systematically too low → keep beating → sustained alpha
                _fwd = min(0.05, (_br - 0.50) * _es / 150.0)
                momentum_raw += _fwd
            elif _br < 0.35 and _es < -2.0:
                # Analysts systematically too high → chronic disappointments ahead
                _fwd = -min(0.05, (0.50 - _br) * abs(_es) / 150.0)
                momentum_raw += _fwd

        # 52-Week High Momentum (George & Hwang 2004)
        # Stocks trading near their 52-week high continue outperforming.
        # Investors anchor to the 52w high as a reference, causing underreaction
        # that resolves in sustained continuation when the level is approached.
        # Signal is entirely forward-looking: today's nearness predicts next 6-12M.
        high_52w = info.get("fiftyTwoWeekHigh")
        if high_52w and float(high_52w) > 0:
            nearness     = min(1.0, price / float(high_52w))
            wh_adj       = (nearness - 0.75) * 0.20   # neutral at 75%; ±0.05 range
            momentum_raw += max(-0.05, min(0.05, wh_adj))

        # ── 2. Volatility + Beta Risk (inverted: low = high score) ──────────
        # Blends realised idiosyncratic volatility with systematic market risk.
        # A high-beta stock that happens to be calm still carries market exposure,
        # so both dimensions penalise the score — favouring stocks that are
        # genuinely less risky, not just lucky in a quiet period.
        daily_ret      = close.pct_change().dropna()
        vol            = float(daily_ret.std()) * math.sqrt(252)
        volatility_raw = -(vol * 0.60 + max(0.0, beta_raw - 0.5) * 0.10)

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

        # Shareholder yield: dividends + share buybacks (Boudoukh et al.)
        # Total cash returned to shareholders is a stronger value signal than
        # dividend yield alone — buybacks are tax-efficient and signal mgmt
        # confidence. Combined yield predicts returns better than P/E alone.
        div_yield_raw  = float(info.get("dividendYield", 0) or 0)
        buyback_y      = float(data.get("buyback_yield") or 0)
        shareholder_yield = min(0.20, div_yield_raw + buyback_y)   # cap at 20%

        # Blend value signals (weighted by availability)
        val_components = []
        if not math.isnan(pe_score):
            val_components.append((pe_score, 0.40))
        if ev_score is not None:
            val_components.append((ev_score, 0.35))
        if fcf_ratio is not None:
            val_components.append((fcf_ratio, 0.25))
        if shareholder_yield > 0:
            val_components.append((shareholder_yield, 0.20))

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

            # ── Beneish M-Score (simplified 2-factor fraud detection) ─────────
            # Beneish (1999): M > -1.78 flags likely earnings manipulation.
            # Two strongest components available from yfinance data:
            #   TATA = Total Accruals / Total Assets (= accruals, already computed)
            #   SGI  = Sales Growth Index (1 + revenue_growth_rate)
            # M ≈ 4.679×TATA + 0.892×SGI − 3.0 (partial approx, others zeroed at mean)
            # This catches Enron-style stocks: earnings not backed by cash AND
            # aggressive revenue growth (two of the strongest manipulation signals).
            try:
                _rev_g    = float(info.get("revenueGrowth") or data.get("revenue_trend") or 0.0)
                _sgi      = 1.0 + _rev_g
                _m_approx = 4.679 * accruals + 0.892 * _sgi - 3.0
                if _m_approx > -1.78:   # above manipulation threshold
                    _beneish_penalty = min(0.12, (_m_approx + 1.78) * 0.07)
                    quality_raw -= _beneish_penalty
            except Exception:
                pass

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

        # ── Source 1d: Asset growth penalty (Cooper et al. 2008) ────────────
        # Companies aggressively expanding total assets destroy return on capital.
        # Normal organic growth (≤15% YoY) is fine and gets no penalty.
        # Above 15% the penalty scales linearly — capped at −0.10.
        asset_growth = data.get("asset_growth")
        if asset_growth is not None:
            ag = float(asset_growth)
            if ag > 0.15:
                quality_raw -= min(0.10, (ag - 0.15) * 0.25)

        # ── Source 1f: EPS consistency (consecutive years of growth) ─────────
        # Consecutive EPS growth is one of the cleanest proxies for a durable
        # competitive advantage. 4 straight years = maximum bonus (+0.08).
        # This rewards compounders that grow earnings through full market cycles.
        eps_cons = data.get("eps_consistency")
        if eps_cons is not None:
            quality_raw += min(0.08, int(eps_cons) / 4.0 * 0.08)

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
        beta       = beta_raw
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
            "ticker":              ticker,
            "sector":              sector,
            "current_price":       price,
            "beta":                beta,
            "vol":                 vol,
            "market_cap":          market_cap,
            "momentum_raw":        momentum_raw,
            "volatility_raw":      volatility_raw,
            "value_raw":           value_raw,
            "quality_raw":         quality_raw,
            "technical_raw":       technical_raw,
            "sentiment_raw":       sentiment_raw,
            "dividend_raw":        dividend_raw,
            "div_pct":             div * 100,
            "distress_flags":      distress,
            "data_quality_score":  float(data.get("data_quality_score", 100.0)),
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
        """
        Rank-percentile normalization (replaces min-max).

        Maps each value to its cross-sectional rank expressed as a 0–100 percentile.
        Robust to extreme outliers: a single stock with a wildly high P/E can't
        compress every other stock into a narrow band near 0, which min-max would do.
        Ensures uniform score distribution across the 0–100 range.
        """
        n = s.notna().sum()
        if n == 0:
            return pd.Series(50.0, index=s.index)
        if n == 1:
            return s.notna().astype(float) * 50.0 + 25.0
        # Rank: 1 = lowest, n = highest; ties averaged
        ranks  = s.rank(method="average", na_option="keep")
        normed = (ranks - 1) / max(n - 1, 1) * 100
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
