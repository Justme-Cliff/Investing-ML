"""
advisor/alternative_data.py — Tier 2 enrichment with alternative free data sources.

Runs on the top-30 stocks after initial scoring to add:
  - Options flow: put/call ratio + IV rank (yfinance option_chain)
  - Google Trends: 90-day retail search interest change (pytrends)
  - Reddit sentiment: r/stocks + r/investing RSS mention sentiment (feedparser)
  - Alpha Vantage: EPS surprise history
  - FMP: analyst revisions + financial rating + revenue growth
  - SEC EDGAR XBRL: revenue validation fallback
  - SEC Smart Money: Form 4 insider cluster + 8-K event sentiment (new)

None of these require paid API keys (optional keys enhance quality).

Usage (called from main.py and app.py after scorer.score_all()):
    from advisor.alternative_data import enrich_top_n
    ranked_df = enrich_top_n(ranked_df, universe_data, macro_data, n=30)
"""

import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import POSITIVE_WORDS, NEGATIVE_WORDS
# smart_money import is lazy (inside enrich_top_n) to prevent crash if module missing

# Estimated sentiment factor weight — used to translate sentiment delta → composite delta.
# Conservative: most profiles weight sentiment 4-7%; we use 6% to stay modest.
_SENTIMENT_W = 0.06


# ─────────────────────────────────────────────────────────────────────────────
# Options flow
# ─────────────────────────────────────────────────────────────────────────────

def fetch_options_data(ticker: str, hist_vol: float) -> dict:
    """
    Compute P/C ratio and options score from the nearest-expiry yfinance option chain.
    Also estimates IV rank using historical vol as a proxy for the 52w IV baseline.

    Returns dict with keys: pc_ratio, current_iv, options_score, iv_rank
    Returns {} on any failure (illiquid tickers, no options listed, etc.).
    """
    try:
        import yfinance as yf
        t        = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return {}

        chain = t.option_chain(expiries[0])
        calls = chain.calls
        puts  = chain.puts

        if len(calls) == 0:
            return {}

        # ── P/C ratio by open interest ────────────────────────────────────────
        call_oi = float(calls["openInterest"].sum() or 0)
        put_oi  = float(puts["openInterest"].sum()  or 0) if len(puts) > 0 else 0
        pc_ratio = put_oi / max(call_oi, 1)

        # ── Current IV (median of ATM calls) ──────────────────────────────────
        call_iv = calls["impliedVolatility"].dropna()
        put_iv  = puts["impliedVolatility"].dropna()  if len(puts) > 0 else call_iv
        current_iv = float((call_iv.median() + put_iv.median()) / 2) if (len(call_iv) > 0 and len(put_iv) > 0) else float(call_iv.median()) if len(call_iv) > 0 else 0.0

        # ── P/C score: low P/C = bullish; high P/C = hedging/bearish ──────────
        # pc_ratio = 0.5 neutral → 50pts; each 0.1 above 0.5 = −6pts
        pc_score = float(max(0, min(100, 100 - (pc_ratio - 0.5) * 60)))

        # ── IV premium vs hist vol ─────────────────────────────────────────────
        # Elevated IV → higher premium → caution flag
        iv_premium = current_iv - hist_vol
        iv_score   = float(max(0, min(100, 50 - iv_premium * 100)))

        options_score = 0.60 * pc_score + 0.40 * iv_score

        # ── IV rank proxy (0 = low fear, 1 = extreme fear) ────────────────────
        # True IV rank needs 52w IV history which yfinance doesn't provide.
        # Proxy: compare current IV to hist_vol * 1.15 (typical ATM IV ≈ 115% HV).
        # iv_rank > 0.70 = elevated; < 0.30 = complacent
        iv_rank = max(0.0, min(1.0, (current_iv / max(hist_vol * 1.15, 0.01) - 0.5) * 1.25))

        return {
            "pc_ratio":     round(pc_ratio, 3),
            "current_iv":   round(current_iv, 4),
            "options_score": round(options_score, 1),
            "iv_rank":      round(iv_rank, 3),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Google Trends
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_google_trends_batch(tickers: list) -> dict:
    """Fetch Google Trends for up to 5 tickers per request (batched). Returns {ticker: score}."""
    results_out: dict = {}
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        for i in range(0, len(tickers), 5):
            batch = tickers[i:i+5]
            try:
                pytrends.build_payload(batch, cat=0, timeframe="today 3-m", geo="US")
                interest = pytrends.interest_over_time()
                for kw in batch:
                    if kw in interest.columns:
                        vals = interest[kw].dropna()
                        if len(vals) >= 4:
                            ratio = float(vals.iloc[-4:].mean()) / max(float(vals.mean()), 1.0)
                            results_out[kw] = max(0.0, min(100.0, 50.0 + (ratio - 1.0) * 100.0))
                        else:
                            results_out[kw] = 50.0
                    else:
                        results_out[kw] = 50.0
                time.sleep(2)   # one sleep per BATCH of 5, not per ticker
            except Exception:
                for kw in batch:
                    results_out[kw] = 50.0
    except Exception:
        for t in tickers:
            results_out[t] = 50.0
    return results_out


def fetch_google_trends(ticker: str, company_name: str = "") -> float:
    """
    Fetch Google Trends interest for ticker over the last 3 months (US).
    Returns a 0–100 score where 50 = average; >50 = rising search interest.
    Falls back to 50.0 on throttle or any error.
    """
    try:
        from pytrends.request import TrendReq
        kw       = ticker.upper()
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload([kw], cat=0, timeframe="today 3-m", geo="US")
        interest = pytrends.interest_over_time()
        if interest.empty or kw not in interest.columns:
            return 50.0
        vals = interest[kw].dropna()
        if len(vals) < 4:
            return 50.0
        # Compare most-recent 4 weeks to overall 3m average
        recent_avg  = float(vals.iloc[-4:].mean())
        overall_avg = float(vals.mean())
        if overall_avg < 1:
            return 50.0
        ratio = recent_avg / overall_avg  # > 1.0 = rising interest
        score = max(0.0, min(100.0, 50.0 + (ratio - 1.0) * 100.0))
        time.sleep(2)                     # be polite — avoid 429s
        return round(score, 1)
    except Exception:
        return 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Reddit sentiment
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reddit_sentiment(ticker: str) -> float:
    """
    Scrape recent posts from r/stocks and r/investing RSS that mention the ticker.
    Returns a 0–100 sentiment score (50 = neutral). Falls back to 50.0 on error.
    """
    try:
        import feedparser
        score_total   = 0.0
        mention_count = 0

        for sub in ("stocks", "investing"):
            url  = (
                f"https://www.reddit.com/r/{sub}/search.rss"
                f"?q={ticker}&sort=new&restrict_sr=1&limit=10"
            )
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = getattr(entry, "title", "").lower()
                if ticker.lower() not in title:
                    continue
                mention_count += 1
                pos = sum(1 for w in POSITIVE_WORDS if w in title)
                neg = sum(1 for w in NEGATIVE_WORDS if w in title)
                score_total += pos - neg

        if mention_count == 0:
            return 50.0
        avg   = score_total / mention_count
        score = max(0.0, min(100.0, 50.0 + avg * 15.0))
        return round(score, 1)
    except Exception:
        return 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Source 4: Alpha Vantage — EPS surprise history
# ─────────────────────────────────────────────────────────────────────────────

def fetch_alpha_vantage_earnings(ticker: str) -> dict:
    """
    Source 4: Fetch detailed EPS surprise history from Alpha Vantage EARNINGS endpoint.
    Free tier: 25 calls/day (sufficient for top-30 enrichment with 12s delay).
    Requires ALPHAVANTAGE_KEY in config/.env — gracefully returns {} if absent.

    Returns:
        av_eps_beat_rate   (0–1)     fraction of last 4 quarters that beat estimates
        av_eps_surprise_avg (float)  average EPS surprise % (positive = beats)
    """
    try:
        from config import ALPHAVANTAGE_KEY
        if not ALPHAVANTAGE_KEY:
            return {}
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "EARNINGS",
                "symbol":   ticker,
                "apikey":   ALPHAVANTAGE_KEY,
            },
            timeout=12,
        )
        if r.status_code != 200:
            return {}
        quarterly = r.json().get("quarterlyEarnings", [])[:4]
        if not quarterly:
            return {}
        beats = 0
        valid_q = 0
        surprises = []
        for q in quarterly:
            try:
                reported = q.get("reportedEPS", "None")
                estimated = q.get("estimatedEPS", "None")
                surprise_pct = q.get("surprisePercentage", "None")
                if reported in (None, "None", "") or estimated in (None, "None", ""):
                    continue
                a, e = float(reported), float(estimated)
                if abs(e) > 0.001:
                    valid_q += 1
                    if a > e:
                        beats += 1
                    if surprise_pct not in (None, "None", ""):
                        surprises.append(float(surprise_pct))
            except Exception:
                pass
        out = {}
        if valid_q > 0:
            out["av_eps_beat_rate"] = beats / valid_q
        if surprises:
            out["av_eps_surprise_avg"] = float(np.mean(surprises))
        time.sleep(12)   # respect 25 calls/day free limit
        return out
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Source 5: Financial Modeling Prep — analyst revisions + financial rating
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fmp_data(ticker: str) -> dict:
    """
    Source 5: Fetch analyst estimate revisions and financial health from FMP.
    Free tier: 250 calls/day — plenty for top-30 enrichment.
    Requires FMP_KEY in config/.env — gracefully returns {} if absent.

    Returns:
        fmp_analyst_revision  (-1 to +1)  direction EPS estimates are moving
        fmp_rating_score      (0–100)     FMP composite financial health rating
        fmp_revenue_growth    (float)     YoY revenue growth from annual income stmt
    """
    try:
        from config import FMP_KEY
        if not FMP_KEY:
            return {}
        base = "https://financialmodelingprep.com/api/v3"
        out  = {}

        # ── Analyst estimate revisions ─────────────────────────────────────────
        try:
            r = requests.get(
                f"{base}/analyst-estimates/{ticker}",
                params={"limit": 4, "apikey": FMP_KEY},
                timeout=8,
            )
            if r.status_code == 200:
                est = r.json()
                if len(est) >= 2:
                    cur_eps  = float(est[0].get("estimatedEpsAvg") or 0)
                    prev_eps = float(est[1].get("estimatedEpsAvg") or 0)
                    if abs(prev_eps) > 0.01:
                        # Positive = analysts are RAISING estimates (bullish)
                        revision = (cur_eps - prev_eps) / abs(prev_eps)
                        out["fmp_analyst_revision"] = max(-1.0, min(1.0, revision))
        except Exception:
            pass

        # ── Financial health rating (1–5 scale → 0–100) ───────────────────────
        try:
            r = requests.get(
                f"{base}/rating/{ticker}",
                params={"apikey": FMP_KEY},
                timeout=8,
            )
            if r.status_code == 200:
                rat = r.json()
                if rat:
                    score = float(rat[0].get("ratingScore") or 3)
                    out["fmp_rating_score"] = (score / 5) * 100
        except Exception:
            pass

        # ── Revenue growth (annual income statement) ───────────────────────────
        try:
            r = requests.get(
                f"{base}/income-statement/{ticker}",
                params={"limit": 2, "period": "annual", "apikey": FMP_KEY},
                timeout=8,
            )
            if r.status_code == 200:
                inc = r.json()
                if len(inc) >= 2:
                    rev_cur  = float(inc[0].get("revenue") or 0)
                    rev_prev = float(inc[1].get("revenue") or 0)
                    if rev_prev > 0:
                        out["fmp_revenue_growth"] = (rev_cur - rev_prev) / rev_prev
        except Exception:
            pass

        time.sleep(0.5)   # polite — 250/day limit is generous
        return out
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Source 6: SEC EDGAR XBRL — revenue validation from 10-Q filings
# ─────────────────────────────────────────────────────────────────────────────

_SEC_CIK_CACHE: dict = {}   # module-level cache

def _get_sec_cik(ticker: str) -> Optional[str]:
    """Look up SEC CIK number for a ticker (cached after first download)."""
    global _SEC_CIK_CACHE
    if not _SEC_CIK_CACHE:
        try:
            r = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": "StockAdvisor contact@stockadvisor.local"},
                timeout=10,
            )
            if r.status_code == 200:
                _SEC_CIK_CACHE = {
                    v["ticker"].upper(): str(v["cik_str"])
                    for v in r.json().values()
                }
        except Exception:
            pass
    return _SEC_CIK_CACHE.get(ticker.upper())


def fetch_sec_revenue_trend(ticker: str) -> dict:
    """
    Source 6: Get quarterly revenue trend directly from SEC EDGAR XBRL API.
    No API key needed — completely free. Provides data straight from 10-Q filings,
    bypassing yfinance gaps.  Used only when yfinance revenue data is missing.

    Returns:
        sec_revenue_trend  (float | None)  QoQ revenue growth from last two 10-Q filings
        sec_revenue_ttm    (float | None)  trailing 12-month revenue (sum of last 4 qtrs)
    """
    try:
        cik = _get_sec_cik(ticker)
        if not cik:
            return {}
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json",
            headers={"User-Agent": "StockAdvisor contact@stockadvisor.local"},
            timeout=15,
        )
        if r.status_code != 200:
            return {}
        facts = r.json().get("facts", {}).get("us-gaap", {})
        out   = {}
        rev_keys = (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        )
        for rk in rev_keys:
            if rk not in facts:
                continue
            units = facts[rk].get("units", {}).get("USD", [])
            # Keep only 10-Q filings (quarterly) with a valid end date
            quarterly = [
                u for u in units
                if u.get("form") == "10-Q" and u.get("val") is not None
            ]
            quarterly.sort(key=lambda x: x.get("end", ""), reverse=True)
            if len(quarterly) >= 2:
                r_new = float(quarterly[0]["val"])
                r_old = float(quarterly[1]["val"])
                if r_old > 0:
                    out["sec_revenue_trend"] = (r_new - r_old) / r_old
            if len(quarterly) >= 4:
                out["sec_revenue_ttm"] = sum(
                    float(q["val"]) for q in quarterly[:4]
                )
            break   # found a valid revenue series
        return out
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 enrichment — top-N
# ─────────────────────────────────────────────────────────────────────────────

def enrich_top_n(
    ranked_df:     pd.DataFrame,
    universe_data: dict,
    macro_data:    dict,
    n:             int = 30,
) -> pd.DataFrame:
    """
    Tier 2 enrichment: runs options flow, Google Trends, and Reddit sentiment
    on the top-N stocks and updates composite_score + sentiment_score in ranked_df.

    Also stores iv_rank back into universe_data[ticker] so risk.py can use it.

    The composite_score delta is approximated as:
        Δcomposite ≈ Δsentiment_score × _SENTIMENT_W
    where _SENTIMENT_W = 0.06 (6% weight, conservative median across profiles).
    """
    if ranked_df.empty:
        return ranked_df

    top_tickers = ranked_df.head(n)["ticker"].tolist()
    total = len(top_tickers)
    print(f"\n  Enriching top {total} stocks with Tier 2 data (options · trends · AV · FMP · SEC · Smart Money)...")

    # Failure tracking — warn if >50% of tickers fail for a source
    _source_failures = {"options": 0, "trends": 0, "reddit": 0, "av": 0,
                        "fmp": 0, "sec": 0, "smart_money": 0}

    # Pre-fetch Google Trends in batches (one sleep per 5 tickers instead of per ticker)
    print("  Pre-fetching Google Trends (batched)...", end="\r", flush=True)
    _trends_batch = _fetch_google_trends_batch(top_tickers)

    # Build SEC CIK map once for smart money lookups.
    # Also populate the module-level _SEC_CIK_CACHE so that fetch_sec_revenue_trend()
    # can reuse it without making a second HTTP request to SEC.
    global _SEC_CIK_CACHE
    _cik_map: dict = {}
    if _SEC_CIK_CACHE:
        _cik_map = _SEC_CIK_CACHE   # already loaded from a prior call
    else:
        try:
            r_cik = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": "StockAdvisor contact@stockadvisor.local"},
                timeout=10,
            )
            if r_cik.status_code == 200:
                _cik_map = {
                    v["ticker"].upper(): str(v["cik_str"])
                    for v in r_cik.json().values()
                }
                _SEC_CIK_CACHE = _cik_map   # share with _get_sec_cik() — no re-download
        except Exception:
            pass

    def _bar(done: int, step: str = "", ticker: str = "") -> None:
        pct    = done / total * 100
        filled = int(pct / 5)
        bar    = "█" * filled + "░" * (20 - filled)
        label  = f"{ticker:<6}  {step}" if ticker else ""
        print(f"  [{bar}] {pct:4.0f}%  {done}/{total}  {label:<30}", end="\r", flush=True)

    for i, ticker in enumerate(top_tickers):
        data = universe_data.get(ticker)
        if not data:
            _bar(i + 1, "skip (no data)", ticker)
            continue

        hist  = data.get("history")
        info  = data.get("info", {})
        if hist is None or len(hist) < 20:
            _bar(i + 1, "skip (no hist)", ticker)
            continue

        close = hist["Close"].dropna()
        if len(close) < 20:
            _bar(i + 1, "skip (no hist)", ticker)
            continue

        hist_vol = float(close.pct_change().dropna().std()) * math.sqrt(252)

        # ── Source 3: Options ─────────────────────────────────────────────────
        _bar(i, "Options...    ", ticker)
        try:
            opt = fetch_options_data(ticker, hist_vol)
        except Exception:
            opt = {}
            _source_failures["options"] += 1
        opt_score = opt.get("options_score", 50.0)
        iv_rank   = opt.get("iv_rank")
        if iv_rank is not None:
            universe_data[ticker]["iv_rank"] = iv_rank   # for risk.py

        # ── Google Trends (from pre-fetched batch) ────────────────────────────
        _bar(i, "Trends...     ", ticker)
        trends_score = _trends_batch.get(ticker, 50.0)
        if trends_score == 50.0 and ticker not in _trends_batch:
            _source_failures["trends"] += 1

        # ── Reddit ────────────────────────────────────────────────────────────
        _bar(i, "Reddit...     ", ticker)
        try:
            reddit_score = fetch_reddit_sentiment(ticker)
        except Exception:
            reddit_score = 50.0
            _source_failures["reddit"] += 1
        retail_score = 0.60 * trends_score + 0.40 * reddit_score

        # ── Source 4: Alpha Vantage — EPS surprise history ────────────────────
        _bar(i, "Alpha Vantage...", ticker)
        try:
            av_data = fetch_alpha_vantage_earnings(ticker)
        except Exception:
            av_data = {}
            _source_failures["av"] += 1
        av_beat_rate = av_data.get("av_eps_beat_rate")
        av_surp_avg  = av_data.get("av_eps_surprise_avg")
        if av_data:
            universe_data[ticker]["av_data"] = av_data

        # ── Source 5: FMP — analyst revisions + financial rating ──────────────
        _bar(i, "FMP...        ", ticker)
        try:
            fmp_data = fetch_fmp_data(ticker)
        except Exception:
            fmp_data = {}
            _source_failures["fmp"] += 1
        fmp_revision = fmp_data.get("fmp_analyst_revision")    # −1 to +1
        fmp_rating   = fmp_data.get("fmp_rating_score")        # 0–100
        fmp_rev_gro  = fmp_data.get("fmp_revenue_growth")      # YoY %
        if fmp_data:
            universe_data[ticker]["fmp_data"] = fmp_data

        # ── Source 6: SEC EDGAR — revenue validation/fallback ────────────────
        # Only fetch from SEC if yfinance revenue is missing (saves bandwidth)
        _bar(i, "SEC EDGAR...  ", ticker)
        sec_data = {}
        try:
            if data.get("revenue_trend") is None and info.get("totalRevenue") is None:
                sec_data = fetch_sec_revenue_trend(ticker)
                if sec_data.get("sec_revenue_trend") is not None:
                    universe_data[ticker]["sec_revenue_trend"] = sec_data["sec_revenue_trend"]
        except Exception:
            _source_failures["sec"] += 1

        # ── Source 7: SEC Smart Money — Form 4 cluster + 8-K NLP ─────────────
        _bar(i, "Smart Money...", ticker)
        sm_data = {}
        try:
            from advisor.smart_money import fetch_smart_money as _fetch_smart_money
            sm_data = _fetch_smart_money(ticker, _cik_map)
            if sm_data:
                universe_data[ticker]["smart_money"] = sm_data
        except Exception:
            _source_failures["smart_money"] += 1

        # ── Build full 5-source sentiment (0–100) ─────────────────────────────
        news_score    = float(data.get("sentiment",     50.0))
        insider_score = float(data.get("insider_score", 50.0))

        # Analyst score — blend base rec with FMP revision if available
        rec_map = {"strong_buy": 90, "buy": 75, "hold": 50, "sell": 25, "strong_sell": 10}
        rec          = (info.get("recommendationKey") or "").lower()
        rec_score    = rec_map.get(rec, 50)
        cur          = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        tgt          = float(info.get("targetMeanPrice") or 0)
        upside_score = float(max(0, min(100, 50 + (tgt - cur) / max(cur, 1) * 200))) if cur and tgt else 50.0
        n_ana        = float(info.get("numberOfAnalystOpinions") or 0)
        cov_score    = float(max(10, min(100, n_ana / 20 * 100)))
        analyst_score = 0.40 * rec_score + 0.40 * upside_score + 0.20 * cov_score

        # Blend FMP analyst revision into the analyst component
        if fmp_revision is not None:
            revision_score = max(0.0, min(100.0, 50.0 + fmp_revision * 50.0))
            analyst_score  = analyst_score * 0.60 + revision_score * 0.40

        new_sentiment = (
            0.30 * news_score    +
            0.25 * insider_score +
            0.20 * analyst_score +
            0.15 * opt_score     +
            0.10 * retail_score
        )

        # ── Direct composite boosts from new sources ──────────────────────────
        # Source 4: earnings momentum (AV EPS beats)
        earnings_boost = 0.0
        if av_beat_rate is not None:
            # 75%+ beat rate → +3 pts; 25% → -3 pts
            earnings_boost += (float(av_beat_rate) - 0.50) * 6.0
        if av_surp_avg is not None:
            # Average EPS surprise >5% → small additional boost (max ±2 pts)
            earnings_boost += max(-2.0, min(2.0, float(av_surp_avg) * 0.05))

        # Source 5: FMP financial rating + revenue growth
        rating_boost = 0.0
        if fmp_rating is not None:
            rating_boost = (float(fmp_rating) - 50.0) * 0.04   # ±2 pts max
        rev_boost = 0.0
        if fmp_rev_gro is not None:
            rev_boost = max(-2.0, min(2.0, float(fmp_rev_gro) * 5.0))

        # Source 6: SEC-derived revenue trend bonus (if yfinance was missing it)
        sec_boost = 0.0
        if sec_data.get("sec_revenue_trend") is not None:
            rv = float(sec_data["sec_revenue_trend"])
            sec_boost = max(-1.5, min(1.5, rv * 4.0))

        # Source 7: SEC smart money boost
        # Form 4 cluster signal: 3+ insiders buying → +2.5 pts max
        # 8-K event quality: positive catalyst items → up to +2 pts
        smart_boost = 0.0
        if sm_data:
            sm_score = sm_data.get("smart_money_score")
            if sm_score is not None:
                # 50 = neutral, 100 = maximum bullish → ±3 pts
                smart_boost = max(-3.0, min(3.0, (float(sm_score) - 50.0) * 0.06))

        # ── Apply all deltas to composite_score ───────────────────────────────
        mask = ranked_df["ticker"] == ticker
        if not mask.any():
            continue
        idx = ranked_df.index[mask][0]

        old_sentiment = float(ranked_df.loc[idx, "sentiment_score"])
        sentiment_delta = (new_sentiment - old_sentiment) * _SENTIMENT_W
        total_boost = earnings_boost + rating_boost + rev_boost + sec_boost + smart_boost

        ranked_df.loc[idx, "sentiment_score"] = round(new_sentiment, 2)
        ranked_df.loc[idx, "composite_score"] = float(max(
            0.0, min(100.0,
                ranked_df.loc[idx, "composite_score"] + sentiment_delta + total_boost
            )
        ))

        # Tick progress bar after fully completing this ticker
        _bar(i + 1, "done          ", ticker)

    print(f"  [{'█' * 20}] 100%  {total}/{total}  Tier 2 enrichment complete." + " " * 10)

    # Print warnings for sources with >50% failure rate
    for src, count in _source_failures.items():
        if total > 0 and count > total / 2:
            print(f"  [!] Tier 2 warning: '{src}' failed for {count}/{total} tickers — check network/API key.")

    # Re-rank
    ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = range(1, len(ranked_df) + 1)
    return ranked_df
