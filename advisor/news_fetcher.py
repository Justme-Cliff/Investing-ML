"""
advisor/news_fetcher.py — Multi-source free news aggregation

Sources (all free, no paid subscription required):
  1. Yahoo Finance via yfinance             — always available, no key
  2. Yahoo Finance RSS feed                 — no key, uses feedparser
  3. Finnhub.io company news               — optional free key (60 calls/min)
  4. NewsAPI.org                           — optional free key (100 req/day)
  5. FRED macro commentary (via FRED API)  — optional free key

Usage:
    nf       = NewsFetcher()
    articles = nf.fetch_ticker_news("AAPL", n=15)
    score    = nf.score_sentiment([a["title"] for a in articles])
"""

import datetime
import os
from typing import List, Dict, Optional

import requests
import yfinance as yf

from config import FINNHUB_KEY, NEWSAPI_KEY, POSITIVE_WORDS, NEGATIVE_WORDS


# ── RSS feed templates ─────────────────────────────────────────────────────────
_YF_RSS        = "https://finance.yahoo.com/rss/headline?s={ticker}&region=US&lang=en-US"
_REUTERS       = "https://feeds.reuters.com/reuters/businessNews"
_AP_BUSINESS   = "https://feeds.apnews.com/apnews/business"
_MARKETWATCH   = "https://feeds.marketwatch.com/marketwatch/topstories"
_BENZINGA      = "https://www.benzinga.com/feed"

# Phrases that negate the keyword appearing immediately after
_NEGATION_PHRASES = (
    "not ", "no ", "never ", "fails to", "unable to", "failed to",
    "didn't ", "doesn't ", "won't ", "cannot ", "can't ", "no longer",
)

# Clause boundaries that reset negation scope (negation can't cross these)
_CLAUSE_BOUNDARIES = (",", ";", " but ", " and ", " however ", " yet ", " while ")


def _is_negated(text_before: str) -> bool:
    """
    Return True if a negation phrase appears in the 50 chars before a keyword
    AND no clause boundary (comma, 'and', 'but', etc.) separates the negation
    from the keyword.  This prevents 'not X and Y' from negating Y.
    """
    snippet = text_before[-50:].lower() if len(text_before) > 50 else text_before.lower()
    for neg in _NEGATION_PHRASES:
        idx = snippet.rfind(neg)     # last occurrence in window
        if idx >= 0:
            after_neg = snippet[idx + len(neg):]
            if not any(b in after_neg for b in _CLAUSE_BOUNDARIES):
                return True
    return False


def _score_headline(title: str) -> float:
    """
    Score a single headline with negation detection.
    Returns a raw score (positive = bullish, negative = bearish).
    """
    t = title.lower()
    score = 0.0
    for kw in POSITIVE_WORDS:
        if kw in t:
            idx = t.find(kw)
            score += -1.0 if _is_negated(t[:idx]) else 1.0
    for kw in NEGATIVE_WORDS:
        if kw in t:
            idx = t.find(kw)
            score += 1.0 if _is_negated(t[:idx]) else -1.0
    return score


class NewsFetcher:
    """
    Aggregates news from multiple free sources, deduplicates by title,
    and returns sorted (newest-first) lists of article dicts.
    """

    _FINNHUB_BASE = "https://finnhub.io/api/v1"
    _NEWSAPI_BASE = "https://newsapi.org/v2"

    def __init__(self):
        self._fh_key  = FINNHUB_KEY  or os.environ.get("FINNHUB_KEY",  "")
        self._na_key  = NEWSAPI_KEY  or os.environ.get("NEWSAPI_KEY",  "")

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_ticker_news(self, ticker: str, n: int = 15) -> List[Dict]:
        """
        Return up to `n` articles for `ticker`, newest first.
        Each dict: { title, source, url, published, summary, sentiment_hint }
        """
        articles: List[Dict] = []
        seen:     set         = set()

        def _add(batch):
            for a in batch:
                key = (a.get("title") or "")[:80].lower()
                if key and key not in seen:
                    seen.add(key)
                    articles.append(a)

        _add(self._yf_news(ticker))
        _add(self._rss_news(ticker))
        _add(self._ap_news(ticker))
        _add(self._marketwatch_news(ticker))
        _add(self._benzinga_news(ticker))
        if self._fh_key:
            _add(self._finnhub_news(ticker))
        if self._na_key and len(articles) < n:
            _add(self._newsapi_news(ticker))

        articles.sort(key=lambda x: x.get("published", ""), reverse=True)
        # Tag each article with a quick sentiment hint
        for a in articles:
            a["sentiment_hint"] = self._classify(a.get("title", ""))
        return articles[:n]

    def fetch_market_news(self, n: int = 20) -> List[Dict]:
        """General market/business news (not ticker-specific)."""
        articles: List[Dict] = []
        seen:     set         = set()
        try:
            import feedparser
            feed = feedparser.parse(_REUTERS)
            for entry in feed.entries[:n]:
                title = getattr(entry, "title", "")
                key   = title[:80].lower()
                if key and key not in seen:
                    seen.add(key)
                    articles.append({
                        "title":          title,
                        "source":         "Reuters",
                        "url":            getattr(entry, "link", ""),
                        "published":      getattr(entry, "published", ""),
                        "summary":        getattr(entry, "summary", "")[:200],
                        "sentiment_hint": self._classify(title),
                    })
        except Exception:
            pass
        return articles[:n]

    def score_sentiment(self, headlines: List[str]) -> float:
        """
        Score a list of headlines 0–100.
        Uses negation detection and recency weighting (articles are assumed newest-first).
        Up to 20 articles scored; recent ones weighted 1.5× (≤3d) or 1.2× (≤7d).
        """
        if not headlines:
            return 50.0
        today = datetime.date.today()
        total_score  = 0.0
        total_weight = 0.0
        for i, h in enumerate(headlines[:20]):
            # Approximate recency: first items in list are newest
            # Index-based weight: 0-2 → 1.5×, 3-6 → 1.2×, older → 1.0×
            weight = 1.5 if i < 3 else (1.2 if i < 7 else 1.0)
            total_score  += _score_headline(h) * weight
            total_weight += weight
        if total_weight == 0:
            return 50.0
        raw = total_score / total_weight      # typically in [-3, +3]
        normalised = (raw + 3) / 6 * 100
        return round(min(100.0, max(0.0, normalised)), 1)

    # ── Sources ────────────────────────────────────────────────────────────────

    def _yf_news(self, ticker: str) -> List[Dict]:
        try:
            raw = yf.Ticker(ticker).news or []
            out = []
            for item in raw[:12]:
                title = item.get("title", "")
                if not title:
                    continue
                ts  = item.get("providerPublishTime", 0)
                pub = (datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                       if ts else "")
                out.append({
                    "title":     title,
                    "source":    item.get("publisher", "Yahoo Finance"),
                    "url":       item.get("link", ""),
                    "published": pub,
                    "summary":   item.get("summary", "")[:200],
                })
            return out
        except Exception:
            return []

    def _rss_news(self, ticker: str) -> List[Dict]:
        try:
            import feedparser
            url  = _YF_RSS.format(ticker=ticker)
            feed = feedparser.parse(url)
            out  = []
            for entry in feed.entries[:10]:
                title = getattr(entry, "title", "")
                if not title:
                    continue
                out.append({
                    "title":     title,
                    "source":    "Yahoo Finance RSS",
                    "url":       getattr(entry, "link", ""),
                    "published": getattr(entry, "published", ""),
                    "summary":   getattr(entry, "summary", "")[:200],
                })
            return out
        except Exception:
            return []

    def _finnhub_news(self, ticker: str) -> List[Dict]:
        try:
            today = datetime.date.today()
            since = (today - datetime.timedelta(days=7)).isoformat()
            to_   = today.isoformat()
            resp  = requests.get(
                f"{self._FINNHUB_BASE}/company-news",
                params={"symbol": ticker, "from": since, "to": to_,
                        "token": self._fh_key},
                timeout=8,
            )
            resp.raise_for_status()
            out = []
            for item in resp.json()[:12]:
                title = item.get("headline", "")
                if not title:
                    continue
                ts  = item.get("datetime", 0)
                pub = (datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                       if ts else "")
                out.append({
                    "title":     title,
                    "source":    item.get("source", "Finnhub"),
                    "url":       item.get("url", ""),
                    "published": pub,
                    "summary":   item.get("summary", "")[:200],
                })
            return out
        except Exception:
            return []

    def _newsapi_news(self, ticker: str) -> List[Dict]:
        try:
            resp = requests.get(
                f"{self._NEWSAPI_BASE}/everything",
                params={
                    "q":        f'"{ticker}" stock',
                    "language": "en",
                    "sortBy":   "publishedAt",
                    "pageSize": 10,
                    "apiKey":   self._na_key,
                },
                timeout=8,
            )
            resp.raise_for_status()
            out = []
            for item in resp.json().get("articles", []):
                title = item.get("title", "")
                if not title or "[Removed]" in title:
                    continue
                pub = (item.get("publishedAt", "") or "")[:16].replace("T", " ")
                out.append({
                    "title":     title,
                    "source":    item.get("source", {}).get("name", "NewsAPI"),
                    "url":       item.get("url", ""),
                    "published": pub,
                    "summary":   (item.get("description") or "")[:200],
                })
            return out
        except Exception:
            return []

    def _ap_news(self, ticker: str) -> List[Dict]:
        """AP Business RSS — filter entries that mention the ticker."""
        try:
            import feedparser
            feed = feedparser.parse(_AP_BUSINESS)
            out  = []
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                if not title or ticker.upper() not in title.upper():
                    continue
                out.append({
                    "title":     title,
                    "source":    "AP Business",
                    "url":       getattr(entry, "link", ""),
                    "published": getattr(entry, "published", ""),
                    "summary":   getattr(entry, "summary", "")[:200],
                })
            return out
        except Exception:
            return []

    def _marketwatch_news(self, ticker: str) -> List[Dict]:
        """MarketWatch top-stories RSS — filter entries that mention the ticker."""
        try:
            import feedparser
            feed = feedparser.parse(_MARKETWATCH)
            out  = []
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                if not title or ticker.upper() not in title.upper():
                    continue
                out.append({
                    "title":     title,
                    "source":    "MarketWatch",
                    "url":       getattr(entry, "link", ""),
                    "published": getattr(entry, "published", ""),
                    "summary":   getattr(entry, "summary", "")[:200],
                })
            return out
        except Exception:
            return []

    def _benzinga_news(self, ticker: str) -> List[Dict]:
        """Benzinga RSS — filter entries that mention the ticker."""
        try:
            import feedparser
            feed = feedparser.parse(_BENZINGA)
            out  = []
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                if not title or ticker.upper() not in title.upper():
                    continue
                out.append({
                    "title":     title,
                    "source":    "Benzinga",
                    "url":       getattr(entry, "link", ""),
                    "published": getattr(entry, "published", ""),
                    "summary":   getattr(entry, "summary", "")[:200],
                })
            return out
        except Exception:
            return []

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _classify(self, title: str) -> str:
        """Quick 3-way sentiment classification for a single headline (negation-aware)."""
        score = _score_headline(title)
        if score > 0:   return "positive"
        if score < 0:   return "negative"
        return "neutral"


# ── FRED macro data helper (optional) ─────────────────────────────────────────

_FRED_SERIES = {
    "cpi":            "CPIAUCSL",   # Consumer Price Index
    "fed_funds":      "FEDFUNDS",   # Federal Funds Rate
    "unemployment":   "UNRATE",     # Unemployment Rate
    "yield_curve":    "T10Y2Y",     # 10Y−2Y spread (recession signal)
    "hy_spread":      "BAMLH0A0HYM2",  # High Yield credit spread
    "m2":             "M2SL",       # M2 Money Supply
}

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_macro(fred_key: str) -> Dict:
    """
    Fetch the latest reading for each FRED series.
    Returns dict: { 'cpi': float, 'fed_funds': float, ... }
    """
    if not fred_key:
        return {}
    out = {}
    for label, series_id in _FRED_SERIES.items():
        try:
            resp = requests.get(
                _FRED_BASE,
                params={
                    "series_id":     series_id,
                    "api_key":       fred_key,
                    "file_type":     "json",
                    "sort_order":    "desc",
                    "limit":         2,
                    "observation_start": "2020-01-01",
                },
                timeout=8,
            )
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            for o in obs:
                val = o.get("value", ".")
                if val != ".":
                    out[label] = float(val)
                    break
        except Exception:
            pass
    return out
