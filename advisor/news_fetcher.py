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
_YF_RSS   = "https://finance.yahoo.com/rss/headline?s={ticker}&region=US&lang=en-US"
_REUTERS  = "https://feeds.reuters.com/reuters/businessNews"


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
        """Score a list of headlines 0–100 using keyword matching."""
        if not headlines:
            return 50.0
        pos = neg = 0
        for h in headlines:
            hl = h.lower()
            pos += sum(1 for w in POSITIVE_WORDS if w in hl)
            neg += sum(1 for w in NEGATIVE_WORDS if w in hl)
        total = max(pos + neg, 1)
        return round(min(100.0, max(0.0, 30.0 + (pos / total) * 70.0)), 1)

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

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _classify(self, title: str) -> str:
        """Quick 3-way sentiment classification for a single headline."""
        hl  = title.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in hl)
        neg = sum(1 for w in NEGATIVE_WORDS if w in hl)
        if pos > neg:   return "positive"
        if neg > pos:   return "negative"
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
