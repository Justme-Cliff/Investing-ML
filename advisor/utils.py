"""advisor/utils.py — shared utility functions used across advisor modules."""

from __future__ import annotations

import math
import re
from typing import List, Optional


# ── Sentiment keyword lists ──────────────────────────────────────────────────

_POS_WORDS = [
    "beat", "beats", "surge", "surges", "record", "profit", "grow", "growth",
    "strong", "rally", "bullish", "upgrade", "outperform", "raise", "raised",
    "exceed", "exceeds", "positive", "gain", "gains", "jump", "jumps",
    "momentum", "expansion", "breakthrough", "partnership", "acquisition",
    "buyback", "dividend", "confident", "guidance raised", "revenue growth",
]

_NEG_WORDS = [
    "miss", "misses", "loss", "losses", "decline", "declines", "weak",
    "warning", "downgrade", "underperform", "lower", "cut", "cuts",
    "disappoint", "disappoints", "investigation", "lawsuit", "recall",
    "bankruptcy", "layoff", "layoffs", "restructure", "charge", "impairment",
    "guidance cut", "revenue decline", "margin pressure", "default",
    "fraud", "restatement", "whistleblower", "probe", "fine",
]

_NEGATION_WORDS = ("not ", "no ", "never ", "fails to ", "unable to ", "won't ", "doesn't ",
                   "didn't ", "cannot ", "can't ", "without ")

_CLAUSE_BOUNDS = (",", ";", " but ", " however ", " yet ", " although ", " though ")


# ── Sentiment helpers ────────────────────────────────────────────────────────

def _is_negated(text_before: str) -> bool:
    """Return True if the 50 chars before a keyword contain a negation phrase
    that is not separated from it by a clause boundary."""
    chunk = text_before[-50:].lower()
    for bound in _CLAUSE_BOUNDS:
        if bound in chunk:
            chunk = chunk.split(bound)[-1]
    return any(neg in chunk for neg in _NEGATION_WORDS)


def score_headline(title: str) -> float:
    """Score a single news headline in [-1, +1] with negation awareness."""
    text = title.lower()
    raw = 0.0
    for word in _POS_WORDS:
        idx = text.find(word)
        if idx >= 0 and not _is_negated(text[:idx]):
            raw += 1.0
        elif idx >= 0:
            raw -= 0.5   # negated positive ≈ mild negative
    for word in _NEG_WORDS:
        idx = text.find(word)
        if idx >= 0 and not _is_negated(text[:idx]):
            raw -= 1.0
        elif idx >= 0:
            raw += 0.3   # negated negative ≈ mild positive
    # normalise to [-1, 1]
    return max(-1.0, min(1.0, raw / max(1, len(_POS_WORDS) + len(_NEG_WORDS)) * 10))


def sentiment_from_headlines(headlines: List[str], weights: Optional[List[float]] = None) -> float:
    """Aggregate a list of headline strings into a 0-100 sentiment score."""
    if not headlines:
        return 50.0
    if weights is None:
        weights = [1.0] * len(headlines)
    total_w = sum(weights)
    if total_w == 0:
        return 50.0
    raw = sum(score_headline(h) * w for h, w in zip(headlines, weights)) / total_w
    # map [-1, 1] → [0, 100]
    return max(0.0, min(100.0, 50.0 + raw * 50.0))


# ── Analyst score ────────────────────────────────────────────────────────────

_REC_MAP = {
    "strong_buy":  90.0,
    "buy":         75.0,
    "hold":        50.0,
    "sell":        25.0,
    "strong_sell": 10.0,
}


def analyst_score(info: dict) -> float:
    """Compute a 0-100 analyst sentiment score from a yfinance info dict."""
    rec       = (info.get("recommendationKey") or "").lower().replace(" ", "_")
    rec_score = _REC_MAP.get(rec, 50.0)

    cur = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    tgt = float(info.get("targetMeanPrice") or 0)
    if cur > 0 and tgt > 0:
        upside_sc = float(max(0.0, min(100.0, 50.0 + (tgt - cur) / cur * 200.0)))
    else:
        upside_sc = 50.0

    n_ana  = float(info.get("numberOfAnalystOpinions") or 0)
    cov_sc = float(max(10.0, min(100.0, n_ana / 20.0 * 100.0)))

    return 0.40 * rec_score + 0.40 * upside_sc + 0.20 * cov_sc


# ── Numeric helpers ──────────────────────────────────────────────────────────

def safe_float(val, default: float = 0.0) -> float:
    """Convert val to float, returning default on failure or NaN."""
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def safe_int(val, default: int = 0) -> int:
    """Convert val to int, returning default on failure."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def is_imminent_earnings(days_away) -> bool:
    """Return True if earnings are 0-7 days away, handling None/NaN safely."""
    try:
        if days_away is None:
            return False
        v = float(days_away)
        return not math.isnan(v) and 0 <= int(v) <= 7
    except (ValueError, TypeError):
        return False
