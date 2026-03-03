"""
advisor/universe.py — Dynamic US stock universe fetcher

Downloads ALL US-listed common stocks from NASDAQ's free public screener API.
No API key required.  Results are cached for 24 hours so each run is fast.
Falls back to the static STOCK_UNIVERSE in config.py if the download fails.

Selection logic (per run):
  - Always keep the top 200 largest-cap stocks (quality anchor)
  - Fill the remaining slots up to UNIVERSE_MAX_TICKERS with a random
    sample from the rest of the universe (variety / discovery)
  → Each run surfaces different mid/small-cap stocks alongside the majors.
"""

import json
import os
import random
import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import requests

warnings.filterwarnings("ignore")

CACHE_FILE    = os.path.join("memory", "universe_cache.json")
CACHE_TTL_HRS = 24

# ── NASDAQ public screener (no API key, covers all US exchanges) ──────────────
_NASDAQ_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=25000&offset=0&download=true"
)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nasdaq.com/",
    "Origin":          "https://www.nasdaq.com",
}

# Keywords in the company name that signal it is NOT a common stock
_SKIP_KEYWORDS = [
    "warrant", "warrants", " right ", " rights ", " unit ", " units ",
    "preferred", "depositary", "debenture", "notes due", "% due",
    " note ", " bond ", "acquisition corp", "blank check",
    "class w", "series a pref", "series b pref", "series c pref",
    "series d pref", "series e pref",
]
_SKIP_SYMBOL_CHARS = set("^/~*")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cap(raw) -> float:
    """Parse a market-cap value from numeric, '$1.23B', '1,234,567,890', etc."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace("$", "").replace(",", "")
    if not s or s in ("N/A", "--", "n/a", ""):
        return 0.0
    try:
        u = s[-1].upper()
        if u == "T":
            return float(s[:-1]) * 1e12
        if u == "B":
            return float(s[:-1]) * 1e9
        if u == "M":
            return float(s[:-1]) * 1e6
        if u == "K":
            return float(s[:-1]) * 1e3
        return float(s)
    except (ValueError, IndexError):
        return 0.0


def _is_common_stock(symbol: str, name: str) -> bool:
    """Return True if this looks like an ordinary common stock."""
    if not symbol or len(symbol) > 5:
        return False
    if any(c in symbol for c in _SKIP_SYMBOL_CHARS):
        return False
    name_lc = name.lower()
    return not any(kw in name_lc for kw in _SKIP_KEYWORDS)


def _load_cache() -> Optional[List[str]]:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as fh:
            data = json.load(fh)
        ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01T00:00:00"))
        if datetime.now() - ts < timedelta(hours=CACHE_TTL_HRS):
            tickers = data.get("tickers", [])
            if tickers:
                return tickers
    except Exception:
        pass
    return None


def _save_cache(tickers: List[str]) -> None:
    os.makedirs("memory", exist_ok=True)
    try:
        with open(CACHE_FILE, "w") as fh:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "tickers":   tickers,
                    "count":     len(tickers),
                },
                fh,
            )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_us_universe(
    min_market_cap: float = 100_000_000,
    max_tickers:    int   = 500,
    verbose:        bool  = True,
) -> List[str]:
    """
    Return a list of US common-stock ticker symbols for the analysis pipeline.

    Steps:
    1. Check 24-hour cache  → return immediately if fresh.
    2. Download all US stocks from NASDAQ's public screener API.
    3. Filter: common stocks only, market cap ≥ min_market_cap.
    4. Sort by market cap descending, cache full list.
    5. Select up to max_tickers using stratified sampling:
         • Always include the top 200 by market cap (quality anchor)
         • Fill remaining slots with a random sample of the rest (variety)
    6. On any failure, fall back to the static STOCK_UNIVERSE from config.py.
    """
    from config import STOCK_UNIVERSE

    # ── Step 1: cache ─────────────────────────────────────────────────────────
    all_cached = _load_cache()
    if all_cached:
        if verbose:
            print(
                f"  Universe: {len(all_cached):,} US stocks loaded "
                f"(24h cache  |  min cap ${min_market_cap / 1e6:.0f}M)"
            )
        return _select(all_cached, max_tickers, verbose)

    # ── Step 2–4: download ────────────────────────────────────────────────────
    if verbose:
        print("  Downloading full US stock universe from NASDAQ…", end=" ", flush=True)

    try:
        resp = requests.get(_NASDAQ_URL, headers=_HEADERS, timeout=25)
        resp.raise_for_status()
        payload = resp.json()

        rows = payload.get("data", {}).get("rows", [])
        if not rows:
            raise ValueError("Empty response from NASDAQ screener")

        entries: List[Tuple[str, float]] = []
        for row in rows:
            sym  = (row.get("symbol") or "").strip().upper()
            name = (row.get("name")   or "")
            if not _is_common_stock(sym, name):
                continue
            cap = _parse_cap(row.get("marketCap"))
            if cap < min_market_cap:
                continue
            entries.append((sym, cap))

        # Sort largest-first so the cache is ordered by market cap
        entries.sort(key=lambda x: x[1], reverse=True)
        full_list = [sym for sym, _ in entries]

        if not full_list:
            raise ValueError("No tickers survived filters")

        _save_cache(full_list)

        if verbose:
            print(f"done — {len(full_list):,} common stocks")

        return _select(full_list, max_tickers, verbose)

    except Exception as exc:
        if verbose:
            print(
                f"failed ({exc})\n"
                f"  Falling back to static stock list ({sum(len(v) for v in STOCK_UNIVERSE.values())} stocks)."
            )
        return [t for lst in STOCK_UNIVERSE.values() for t in lst]


def _select(all_tickers: List[str], max_tickers: int, verbose: bool) -> List[str]:
    """
    Stratified selection:
      1. Top-200 by market cap — quality anchor (large-caps always present)
      2. Full curated STOCK_UNIVERSE — guaranteed sector breadth and low-beta
         defensive stocks for portfolio beta management (utilities, staples, etc.)
      3. Random sample of remaining tickers — variety / discovery

    Guaranteeing the curated list solves a key problem: the random sample may
    entirely skip defensive sectors, leaving the portfolio with no low-beta
    swap candidates when the beta cap fires.
    """
    from config import STOCK_UNIVERSE
    curated = [t for lst in STOCK_UNIVERSE.values() for t in lst]

    if len(all_tickers) <= max_tickers:
        # Even in a small universe, inject curated tickers that aren't present
        merged = list(dict.fromkeys(all_tickers + curated))
        return merged[:max_tickers]

    top_n = min(200, max_tickers)
    top   = all_tickers[:top_n]

    # Guaranteed base: top-200 by cap + any curated ticker not already in top-200
    top_set       = set(top)
    extra_curated = [t for t in curated if t not in top_set]
    guaranteed    = top + extra_curated          # deduplicated by construction
    guaranteed_set = top_set | set(extra_curated)

    remaining = max_tickers - len(guaranteed)
    if remaining <= 0:
        # Curated list filled the budget — trim and return
        return list(dict.fromkeys(guaranteed))[:max_tickers]

    # Fill leftover slots with a random mid/small-cap sample
    pool    = [t for t in all_tickers[top_n:] if t not in guaranteed_set]
    random.shuffle(pool)
    sampled = pool[:remaining]

    selected = list(dict.fromkeys(guaranteed + sampled))

    if verbose:
        print(
            f"  Selected {len(selected):,} stocks for this run  "
            f"({top_n} large-cap anchor + {len(extra_curated)} curated defensive"
            f" + {len(sampled)} random mid/small-cap)"
        )
    return selected


def clear_universe_cache() -> None:
    """Force a fresh download on the next run by deleting the cached list."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("  Universe cache cleared — will re-download on next run.")
