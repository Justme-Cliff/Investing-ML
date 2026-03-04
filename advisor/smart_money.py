"""
advisor/smart_money.py — SEC EDGAR smart-money signals (no API key required).

Two free signals derived from SEC EDGAR public filing APIs:

  1. Form 4 cluster — multiple distinct insiders buying the same company within
     60 days. One insider buy = noise; 3+ = institutional-grade conviction signal.
     Source: EDGAR submissions API + Form 4 XML sampling.
     Academic basis: Lakonishok & Lee (2001) — cluster buys predict +5-7% excess return.

  2. 8-K event sentiment — item type classification of recent 8-K filings.
     Positive items (earnings beats, new deals) vs negative items (impairments,
     restatements, bankruptcy). Score is based on item type, not NLP.
     Source: EDGAR submissions API (filingDate + items fields).

Usage (called from alternative_data.enrich_top_n):
    from advisor.smart_money import fetch_smart_money
    result = fetch_smart_money("AAPL", cik_map)
    # result keys: cluster_count, cluster_score, event_score, event_count,
    #              smart_money_score
"""

import io
import re
import time
import datetime
import zipfile
import requests
from typing import Dict, Optional

_SEC_HEADERS = {"User-Agent": "StockAdvisor contact@stockadvisor.local"}
_BASE = "https://data.sec.gov"

# ── 8-K item classification ───────────────────────────────────────────────────
# Item numbers that typically signal positive catalyst
_8K_ITEMS_POS = {
    "1.01",  # Material Definitive Agreement (new deals, partnerships)
    "2.02",  # Results of Operations and Financial Condition (earnings)
    "2.03",  # Creation of a Direct Financial Obligation (new credit line)
    "7.01",  # Regulation FD Disclosure (often guidance updates)
    "8.01",  # Other Events (share buybacks, major wins)
    "9.01",  # Financial Statements (often filed with positive results)
}
# Item numbers that typically signal negative catalyst
_8K_ITEMS_NEG = {
    "1.02",  # Termination of a Material Definitive Agreement
    "1.03",  # Bankruptcy or Receivership
    "2.05",  # Costs Associated with Exit or Disposal
    "2.06",  # Material Impairments
    "4.01",  # Changes in Registrant's Certifying Accountant
    "4.02",  # Non-Reliance on Previously Issued Financial Statements (restatement)
    "5.02",  # Departure of Directors or Officers (CEO/CFO resignation)
}


# ── Form 4 cluster ────────────────────────────────────────────────────────────

def fetch_form4_cluster(cik: str, days: int = 60) -> dict:
    """
    Count distinct insiders who filed Form 4 purchases in the last `days` days.

    Strategy:
      - Fetch recent filings list from EDGAR submissions API.
      - Identify Form 4 filings within the window.
      - Sample up to 8 of them; parse the XML to check for transaction code 'P'
        (open market purchase) and extract the reporter name.
      - Count unique buyers → higher count = stronger cluster signal.

    Returns dict with: cluster_count (int), cluster_score (0–100 float)
    """
    try:
        padded = str(cik).zfill(10)
        r = requests.get(
            f"{_BASE}/submissions/CIK{padded}.json",
            headers=_SEC_HEADERS, timeout=12,
        )
        if r.status_code != 200:
            return {}

        filings  = r.json().get("filings", {}).get("recent", {})
        forms    = filings.get("form", [])
        dates    = filings.get("filingDate", [])
        accnums  = filings.get("accessionNumber", [])
        cik_int  = int(cik)

        cutoff = (
            datetime.date.today() - datetime.timedelta(days=days)
        ).isoformat()

        # Gather Form 4 accession numbers within the window
        candidates = []
        for i, (f, d) in enumerate(zip(forms, dates)):
            if f == "4" and d >= cutoff:
                candidates.append(accnums[i])
            if len(candidates) >= 8:
                break

        if not candidates:
            return {"cluster_count": 0, "cluster_score": 50.0}

        buy_reporters: set = set()

        for acc in candidates:
            try:
                acc_clean = acc.replace("-", "")
                xml_index_url = (
                    f"{_BASE}/Archives/edgar/data/{cik_int}"
                    f"/{acc_clean}/{acc}-index.json"
                )
                r2 = requests.get(xml_index_url, headers=_SEC_HEADERS, timeout=8)
                if r2.status_code != 200:
                    continue

                # Find the primary Form 4 XML document
                for doc in r2.json().get("documents", []):
                    doc_name = doc.get("document", "")
                    if doc.get("type") == "4" and doc_name.endswith(".xml"):
                        xml_url = (
                            f"{_BASE}/Archives/edgar/data/{cik_int}"
                            f"/{acc_clean}/{doc_name}"
                        )
                        r3 = requests.get(xml_url, headers=_SEC_HEADERS, timeout=8)
                        if r3.status_code == 200:
                            xml = r3.text
                            # Check for purchase transaction code
                            if "<transactionCode>P</transactionCode>" in xml:
                                name_match = re.search(
                                    r"<rptOwnerName>(.*?)</rptOwnerName>", xml
                                )
                                if name_match:
                                    buy_reporters.add(name_match.group(1).strip())
                        break   # only need the primary doc

                time.sleep(0.08)   # be polite — SEC rate limit is 10 req/s
            except Exception:
                continue

        n = len(buy_reporters)
        # Score: 0→50, 1→55, 2→65, 3→78, 4→88, 5+→95
        score_map = {0: 50.0, 1: 55.0, 2: 65.0, 3: 78.0, 4: 88.0}
        score = score_map.get(n, 95.0)

        return {"cluster_count": n, "cluster_score": score}

    except Exception:
        return {}


# ── 8-K event sentiment ───────────────────────────────────────────────────────

def fetch_8k_sentiment(cik: str, days: int = 90) -> dict:
    """
    Classify recent 8-K filings by item type to estimate catalyst sentiment.

    Uses the EDGAR submissions API `items` field — no XML parsing required.
    Each 8-K lists the item numbers it covers (e.g. "2.02, 9.01").
    Positive items score +1, negative items score -1.

    Returns dict with: event_score (0–100 float), event_count (int)
    """
    try:
        padded = str(cik).zfill(10)
        r = requests.get(
            f"{_BASE}/submissions/CIK{padded}.json",
            headers=_SEC_HEADERS, timeout=12,
        )
        if r.status_code != 200:
            return {}

        filings = r.json().get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
        items   = filings.get("items", [])

        cutoff = (
            datetime.date.today() - datetime.timedelta(days=days)
        ).isoformat()

        scores = []
        for i, (f, d) in enumerate(zip(forms, dates)):
            if f not in ("8-K", "8-K/A"):
                continue
            if d < cutoff:
                continue

            item_str = str(items[i] if i < len(items) else "")
            # Items are comma-separated, e.g. "2.02, 9.01"
            item_list = [s.strip() for s in item_str.replace(";", ",").split(",")]

            score = 0
            for item in item_list:
                if item in _8K_ITEMS_POS:
                    score += 1
                elif item in _8K_ITEMS_NEG:
                    score -= 1
            scores.append(score)

        if not scores:
            return {"event_score": 50.0, "event_count": 0}

        avg = sum(scores) / len(scores)
        # avg of +1 → 62; avg of -1 → 38; +2 → 74; -2 → 26
        event_score = float(max(0.0, min(100.0, 50.0 + avg * 12.0)))
        return {"event_score": event_score, "event_count": len(scores)}

    except Exception:
        return {}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def fetch_smart_money(ticker: str, cik_map: dict) -> dict:
    """
    Run all smart-money signals for a ticker and return a composite score.

    Args:
        ticker:  Stock ticker symbol (e.g. "AAPL")
        cik_map: Dict of {ticker → CIK string} from SEC EDGAR company_tickers.json

    Returns dict with:
        cluster_count     int    Number of distinct insider buyers (60d)
        cluster_score     float  0–100
        event_score       float  0–100 (8-K catalyst sentiment)
        event_count       int    Number of 8-K filings analyzed
        smart_money_score float  0–100 (equal-weighted composite)
    """
    cik = cik_map.get(ticker.upper())
    if not cik:
        return {}

    result: dict = {}

    form4 = fetch_form4_cluster(cik)
    result.update(form4)
    time.sleep(0.1)

    eightk = fetch_8k_sentiment(cik)
    result.update(eightk)

    # Composite: equal-weight the two signals
    scores = []
    if "cluster_score" in result:
        scores.append(result["cluster_score"])
    if "event_score" in result:
        scores.append(result["event_score"])

    if scores:
        result["smart_money_score"] = float(sum(scores) / len(scores))

    return result
