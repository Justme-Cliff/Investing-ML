# advisor/fetcher.py — DataFetcher (stock data + technicals + sentiment) and MacroFetcher

import io
import time
import math
import datetime
import contextlib
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import requests

from config import (
    SP500_TICKER, VIX_TICKER, YIELD_10Y_TICKER,
    SECTOR_ETFS, STOCK_UNIVERSE, MACRO_TILTS,
    POSITIVE_WORDS, NEGATIVE_WORDS,
    FINNHUB_KEY, ALPHAVANTAGE_KEY,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicator helpers (pure functions)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist   = line - signal
    return line, signal, hist


def _technical_score(history: pd.DataFrame) -> float:
    """
    RSI (25%) + MACD (30%) + MA crossover (20%) + Bollinger %B (15%) + OBV trend (10%)
    → 0–100
    """
    close = history["Close"].dropna()
    if len(close) < 50:
        return 50.0

    # ── RSI score ─────────────────────────────────────────────────────────────
    rsi_series = _rsi(close)
    last_rsi   = float(rsi_series.dropna().iloc[-1]) if len(rsi_series.dropna()) else 50.0
    if 40 <= last_rsi <= 65:
        rsi_score = 80 + (1 - abs(last_rsi - 52.5) / 12.5) * 20
    elif last_rsi < 30:
        rsi_score = 72   # oversold — contrarian buy
    elif last_rsi > 80:
        rsi_score = 10   # very overbought
    elif last_rsi > 70:
        rsi_score = 30
    elif last_rsi < 40:
        rsi_score = 45
    else:
        rsi_score = 55

    # ── MACD score ────────────────────────────────────────────────────────────
    macd_line, sig_line, hist_series = _macd(close)
    if len(hist_series.dropna()) < 2:
        macd_score = 50.0
    else:
        lm = float(macd_line.iloc[-1])
        ls = float(sig_line.iloc[-1])
        lh = float(hist_series.iloc[-1])
        ph = float(hist_series.iloc[-2])
        if lm > ls and lh > 0:
            macd_score = 88
        elif lm > ls and lh < 0:
            macd_score = 63
        elif lm < ls and lh > ph:
            macd_score = 42
        elif lm < ls and lh < 0 and lh < ph:
            macd_score = 15
        else:
            macd_score = 50

    # ── MA crossover score ────────────────────────────────────────────────────
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    cur    = float(close.iloc[-1])
    s50    = float(sma50.iloc[-1])
    s200   = float(sma200.dropna().iloc[-1]) if len(sma200.dropna()) >= 1 else None

    if s200 is not None and not math.isnan(s200):
        if cur > s50 > s200:
            ma_score = 90
        elif cur > s200 and cur < s50:
            ma_score = 65
        elif cur < s200 and cur > s50:
            ma_score = 40
        else:
            ma_score = 15
        if len(sma50.dropna()) >= 2 and len(sma200.dropna()) >= 2:
            ps50  = float(sma50.dropna().iloc[-2])
            ps200 = float(sma200.dropna().iloc[-2])
            if s50 > s200 and ps50 <= ps200:
                ma_score = min(ma_score + 12, 100)
    else:
        ma_score = 75 if cur > s50 else 35

    # ── Bollinger %B score ────────────────────────────────────────────────────
    # %B = 0 → at lower band  |  %B = 1 → at upper band
    # Ideal buy zone: 0.20–0.65  (neither overbought nor oversold)
    bb_score = 50.0
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        bw    = float(upper.iloc[-1] - lower.iloc[-1])
        if bw > 0:
            pctb = float(close.iloc[-1] - lower.iloc[-1]) / bw
            if 0.20 <= pctb <= 0.65:
                bb_score = 78 + (1 - abs(pctb - 0.425) / 0.225) * 12
            elif pctb < 0.10:
                bb_score = 68    # near lower band → potential oversold bounce
            elif pctb > 0.90:
                bb_score = 12    # near upper band → overbought warning
            elif pctb > 0.80:
                bb_score = 30
            else:
                bb_score = 55

    # ── OBV trend score ───────────────────────────────────────────────────────
    # Rising OBV (20d SMA > 50d SMA) = smart-money accumulation = bullish
    obv_score = 50.0
    vol_col = "Volume" if "Volume" in history.columns else None
    if vol_col and len(close) >= 50:
        vol_series = history[vol_col].fillna(0)
        obv = (np.sign(close.diff()) * vol_series).fillna(0).cumsum()
        if len(obv) >= 50:
            obv20 = obv.rolling(20).mean().dropna()
            obv50 = obv.rolling(50).mean().dropna()
            if len(obv20) > 0 and len(obv50) > 0:
                obv_score = 80 if float(obv20.iloc[-1]) > float(obv50.iloc[-1]) else 28

    return (0.25 * rsi_score + 0.30 * macd_score + 0.20 * ma_score
            + 0.15 * bb_score + 0.10 * obv_score)


def _piotroski(info: dict) -> float:
    """Simplified 8-point Piotroski F-score → 0–100."""
    score = 0

    # Profitability (3 pts)
    roa = info.get("returnOnAssets")
    if roa is not None and float(roa) > 0.04:
        score += 1

    op_cf = info.get("operatingCashflow")
    if op_cf is not None and float(op_cf) > 0:
        score += 1

    fcf = info.get("freeCashflow")
    if fcf is not None and float(fcf) > 0:
        score += 1

    # Leverage / Liquidity (2 pts)
    de = info.get("debtToEquity")
    if de is not None:
        de_val = float(de)
        de_ratio = de_val / 100 if de_val > 10 else de_val   # handle % vs ratio form
        if de_ratio < 1.0:
            score += 1

    cr = info.get("currentRatio")
    if cr is not None and float(cr) > 1.5:
        score += 1

    # Efficiency / Growth (3 pts)
    pm = info.get("profitMargins")
    if pm is not None and float(pm) > 0.10:
        score += 1

    rg = info.get("revenueGrowth")
    if rg is not None and float(rg) > 0:
        score += 1

    eg = info.get("earningsGrowth")
    if eg is not None and float(eg) > 0:
        score += 1

    return (score / 8) * 100


def _sentiment(news: list) -> float:
    """Score news headlines → 0–100 (50 = neutral)."""
    if not news:
        return 50.0
    total = 0
    count = 0
    for item in news[:7]:
        title = (item.get("title") or "").lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in title)
        neg = sum(1 for w in NEGATIVE_WORDS if w in title)
        total += (pos - neg)
        count += 1
    if count == 0:
        return 50.0
    raw = total / count                        # typically in [-3, +3]
    normalised = (raw + 3) / 6 * 100          # map to 0–100
    return float(max(0.0, min(100.0, normalised)))


# ─────────────────────────────────────────────────────────────────────────────
# DataFetcher
# ─────────────────────────────────────────────────────────────────────────────

class DataFetcher:
    """Fetches per-stock data (price history, fundamentals, technicals, sentiment)."""

    def __init__(self, yf_period: str):
        self.yf_period = yf_period
        self.failed: List[str] = []

    def fetch_universe(self, tickers: List[str]) -> Dict:
        results: Dict = {}
        total = len(tickers)
        done  = 0
        batches = [tickers[i:i + 20] for i in range(0, total, 20)]

        print(f"\nFetching data for {total} stocks  ({len(batches)} batches)...")
        for bi, batch in enumerate(batches, 1):
            for ticker in batch:
                data = self._fetch_one(ticker)
                if data:
                    results[ticker] = data
                else:
                    self.failed.append(ticker)
                done += 1
                pct    = done / total * 100
                filled = int(pct / 5)
                bar    = "█" * filled + "░" * (20 - filled)
                print(f"  [{bar}] {pct:4.0f}%  {done}/{total}", end="\r", flush=True)
            if bi < len(batches):
                time.sleep(0.4)

        print(f"  [{'█'*20}] 100%  {done}/{done}" + " " * 15)
        print(f"  Loaded: {len(results)}  |  Skipped: {len(self.failed)}")
        return results

    def fetch_sp500(self) -> Optional[pd.DataFrame]:
        print(f"Fetching S&P 500 benchmark ({SP500_TICKER})...")
        for attempt in range(3):
            try:
                sink = io.StringIO()
                with contextlib.redirect_stderr(sink):
                    hist = yf.Ticker(SP500_TICKER).history(period=self.yf_period)
                if hist is not None and len(hist) > 10:
                    print("  S&P 500 loaded.\n")
                    return hist
            except Exception:
                if attempt < 2:
                    time.sleep(1)
        print("  Warning: S&P 500 unavailable — benchmark line omitted.\n")
        return None

    def _fetch_one(self, ticker: str) -> Optional[dict]:
        for attempt in range(3):
            try:
                sink = io.StringIO()
                with contextlib.redirect_stderr(sink):
                    t       = yf.Ticker(ticker)
                    info    = t.info
                    if not info or len(info) < 5:
                        return None
                    history = t.history(period=self.yf_period)
                    news    = t.news or []

                if history is None or len(history) < 63:
                    return None

                sector      = self._map_sector(info.get("sector", ""), ticker)
                tech        = _technical_score(history)
                piotr       = _piotroski(info)
                sent        = _sentiment(news)
                news_titles = [
                    item.get("title", "") for item in news[:12]
                    if item.get("title")
                ]

                # ── Next earnings date ────────────────────────────────────
                earnings_date      = None
                earnings_days_away = None
                try:
                    cal   = t.calendar
                    today = datetime.date.today()
                    dates = []
                    if isinstance(cal, dict):
                        raw = cal.get("Earnings Date", [])
                        dates = raw if isinstance(raw, list) else [raw]
                    elif hasattr(cal, "loc"):
                        try:
                            dates = list(cal.loc["Earnings Date"].values)
                        except Exception:
                            pass
                    for d in sorted(dates):
                        try:
                            d_date = d.date() if hasattr(d, "date") else (
                                datetime.date.fromisoformat(str(d)[:10])
                                if isinstance(d, str) else d
                            )
                            if d_date >= today:
                                earnings_date      = str(d_date)
                                earnings_days_away = (d_date - today).days
                                break
                        except Exception:
                            pass
                except Exception:
                    pass

                # ── Finnhub supplemental signals (optional, non-blocking) ──────
                finnhub = self._fetch_finnhub_signals(ticker)

                return {
                    "info":                  info,
                    "history":               history,
                    "sector":                sector,
                    "technical":             tech,
                    "piotroski":             piotr,
                    "sentiment":             sent,
                    "news_titles":           news_titles,
                    "earnings_date":         earnings_date,
                    "earnings_days_away":    earnings_days_away,
                    "insider_signal":        finnhub.get("insider_signal",        0.0),
                    "earnings_surprise_avg": finnhub.get("earnings_surprise_avg", None),
                }
            except Exception:
                if attempt < 2:
                    time.sleep(1)
        return None

    def _map_sector(self, yf_sector: str, ticker: str) -> str:
        MAP = {
            "Technology": "Technology",
            "Healthcare": "Healthcare",
            "Financial Services": "Financials",
            "Financials": "Financials",
            "Consumer Cyclical": "Consumer",
            "Consumer Defensive": "Consumer",
            "Consumer Staples": "Consumer",
            "Consumer Discretionary": "Consumer",
            "Energy": "Energy",
            "Industrials": "Industrials",
            "Utilities": "Utilities",
            "Real Estate": "Real Estate",
            "Basic Materials": "Materials",
            "Materials": "Materials",
            "Communication Services": "Communication",
        }
        if yf_sector in MAP:
            return MAP[yf_sector]
        for sector, tlist in STOCK_UNIVERSE.items():
            if ticker in tlist:
                return sector
        return "Unknown"

    def _fetch_finnhub_signals(self, ticker: str) -> dict:
        """
        Fetch supplemental signals from Finnhub (free tier: 60 calls/min).
        Returns:
          insider_signal       float  [-1, +1]  net insider buy pressure
          earnings_surprise_avg float | None    avg EPS surprise % (last 4 qtrs)
        Returns empty dict if no API key or request fails.
        """
        if not FINNHUB_KEY:
            return {}

        result = {}
        base   = "https://finnhub.io/api/v1"
        headers = {"X-Finnhub-Token": FINNHUB_KEY}

        # ── 1. Insider transactions (last 90 days) ────────────────────────────
        try:
            today      = datetime.date.today()
            from_date  = (today - datetime.timedelta(days=90)).isoformat()
            r = requests.get(
                f"{base}/stock/insider-transactions",
                params={"symbol": ticker, "from": from_date},
                headers=headers,
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                buys  = sum(float(tx.get("share", 0)) for tx in data if str(tx.get("transactionCode", "")).upper() == "P")
                sells = sum(abs(float(tx.get("share", 0))) for tx in data if str(tx.get("transactionCode", "")).upper() == "S")
                total = buys + sells
                if total > 0:
                    result["insider_signal"] = (buys - sells) / total   # -1 to +1
        except Exception:
            pass

        # ── 2. Earnings surprise (last 4 quarters) ────────────────────────────
        try:
            r = requests.get(
                f"{base}/stock/earnings",
                params={"symbol": ticker, "limit": 4},
                headers=headers,
                timeout=5,
            )
            if r.status_code == 200:
                quarters = r.json()
                surprises = []
                for q in quarters:
                    actual   = q.get("actual")
                    estimate = q.get("estimate")
                    if actual is not None and estimate and abs(float(estimate)) > 0.01:
                        surprises.append((float(actual) - float(estimate)) / abs(float(estimate)))
                if surprises:
                    result["earnings_surprise_avg"] = float(np.mean(surprises))
        except Exception:
            pass

        return result

    @staticmethod
    def strip_tz(s: pd.Series) -> pd.Series:
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# MacroFetcher
# ─────────────────────────────────────────────────────────────────────────────

class MacroFetcher:
    """Fetches macro indicators (VIX, yields, sector ETFs) from yfinance."""

    def fetch(self) -> dict:
        print("Fetching macro data (VIX · 10Y yield · sector ETFs · crash signals)...")
        result = {
            "vix":                 None,
            "vix_hist":            None,
            "yield_10y":           None,
            "yield_hist":          None,
            "yield_3m":            None,
            "vix_velocity":        None,
            "credit_stress":       False,
            "market_drawdown_pct": 0.0,
            "crash_signals":       0,
            "sector_etf":          {},
            "regime":              "neutral",
            "regime_reasons":      [],
        }
        sink = io.StringIO()

        # VIX — 3-month history; compute 5-day velocity for panic detection
        try:
            with contextlib.redirect_stderr(sink):
                vix_hist = yf.Ticker(VIX_TICKER).history(period="3mo")
            if vix_hist is not None and len(vix_hist) > 5:
                result["vix"]      = float(vix_hist["Close"].iloc[-1])
                result["vix_hist"] = vix_hist
                if len(vix_hist) >= 6:
                    result["vix_velocity"] = float(
                        vix_hist["Close"].iloc[-1] - vix_hist["Close"].iloc[-6]
                    )
        except Exception:
            pass

        # 10Y Treasury yield
        try:
            with contextlib.redirect_stderr(sink):
                tnx_hist = yf.Ticker(YIELD_10Y_TICKER).history(period="3mo")
            if tnx_hist is not None and len(tnx_hist) > 5:
                result["yield_10y"]  = float(tnx_hist["Close"].iloc[-1])
                result["yield_hist"] = tnx_hist
        except Exception:
            pass

        # 3-month T-bill yield (^IRX) — yield curve inversion check
        try:
            with contextlib.redirect_stderr(sink):
                irx_hist = yf.Ticker("^IRX").history(period="5d")
            if irx_hist is not None and len(irx_hist) > 0:
                result["yield_3m"] = float(irx_hist["Close"].iloc[-1])
        except Exception:
            pass

        # HYG (iShares High Yield ETF) — credit stress proxy
        try:
            with contextlib.redirect_stderr(sink):
                hyg_hist = yf.Ticker("HYG").history(period="2mo")
            if hyg_hist is not None and len(hyg_hist) > 20:
                hyg_1m_ret = float(
                    hyg_hist["Close"].iloc[-1] / hyg_hist["Close"].iloc[-21] - 1
                )
                result["credit_stress"] = hyg_1m_ret < -0.03
        except Exception:
            pass

        # SPY (1-year) — market drawdown from 52-week high
        try:
            with contextlib.redirect_stderr(sink):
                spy_hist = yf.Ticker("SPY").history(period="1y")
            if spy_hist is not None and len(spy_hist) > 20:
                spy_high = float(spy_hist["Close"].max())
                spy_now  = float(spy_hist["Close"].iloc[-1])
                result["market_drawdown_pct"] = (spy_now / spy_high - 1) * 100
        except Exception:
            pass

        # Sector ETF 3-month returns
        for sector, etf in SECTOR_ETFS.items():
            try:
                with contextlib.redirect_stderr(sink):
                    hist = yf.Ticker(etf).history(period="3mo")
                if hist is not None and len(hist) > 20:
                    ret = float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
                    result["sector_etf"][sector] = round(ret * 100, 2)
            except Exception:
                pass

        result["regime"], result["regime_reasons"] = self._classify(result)
        crash_n = result.get("crash_signals", 0)
        print(f"  Macro regime: {result['regime'].upper()}", end="")
        if crash_n > 0:
            print(f"  |  Crash signals active: {crash_n}/5", end="")
        print()
        print()
        return result

    def _classify(self, m: dict):
        vix          = m.get("vix")
        vix_velocity = m.get("vix_velocity")
        yield_10y    = m.get("yield_10y")
        yield_3m     = m.get("yield_3m")
        credit_stress= m.get("credit_stress", False)
        drawdown_pct = m.get("market_drawdown_pct", 0.0)
        y_hist       = m.get("yield_hist")
        reasons      = []
        regime       = "neutral"

        # ── Count crash signals (0–5) ──────────────────────────────────────────
        crash_signals = 0
        crash_reasons = []

        # Signal 1: VIX velocity > 7 points in 5 days (real-time panic)
        if vix_velocity is not None and vix_velocity > 7:
            crash_signals += 1
            crash_reasons.append(f"VIX velocity +{vix_velocity:.1f}pts/5d (panic spike)")

        # Signal 2: HYG 1-month return < -3% (credit markets seizing)
        if credit_stress:
            crash_signals += 1
            crash_reasons.append("HYG 1m return < -3% (credit stress)")

        # Signal 3: Yield curve inverted (3M T-bill > 10Y Treasury)
        if yield_3m is not None and yield_10y is not None and yield_3m > yield_10y:
            crash_signals += 1
            crash_reasons.append(
                f"Yield curve inverted: 3M={yield_3m:.2f}% > 10Y={yield_10y:.2f}%"
            )

        # Signal 4: SPY more than 12% below 52-week high (bear territory)
        if drawdown_pct < -12.0:
            crash_signals += 1
            crash_reasons.append(f"SPY {drawdown_pct:.1f}% off 52-week high (bear territory)")

        # Signal 5: VIX absolute level > 35 (extreme fear)
        if vix is not None and vix > 35:
            crash_signals += 1
            crash_reasons.append(f"VIX={vix:.1f} (extreme fear)")

        m["crash_signals"] = crash_signals

        # ── Regime classification (crash signals override legacy logic) ────────
        if crash_signals >= 3:
            regime = "crisis"
            reasons.append(f"CRISIS DETECTED — {crash_signals}/5 crash signals active")
            reasons.extend(crash_reasons)
            return regime, reasons

        if crash_signals == 2:
            regime = "pre_crisis"
            reasons.append(f"PRE-CRISIS — {crash_signals}/5 crash signals active")
            reasons.extend(crash_reasons)
            return regime, reasons

        # Legacy VIX-based logic
        if vix is not None:
            if vix < 16:
                reasons.append(f"VIX={vix:.1f} (low fear → risk-on)")
                regime = "risk_on"
            elif vix > 27:
                reasons.append(f"VIX={vix:.1f} (high fear → risk-off)")
                regime = "risk_off"
            else:
                reasons.append(f"VIX={vix:.1f} (moderate)")

        # Yield change over 1 month
        if y_hist is not None and len(y_hist) > 20:
            y_recent = float(y_hist["Close"].iloc[-1])
            y_month  = float(y_hist["Close"].iloc[-20])
            dy       = y_recent - y_month
            if dy > 0.35:
                reasons.append(f"10Y yield ↑{dy:+.2f}% → rising rate env")
                if regime in ("neutral", "risk_on"):
                    regime = "rising_rate"
            elif dy < -0.30:
                reasons.append(f"10Y yield ↓{dy:+.2f}% → falling rate env")
                if regime in ("neutral",):
                    regime = "falling_rate"

        # Surface any sub-threshold crash signals for transparency
        if crash_reasons:
            reasons.append(f"Crash signals ({crash_signals}/5): {'; '.join(crash_reasons)}")

        return regime, reasons
