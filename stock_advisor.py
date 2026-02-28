# ─────────────────────────────────────────────────────────────────────────────
# STOCK RANKING ADVISOR
# Real-time personalized stock rankings using Yahoo Finance (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

# SECTION 1: Imports + Constants
import sys
import io
import time
import math
import warnings
import contextlib
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box as rich_box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ── Stock universe: ~110 liquid large/mid cap stocks across 9 sectors ─────────
STOCK_UNIVERSE = {
    "Technology":  [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL",
        "CRM", "ADBE", "AMD", "QCOM", "TXN", "NOW", "INTU",
        "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "PANW",
    ],
    "Healthcare":  [
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        "DHR", "AMGN", "BMY", "GILD", "CI", "ISRG", "SYK",
        "BDX", "ZTS", "VRTX", "REGN", "HCA",
    ],
    "Financials":  [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP",
        "V", "MA", "C", "SCHW", "USB", "PNC", "COF",
        "ICE", "CME", "MCO", "SPGI", "TFC",
    ],
    "Consumer":    [
        "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX",
        "COST", "PG", "KO", "PEP", "PM", "CL", "KMB",
        "GIS", "K", "HSY", "MDLZ", "YUM",
    ],
    "Energy":      [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX",
        "OXY", "KMI", "WMB", "LNG", "DVN", "HES", "BKR",
    ],
    "Industrials": [
        "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE",
        "EMR", "ETN", "PH", "ROK", "ITW", "NSC", "UNP",
        "CSX", "FDX", "WM", "RSG", "FAST",
    ],
    "Utilities":   [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ED",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "AVB", "EQR",
    ],
    "Materials":   [
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "ALB",
    ],
}

SP500_TICKER = "^GSPC"

# Approximate sector median P/E ratios (used for value scoring)
SECTOR_MEDIAN_PE = {
    "Technology":  28,
    "Healthcare":  22,
    "Financials":  14,
    "Consumer":    24,
    "Energy":      12,
    "Industrials": 20,
    "Utilities":   18,
    "Real Estate": 35,
    "Materials":   16,
    "Unknown":     20,
}

# Weight matrix: [momentum, volatility, value, quality, dividend]
# Key: (risk_level 1–4, time_horizon "short"/"medium"/"long")
WEIGHT_MATRIX = {
    (1, "short"):  [0.10, 0.35, 0.20, 0.15, 0.20],
    (1, "medium"): [0.10, 0.30, 0.25, 0.15, 0.20],
    (1, "long"):   [0.10, 0.25, 0.30, 0.15, 0.20],
    (2, "short"):  [0.25, 0.20, 0.20, 0.20, 0.15],
    (2, "medium"): [0.20, 0.20, 0.25, 0.25, 0.10],
    (2, "long"):   [0.15, 0.20, 0.30, 0.30, 0.05],
    (3, "short"):  [0.45, 0.10, 0.15, 0.25, 0.05],
    (3, "medium"): [0.35, 0.10, 0.20, 0.30, 0.05],
    (3, "long"):   [0.25, 0.10, 0.25, 0.35, 0.05],
    (4, "short"):  [0.55, 0.05, 0.10, 0.25, 0.05],
    (4, "medium"): [0.45, 0.05, 0.15, 0.30, 0.05],
    (4, "long"):   [0.35, 0.05, 0.20, 0.35, 0.05],
}

RISK_LABELS = {
    1: "Low / Conservative",
    2: "Moderate / Balanced",
    3: "High / Aggressive",
    4: "Very High / Speculative",
}

HORIZON_LABELS = {
    "short":  "Short (1 year)",
    "medium": "Medium (3 years)",
    "long":   "Long (5 years)",
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: UserProfile dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    portfolio_size: float
    time_horizon: str       # "short" | "medium" | "long"
    time_horizon_years: int
    yf_period: str          # "1y" | "3y" | "5y"
    risk_label: str
    risk_level: int         # 1–4


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: InputCollector
# ─────────────────────────────────────────────────────────────────────────────
class InputCollector:
    """Collects and validates all user preferences via CLI prompts."""

    def collect(self) -> UserProfile:
        self._print_welcome()
        while True:
            portfolio_size = self._ask_portfolio_size()
            time_horizon, time_horizon_years, yf_period = self._ask_time_horizon()
            risk_label, risk_level = self._ask_risk_tolerance()
            profile = UserProfile(
                portfolio_size=portfolio_size,
                time_horizon=time_horizon,
                time_horizon_years=time_horizon_years,
                yf_period=yf_period,
                risk_label=risk_label,
                risk_level=risk_level,
            )
            if self._confirm_profile(profile):
                return profile
            print("\nLet's start over.\n")

    # ── Welcome ───────────────────────────────────────────────────────────────
    def _print_welcome(self):
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║          PERSONALIZED  STOCK  RANKING  ADVISOR               ║",
            "║   Real-time data from Yahoo Finance  ·  No API key needed    ║",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        print("\n" + "\n".join(lines) + "\n")

    # ── Portfolio size ────────────────────────────────────────────────────────
    def _ask_portfolio_size(self) -> float:
        print("─" * 60)
        print("STEP 1 of 3 — Portfolio Size")
        print("─" * 60)
        while True:
            raw = input(
                "How much are you planning to invest?\n"
                "(e.g.  5000 · 10k · 1.5m · $50,000): "
            ).strip()
            try:
                value = self._parse_amount(raw)
                if value <= 0:
                    print("  Must be a positive amount. Try again.\n")
                    continue
                if value < 1_000:
                    print(f"  Note: Small portfolio (${value:,.0f}) — diversification may be limited.\n")
                elif value > 10_000_000:
                    print("  Note: Large portfolio — consider consulting a licensed financial advisor.\n")
                print()
                return value
            except (ValueError, ZeroDivisionError):
                print(f"  Could not parse '{raw}'. Please enter a number (e.g. 10000 or 10k).\n")

    def _parse_amount(self, raw: str) -> float:
        raw = raw.strip().replace(",", "").replace("$", "").replace(" ", "")
        if not raw:
            raise ValueError("empty")
        multiplier = 1
        if raw.lower().endswith("k"):
            multiplier = 1_000
            raw = raw[:-1]
        elif raw.lower().endswith("m"):
            multiplier = 1_000_000
            raw = raw[:-1]
        return float(raw) * multiplier

    # ── Time horizon ──────────────────────────────────────────────────────────
    def _ask_time_horizon(self) -> Tuple[str, int, str]:
        print("─" * 60)
        print("STEP 2 of 3 — Investment Time Horizon")
        print("─" * 60)
        print("  1.  Short  term  (1 year or less)")
        print("  2.  Medium term  (2–5 years)")
        print("  3.  Long   term  (6+ years)")
        mapping: Dict[str, Tuple[str, int, str]] = {
            "1":      ("short",  1, "1y"),
            "short":  ("short",  1, "1y"),
            "2":      ("medium", 3, "3y"),
            "medium": ("medium", 3, "3y"),
            "3":      ("long",   5, "5y"),
            "long":   ("long",   5, "5y"),
        }
        while True:
            raw = input("Enter choice (1 / 2 / 3): ").strip().lower()
            if raw in mapping:
                print()
                return mapping[raw]
            print(f"  Invalid choice '{raw}'. Please enter 1, 2, or 3.\n")

    # ── Risk tolerance ────────────────────────────────────────────────────────
    def _ask_risk_tolerance(self) -> Tuple[str, int]:
        print("─" * 60)
        print("STEP 3 of 3 — Risk Tolerance")
        print("─" * 60)
        print("  1.  Low / Conservative     – Capital preservation, stable returns")
        print("  2.  Moderate / Balanced    – Mix of growth and stability")
        print("  3.  High / Aggressive      – Growth-focused, comfortable with swings")
        print("  4.  Very High / Speculative – Maximum growth, high volatility OK")
        mapping: Dict[str, Tuple[str, int]] = {
            "1": (RISK_LABELS[1], 1), "low": (RISK_LABELS[1], 1), "conservative": (RISK_LABELS[1], 1),
            "2": (RISK_LABELS[2], 2), "moderate": (RISK_LABELS[2], 2), "balanced": (RISK_LABELS[2], 2),
            "medium": (RISK_LABELS[2], 2),
            "3": (RISK_LABELS[3], 3), "high": (RISK_LABELS[3], 3), "aggressive": (RISK_LABELS[3], 3),
            "4": (RISK_LABELS[4], 4), "very high": (RISK_LABELS[4], 4),
            "very_high": (RISK_LABELS[4], 4), "speculative": (RISK_LABELS[4], 4),
        }
        while True:
            raw = input("Enter choice (1 / 2 / 3 / 4): ").strip().lower()
            if raw in mapping:
                print()
                return mapping[raw]
            print(f"  Invalid choice '{raw}'. Please enter 1, 2, 3, or 4.\n")

    # ── Confirmation ──────────────────────────────────────────────────────────
    def _confirm_profile(self, p: UserProfile) -> bool:
        print("═" * 60)
        print("  YOUR INVESTMENT PROFILE")
        print("═" * 60)
        print(f"  Portfolio Size  :  ${p.portfolio_size:>12,.2f}")
        print(f"  Time Horizon    :  {HORIZON_LABELS[p.time_horizon]}")
        print(f"  Risk Tolerance  :  {p.risk_label}")
        print("═" * 60)
        while True:
            ans = input("  Proceed with this profile? (y / n): ").strip().lower()
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False
            print("  Please enter y or n.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DataFetcher
# ─────────────────────────────────────────────────────────────────────────────
class DataFetcher:
    """Fetches stock data from Yahoo Finance in batches, gracefully handles failures."""

    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.failed: List[str] = []

    def fetch_universe_data(self, tickers: List[str]) -> Dict:
        results: Dict = {}
        total = len(tickers)
        done = 0
        batches = [tickers[i:i + 20] for i in range(0, total, 20)]

        print(f"\nFetching data for {total} stocks  ({len(batches)} batches) ...")
        for batch_idx, batch in enumerate(batches, 1):
            for ticker in batch:
                data = self._fetch_single(ticker)
                if data:
                    results[ticker] = data
                else:
                    self.failed.append(ticker)
                done += 1
                pct = done / total * 100
                filled = int(pct / 5)
                bar = "█" * filled + "░" * (20 - filled)
                print(f"  [{bar}] {pct:4.0f}%  {done}/{total}", end="\r", flush=True)
            if batch_idx < len(batches):
                time.sleep(0.4)

        print(f"  [{'█'*20}] 100%  {done}/{done}" + " " * 15)
        print(f"  Loaded: {len(results)} stocks  |  Skipped: {len(self.failed)}")
        return results

    def _fetch_single(self, ticker: str) -> Optional[dict]:
        for attempt in range(3):
            try:
                sink = io.StringIO()
                with contextlib.redirect_stderr(sink):
                    t       = yf.Ticker(ticker)
                    info    = t.info
                    if not info or len(info) < 5:
                        return None
                    history = t.history(period=self.profile.yf_period)
                if history is None or len(history) < 63:
                    return None
                sector = self._map_sector(info.get("sector", ""), ticker)
                return {"info": info, "history": history, "sector": sector}
            except Exception:
                if attempt < 2:
                    time.sleep(1)
        return None

    def _map_sector(self, yf_sector: str, ticker: str) -> str:
        mapping = {
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
            "Communication Services": "Technology",
        }
        if yf_sector in mapping:
            return mapping[yf_sector]
        for sector, tlist in STOCK_UNIVERSE.items():
            if ticker in tlist:
                return sector
        return "Unknown"

    def fetch_sp500(self) -> Optional[pd.DataFrame]:
        print(f"Fetching S&P 500 benchmark ({SP500_TICKER}) ...")
        for attempt in range(3):
            try:
                sink = io.StringIO()
                with contextlib.redirect_stderr(sink):
                    sp   = yf.Ticker(SP500_TICKER)
                    hist = sp.history(period=self.profile.yf_period)
                if hist is not None and len(hist) > 10:
                    print("  S&P 500 loaded.\n")
                    return hist
            except Exception:
                if attempt < 2:
                    time.sleep(1)
        print("  Warning: S&P 500 data unavailable — benchmark line will be omitted.\n")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: StockScorer
# ─────────────────────────────────────────────────────────────────────────────
class StockScorer:
    """Computes five component scores per stock, normalizes cross-sectionally,
    and produces a weighted composite score tuned to the user's risk profile."""

    def __init__(self, profile: UserProfile):
        self.profile = profile

    # ── Main entry ────────────────────────────────────────────────────────────
    def score_all(self, universe_data: Dict) -> pd.DataFrame:
        rows = [self._score_ticker(t, d) for t, d in universe_data.items()]
        rows = [r for r in rows if r is not None]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = self._apply_filter(df, strict=True)

        # Relax if we can't form a top 10
        if len(df) < 10:
            df_all = pd.DataFrame([r for r in rows if r is not None])
            df = self._apply_filter(df_all, strict=False)

        # Cross-sectional normalization (0–100) per component
        raw_to_score = {
            "momentum_raw":   "momentum_score",
            "volatility_raw": "volatility_score",
            "value_raw":      "value_score",
            "quality_raw":    "quality_score",
            "dividend_raw":   "dividend_score",
        }
        for raw_col, score_col in raw_to_score.items():
            df[score_col] = self._normalize(df[raw_col])

        # Weighted composite
        w = WEIGHT_MATRIX[(self.profile.risk_level, self.profile.time_horizon)]
        df["composite_score"] = (
            w[0] * df["momentum_score"]   +
            w[1] * df["volatility_score"] +
            w[2] * df["value_score"]      +
            w[3] * df["quality_score"]    +
            w[4] * df["dividend_score"]
        )

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    # ── Per-ticker scoring ────────────────────────────────────────────────────
    def _score_ticker(self, ticker: str, data: dict) -> Optional[dict]:
        info    = data["info"]
        history = data["history"]
        sector  = data["sector"]
        close   = history["Close"].dropna()

        if len(close) < 63:
            return None

        price = float(close.iloc[-1])
        if price < 5:
            return None

        # ── Momentum
        r1m = self._ret(close, 21)
        r3m = self._ret(close, 63)
        r6m = self._ret(close, 126)
        momentum_raw = (
            0.20 * (r1m if r1m is not None else 0.0) +
            0.35 * (r3m if r3m is not None else 0.0) +
            0.45 * (r6m if r6m is not None else 0.0)
        )

        # ── Volatility (negated: low vol → higher score)
        daily_ret = close.pct_change().dropna()
        vol = float(daily_ret.std()) * math.sqrt(252)
        volatility_raw = -vol

        # ── Value (lower P/E relative to sector = better)
        pe = info.get("trailingPE", None)
        if pe and 0 < float(pe) <= 1000:
            sector_pe  = SECTOR_MEDIAN_PE.get(sector, 20)
            value_raw  = max(-(float(pe) / sector_pe), -3.0)
        else:
            value_raw = float("nan")

        # ── Quality
        pm  = info.get("profitMargins", None)
        roe = info.get("returnOnEquity", None)
        pm  = self._clamp(pm,  -1.0, 1.0)
        roe = self._clamp(roe, -1.0, 1.0)
        valid_q = [v for v in [pm, roe] if v is not None]
        quality_raw = float(np.mean(valid_q)) if valid_q else float("nan")

        # ── Dividend
        div = float(info.get("dividendYield", 0) or 0)
        div = min(div, 0.15)

        # ── Filter-relevant fields
        beta       = float(info.get("beta", 1.0) or 1.0)
        market_cap = float(info.get("marketCap", 0) or 0)

        return {
            "ticker":        ticker,
            "sector":        sector,
            "current_price": price,
            "beta":          beta,
            "vol":           vol,
            "market_cap":    market_cap,
            "momentum_raw":  momentum_raw,
            "volatility_raw":volatility_raw,
            "value_raw":     value_raw,
            "quality_raw":   quality_raw,
            "dividend_raw":  div,
            "div_pct":       div * 100,
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

    def _normalize(self, s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) == 0 or valid.max() == valid.min():
            return pd.Series(50.0, index=s.index)
        normalized = (s - valid.min()) / (valid.max() - valid.min()) * 100
        return normalized.fillna(50.0)

    def _apply_filter(self, df: pd.DataFrame, strict: bool) -> pd.DataFrame:
        level = self.profile.risk_level
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Visualizer
# ─────────────────────────────────────────────────────────────────────────────
class Visualizer:
    """Generates two matplotlib figures: score bar chart + performance line chart."""

    SECTOR_COLORS = {
        "Technology":  "#58a6ff",
        "Healthcare":  "#3fb950",
        "Financials":  "#f78166",
        "Consumer":    "#d2a8ff",
        "Energy":      "#e3b341",
        "Industrials": "#f0883e",
        "Utilities":   "#79c0ff",
        "Real Estate": "#56d364",
        "Materials":   "#ffa657",
        "Unknown":     "#8b949e",
    }
    BG_DARK  = "#0d1117"
    BG_PANEL = "#161b22"
    BORDER   = "#30363d"

    def __init__(self, profile: UserProfile, sp500: Optional[pd.DataFrame]):
        self.profile = profile
        self.sp500   = sp500

    # ── Figure 1: Score bar chart ─────────────────────────────────────────────
    def score_chart(self, top10: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(self.BG_DARK)
        ax.set_facecolor(self.BG_PANEL)

        tickers = top10["ticker"].tolist()[::-1]
        scores  = top10["composite_score"].tolist()[::-1]
        sectors = top10["sector"].tolist()[::-1]
        colors  = [self.SECTOR_COLORS.get(s, self.SECTOR_COLORS["Unknown"]) for s in sectors]

        bars = ax.barh(tickers, scores, color=colors, edgecolor=self.BORDER, height=0.6)
        for bar, score in zip(bars, scores):
            ax.text(
                min(score + 1.0, 101),
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}",
                va="center", ha="left", fontsize=10,
                color="white", fontweight="bold",
            )

        ax.set_xlim(0, 107)
        ax.set_xlabel("Composite Score  (0 – 100)", color="#8b949e", fontsize=11)
        ax.set_title(
            f"Top 10 Stocks  —  {self.profile.risk_label}\n"
            f"{HORIZON_LABELS[self.profile.time_horizon]} Horizon",
            color="white", fontsize=13, fontweight="bold", pad=14,
        )
        ax.tick_params(colors="#8b949e", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.BORDER)
        ax.grid(axis="x", alpha=0.15, color="white")
        ax.set_axisbelow(True)

        # Sector colour legend
        seen: Dict[str, str] = {}
        handles = []
        for s, c in zip(sectors, colors):
            if s not in seen:
                seen[s] = c
                handles.append(mpatches.Patch(color=c, label=s))
        ax.legend(
            handles=handles, loc="lower right",
            facecolor="#1c2128", labelcolor="white",
            edgecolor=self.BORDER, fontsize=9,
        )

        fig.tight_layout(pad=1.5)
        return fig

    # ── Figure 2: Performance line chart ─────────────────────────────────────
    def performance_chart(self, top10: pd.DataFrame, universe_data: Dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(self.BG_DARK)
        ax.set_facecolor(self.BG_PANEL)

        cmap = plt.cm.get_cmap("tab10")
        start_date = None

        # S&P 500 benchmark
        if self.sp500 is not None and len(self.sp500) > 0:
            sp_close = self._strip_tz(self.sp500["Close"].dropna())
            start_date = sp_close.index[0]
            sp_norm = sp_close / sp_close.iloc[0] * 100
            sp_ret  = (sp_close.iloc[-1] / sp_close.iloc[0] - 1) * 100
            ax.plot(
                sp_norm.index, sp_norm.values,
                color="white", linewidth=2.8, zorder=10,
                label=f"S&P 500  ({sp_ret:+.1f}%)",
            )

        ax.axhline(100, color="#555", linestyle="--", linewidth=1, alpha=0.6)

        # Top 10 stocks
        for i, (_, row) in enumerate(top10.iterrows()):
            ticker = row["ticker"]
            if ticker not in universe_data:
                continue
            close = self._strip_tz(universe_data[ticker]["history"]["Close"].dropna())
            if start_date is not None:
                close = close[close.index >= start_date]
            if len(close) < 10:
                continue

            norm    = close / close.iloc[0] * 100
            tot_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
            color   = cmap(i / 10)

            ax.plot(
                norm.index, norm.values,
                color=color, linewidth=1.6, alpha=0.88,
                label=f"{ticker}  ({tot_ret:+.1f}%)",
            )
            # Ticker label at line end
            ax.annotate(
                ticker,
                xy=(norm.index[-1], float(norm.values[-1])),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=8, va="center",
            )

        ax.set_title(
            f"Historical Performance vs S&P 500  —  {HORIZON_LABELS[self.profile.time_horizon]}",
            color="white", fontsize=13, fontweight="bold", pad=14,
        )
        ax.set_xlabel("Date", color="#8b949e", fontsize=11)
        ax.set_ylabel("Normalised Price  (Base = 100)", color="#8b949e", fontsize=11)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor(self.BORDER)
        ax.grid(alpha=0.12, color="white")

        leg = ax.legend(
            loc="upper left", facecolor="#1c2128", labelcolor="white",
            edgecolor=self.BORDER, fontsize=8, ncol=2,
        )

        fig.tight_layout(pad=1.5)
        return fig

    def _strip_tz(self, s: pd.Series) -> pd.Series:
        """Remove timezone info from a Series index so comparisons are safe."""
        if hasattr(s.index, "tz") and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TerminalDisplay
# ─────────────────────────────────────────────────────────────────────────────
class TerminalDisplay:
    """Prints ranked table, allocation suggestion, and disclaimer to stdout."""

    def show_results(self, top10: pd.DataFrame):
        W = 92
        print("\n" + "═" * W)
        print("  TOP 10 STOCKS FOR YOUR PROFILE")
        print("═" * W)
        hdr = (
            f"  {'#':<4} {'Ticker':<8} {'Sector':<15}"
            f" {'Score':>6} {'Mom':>7} {'Vol':>7} {'Val':>7} {'Qual':>7} {'Div%':>6}"
        )
        print(hdr)
        print("─" * W)
        for _, row in top10.iterrows():
            print(
                f"  {int(row['rank']):<4} "
                f"{row['ticker']:<8} "
                f"{row['sector']:<15} "
                f"{row['composite_score']:>6.1f} "
                f"{row['momentum_score']:>7.1f} "
                f"{row['volatility_score']:>7.1f} "
                f"{row['value_score']:>7.1f} "
                f"{row['quality_score']:>7.1f} "
                f"{row['div_pct']:>5.1f}%"
            )
        print("═" * W)

    def show_allocation(self, top10: pd.DataFrame, portfolio_size: float):
        W = 65
        print(f"\n{'─'*W}")
        print(f"  SUGGESTED PORTFOLIO ALLOCATION  (${portfolio_size:,.0f} total)")
        print(f"  Strategy: score-weighted across top 10")
        print(f"{'─'*W}")
        total_score = top10["composite_score"].sum()
        for _, row in top10.iterrows():
            weight        = row["composite_score"] / total_score
            amount        = weight * portfolio_size
            price         = row["current_price"]
            approx_shares = int(amount / price) if price > 0 else 0
            print(
                f"  #{int(row['rank']):<2}  {row['ticker']:<6}  │"
                f"  {weight*100:>5.1f}%  │"
                f"  ${amount:>9,.0f}  │"
                f"  ~{approx_shares} shares @ ${price:,.2f}"
            )
        print(f"{'─'*W}")

    def show_disclaimer(self):
        W = 65
        print(f"\n{'─'*W}")
        print("  DISCLAIMER")
        print(f"{'─'*W}")
        print("  This tool is for educational purposes only.")
        print("  Past performance does not guarantee future results.")
        print("  Rankings are quantitative and are NOT financial advice.")
        print("  Always do your own research before investing.")
        print(f"{'─'*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: StockAdvisor (Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────
class StockAdvisor:
    def run(self):
        # 1. Collect user profile
        profile = InputCollector().collect()

        # 2. Build ticker list
        all_tickers = [t for tlist in STOCK_UNIVERSE.values() for t in tlist]

        # 3. Fetch data
        fetcher       = DataFetcher(profile)
        universe_data = fetcher.fetch_universe_data(all_tickers)
        sp500_history = fetcher.fetch_sp500()

        if not universe_data:
            print("\nError: No stock data could be loaded. Check your internet connection.")
            sys.exit(1)

        # 4. Score and rank
        print("Scoring and ranking stocks ...")
        scorer    = StockScorer(profile)
        ranked_df = scorer.score_all(universe_data)

        if ranked_df.empty:
            print("No stocks passed the filters. Try a different risk profile.")
            sys.exit(0)

        top10 = ranked_df.head(10).copy()
        print(f"  Done — {len(ranked_df)} stocks scored, top 10 selected.\n")

        # 5. Terminal output
        display = TerminalDisplay()
        display.show_results(top10)
        display.show_allocation(top10, profile.portfolio_size)

        # 6. Charts
        print("\nGenerating charts ...")
        viz  = Visualizer(profile, sp500_history)
        fig1 = viz.score_chart(top10)
        fig2 = viz.performance_chart(top10, universe_data)

        fig1.savefig("stock_scores.png",       dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
        fig2.savefig("stock_performance.png",  dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
        print("  Saved:  stock_scores.png  |  stock_performance.png")

        # 7. Disclaimer
        display.show_disclaimer()

        # 8. Show charts interactively
        try:
            plt.show()
        except Exception:
            print("  (Interactive display unavailable — open the saved PNG files.)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    try:
        StockAdvisor().run()
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
