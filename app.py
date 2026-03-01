"""
app.py — Streamlit dashboard for Stock Ranking Advisor v3

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st

st.set_page_config(
    page_title="Stock Advisor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    STOCK_UNIVERSE, WEIGHT_MATRIX, FACTOR_NAMES,
    RISK_LABELS, GOAL_LABELS, HORIZON_LABELS,
)
from advisor.collector    import UserProfile
from advisor.fetcher      import DataFetcher, MacroFetcher
from advisor.scorer       import MultiFactorScorer
from advisor.portfolio    import PortfolioConstructor
from advisor.learner      import SessionMemory
from advisor.protocol     import ProtocolAnalyzer, GATE_SHORT, GATE_NAMES
from advisor.valuation    import ValuationEngine
from advisor.risk         import RiskEngine
from advisor.news_fetcher import NewsFetcher


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BLUE       = "#2563EB"
BLUE_LT    = "#EFF6FF"
GREEN      = "#059669"
GREEN_LT   = "#ECFDF5"
AMBER      = "#D97706"
AMBER_LT   = "#FFFBEB"
RED        = "#DC2626"
RED_LT     = "#FEF2F2"
ORANGE     = "#EA580C"
ORANGE_LT  = "#FFF7ED"
TEXT       = "#111827"
MUTED      = "#6B7280"
MUTED2     = "#9CA3AF"
BORDER     = "#E5E7EB"
GRAY_LT    = "#F9FAFB"

FACTOR_COLORS = ["#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4","#84CC16"]

SECTOR_COLORS = {
    "Technology":  "#3B82F6", "Healthcare":  "#10B981",
    "Financials":  "#F59E0B", "Consumer":    "#8B5CF6",
    "Energy":      "#F97316", "Industrials": "#6B7280",
    "Utilities":   "#06B6D4", "Real Estate": "#84CC16",
    "Materials":   "#EC4899", "Unknown":     "#9CA3AF",
}

SIGNAL_META = {
    "STRONG_BUY":        (GREEN,  GREEN_LT,  "STRONG BUY"),
    "BUY":               (BLUE,   BLUE_LT,   "BUY"),
    "HOLD_WATCH":        (AMBER,  AMBER_LT,  "HOLD / WATCH"),
    "WAIT":              (ORANGE, ORANGE_LT, "WAIT"),
    "AVOID_PEAK":        (RED,    RED_LT,    "AVOID PEAK"),
    "INSUFFICIENT_DATA": (MUTED,  GRAY_LT,   "NO DATA"),
}

# Left-border accent color per signal (for pick cards)
SIGNAL_ACCENT = {
    "STRONG_BUY": GREEN, "BUY": BLUE, "HOLD_WATCH": AMBER,
    "WAIT": ORANGE, "AVOID_PEAK": RED, "INSUFFICIENT_DATA": BORDER,
}

CONV_META = {
    "HIGH":   (GREEN, GREEN_LT),
    "MEDIUM": (AMBER, AMBER_LT),
    "LOW":    (RED,   RED_LT),
}
ZONE_META = {
    "SAFE":     (GREEN, GREEN_LT),
    "GRAY":     (AMBER, AMBER_LT),
    "DISTRESS": (RED,   RED_LT),
}


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
[data-testid="stSidebar"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    letter-spacing: .01em;
}

/* ── Tabs — underline style ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 2px solid #E5E7EB;
    padding: 0;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 10px 24px;
    font-size: 13px;
    font-weight: 500;
    color: #9CA3AF;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent !important;
    transition: color .15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #374151 !important; }
.stTabs [aria-selected="true"] {
    color: #111827 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #2563EB !important;
    box-shadow: none !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Animations ──────────────────────────────────────────────────────── */
@keyframes barGrow {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Metric tile — with left accent border ───────────────────────────── */
.mtile {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 3px solid var(--accent, #E5E7EB);
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
    transition: box-shadow .2s ease, transform .2s ease;
}
.mtile:hover {
    box-shadow: 0 4px 18px rgba(0,0,0,.07);
    transform: translateY(-1px);
}
.mtile-lbl {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #9CA3AF;
    margin-bottom: 8px;
}
.mtile-val {
    font-size: 22px;
    font-weight: 800;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
}
.mtile-sub {
    font-size: 11px;
    color: #9CA3AF;
    margin-top: 5px;
    line-height: 1.4;
}

/* ── Badge ───────────────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .05em;
    text-transform: uppercase;
    white-space: nowrap;
}
.badge::before {
    content: '';
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
}

/* ── Table ───────────────────────────────────────────────────────────── */
.qt { width: 100%; border-collapse: collapse; font-size: 13px; }
.qt th {
    background: #F9FAFB;
    color: #374151;
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .08em;
    padding: 10px 14px;
    border-bottom: 2px solid #E5E7EB;
    text-align: left;
    white-space: nowrap;
}
.qt td {
    padding: 10px 14px;
    border-bottom: 1px solid #F3F4F6;
    color: #111827;
    vertical-align: middle;
}
.qt tr:last-child td { border-bottom: none; }
.qt tbody tr { transition: background .1s ease; }
.qt tbody tr:hover td { background: #F0F7FF; }

/* ── Rank chip ───────────────────────────────────────────────────────── */
.rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 6px;
    background: #1E3A8A;
    color: #fff;
    font-size: 10.5px;
    font-weight: 800;
}

/* ── Animated score bar ──────────────────────────────────────────────── */
.sbar-wrap {
    background: #F3F4F6;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
}
.sbar {
    height: 5px;
    border-radius: 4px;
    width: var(--w, 0%);
    transform-origin: left center;
    animation: barGrow .6s cubic-bezier(.4,0,.2,1) both;
}

/* ── Section header ──────────────────────────────────────────────────── */
.shdr {
    font-size: 14px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -.01em;
    margin-bottom: 2px;
}
.ssub { font-size: 11.5px; color: #9CA3AF; margin-bottom: 14px; }

/* ── Pick card — left border colored by signal ───────────────────────── */
.pick-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 4px solid var(--signal-color, #E5E7EB);
    border-radius: 10px;
    padding: 22px;
    height: 100%;
    animation: fadeUp .4s ease both;
    transition: transform .2s ease, box-shadow .2s ease;
}
.pick-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,.09);
}

/* ── Feature card (welcome page) ─────────────────────────────────────── */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-top: 3px solid var(--fc, #2563EB);
    border-radius: 10px;
    padding: 22px;
    height: 100%;
    animation: fadeUp .5s ease both;
    transition: transform .18s ease, box-shadow .18s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,.07);
}
.fc-num {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--fc, #2563EB);
    margin-bottom: 10px;
}
.fc-title {
    font-size: 14px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 7px;
}
.fc-desc { font-size: 12px; color: #6B7280; line-height: 1.65; }

/* ── Hero (welcome) ──────────────────────────────────────────────────── */
.hero { text-align: center; padding: 72px 20px 56px; }
.hero-eyebrow {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: #2563EB;
    margin-bottom: 18px;
}
.hero-title {
    font-size: 52px;
    font-weight: 900;
    color: #111827;
    letter-spacing: -.045em;
    line-height: 1;
    margin-bottom: 20px;
}
.hero-sub {
    font-size: 17px;
    color: #6B7280;
    max-width: 480px;
    margin: 0 auto 36px;
    line-height: 1.7;
    font-weight: 400;
}
.hero-note {
    font-size: 12px;
    color: #9CA3AF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    display: inline-block;
    padding: 8px 20px;
}

/* ── Page title ──────────────────────────────────────────────────────── */
.ptitle {
    font-size: 22px;
    font-weight: 800;
    color: #111827;
    letter-spacing: -.03em;
}
.psub { font-size: 12px; color: #9CA3AF; margin-top: 3px; margin-bottom: 20px; }

/* ── Sidebar brand ───────────────────────────────────────────────────── */
.sb-brand { font-size: 15px; font-weight: 800; color: #111827; letter-spacing: -.02em; }
.sb-brand-sub { font-size: 11px; color: #9CA3AF; margin-top: 2px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def badge(text, color, bg):
    return f'<span class="badge" style="color:{color};background:{bg}">{text}</span>'

def signal_badge(sig):
    c, bg, lbl = SIGNAL_META.get(sig, (MUTED, GRAY_LT, sig))
    return badge(lbl, c, bg)

def conv_badge(conv):
    c, bg = CONV_META.get(conv, (MUTED, GRAY_LT))
    return badge(conv, c, bg)

def zone_badge(zone):
    c, bg = ZONE_META.get(zone, (MUTED, GRAY_LT))
    return badge(zone, c, bg)

_NA = f'<span style="color:{BORDER};font-size:11px">n/a</span>'

def fmt_price(v):
    if v is None: return _NA
    try:    return f"${float(v):,.2f}"
    except: return _NA

def fmt_pct(v, plus=True):
    if v is None: return _NA
    try:    return f"{float(v):+.1f}%" if plus else f"{float(v):.1f}%"
    except: return _NA

def fmt_2(v):
    if v is None: return _NA
    try:    return f"{float(v):.2f}"
    except: return _NA

def score_color(s):
    s = s or 0
    if s >= 70: return GREEN
    if s >= 45: return AMBER
    return RED

def sbar(s, color=None):
    c = color or score_color(s or 0)
    w = min(s or 0, 100)
    return (
        f'<div class="sbar-wrap">'
        f'<div class="sbar" style="--w:{w:.0f}%;background:{c}"></div>'
        f'</div>'
    )

def mtile(label, value, sub="", color=TEXT, accent=None):
    acc = accent or color
    return (
        f'<div class="mtile" style="--accent:{acc}">'
        f'<div class="mtile-lbl">{label}</div>'
        f'<div class="mtile-val" style="color:{color}">{value}</div>'
        f'{"<div class=mtile-sub>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )

def shdr(title, sub=""):
    out = f'<div class="shdr">{title}</div>'
    if sub: out += f'<div class="ssub">{sub}</div>'
    return out

def _stat(label, value, color=TEXT):
    """Tiny inline stat block used inside pick cards."""
    return (
        f'<div>'
        f'<div style="font-size:9.5px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.09em;color:{MUTED2};margin-bottom:4px">{label}</div>'
        f'<div style="font-size:14px;font-weight:700;color:{color};'
        f'font-variant-numeric:tabular-nums">{value}</div>'
        f'</div>'
    )

def _plotly_base():
    """Common Plotly layout kwargs."""
    return dict(
        template="plotly_white",
        font=dict(family="Inter, -apple-system, sans-serif", size=12),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUANT THESIS GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _generate_quant_thesis(ticker: str, val: dict, risk: dict, proto_dict: dict) -> str:
    """
    Auto-generate a Citadel-style quant thesis paragraph for a stock.
    Returns an HTML string for use in a styled card.
    """
    parts = []

    # Valuation line
    sig      = val.get("signal", "INSUFFICIENT_DATA")
    fv       = val.get("fair_value")
    price    = val.get("current_price")
    prem     = val.get("premium_pct")
    rr       = val.get("rr_ratio")
    methods  = val.get("methods_count", 0)
    upside   = val.get("upside_pct")
    tgt      = val.get("target_price")

    if fv and price:
        direction = "below" if (prem or 0) < 0 else "above"
        abs_pct   = abs(prem or 0)
        sig_map   = {
            "STRONG_BUY":  "firmly in the STRONG BUY zone",
            "BUY":         "in the BUY zone",
            "HOLD_WATCH":  "at HOLD/WATCH — approaching fair value",
            "WAIT":        "slightly above fair value — best to WAIT for a pullback",
            "AVOID_PEAK":  "above our AVOID PEAK threshold — not a clean entry",
        }
        sig_text = sig_map.get(sig, "")
        rr_text  = f" Risk/reward of <b>{rr:.1f}:1</b>." if rr else ""
        upside_text = f" Price target <b>${tgt:,.2f}</b> implies <b>{upside:+.1f}%</b> upside." if (tgt and upside) else ""
        parts.append(
            f"<b>{ticker}</b> trades at <b>${price:,.2f}</b> — "
            f"<b>{abs_pct:.1f}%</b> {direction} our {methods}-method median fair value of "
            f"<b>${fv:,.2f}</b>, {sig_text}.{rr_text}{upside_text}"
        )

    # ROIC/WACC line
    rw = risk.get("roic_wacc", {})
    if rw.get("spread") is not None and rw.get("verdict"):
        spread  = rw["spread"]
        verdict = rw["verdict"]
        roic    = rw.get("roic", 0)
        wacc    = rw.get("wacc", 0)
        parts.append(
            f"ROIC/WACC spread of <b>{spread:+.1f}%</b> (<b>{roic:.1f}%</b> vs <b>{wacc:.1f}%</b> cost of capital) "
            f"— <b>{verdict}</b> value creation."
        )

    # Piotroski line
    pf = risk.get("piotroski", {})
    if pf.get("score") is not None:
        pscore = pf["score"]
        pinterp = pf.get("interpretation", "")
        pfcolor = "strong" if pscore >= 7 else "weak" if pscore <= 3 else "average"
        parts.append(
            f"Piotroski F-Score <b>{pscore}/9</b> ({pinterp}) — "
            f"{'fundamentals are robust' if pscore >= 7 else 'watch for deterioration' if pscore <= 3 else 'adequate fundamentals'}."
        )

    # Altman Z line
    az = risk.get("altman_z", {})
    if az.get("score") is not None:
        az_score = az["score"]
        az_zone  = az.get("zone", "—")
        az_text  = {
            "SAFE":     "balance sheet is healthy, distress risk is low",
            "GRAY":     "in the gray zone — monitor leverage closely",
            "DISTRESS": "distress risk elevated — high-risk position",
        }.get(az_zone, "")
        parts.append(
            f"Altman Z of <b>{az_score:.2f}</b> [{az_zone}] — {az_text}."
        )

    # Sharpe line
    sharpe  = risk.get("sharpe")
    sortino = risk.get("sortino")
    maxdd   = risk.get("max_drawdown_pct")
    if sharpe is not None:
        sharpe_quality = "excellent" if sharpe > 1.5 else "good" if sharpe > 0.8 else "poor"
        s_extra = f" Sortino <b>{sortino:.2f}</b>." if sortino else ""
        dd_extra = f" Max drawdown <b>{maxdd:.1f}%</b>." if maxdd else ""
        parts.append(
            f"Sharpe ratio of <b>{sharpe:.2f}</b> ({sharpe_quality} risk-adjusted return).{s_extra}{dd_extra}"
        )

    # Protocol line
    if proto_dict:
        pconv    = proto_dict.get("conviction", "—")
        pass_cnt = proto_dict.get("pass_count", 0)
        warn_cnt = proto_dict.get("warn_count", 0)
        fail_cnt = proto_dict.get("fail_count", 0)
        overall  = proto_dict.get("overall_score", 0)
        parts.append(
            f"7-gate protocol: <b>{pass_cnt} pass / {warn_cnt} warn / {fail_cnt} fail</b> "
            f"→ overall score <b>{overall:.0f}</b>, conviction <b>{pconv}</b>."
        )

    if not parts:
        return ""

    thesis = " ".join(parts)
    return thesis


def _analyst_targets_html(info: dict) -> str:
    """Render analyst target price distribution as styled HTML."""
    mean_t = info.get("targetMeanPrice")
    high_t = info.get("targetHighPrice")
    low_t  = info.get("targetLowPrice")
    med_t  = info.get("targetMedianPrice")
    n_ana  = info.get("numberOfAnalystOpinions")
    rec_key= info.get("recommendationKey", "")
    rec_mean = info.get("recommendationMean")

    if not mean_t:
        return ""

    def fp(v): return f"${float(v):,.2f}" if v else "—"
    cur = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    upside_pct = ((float(mean_t) / float(cur)) - 1) * 100 if cur and mean_t else None

    rec_label_map = {
        "strong_buy":  ("STRONG BUY", GREEN),
        "buy":         ("BUY", BLUE),
        "hold":        ("HOLD", AMBER),
        "underperform":("UNDERPERFORM", ORANGE),
        "sell":        ("SELL", RED),
    }
    rec_lbl, rec_c = rec_label_map.get(rec_key, (rec_key.upper() if rec_key else "—", MUTED))

    upside_html = ""
    if upside_pct is not None:
        uc = GREEN if upside_pct > 5 else RED if upside_pct < -5 else AMBER
        upside_html = f'<span style="color:{uc};font-weight:700;font-size:13px">{upside_pct:+.1f}%</span>'

    rows = [
        ("Analyst Consensus", f'<span class="badge" style="color:{rec_c};background:{rec_c}22">{rec_lbl}</span>' if rec_lbl != "—" else "—"),
        ("# Analysts",        str(int(n_ana)) if n_ana else "—"),
        ("Mean Target",       f"{fp(mean_t)}  {upside_html}"),
        ("Median Target",     fp(med_t)),
        ("High Target",       fp(high_t)),
        ("Low Target",        fp(low_t)),
    ]
    rows_html = "".join(
        f'<tr><td style="color:{MUTED};padding:6px 14px;font-size:12.5px">{k}</td>'
        f'<td style="padding:6px 14px;font-size:12.5px;text-align:right">{v}</td></tr>'
        for k, v in rows
    )
    return f'<table class="qt" style="width:100%"><tbody>{rows_html}</tbody></table>'


def _technical_summary_html(info: dict, hist) -> str:
    """Generate a plain-English technical status summary with color-coded metrics."""
    sma50  = info.get("fiftyDayAverage")
    sma200 = info.get("twoHundredDayAverage")
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    low52  = info.get("fiftyTwoWeekLow")
    high52 = info.get("fiftyTwoWeekHigh")

    rows = []

    # Price vs SMAs
    if price and sma200:
        price, sma200 = float(price), float(sma200)
        trend_c   = GREEN if price > sma200 else RED
        trend_lbl = "above SMA 200 (uptrend)" if price > sma200 else "below SMA 200 (downtrend)"
        vs_200    = ((price / sma200) - 1) * 100
        rows.append(("SMA 200",
                      f'<span style="color:{trend_c};font-weight:700">{trend_lbl}</span>'
                      f'<span style="color:{MUTED};font-size:11px"> ({vs_200:+.1f}%)</span>'))
    if price and sma50:
        price, sma50 = float(price), float(sma50)
        c50   = GREEN if price > sma50 else RED
        l50   = "above SMA 50" if price > sma50 else "below SMA 50"
        v50   = ((price / sma50) - 1) * 100
        rows.append(("SMA 50",
                      f'<span style="color:{c50};font-weight:700">{l50}</span>'
                      f'<span style="color:{MUTED};font-size:11px"> ({v50:+.1f}%)</span>'))

    # 52-week positioning
    if price and low52 and high52:
        price, low52, high52 = float(price), float(low52), float(high52)
        rng = high52 - low52
        if rng > 0:
            pct_range = (price - low52) / rng * 100
            pct_c     = RED if pct_range > 85 else GREEN if pct_range < 30 else AMBER
            rows.append(("52W Position",
                          f'<span style="color:{pct_c};font-weight:700">{pct_range:.0f}%</span>'
                          f'<span style="color:{MUTED};font-size:11px"> of 52W range</span>'
                          f'<span style="color:{MUTED2};font-size:11px"> (${low52:,.2f}–${high52:,.2f})</span>'))

    # RSI from price history
    if hist is not None and len(hist) >= 15:
        try:
            closes = hist["Close"].dropna().values.astype(float)
            deltas = np.diff(closes[-15:])
            gains  = deltas[deltas > 0].mean() if (deltas > 0).any() else 0
            losses = (-deltas[deltas < 0]).mean() if (deltas < 0).any() else 0
            if losses > 0:
                rs  = gains / losses
                rsi = 100 - (100 / (1 + rs))
                rsi_c   = RED if rsi > 70 else GREEN if rsi < 30 else AMBER
                rsi_lbl = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                rows.append(("RSI (14)",
                              f'<span style="color:{rsi_c};font-weight:700">{rsi:.1f} — {rsi_lbl}</span>'))
        except Exception:
            pass

    # Momentum from hist
    if hist is not None and len(hist) >= 63:
        try:
            closes   = hist["Close"].dropna().values.astype(float)
            mom_3m   = (closes[-1] / closes[-63] - 1) * 100
            mom_c    = GREEN if mom_3m > 5 else RED if mom_3m < -5 else AMBER
            rows.append(("3M Momentum",
                          f'<span style="color:{mom_c};font-weight:700">{mom_3m:+.1f}%</span>'))
        except Exception:
            pass

    if not rows:
        return ""

    rows_html = "".join(
        f'<tr><td style="color:{MUTED};padding:6px 14px;font-size:12.5px">{k}</td>'
        f'<td style="padding:6px 14px;font-size:12.5px;text-align:right">{v}</td></tr>'
        for k, v in rows
    )
    return f'<table class="qt" style="width:100%"><tbody>{rows_html}</tbody></table>'


def _factor_bars_html(ticker: str) -> str:
    """
    Render 7-factor score bars for a ticker using ranked_df from session state.
    Returns empty string if not available.
    """
    results = st.session_state.get("results") or {}
    ranked_df = results.get("ranked_df")
    if ranked_df is None or ranked_df.empty:
        return ""
    row = ranked_df[ranked_df["ticker"] == ticker]
    if row.empty:
        return ""
    row = row.iloc[0]

    bars = []
    for i, factor in enumerate(FACTOR_NAMES):
        col = f"{factor}_score"
        score = float(row.get(col, 0))
        color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
        pct   = min(max(score, 0), 100)
        bars.append(
            f'<div style="margin-bottom:6px">'
            f'  <div style="display:flex;justify-content:space-between;'
            f'       font-size:11px;color:{MUTED};margin-bottom:3px">'
            f'    <span style="font-weight:600;text-transform:capitalize">{factor}</span>'
            f'    <span style="font-family:monospace;font-weight:700;color:{TEXT}">{score:.0f}</span>'
            f'  </div>'
            f'  <div style="background:{BORDER};border-radius:4px;height:7px;overflow:hidden">'
            f'    <div style="width:{pct}%;height:100%;background:{color};'
            f'         border-radius:4px;transition:width .6s"></div>'
            f'  </div>'
            f'</div>'
        )
    return (
        f'<div style="background:{GRAY_LT};border:1px solid {BORDER};border-radius:10px;'
        f'padding:16px 18px">'
        + "".join(bars)
        + "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(profile: UserProfile) -> dict:
    res  = {}
    prog = st.progress(0, text="Fetching stock data from Yahoo Finance…")

    all_tickers = [
        t for sector, tlist in STOCK_UNIVERSE.items()
        for t in tlist
        if sector not in profile.excluded_sectors
        and t not in profile.existing_tickers
    ]
    all_tickers = list(dict.fromkeys(all_tickers))

    fetcher              = DataFetcher(profile.yf_period)
    res["universe_data"] = fetcher.fetch_universe(all_tickers)
    res["sp500_hist"]    = fetcher.fetch_sp500()
    prog.progress(28, text="Fetching macro data (VIX · 10Y yield · sector ETFs)…")

    res["macro_data"] = MacroFetcher().fetch()
    rf_rate           = (res["macro_data"].get("yield_10y") or 4.5) / 100
    res["rf_rate"]    = rf_rate
    prog.progress(42, text="Scoring stocks on 7 factors…")

    memory  = SessionMemory(); memory.load()
    adapted = memory.get_adapted_weights(profile.risk_level, profile.time_horizon)
    scorer  = MultiFactorScorer(profile, res["macro_data"], adapted)
    ranked_df = scorer.score_all(res["universe_data"])

    # Fresh picks mode: penalise tickers from last 2 sessions
    if profile.avoid_recent:
        recent_tickers = memory.get_recent_tickers(n_sessions=2)
        if recent_tickers:
            PENALTY = 22.0
            mask = ranked_df["ticker"].isin(recent_tickers)
            ranked_df.loc[mask, "composite_score"] = (
                ranked_df.loc[mask, "composite_score"] - PENALTY
            ).clip(lower=0)
            ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    res["ranked_df"] = ranked_df
    prog.progress(56, text="Building portfolio (correlation-aware selection)…")

    constructor  = PortfolioConstructor()
    top10        = constructor.select(res["ranked_df"], res["universe_data"])
    res["top10"] = constructor.size_positions(top10, profile.portfolio_size)
    prog.progress(66, text="Running multi-method valuation (DCF · Graham · EV/EBITDA · FCF)…")

    res["valuation"] = ValuationEngine(rf_rate).analyze_all(res["top10"], res["universe_data"])
    prog.progress(78, text="Computing risk metrics (Altman Z · Sharpe · Sortino · Piotroski)…")

    res["risk"] = RiskEngine().analyze_all(res["top10"], res["universe_data"], rf_rate)
    prog.progress(90, text="Running 7-gate investment protocol…")

    res["protocol"] = ProtocolAnalyzer().analyze_all(
        res["top10"], res["universe_data"], res["valuation"]
    )
    prog.progress(100, text="Done.")
    prog.empty()
    return res


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="sb-brand">Stock Advisor</div>'
            '<div class="sb-brand-sub">Hedge-fund grade · Free data only</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**Investor Profile**")

        portfolio_size = st.number_input(
            "Portfolio Size ($)",
            min_value=1_000, max_value=100_000_000,
            value=50_000, step=1_000, format="%d",
        )
        time_horizon_key = st.selectbox(
            "Time Horizon",
            options=["short", "medium", "long"],
            format_func=lambda x: HORIZON_LABELS[x],
        )
        horizon_map = {"short": (1, "1y"), "medium": (3, "3y"), "long": (5, "5y")}
        hy, yf_period = horizon_map[time_horizon_key]

        risk_level = st.selectbox(
            "Risk Level",
            options=[1, 2, 3, 4],
            format_func=lambda x: RISK_LABELS[x],
            index=1,
        )
        goal_key = st.selectbox(
            "Investment Goal",
            options=["retirement", "wealth", "income", "speculative"],
            format_func=lambda x: GOAL_LABELS[x],
        )
        drawdown_val = st.select_slider(
            "Max Drawdown Tolerance",
            options=[0.10, 0.15, 0.25, 0.40],
            value=0.25,
            format_func=lambda x: f"{x*100:.0f}%",
        )

        all_sectors = list(STOCK_UNIVERSE.keys())
        excluded  = st.multiselect("Exclude Sectors", options=all_sectors)
        preferred = st.multiselect(
            "Focus Sectors (score bonus)",
            options=[s for s in all_sectors if s not in excluded],
        )
        existing_raw     = st.text_input("Existing Holdings", placeholder="AAPL, MSFT, …")
        existing_tickers = [t.strip().upper() for t in existing_raw.split(",") if t.strip()]
        avoid_recent     = st.checkbox(
            "Fresh picks mode",
            value=False,
            help="Penalises stocks from your last 2 sessions so the tool recommends new ideas each run.",
        )

        st.divider()
        run_btn   = st.button("Run Analysis",   type="primary",    use_container_width=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            hist_btn = st.button("Past Sessions", type="secondary", use_container_width=True)
        with col_b2:
            bt_btn   = st.button("Backtest",      type="secondary", use_container_width=True)

    profile = UserProfile(
        portfolio_size    = float(portfolio_size),
        time_horizon      = time_horizon_key,
        time_horizon_years= hy,
        yf_period         = yf_period,
        risk_label        = RISK_LABELS[risk_level],
        risk_level        = risk_level,
        goal              = goal_key,
        goal_label        = GOAL_LABELS[goal_key],
        drawdown_ok       = drawdown_val,
        preferred_sectors = preferred,
        excluded_sectors  = excluded,
        existing_tickers  = existing_tickers,
        avoid_recent      = avoid_recent,
    )
    return profile, run_btn, hist_btn, bt_btn


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME
# ─────────────────────────────────────────────────────────────────────────────
def render_welcome():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Pure Quantitative · No AI APIs · Free Data Only</div>
      <div class="hero-title">Stock Ranking<br>Advisor</div>
      <div class="hero-sub">
        Hedge-fund grade analysis built entirely on Yahoo Finance and serious math.
        No subscriptions. No API keys.
      </div>
      <div class="hero-note">Configure your profile in the sidebar, then click Run Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("#2563EB", "01", "7-Factor Scoring",   "12-1 momentum · EV/EBITDA value · Novy-Marx quality · 5-factor technicals"),
        ("#059669", "02", "4-Method Valuation",  "DCF (2-stage) · Graham Number · EV/EBITDA target · FCF yield"),
        ("#D97706", "03", "Full Risk Suite",     "Altman Z · Sharpe · Sortino · Max DD · VaR 95% · ROIC/WACC"),
        ("#7C3AED", "04", "7-Gate Protocol",     "Warren Buffett quality screen — every stock must earn its place"),
    ]
    cols = st.columns(4)
    for i, (col, (fc, num, title, desc)) in enumerate(zip(cols, features)):
        with col:
            st.markdown(
                f'<div class="feature-card" style="--fc:{fc};animation-delay:{i*0.08}s">'
                f'<div class="fc-num">{num}</div>'
                f'<div class="fc-title">{title}</div>'
                f'<div class="fc-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# MACRO STRIP
# ─────────────────────────────────────────────────────────────────────────────
def render_macro_strip(macro, top10, rf_rate):
    vix    = macro.get("vix")
    y10    = macro.get("yield_10y")
    regime = macro.get("regime", "neutral").upper().replace("_", " ")
    reasons = "  ·  ".join(macro.get("regime_reasons", []))
    n      = len(top10)
    avg_sc = float(top10["composite_score"].mean()) if "composite_score" in top10.columns else 0

    r_color  = GREEN if "RISK ON" in regime else RED if "RISK OFF" in regime else AMBER
    vix_color = RED if (vix or 0) > 25 else AMBER if (vix or 0) > 18 else GREEN
    avg_clr  = score_color(avg_sc)

    c1, c2, c3, c4, c5 = st.columns(5)
    tiles = [
        (c1, "VIX",           f"{vix:.1f}" if vix else "N/A",    "Elevated — use caution" if (vix or 0) > 25 else "Normal range", vix_color, vix_color),
        (c2, "10-Year Yield", f"{y10:.2f}%" if y10 else "N/A",   f"Risk-free rate  ·  rf = {rf_rate*100:.2f}%", TEXT, BLUE),
        (c3, "Regime",        regime,                             reasons[:52] if reasons else "—", r_color, r_color),
        (c4, "In Portfolio",  str(n),                             "Stocks selected", BLUE, BLUE),
        (c5, "Avg Score",     f"{avg_sc:.1f}",                    "Portfolio composite / 100", avg_clr, avg_clr),
    ]
    for col, lbl, val, sub, color, accent in tiles:
        with col:
            st.markdown(mtile(lbl, val, sub, color, accent), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
def tab_rankings(top10, profile, valuation, protocol, risk=None):
    proto_map = {p["ticker"]: p for p in protocol}
    uni_data  = st.session_state.results.get("universe_data", {}) if st.session_state.results else {}

    # ── Top-3 highlight cards ──────────────────────────────────────────────
    cols = st.columns(3)
    for i, (col, (_, row)) in enumerate(zip(cols, top10.head(3).iterrows())):
        t      = row["ticker"]
        val    = valuation.get(t, {})
        sig    = val.get("signal", "INSUFFICIENT_DATA")
        sc     = float(row["composite_score"])
        c_sig, bg_sig, lbl_sig = SIGNAL_META.get(sig, (MUTED, GRAY_LT, sig))
        accent = SIGNAL_ACCENT.get(sig, BORDER)
        conv   = proto_map.get(t, {}).get("conviction", "—")
        c_cv, bg_cv = CONV_META.get(conv, (MUTED, GRAY_LT))
        fv     = val.get("fair_value")
        prem   = val.get("premium_pct")
        prem_str = f"{prem:+.1f}%" if prem is not None else "—"
        prem_clr  = RED if (prem or 0) > 5 else GREEN if (prem or 0) < -10 else AMBER
        rank      = int(row.get("rank", i + 1))
        days_away = uni_data.get(t, {}).get("earnings_days_away")
        edate     = uni_data.get(t, {}).get("earnings_date", "")

        # Earnings badge
        if days_away is not None and days_away <= 30:
            earn_c  = RED   if days_away <= 7  else AMBER if days_away <= 14 else MUTED
            earn_bg = "#FEF2F2" if days_away <= 7 else "#FFFBEB" if days_away <= 14 else GRAY_LT
            earn_badge = (f'<span class="badge" style="color:{earn_c};background:{earn_bg};'
                          f'font-size:9px">EARNINGS {days_away}d</span>')
        else:
            earn_badge = ""

        with col:
            st.markdown(
                f'<div class="pick-card" style="--signal-color:{accent};animation-delay:{i*0.08}s">'

                # Header row
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:flex-start;margin-bottom:18px">'
                f'  <div>'
                f'    <div style="font-size:10px;font-weight:700;text-transform:uppercase;'
                f'         letter-spacing:.1em;color:{MUTED2};margin-bottom:5px">Rank #{rank}</div>'
                f'    <div style="font-size:28px;font-weight:900;color:{TEXT};'
                f'         letter-spacing:-.03em;line-height:1">{t}</div>'
                f'    <div style="font-size:12px;color:{MUTED};margin-top:4px">{row["sector"]}</div>'
                f'  </div>'
                f'  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:5px">'
                f'    <span class="badge" style="color:{c_sig};background:{bg_sig}">{lbl_sig}</span>'
                f'    <span class="badge" style="color:{c_cv};background:{bg_cv}">{conv}</span>'
                f'    {earn_badge}'
                f'  </div>'
                f'</div>'

                # Score
                f'<div style="margin-bottom:10px">'
                f'  <span style="font-size:42px;font-weight:900;letter-spacing:-.04em;'
                f'       color:{score_color(sc)};line-height:1">{sc:.1f}</span>'
                f'  <span style="font-size:16px;color:{MUTED2};font-weight:500"> /100</span>'
                f'</div>'
                f'{sbar(sc)}'

                # Divider
                f'<div style="height:1px;background:#F3F4F6;margin:16px 0"></div>'

                # Stats grid
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">'
                f'{_stat("Fair Value", fmt_price(fv))}'
                f'{_stat("Premium", prem_str, prem_clr)}'
                f'{_stat("Entry Zone", fmt_price(val.get("entry_low")), GREEN)}'
                f'</div>'

                f'</div>',
                unsafe_allow_html=True,
            )
            # Clickable "View Analysis" button below card
            if st.button("View Full Analysis", key=f"pick_detail_{t}_{i}",
                         use_container_width=True):
                st.session_state.rankings_selected = t
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Rankings table ─────────────────────────────────────────────────────
    st.markdown(shdr("All Rankings", "Complete top-10 with valuation signal and protocol conviction"),
                unsafe_allow_html=True)
    rows = ""
    for _, row in top10.iterrows():
        t     = row["ticker"]
        val   = valuation.get(t, {})
        proto = proto_map.get(t, {})
        sig   = val.get("signal", "INSUFFICIENT_DATA")
        conv  = proto.get("conviction", "—")
        sc    = float(row["composite_score"])
        prem  = val.get("premium_pct")
        p_clr = RED if (prem or 0) > 5 else GREEN if (prem or 0) < -10 else AMBER
        tda   = uni_data.get(t, {}).get("earnings_days_away")
        earn_cell = ""
        if tda is not None and tda <= 30:
            ec = RED if tda <= 7 else AMBER if tda <= 14 else MUTED
            earn_cell = f'<span style="color:{ec};font-size:10px;font-weight:700">{tda}d</span>'
        rows += (
            f'<tr>'
            f'<td><span class="rank">{int(row["rank"])}</span></td>'
            f'<td><b style="font-size:13.5px;letter-spacing:-.01em">{t}</b></td>'
            f'<td style="color:{MUTED}">{row["sector"]}</td>'
            f'<td style="min-width:130px">'
            f'  <div style="display:flex;align-items:center;gap:8px">'
            f'    <span style="font-weight:700;color:{score_color(sc)};width:34px;'
            f'          font-variant-numeric:tabular-nums">{sc:.1f}</span>'
            f'    <div style="flex:1">{sbar(sc)}</div>'
            f'  </div>'
            f'</td>'
            f'<td>{signal_badge(sig)}</td>'
            f'<td>{conv_badge(conv)}</td>'
            f'<td style="font-family:monospace;font-weight:600">{fmt_price(row.get("current_price"))}</td>'
            f'<td style="font-family:monospace">{fmt_price(val.get("fair_value"))}</td>'
            f'<td style="font-family:monospace;color:{p_clr};font-weight:600">{fmt_pct(prem)}</td>'
            f'<td style="font-family:monospace;color:{GREEN};font-weight:600">{fmt_price(val.get("entry_low"))}</td>'
            f'<td style="font-family:monospace;color:{RED}">{fmt_price(val.get("stop_loss"))}</td>'
            f'<td style="text-align:center">{earn_cell}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="qt"><thead><tr>'
        f'<th>#</th><th>Ticker</th><th>Sector</th><th>Score</th>'
        f'<th>Signal</th><th>Conviction</th><th>Price</th>'
        f'<th>Fair Value</th><th>Premium</th><th>Entry (−20%)</th><th>Stop Loss</th>'
        f'<th>Earnings</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Factor breakdown chart ─────────────────────────────────────────────
    st.markdown(shdr("Factor Score Breakdown", "Weighted contribution of each factor to composite score"),
                unsafe_allow_html=True)

    weights   = WEIGHT_MATRIX.get((profile.risk_level, profile.time_horizon), [1/7]*7)
    tickers_r = top10["ticker"].tolist()[::-1]
    scores_r  = top10["composite_score"].tolist()[::-1]

    fig = go.Figure()
    for i, factor in enumerate(FACTOR_NAMES):
        col_name = f"{factor}_score"
        if col_name not in top10.columns:
            continue
        vals = (top10[col_name] * weights[i]).tolist()[::-1]
        fig.add_trace(go.Bar(
            name=factor.capitalize(), y=tickers_r, x=vals, orientation="h",
            marker_color=FACTOR_COLORS[i % len(FACTOR_COLORS)],
            hovertemplate=f"<b>{factor.capitalize()}</b><br>Contribution: %{{x:.1f}}<extra></extra>",
        ))
    for i, (t, sc) in enumerate(zip(tickers_r, scores_r)):
        fig.add_annotation(
            x=sc + 0.8, y=i, text=f"<b>{sc:.1f}</b>",
            showarrow=False, xanchor="left",
            font=dict(size=11, color=TEXT, family="monospace"),
        )
    fig.update_layout(
        **_plotly_base(),
        barmode="stack", height=390,
        margin=dict(l=0, r=70, t=6, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(title="Weighted Score Contribution (0–100)", range=[0, 108],
                   gridcolor="#F3F4F6"),
        yaxis=dict(tickfont=dict(size=12, family="monospace"), gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Quick-select ticker buttons ────────────────────────────────────────────
    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:24px 0 16px"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.08em;color:{MUTED2};margin-bottom:10px">'
        f'Click a ticker for full analysis</div>',
        unsafe_allow_html=True,
    )
    btn_cols = st.columns(len(top10))
    for i, (_, row) in enumerate(top10.iterrows()):
        t = row["ticker"]
        with btn_cols[i]:
            if st.button(t, key=f"rank_btn_{t}", use_container_width=True):
                st.session_state.rankings_selected = t
                st.rerun()

    # ── Stock detail panel ─────────────────────────────────────────────────────
    st.markdown(shdr("Stock Detail", "Full quantitative analysis — candlestick, news, valuation, risk, protocol"),
                unsafe_allow_html=True)

    tickers_list = top10["ticker"].tolist()
    default_idx  = 0
    if st.session_state.get("rankings_selected") in tickers_list:
        default_idx = tickers_list.index(st.session_state.rankings_selected)

    col_sel, col_per = st.columns([3, 1])
    with col_sel:
        selected = st.selectbox("Select stock", tickers_list, index=default_idx,
                                label_visibility="collapsed")
    with col_per:
        detail_period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"],
                                     index=3, label_visibility="collapsed",
                                     key="rankings_period")

    if selected:
        _render_stock_detail(
            ticker        = selected,
            universe_data = st.session_state.results.get("universe_data", {}),
            valuation     = valuation,
            risk          = risk or {},
            protocol      = protocol,
            rf_rate       = st.session_state.results.get("rf_rate", 0.045),
            period        = detail_period,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VALUATION
# ─────────────────────────────────────────────────────────────────────────────
def tab_valuation(top10, valuation):
    st.markdown(
        shdr("Multi-Method Valuation Matrix",
             "4 independent methods · Fair value = median · Entry zone = 80–90% of FV · Stop loss = entry −8%"),
        unsafe_allow_html=True,
    )

    rows = ""
    for _, row in top10.iterrows():
        t    = row["ticker"]
        val  = valuation.get(t, {})
        est  = val.get("estimates", {})
        sig  = val.get("signal", "INSUFFICIENT_DATA")
        prem = val.get("premium_pct")
        rr   = val.get("rr_ratio")
        p_clr = RED if (prem or 0) > 5 else GREEN if (prem or 0) < -10 else AMBER
        rows += (
            f'<tr>'
            f'<td><b>{t}</b></td>'
            f'<td style="font-family:monospace">{fmt_price(est.get("dcf"))}</td>'
            f'<td style="font-family:monospace">{fmt_price(est.get("graham"))}</td>'
            f'<td style="font-family:monospace">{fmt_price(est.get("ev_ebitda"))}</td>'
            f'<td style="font-family:monospace">{fmt_price(est.get("fcf_yield"))}</td>'
            f'<td style="font-family:monospace;font-weight:700">{fmt_price(val.get("fair_value"))}</td>'
            f'<td style="font-family:monospace;color:{GREEN};font-weight:600">{fmt_price(val.get("entry_low"))}</td>'
            f'<td style="font-family:monospace;color:{MUTED}">{fmt_price(val.get("entry_high"))}</td>'
            f'<td style="font-family:monospace;font-weight:600">{fmt_price(val.get("current_price") or row.get("current_price"))}</td>'
            f'<td style="font-family:monospace;color:{p_clr};font-weight:700">{fmt_pct(prem)}</td>'
            f'<td style="font-family:monospace;color:{RED}">{fmt_price(val.get("stop_loss"))}</td>'
            f'<td style="font-family:monospace;color:{BLUE}">{f"{rr:.1f}:1" if rr else "—"}</td>'
            f'<td>{signal_badge(sig)}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="qt"><thead><tr>'
        f'<th>Ticker</th><th>DCF</th><th>Graham</th><th>EV/EBITDA</th><th>FCF Yield</th>'
        f'<th>Fair Value</th><th>Entry −20%</th><th>Entry −10%</th>'
        f'<th>Current</th><th>Premium</th><th>Stop Loss</th><th>R/R</th><th>Signal</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(
            shdr("Entry Price Positioning", "% vs estimated fair value · Negative = discount to fair value"),
            unsafe_allow_html=True,
        )
        tickers = top10["ticker"].tolist()
        prems, colors, htexts = [], [], []
        for t in tickers:
            val  = valuation.get(t, {})
            p    = val.get("premium_pct") or 0
            sig  = val.get("signal", "INSUFFICIENT_DATA")
            prems.append(p)
            colors.append(SIGNAL_META.get(sig, (MUTED, GRAY_LT, sig))[0])
            htexts.append(
                f"<b>{t}</b><br>Premium: {p:+.1f}%<br>"
                f"Fair Value: {fmt_price(val.get('fair_value'))}<br>"
                f"Entry: {fmt_price(val.get('entry_low'))}<br>"
                f"Current: {fmt_price(val.get('current_price'))}"
            )

        fig = go.Figure()
        for x0, x1, fill, label, lx in [
            (-42, -20, "rgba(5,150,105,.07)",  "STRONG BUY", -31),
            (-20,   0, "rgba(37,99,235,.07)",  "BUY",         -10),
            (  0,  10, "rgba(217,119,6,.07)",  "WATCH",         5),
            ( 10,  50, "rgba(220,38,38,.05)",  "EXPENSIVE",    28),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_width=0)
            fig.add_annotation(x=lx, y=len(tickers)-0.3, text=label,
                               showarrow=False, font=dict(size=9, color="#9CA3AF"), yref="y")
        fig.add_vline(x=0, line_dash="dash", line_color="#D1D5DB", line_width=1.2)
        fig.add_vline(x=-20, line_dash="dot", line_color=GREEN, line_width=1)
        fig.add_trace(go.Bar(
            x=prems, y=tickers, orientation="h",
            marker_color=colors, opacity=0.85,
            text=[f"{p:+.1f}%" for p in prems], textposition="outside",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=htexts,
        ))
        fig.update_layout(
            **_plotly_base(), height=370,
            margin=dict(l=0, r=80, t=6, b=36),
            xaxis=dict(title="% vs Fair Value", range=[-47, 58],
                       zeroline=True, zerolinecolor="#D1D5DB", gridcolor="#F3F4F6"),
            yaxis=dict(tickfont=dict(size=12, family="monospace"), autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(
            shdr("Valuation Method Spread", "Dollar value from each method per stock"),
            unsafe_allow_html=True,
        )
        method_names = ["DCF", "Graham", "EV/EBITDA", "FCF Yield"]
        method_keys  = ["dcf", "graham", "ev_ebitda", "fcf_yield"]
        tickers = top10["ticker"].tolist()

        fig2 = go.Figure()
        for mi, (mn, mk) in enumerate(zip(method_names, method_keys)):
            vals = [valuation.get(t, {}).get("estimates", {}).get(mk) for t in tickers]
            fig2.add_trace(go.Scatter(
                x=tickers, y=vals, mode="markers", name=mn,
                marker=dict(size=11, color=FACTOR_COLORS[mi], symbol="circle",
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"<b>{mn}</b>: $%{{y:,.0f}}<extra></extra>",
            ))
        fvs = [valuation.get(t, {}).get("fair_value") for t in tickers]
        fig2.add_trace(go.Scatter(
            x=tickers, y=fvs, mode="lines+markers", name="Fair Value",
            line=dict(color=TEXT, width=2, dash="dash"),
            marker=dict(size=8, color=TEXT),
            hovertemplate="<b>Fair Value</b>: $%{y:,.0f}<extra></extra>",
        ))
        fig2.update_layout(
            **_plotly_base(), height=370,
            margin=dict(l=0, r=0, t=6, b=36),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            yaxis=dict(title="Price ($)", tickformat="$,.0f", gridcolor="#F3F4F6"),
            xaxis=dict(tickfont=dict(size=10, family="monospace")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── DCF Sensitivity — Bear / Base / Bull ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        shdr("DCF Sensitivity — Bear / Base / Bull",
             "How fair value shifts with different growth assumptions · Bear = ½ base growth · Bull = 1.5× base growth"),
        unsafe_allow_html=True,
    )
    sens_rows = ""
    for _, row in top10.iterrows():
        t    = row["ticker"]
        val  = valuation.get(t, {})
        sens = val.get("sensitivity", {})
        cur  = val.get("current_price") or row.get("current_price")
        if not sens:
            continue
        for sname, sv in sens.items():
            fv   = sv.get("fair_value")
            gr   = sv.get("growth_rate")
            sig  = sv.get("signal", "—")
            prem = sv.get("premium_pct")
            c_s, bg_s, lbl_s = SIGNAL_META.get(sig, (MUTED, GRAY_LT, sig))
            p_clr = RED if (prem or 0) > 5 else GREEN if (prem or 0) < -10 else AMBER
            row_bg = "#FFFBEB" if sname == "Base" else "#FFFFFF"
            sens_rows += (
                f'<tr style="background:{row_bg}">'
                f'<td><b>{t}</b></td>'
                f'<td style="color:{MUTED}">{sname}</td>'
                f'<td style="font-family:monospace;color:{MUTED}">{gr:+.1f}%</td>'
                f'<td style="font-family:monospace;font-weight:700">{fmt_price(fv)}</td>'
                f'<td style="font-family:monospace">{fmt_price(cur)}</td>'
                f'<td style="font-family:monospace;color:{p_clr};font-weight:600">{fmt_pct(prem)}</td>'
                f'<td><span class="badge" style="color:{c_s};background:{bg_s};font-size:10px">{lbl_s}</span></td>'
                f'</tr>'
            )
    if sens_rows:
        st.markdown(
            f'<table class="qt"><thead><tr>'
            f'<th>Ticker</th><th>Scenario</th><th>Growth Rate</th>'
            f'<th>DCF Fair Value</th><th>Current</th><th>Premium</th><th>Signal</th>'
            f'</tr></thead><tbody>{sens_rows}</tbody></table>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Sensitivity data not available (requires free cash flow data).")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RISK & QUALITY
# ─────────────────────────────────────────────────────────────────────────────
def tab_risk(top10, risk):
    st.markdown(
        shdr("Risk & Quality Metrics",
             "Altman Z · Sharpe · Sortino · Max Drawdown · VaR 95% · ROIC/WACC · Piotroski"),
        unsafe_allow_html=True,
    )

    rows = ""
    for _, row in top10.iterrows():
        t  = row["ticker"]
        r  = risk.get(t, {})
        az = r.get("altman_z", {})
        rw = r.get("roic_wacc", {})
        pf = r.get("piotroski", {})
        pf_sc  = pf.get("score")
        pf_clr = GREEN if (pf_sc or 0) >= 7 else AMBER if (pf_sc or 0) >= 4 else RED
        rw_sp  = rw.get("spread")
        rw_clr = GREEN if (rw_sp or 0) > 5 else AMBER if (rw_sp or 0) > 0 else RED
        sh     = r.get("sharpe")
        sh_clr = GREEN if (sh or 0) > 1 else AMBER if (sh or 0) > 0 else RED
        z_badge = zone_badge(az.get("zone", "—")) if az.get("zone") else "—"
        rows += (
            f'<tr>'
            f'<td><b>{t}</b></td>'
            f'<td style="font-family:monospace">{fmt_2(az.get("score"))}</td>'
            f'<td>{z_badge}</td>'
            f'<td style="font-family:monospace;color:{sh_clr};font-weight:600">{fmt_2(sh)}</td>'
            f'<td style="font-family:monospace">{fmt_2(r.get("sortino"))}</td>'
            f'<td style="font-family:monospace;color:{RED}">{fmt_pct(r.get("max_drawdown_pct"), False)}</td>'
            f'<td style="font-family:monospace;color:{AMBER}">{fmt_pct(r.get("var_95_pct"), False)}</td>'
            f'<td style="font-family:monospace;color:{rw_clr};font-weight:700">{fmt_pct(rw_sp)}</td>'
            f'<td style="font-size:11px;color:{MUTED}">{rw.get("verdict","—")}</td>'
            f'<td style="font-weight:700;color:{pf_clr}">'
            f'{pf_sc if pf_sc is not None else "—"}/9</td>'
            f'<td style="font-family:monospace;color:{MUTED}">{fmt_2(r.get("accruals"))}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="qt"><thead><tr>'
        f'<th>Ticker</th><th>Altman Z</th><th>Zone</th>'
        f'<th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>VaR 95%</th>'
        f'<th>ROIC/WACC</th><th>Verdict</th><th>Piotroski</th><th>Accruals</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            shdr("Risk / Return Scatter", "X = Sharpe · Y = ROIC/WACC spread · Size = |spread|"),
            unsafe_allow_html=True,
        )
        xs, ys, szs, dot_clrs, texts = [], [], [], [], []
        for _, row in top10.iterrows():
            t  = row["ticker"]
            r  = risk.get(t, {})
            sh = r.get("sharpe")
            rw = r.get("roic_wacc", {})
            az = r.get("altman_z", {})
            if sh is None: continue
            spread = rw.get("spread") or 0
            zone   = az.get("zone", "GRAY")
            xs.append(sh)
            ys.append(spread)
            szs.append(max(14, min(42, abs(spread) * 2 + 14)))
            dot_clrs.append(ZONE_META.get(zone, (MUTED, GRAY_LT))[0])
            texts.append(t)

        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=texts, textposition="top center",
            textfont=dict(size=10, family="monospace"),
            marker=dict(size=szs, color=dot_clrs, opacity=0.85,
                        line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{text}</b><br>Sharpe: %{x:.2f}<br>ROIC/WACC: %{y:+.1f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#E5E7EB", line_width=1)
        fig.add_vline(x=0, line_dash="dash", line_color="#E5E7EB", line_width=1)
        if xs and ys:
            fig.add_annotation(x=max(xs)*0.88, y=max(ys)*0.88,
                               text="IDEAL", showarrow=False,
                               font=dict(size=10, color=GREEN), opacity=0.7)
        fig.update_layout(
            **_plotly_base(), height=340,
            margin=dict(l=0, r=0, t=6, b=40),
            xaxis=dict(title="Sharpe Ratio", gridcolor="#F3F4F6"),
            yaxis=dict(title="ROIC / WACC Spread (%)", gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(
            shdr("Piotroski F-Score", "9-point quality score · Strong ≥ 7 · Average 4–6 · Weak ≤ 3"),
            unsafe_allow_html=True,
        )
        tickers = top10["ticker"].tolist()
        scores, clrs = [], []
        for t in tickers:
            sc = risk.get(t, {}).get("piotroski", {}).get("score")
            scores.append(sc if sc is not None else 0)
            clrs.append(GREEN if (sc or 0) >= 7 else AMBER if (sc or 0) >= 4 else RED)

        fig2 = go.Figure(go.Bar(
            x=tickers, y=scores, marker_color=clrs, opacity=0.85,
            text=scores, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Piotroski: %{y}/9<extra></extra>",
        ))
        fig2.add_hline(y=7, line_dash="dot", line_color=GREEN,
                       annotation_text="Strong ≥ 7", annotation_position="top right",
                       annotation_font=dict(color=GREEN, size=10))
        fig2.add_hline(y=3, line_dash="dot", line_color=RED,
                       annotation_text="Weak ≤ 3", annotation_position="top right",
                       annotation_font=dict(color=RED, size=10))
        fig2.update_layout(
            **_plotly_base(), height=340,
            margin=dict(l=0, r=70, t=6, b=40),
            yaxis=dict(range=[0, 11], title="Score / 9", gridcolor="#F3F4F6"),
            xaxis=dict(tickfont=dict(size=11, family="monospace")),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PROTOCOL GATES
# ─────────────────────────────────────────────────────────────────────────────
def tab_protocol(top10, protocol):
    proto_map = {p["ticker"]: p for p in protocol}
    tickers   = top10["ticker"].tolist()

    st.markdown(
        shdr("7-Gate Investment Protocol",
             "PASS ≥ 60 · WARN 35–59 · FAIL < 35 · Gate 3 includes Altman Z · Gate 4 uses ValuationEngine signal"),
        unsafe_allow_html=True,
    )

    gate_matrix = []
    text_matrix = []
    for t in tickers:
        gates = proto_map.get(t, {}).get("gates", [50] * 7)
        gate_matrix.append([float(g) for g in gates[:7]])
        text_matrix.append([f"{float(g):.0f}" for g in gates[:7]])

    colorscale = [
        [0.00, "#FEF2F2"], [0.35, "#FEF3C7"],
        [0.60, "#ECFDF5"], [1.00, "#065F46"],
    ]
    fig = go.Figure(go.Heatmap(
        z=gate_matrix, x=GATE_SHORT, y=tickers,
        text=text_matrix, texttemplate="<b>%{text}</b>",
        colorscale=colorscale, zmin=0, zmax=100,
        hovertemplate="<b>%{y}</b> — %{x}<br>Score: %{z:.0f}<extra></extra>",
        colorbar=dict(
            title="Score", tickvals=[0, 35, 60, 100],
            ticktext=["0", "FAIL", "PASS", "100"],
            thickness=14, len=0.9,
        ),
    ))
    fig.update_layout(
        **_plotly_base(), height=370,
        margin=dict(l=0, r=90, t=6, b=6),
        xaxis=dict(side="top", tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12, family="monospace"), autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(shdr("Protocol Summary"), unsafe_allow_html=True)
    rows = ""
    for _, row in top10.iterrows():
        t    = row["ticker"]
        p    = proto_map.get(t, {})
        ea   = p.get("entry_analysis", {})
        conv = p.get("conviction", "—")
        over = p.get("overall_score", 0)

        gate_dots = ""
        for status in p.get("gate_statuses", []):
            c   = GREEN if status == "pass" else AMBER if status == "warn" else RED
            sym = "●" if status == "pass" else "◐" if status == "warn" else "○"
            gate_dots += f'<span style="color:{c};font-size:15px">{sym}</span> '

        rows += (
            f'<tr>'
            f'<td><b>{t}</b></td>'
            f'<td>{gate_dots}</td>'
            f'<td style="color:{GREEN};font-weight:700">{p.get("pass_count",0)}</td>'
            f'<td style="color:{AMBER};font-weight:700">{p.get("warn_count",0)}</td>'
            f'<td style="color:{RED};font-weight:700">{p.get("fail_count",0)}</td>'
            f'<td style="font-weight:700;color:{score_color(over)}">{over:.1f}</td>'
            f'<td>{conv_badge(conv)}</td>'
            f'<td>{signal_badge(ea.get("signal","INSUFFICIENT_DATA"))}</td>'
            f'<td style="font-family:monospace;font-weight:600">{fmt_price(ea.get("fair_value"))}</td>'
            f'<td style="font-family:monospace;color:{GREEN};font-weight:600">{fmt_price(ea.get("entry_target"))}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="qt"><thead><tr>'
        f'<th>Ticker</th><th>Gates</th><th>Pass</th><th>Warn</th><th>Fail</th>'
        f'<th>Score</th><th>Conviction</th><th>Signal</th>'
        f'<th>Fair Value</th><th>Entry Target</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
def tab_portfolio(top10, profile):
    col_a, col_b = st.columns([2, 3])

    with col_a:
        st.markdown(shdr("Allocation by Sector"), unsafe_allow_html=True)
        tickers = top10["ticker"].tolist()
        sectors = top10["sector"].tolist()
        weights = (top10["weight"].tolist()
                   if "weight" in top10.columns
                   else [1/len(top10)] * len(top10))
        colors  = [SECTOR_COLORS.get(s, "#9CA3AF") for s in sectors]

        fig = go.Figure(go.Pie(
            labels=tickers,
            values=weights, hole=0.60,
            marker=dict(colors=colors, line=dict(color="white", width=2.5)),
            textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value:.1%}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>${profile.portfolio_size:,.0f}</b>",
            x=0.5, y=0.5, font=dict(size=14, color=TEXT), showarrow=False,
        )
        fig.update_layout(
            **_plotly_base(), height=360,
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        n_sectors = len(set(sectors))
        max_w     = max(weights) * 100 if weights else 0
        m1, m2, m3 = st.columns(3)
        with m1:
            with st.container(border=True):
                st.metric("Positions", len(top10))
        with m2:
            with st.container(border=True):
                st.metric("Sectors", n_sectors)
        with m3:
            with st.container(border=True):
                st.metric("Max Position", f"{max_w:.1f}%")

    with col_b:
        st.markdown(shdr("Position Breakdown"), unsafe_allow_html=True)
        rows  = ""
        total = profile.portfolio_size
        for _, row in top10.iterrows():
            t   = row["ticker"]
            w   = float(row.get("weight", 0))
            amt = float(row.get("dollar_amount", w * total))
            sh  = row.get("approx_shares", "—")
            px  = float(row.get("current_price", 0))
            sec = row.get("sector", "—")
            sc  = SECTOR_COLORS.get(sec, "#9CA3AF")
            bar = (
                f'<div style="background:#F3F4F6;border-radius:3px;height:5px;width:100%">'
                f'<div style="background:{sc};height:5px;border-radius:3px;width:{w*100:.0f}%"></div></div>'
            )
            rows += (
                f'<tr>'
                f'<td><b>{t}</b><br><span style="font-size:11px;color:{MUTED}">{sec}</span></td>'
                f'<td style="min-width:130px">'
                f'  <div style="display:flex;align-items:center;gap:8px">'
                f'    <span style="font-weight:700;color:{sc};width:40px;'
                f'          font-variant-numeric:tabular-nums">{w*100:.1f}%</span>'
                f'    <div style="flex:1">{bar}</div>'
                f'  </div>'
                f'</td>'
                f'<td style="font-weight:700;font-family:monospace">${amt:,.0f}</td>'
                f'<td style="font-family:monospace;color:{MUTED}">{sh}</td>'
                f'<td style="font-family:monospace">${px:,.2f}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<table class="qt"><thead><tr>'
            f'<th>Ticker</th><th>Weight</th><th>$ Amount</th><th>Shares</th><th>Price</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(shdr("Position Weights"), unsafe_allow_html=True)
        clrs = [SECTOR_COLORS.get(s, "#9CA3AF") for s in top10["sector"].tolist()]
        fig2 = go.Figure(go.Bar(
            x=top10["ticker"].tolist(),
            y=(top10["weight"] * 100).tolist() if "weight" in top10.columns else [],
            marker_color=clrs, opacity=0.85,
            text=[f"{w*100:.1f}%" for w in top10.get("weight", [])],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Weight: %{y:.1f}%<extra></extra>",
        ))
        fig2.add_hline(y=10, line_dash="dot", line_color="#D1D5DB", line_width=1,
                       annotation_text="Equal weight 10%",
                       annotation_font=dict(size=9, color=MUTED2))
        fig2.update_layout(
            **_plotly_base(), height=240,
            margin=dict(l=0, r=0, t=6, b=36),
            yaxis=dict(title="Weight (%)",
                       range=[0, max((top10["weight"]*100).tolist() or [20]) * 1.25],
                       gridcolor="#F3F4F6"),
            xaxis=dict(tickfont=dict(size=11, family="monospace")),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — MACRO & PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def tab_macro(top10, macro, universe_data, sp500_hist, profile):
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(shdr("Macro Environment"), unsafe_allow_html=True)
        vix    = macro.get("vix")
        y10    = macro.get("yield_10y")
        regime = macro.get("regime", "neutral").upper().replace("_", " ")
        rc = GREEN if "RISK ON" in regime else RED if "RISK OFF" in regime else AMBER
        vc = RED if (vix or 0) > 25 else AMBER if (vix or 0) > 18 else GREEN
        st.markdown(
            mtile("VIX", f"{vix:.1f}" if vix else "N/A",
                  "Elevated — use caution" if (vix or 0) > 25 else "Normal range", vc, vc)
            + "<br>"
            + mtile("10-Year Yield", f"{y10:.2f}%" if y10 else "N/A",
                    "Risk-free rate basis", TEXT, BLUE)
            + "<br>"
            + mtile("Regime", regime,
                    "  ·  ".join(macro.get("regime_reasons", []))[:52], rc, rc),
            unsafe_allow_html=True,
        )

        etf = macro.get("sector_etf", {})
        if etf:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(shdr("Sector ETF 3-Month Returns"), unsafe_allow_html=True)
            sorted_etf = sorted(etf.items(), key=lambda x: x[1], reverse=True)
            fig_etf = go.Figure(go.Bar(
                x=[r for _, r in sorted_etf],
                y=[s for s, _ in sorted_etf],
                orientation="h",
                marker_color=[GREEN if r >= 0 else RED for _, r in sorted_etf],
                opacity=0.85,
                text=[f"{r:+.1f}%" for _, r in sorted_etf],
                textposition="outside",
                hovertemplate="<b>%{y}</b>: %{x:+.1f}%<extra></extra>",
            ))
            fig_etf.update_layout(
                **_plotly_base(), height=290,
                margin=dict(l=0, r=60, t=0, b=30),
                xaxis=dict(zeroline=True, zerolinecolor="#D1D5DB", gridcolor="#F3F4F6"),
                showlegend=False,
            )
            st.plotly_chart(fig_etf, use_container_width=True)

    with col_b:
        st.markdown(
            shdr("Historical Performance vs S&P 500",
                 f"Normalised to 100 at start  ·  {HORIZON_LABELS[profile.time_horizon]}"),
            unsafe_allow_html=True,
        )
        fig_perf = go.Figure()
        start_date = None
        if sp500_hist is not None and len(sp500_hist) > 0:
            sp         = DataFetcher.strip_tz(sp500_hist["Close"].dropna())
            start_date = sp.index[0]
            sp_norm    = sp / sp.iloc[0] * 100
            sp_ret     = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
            fig_perf.add_trace(go.Scatter(
                x=sp_norm.index, y=sp_norm.values,
                name=f"S&P 500 ({sp_ret:+.1f}%)",
                line=dict(color=TEXT, width=2.5),
                hovertemplate="S&P 500: %{y:.1f}<extra></extra>",
            ))

        for _, row in top10.iterrows():
            t = row["ticker"]
            if t not in universe_data: continue
            close = DataFetcher.strip_tz(universe_data[t]["history"]["Close"].dropna())
            if start_date is not None:
                close = close[close.index >= start_date]
            if len(close) < 10: continue
            norm    = close / close.iloc[0] * 100
            tot_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
            color   = SECTOR_COLORS.get(row["sector"], "#9CA3AF")
            fig_perf.add_trace(go.Scatter(
                x=norm.index, y=norm.values, name=f"{t} ({tot_ret:+.1f}%)",
                line=dict(color=color, width=1.5), opacity=0.8,
                hovertemplate=f"<b>{t}</b>: %{{y:.1f}}<extra></extra>",
            ))

        fig_perf.add_hline(y=100, line_dash="dot", line_color="#D1D5DB", line_width=1)
        fig_perf.update_layout(
            **_plotly_base(), height=360,
            margin=dict(l=0, r=0, t=6, b=40),
            legend=dict(font=dict(size=10), orientation="v",
                        yanchor="top", y=1, xanchor="left", x=1.01),
            xaxis=dict(title="Date", gridcolor="#F3F4F6"),
            yaxis=dict(title="Normalised (Base = 100)", gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # Correlation heatmap
        st.markdown(
            shdr("Return Correlation Matrix", "Daily returns · Lower = better diversification"),
            unsafe_allow_html=True,
        )
        ret_dict = {}
        for _, row in top10.iterrows():
            t = row["ticker"]
            if t in universe_data:
                close = DataFetcher.strip_tz(universe_data[t]["history"]["Close"].dropna())
                ret_dict[t] = close.pct_change().dropna()

        if len(ret_dict) >= 3:
            ret_df   = pd.DataFrame(ret_dict).dropna()
            corr_mat = ret_df.corr()
            tks      = corr_mat.columns.tolist()
            txt_m    = [[f"{corr_mat.iloc[r, c]:.2f}" for c in range(len(tks))]
                        for r in range(len(tks))]
            fig_corr = go.Figure(go.Heatmap(
                z=corr_mat.values, x=tks, y=tks,
                text=txt_m, texttemplate="%{text}",
                colorscale=[[0, "#F0FDF4"], [0.5, "#93C5FD"], [1, "#1D4ED8"]],
                zmin=0, zmax=1,
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
                colorbar=dict(thickness=14, len=0.85),
            ))
            fig_corr.update_layout(
                **_plotly_base(), height=290,
                margin=dict(l=0, r=60, t=6, b=0),
                xaxis=dict(tickfont=dict(size=10, family="monospace")),
                yaxis=dict(tickfont=dict(size=10, family="monospace"), autorange="reversed"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CANDLESTICK CHART (Plotly — used in detail view)
# ─────────────────────────────────────────────────────────────────────────────
def _candlestick_fig(ticker: str, hist: pd.DataFrame, period: str = "1y") -> go.Figure:
    """
    Returns a Plotly figure with:
      Row 1 — Candlestick + SMA 20 / 50 / 200
      Row 2 — Volume (green/red bars)
      Row 3 — RSI(14) with overbought / oversold bands
    """
    period_days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
    days = period_days.get(period, 252)
    df   = hist.tail(days).copy()
    if df.empty:
        return go.Figure()

    df["sma20"]  = df["Close"].rolling(20).mean()
    df["sma50"]  = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()
    delta        = df["Close"].diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    rs           = gain / loss
    df["rsi"]    = 100 - (100 / (1 + rs))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.60, 0.18, 0.22],
        subplot_titles=(f"{ticker}  OHLCV", "Volume", "RSI (14)"),
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker,
        increasing_line_color=GREEN, increasing_fillcolor=GREEN,
        decreasing_line_color=RED,   decreasing_fillcolor=RED,
        line_width=1,
    ), row=1, col=1)

    # ── SMAs ──
    for col, color, label, dash in [
        ("sma20",  AMBER, "SMA 20",  "solid"),
        ("sma50",  BLUE,  "SMA 50",  "solid"),
        ("sma200", RED,   "SMA 200", "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=label,
            line=dict(color=color, width=1.2, dash=dash),
            opacity=0.85,
        ), row=1, col=1)

    # ── Volume ──
    vol_colors = [GREEN if c >= o else RED
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.55, showlegend=False,
    ), row=2, col=1)

    # ── RSI ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"], name="RSI(14)",
        line=dict(color="#8B5CF6", width=1.5),
    ), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=f"rgba(220,38,38,.06)",
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=f"rgba(5,150,105,.06)",
                  line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=RED,   line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=GREEN, line_width=1, row=3, col=1)

    fig.update_layout(
        **_plotly_base(), height=620,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=36, b=0),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1,
                     gridcolor="#F3F4F6", tickformat="$,.2f")
    fig.update_yaxes(title_text="Volume",   row=2, col=1,
                     gridcolor="#F3F4F6", showticklabels=False)
    fig.update_yaxes(title_text="RSI",      row=3, col=1,
                     gridcolor="#F3F4F6", range=[0, 100])
    fig.update_xaxes(gridcolor="#F3F4F6", row=3, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STOCK DETAIL PANEL (shared by Rankings detail + Stock Lookup tab)
# ─────────────────────────────────────────────────────────────────────────────
def _render_stock_detail(
    ticker:      str,
    universe_data: dict,
    valuation:   dict,
    risk:        dict,
    protocol:    list,
    rf_rate:     float,
    period:      str = "1y",
    fetch_fresh: bool = False,
):
    # ── Fetch data if needed ──────────────────────────────────────────────────
    if fetch_fresh and ticker not in universe_data:
        with st.spinner(f"Fetching data for {ticker}…"):
            fetcher    = DataFetcher("2y")
            fresh_data = fetcher.fetch_universe([ticker])
            if fresh_data:
                universe_data = {**universe_data, **fresh_data}

    data = universe_data.get(ticker, {})
    info = data.get("info", {})
    hist = data.get("history")

    name    = info.get("longName", ticker)
    price   = info.get("currentPrice") or info.get("regularMarketPrice")
    mktcap  = info.get("marketCap", 0)
    sector  = data.get("sector", info.get("sector", "—"))
    website = info.get("website", "")
    mktcap_s = (f"${mktcap/1e12:.2f}T" if mktcap >= 1e12
                else f"${mktcap/1e9:.1f}B" if mktcap >= 1e9
                else "n/a")

    val  = valuation.get(ticker, {})
    r    = risk.get(ticker, {})
    sig  = val.get("signal", "INSUFFICIENT_DATA")
    c_sig, bg_sig, lbl_sig = SIGNAL_META.get(sig, (MUTED, GRAY_LT, sig))
    az   = r.get("altman_z", {})

    # ── HERO HEADER ──────────────────────────────────────────────────────────
    sig_accent = SIGNAL_ACCENT.get(sig, BORDER)
    price_chg  = ""
    if hist is not None and not hist.empty and price:
        try:
            prev = float(hist["Close"].dropna().iloc[-2])
            chg  = ((float(price) - prev) / prev) * 100
            chg_col = GREEN if chg >= 0 else RED
            price_chg = f'<span style="font-size:16px;font-weight:600;color:{chg_col};margin-left:10px">{chg:+.2f}%</span>'
        except Exception:
            pass

    website_html = ""
    if website:
        domain = website.replace("https://","").replace("http://","").rstrip("/")
        website_html = f'<a href="{website}" target="_blank" style="color:{BLUE};text-decoration:none;font-size:12px">{domain}</a>  ·  '

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{GRAY_LT} 0%,#ffffff 100%);'
        f'border:1px solid {BORDER};border-left:5px solid {sig_accent};'
        f'border-radius:12px;padding:22px 26px 18px;margin-bottom:20px">'
        f'  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px">'
        f'    <div>'
        f'      <div style="font-size:32px;font-weight:900;color:{TEXT};letter-spacing:-.04em;line-height:1">{ticker}</div>'
        f'      <div style="font-size:15px;font-weight:600;color:{MUTED};margin-top:5px">{name}</div>'
        f'      <div style="font-size:12px;color:{MUTED2};margin-top:4px">'
        f'        {website_html}{sector}  ·  Market Cap {mktcap_s}'
        f'      </div>'
        f'    </div>'
        f'    <div style="text-align:right">'
        f'      <div style="display:flex;align-items:baseline;gap:4px;justify-content:flex-end">'
        f'        <span style="font-size:36px;font-weight:900;color:{TEXT};font-variant-numeric:tabular-nums">'
        f'          {"${:,.2f}".format(price) if price else "—"}'
        f'        </span>'
        f'        {price_chg}'
        f'      </div>'
        f'      <div style="display:flex;gap:6px;justify-content:flex-end;margin-top:8px;flex-wrap:wrap">'
        f'        <span class="badge" style="font-size:11px;font-weight:700;color:{c_sig};background:{bg_sig};padding:4px 10px">{lbl_sig}</span>'
        f'        {zone_badge(az.get("zone","")) if az.get("zone") else ""}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── EARNINGS BANNER ───────────────────────────────────────────────────────
    days_away = data.get("earnings_days_away")
    edate     = data.get("earnings_date", "")
    if days_away is not None and days_away <= 90:
        if days_away <= 7:
            earn_c, earn_bg = RED, "#FEF2F2"
            earn_icon = "⚠"
            earn_msg = f"Earnings in {days_away} days ({edate}) — high event risk, consider waiting for results"
        elif days_away <= 14:
            earn_c, earn_bg = AMBER, "#FFFBEB"
            earn_icon = "!"
            earn_msg = f"Earnings in {days_away} days ({edate}) — elevated risk, size position carefully"
        elif days_away <= 30:
            earn_c, earn_bg = BLUE, BLUE_LT
            earn_icon = "i"
            earn_msg = f"Earnings in {days_away} days ({edate})"
        else:
            earn_c, earn_bg = MUTED, GRAY_LT
            earn_icon = "cal"
            earn_msg = f"Next earnings: {edate} ({days_away} days)"
        st.markdown(
            f'<div style="background:{earn_bg};border-left:4px solid {earn_c};'
            f'border-radius:8px;padding:11px 16px;margin-bottom:16px;font-size:13px;'
            f'font-weight:600;color:{earn_c};display:flex;align-items:center;gap:8px">'
            f'<span style="font-size:16px">{earn_icon}</span> {earn_msg}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif days_away is None:
        st.markdown(
            f'<div style="background:{GRAY_LT};border-left:4px solid {BORDER};'
            f'border-radius:8px;padding:8px 16px;margin-bottom:16px;font-size:12px;'
            f'color:{MUTED2}">Earnings date not available from data provider</div>',
            unsafe_allow_html=True,
        )

    # ── QUANT THESIS ──────────────────────────────────────────────────────────
    proto_map_local  = {p["ticker"]: p for p in protocol}
    proto_for_ticker = proto_map_local.get(ticker, {})
    thesis_html = _generate_quant_thesis(ticker, val, r, proto_for_ticker)
    if thesis_html:
        st.markdown(
            f'<div style="background:#F8FAFF;border:1px solid #DBEAFE;border-left:4px solid {sig_accent};'
            f'border-radius:10px;padding:16px 20px;margin-bottom:18px">'
            f'<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
            f'letter-spacing:.10em;color:{BLUE};margin-bottom:8px">Quant Thesis</div>'
            f'<div style="font-size:13.5px;line-height:1.8;color:{TEXT}">{thesis_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── BUSINESS DESCRIPTION ──────────────────────────────────────────────────
    desc = (info.get("longBusinessSummary") or "")[:420]
    if desc:
        with st.expander("Business description", expanded=False):
            st.markdown(
                f'<div style="font-size:13px;color:{MUTED};line-height:1.75">'
                f'{desc}{"…" if len(desc)==420 else ""}</div>',
                unsafe_allow_html=True,
            )

    # ── KEY METRICS STRIP ─────────────────────────────────────────────────────
    def _quick_pct(key, mult=100):
        v = info.get(key)
        if v is None: return "n/a"
        try: return f"{float(v)*mult:.1f}%"
        except: return "n/a"
    def _quick_n(key, fmt=".1f"):
        v = info.get(key)
        if v is None: return "n/a"
        try: return format(float(v), fmt)
        except: return "n/a"

    pf    = r.get("piotroski", {})
    rw    = r.get("roic_wacc", {})
    sharpe_v = r.get("sharpe")

    metrics = [
        ("P/E",           _quick_n("trailingPE", ".1f"),  MUTED),
        ("Forward P/E",   _quick_n("forwardPE",  ".1f"),  MUTED),
        ("EV/EBITDA",     _quick_n("enterpriseToEbitda", ".1f"), MUTED),
        ("Gross Margin",  _quick_pct("grossMargins"),     MUTED),
        ("ROE",           _quick_pct("returnOnEquity"),   MUTED),
        ("Beta",          _quick_n("beta", ".2f"),         MUTED),
        ("Piotroski",     f'{pf.get("score","n/a")}/9' if pf.get("score") is not None else "n/a",
                          GREEN if (pf.get("score") or 0) >= 7 else RED if (pf.get("score") or 0) <= 3 else AMBER),
        ("Altman Z",      f'{az.get("score","n/a"):.2f} [{az.get("zone","?")}]' if az.get("score") else "n/a",
                          GREEN if az.get("zone") == "SAFE" else RED if az.get("zone") == "DISTRESS" else AMBER),
        ("Sharpe",        f'{sharpe_v:.2f}' if sharpe_v else "n/a",
                          GREEN if (sharpe_v or 0) > 1.2 else RED if (sharpe_v or 0) < 0.5 else AMBER),
    ]
    pills = "".join(
        f'<div style="background:#fff;border:1px solid {BORDER};border-radius:8px;'
        f'padding:10px 14px;text-align:center;min-width:90px">'
        f'<div style="font-size:9.5px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.08em;color:{MUTED2};margin-bottom:5px">{lbl}</div>'
        f'<div style="font-size:15px;font-weight:800;color:{col};font-variant-numeric:tabular-nums">{val2}</div>'
        f'</div>'
        for lbl, val2, col in metrics
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:22px">{pills}</div>',
        unsafe_allow_html=True,
    )

    # ── FACTOR SCORE BARS ─────────────────────────────────────────────────────
    factor_bars = _factor_bars_html(ticker)
    if factor_bars:
        with st.expander("7-Factor Score Breakdown", expanded=False):
            st.markdown(factor_bars, unsafe_allow_html=True)

    # ── CANDLESTICK CHART ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin-bottom:8px">Price Chart</div>',
        unsafe_allow_html=True,
    )
    if hist is not None and not hist.empty:
        try:
            hist_c = hist.copy()
            if hasattr(hist_c.index, "tz") and hist_c.index.tz is not None:
                hist_c.index = hist_c.index.tz_localize(None)
            st.plotly_chart(_candlestick_fig(ticker, hist_c, period), use_container_width=True)
        except Exception:
            st.info("Chart unavailable.")
    else:
        st.info("Price history not available.")

    # ── VALUATION DEEP DIVE (full width card) ─────────────────────────────────
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin:6px 0 12px">Valuation Deep Dive</div>',
        unsafe_allow_html=True,
    )
    est = val.get("estimates", {})
    current_price = price or 0
    fv   = val.get("fair_value")
    el   = val.get("entry_low")
    eh   = val.get("entry_high")
    sl   = val.get("stop_loss")
    rr   = val.get("rr_ratio")
    upd  = val.get("upside_pct")
    tgt  = val.get("target_price")
    prem = val.get("premium_pct")
    mc   = val.get("methods_count", 0)

    method_rows = [
        ("DCF (2-stage)",    est.get("dcf"),      "2-stage FCF model, rf + 5.5% discount"),
        ("Graham Number",    est.get("graham"),    "√(22.5 × EPS × Book Value)"),
        ("EV/EBITDA Target", est.get("ev_ebitda"), "Sector-median multiple → implied price"),
        ("FCF Yield @4.5%",  est.get("fcf_yield"), "FCF/share ÷ 4.5% target yield"),
    ]

    val_rows_html = ""
    for method, ep, desc_txt in method_rows:
        if ep is None:
            val_rows_html += (
                f'<tr>'
                f'<td style="padding:10px 16px">'
                f'  <div style="font-weight:600;font-size:13px;color:{TEXT}">{method}</div>'
                f'  <div style="font-size:11px;color:{MUTED2};margin-top:2px">{desc_txt}</div>'
                f'</td>'
                f'<td style="padding:10px 16px;text-align:right;font-size:12px;color:{MUTED2}">Insufficient data</td>'
                f'<td style="padding:10px 16px"></td>'
                f'</tr>'
            )
            continue
        diff = ((current_price / ep) - 1) * 100 if ep else 0
        dc   = RED if diff > 5 else GREEN if diff < -10 else AMBER
        dir_lbl = "premium" if diff > 0 else "discount"
        val_rows_html += (
            f'<tr style="border-bottom:1px solid {BORDER}">'
            f'<td style="padding:10px 16px">'
            f'  <div style="font-weight:600;font-size:13px;color:{TEXT}">{method}</div>'
            f'  <div style="font-size:11px;color:{MUTED2};margin-top:2px">{desc_txt}</div>'
            f'</td>'
            f'<td style="padding:10px 16px;text-align:right">'
            f'  <span style="font-size:18px;font-weight:800;font-family:monospace;color:{TEXT}">${ep:,.2f}</span>'
            f'</td>'
            f'<td style="padding:10px 16px;text-align:right">'
            f'  <span style="font-size:13px;font-weight:700;color:{dc}">{diff:+.1f}% {dir_lbl}</span>'
            f'</td>'
            f'</tr>'
        )

    # Entry / target / stop summary bar
    summary_html = ""
    if fv:
        prem_c = RED if (prem or 0) > 10 else GREEN if (prem or 0) < -10 else AMBER
        prem_lbl = "above FV" if (prem or 0) > 0 else "below FV"
        summary_html = (
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));'
            f'gap:12px;padding:16px;background:{GRAY_LT};border-radius:0 0 10px 10px">'
            + "".join([
                f'<div style="text-align:center">'
                f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{MUTED2};margin-bottom:4px">{lbl}</div>'
                f'<div style="font-size:{"18" if i==0 else "15"}px;font-weight:800;color:{clr};font-variant-numeric:tabular-nums">{val_s}</div>'
                f'</div>'
                for i, (lbl, val_s, clr) in enumerate([
                    (f"{mc}-Method FV", f"${fv:,.2f}", TEXT),
                    ("Entry Zone",      f"${el:,.2f}" if el else "n/a", GREEN),
                    ("Target",          f"${tgt:,.2f}" if tgt else "n/a", BLUE),
                    ("Stop Loss",       f"${sl:,.2f}" if sl else "n/a", RED),
                    ("R/R Ratio",       f"{rr:.1f}:1" if rr else "n/a", BLUE),
                    ("Upside",          f"{upd:+.1f}%" if upd else "n/a", GREEN if (upd or 0)>0 else RED),
                    ("Premium/Disc",    f"{prem:+.1f}%" if prem is not None else "n/a", prem_c),
                ])
            ])
            + "</div>"
        )

    st.markdown(
        f'<div style="border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:20px">'
        f'<table style="width:100%;border-collapse:collapse"><tbody>{val_rows_html}</tbody></table>'
        f'{summary_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # DCF Sensitivity
    sens = (valuation.get(ticker) or {}).get("sensitivity", {})
    if sens:
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.08em;color:{MUTED2};margin:0 0 8px">DCF Sensitivity — Bear / Base / Bull</div>',
            unsafe_allow_html=True,
        )
        s_rows = ""
        for sname, sv in sens.items():
            sfv   = sv.get("fair_value")
            sgr   = sv.get("growth_rate")
            ssig  = sv.get("signal", "INSUFFICIENT_DATA")
            sprem = sv.get("premium_pct")
            sc_s, sbg_s, slbl_s = SIGNAL_META.get(ssig, (MUTED, GRAY_LT, ssig))
            sp_clr = RED if (sprem or 0) > 5 else GREEN if (sprem or 0) < -10 else AMBER
            row_bg = "#FFFBEB" if sname == "Base" else "#FAFAFA"
            s_rows += (
                f'<tr style="background:{row_bg};border-bottom:1px solid {BORDER}">'
                f'<td style="font-weight:800;color:{MUTED};padding:9px 16px;font-size:12px">{sname}</td>'
                f'<td style="font-family:monospace;color:{MUTED};padding:9px 16px;font-size:12px">{sgr:+.1f}% growth</td>'
                f'<td style="font-family:monospace;font-weight:800;font-size:14px;padding:9px 16px">{fmt_price(sfv)}</td>'
                f'<td style="font-family:monospace;color:{sp_clr};font-weight:700;padding:9px 16px">'
                f'  {sprem:+.1f}% vs current' if sprem is not None else "n/a"
                f'</td>'
                f'<td style="padding:9px 16px"><span class="badge" style="color:{sc_s};background:{sbg_s}">{slbl_s}</span></td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="border:1px solid {BORDER};border-radius:10px;overflow:hidden;margin-bottom:22px">'
            f'<table style="width:100%;border-collapse:collapse"><tbody>{s_rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    # ── RISK & QUALITY + ANALYST + TECHNICAL (3 columns) ─────────────────────
    col_risk, col_analyst, col_tech = st.columns([1, 1, 1])

    with col_risk:
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin-bottom:12px">Risk & Quality</div>',
            unsafe_allow_html=True,
        )
        risk_rows = [
            ("Altman Z",
             f'{az.get("score","n/a"):.2f}' if az.get("score") else "n/a",
             az.get("zone",""),
             GREEN if az.get("zone")=="SAFE" else RED if az.get("zone")=="DISTRESS" else AMBER),
            ("Sharpe Ratio",    fmt_2(r.get("sharpe")),      "",
             GREEN if (r.get("sharpe") or 0)>1.2 else RED if (r.get("sharpe") or 0)<0.5 else AMBER),
            ("Sortino Ratio",   fmt_2(r.get("sortino")),     "",
             GREEN if (r.get("sortino") or 0)>1.5 else RED if (r.get("sortino") or 0)<0.5 else AMBER),
            ("Max Drawdown",    fmt_pct(r.get("max_drawdown_pct"), False), "",  RED),
            ("VaR 95% (1mo)",   fmt_pct(r.get("var_95_pct"), False),       "",  AMBER),
            ("ROIC",            f'{rw.get("roic","n/a"):.1f}%' if rw.get("roic") else "n/a", "", BLUE),
            ("WACC",            f'{rw.get("wacc","n/a"):.1f}%' if rw.get("wacc") else "n/a", "", MUTED),
            ("ROIC/WACC Spread",f'{rw.get("spread","n/a"):+.1f}%' if rw.get("spread") is not None else "n/a", rw.get("verdict",""),
             GREEN if (rw.get("spread") or 0)>8 else RED if (rw.get("spread") or 0)<0 else AMBER),
            ("Piotroski",
             f'{pf.get("score","n/a")}/9' if pf.get("score") is not None else "n/a",
             pf.get("interpretation",""),
             GREEN if (pf.get("score") or 0)>=7 else RED if (pf.get("score") or 0)<=3 else AMBER),
            ("Accruals Ratio",  fmt_2(r.get("accruals")),    "",  MUTED),
            ("Gross Profit/Assets", fmt_2(r.get("gross_prof")), "", MUTED),
        ]
        rrows_html = ""
        for k, v, sub, vc in risk_rows:
            rrows_html += (
                f'<tr style="border-bottom:1px solid {BORDER}">'
                f'<td style="color:{MUTED};padding:8px 14px;font-size:12px">{k}</td>'
                f'<td style="padding:8px 14px;text-align:right">'
                f'  <div style="font-family:monospace;font-weight:700;font-size:13px;color:{vc}">{v}</div>'
                f'  {"<div style=font-size:10px;color:" + MUTED2 + ">" + sub + "</div>" if sub else ""}'
                f'</td></tr>'
            )
        st.markdown(
            f'<div style="border:1px solid {BORDER};border-radius:10px;overflow:hidden">'
            f'<table style="width:100%;border-collapse:collapse"><tbody>{rrows_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    with col_analyst:
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin-bottom:12px">Analyst Targets</div>',
            unsafe_allow_html=True,
        )
        analyst_html = _analyst_targets_html(info)
        if analyst_html:
            st.markdown(analyst_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="color:{MUTED2};font-size:12px;padding:20px;text-align:center;'
                f'border:1px solid {BORDER};border-radius:10px">No analyst coverage data</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin-bottom:12px">Key Financials</div>',
            unsafe_allow_html=True,
        )
        def _v(key, fmt=".2f", pct=False):
            val2 = info.get(key)
            if val2 is None: return "n/a"
            try:
                f2 = float(val2)
                return f"{f2*100:.1f}%" if pct else format(f2, fmt)
            except: return "n/a"
        fin_rows = [
            ("Revenue Growth",   _v("revenueGrowth",   pct=True)),
            ("Gross Margin",     _v("grossMargins",     pct=True)),
            ("Operating Margin", _v("operatingMargins", pct=True)),
            ("Net Margin",       _v("profitMargins",    pct=True)),
            ("ROE",              _v("returnOnEquity",   pct=True)),
            ("ROA",              _v("returnOnAssets",   pct=True)),
            ("Debt / Equity",    _v("debtToEquity",     ".2f")),
            ("Current Ratio",    _v("currentRatio",     ".2f")),
            ("Free Cash Flow",   f'${float(info["freeCashflow"])/1e6:,.0f}M' if info.get("freeCashflow") else "n/a"),
            ("Dividend Yield",   _v("dividendYield",    pct=True)),
            ("52W High",         f'${float(info["fiftyTwoWeekHigh"]):,.2f}' if info.get("fiftyTwoWeekHigh") else "n/a"),
            ("52W Low",          f'${float(info["fiftyTwoWeekLow"]):,.2f}' if info.get("fiftyTwoWeekLow") else "n/a"),
        ]
        fr_html = "".join(
            f'<tr style="border-bottom:1px solid {BORDER}">'
            f'<td style="color:{MUTED};padding:7px 14px;font-size:12px">{k}</td>'
            f'<td style="font-family:monospace;font-weight:600;padding:7px 14px;font-size:12px;text-align:right;color:{TEXT}">{v}</td>'
            f'</tr>'
            for k, v in fin_rows
        )
        st.markdown(
            f'<div style="border:1px solid {BORDER};border-radius:10px;overflow:hidden">'
            f'<table style="width:100%;border-collapse:collapse"><tbody>{fr_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    with col_tech:
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin-bottom:12px">Technical Status</div>',
            unsafe_allow_html=True,
        )
        tech_html = _technical_summary_html(info, hist)
        if tech_html:
            st.markdown(tech_html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="color:{MUTED2};font-size:12px;padding:20px;text-align:center;'
                f'border:1px solid {BORDER};border-radius:10px">Technical data unavailable</div>',
                unsafe_allow_html=True,
            )

    # ── NEWS ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin-bottom:12px">'
        f'Latest News <span style="font-size:11px;font-weight:400;color:{MUTED2}">· multi-source · sentiment scored</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Loading news…"):
        articles = NewsFetcher().fetch_ticker_news(ticker, n=12)

    if not articles:
        st.markdown(
            f'<div style="color:{MUTED2};font-size:12px;padding:16px;text-align:center;'
            f'border:1px solid {BORDER};border-radius:10px">No recent news found</div>',
            unsafe_allow_html=True,
        )
    else:
        score    = NewsFetcher().score_sentiment([a["title"] for a in articles])
        sc_color = GREEN if score >= 60 else RED if score < 40 else AMBER
        sc_bg    = "#ECFDF5" if score >= 60 else "#FEF2F2" if score < 40 else "#FFFBEB"
        sc_label = "Positive" if score >= 60 else "Negative" if score < 40 else "Neutral"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
            f'<span class="badge" style="color:{sc_color};background:{sc_bg};font-size:12px;padding:5px 12px">'
            f'Sentiment {score:.0f}/100 — {sc_label}</span>'
            f'<span style="font-size:11px;color:{MUTED2}">{len(articles)} articles</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        news_html = ""
        for a in articles:
            hint    = a.get("sentiment_hint", "neutral")
            dot_col = GREEN if hint == "positive" else RED if hint == "negative" else MUTED2
            title_html = (
                f'<a href="{a["url"]}" target="_blank" style="color:{TEXT};text-decoration:none;'
                f'font-weight:600">{a["title"]}</a>'
                if a.get("url") else
                f'<span style="font-weight:600">{a["title"]}</span>'
            )
            news_html += (
                f'<div style="display:flex;gap:10px;padding:11px 0;border-bottom:1px solid {BORDER}">'
                f'  <span style="color:{dot_col};font-size:8px;margin-top:5px;flex-shrink:0">●</span>'
                f'  <div>'
                f'    <div style="font-size:13px;line-height:1.5;color:{TEXT}">{title_html}</div>'
                f'    <div style="font-size:11px;color:{MUTED2};margin-top:4px">'
                f'      {a.get("source","—")}  ·  {a.get("published","—")}'
                f'    </div>'
                f'  </div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="border:1px solid {BORDER};border-radius:10px;padding:0 16px">{news_html}</div>',
            unsafe_allow_html=True,
        )

    # ── PROTOCOL GATES ────────────────────────────────────────────────────────
    p = proto_map_local.get(ticker)
    if p:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{TEXT};margin-bottom:4px">Investment Protocol Gates</div>'
            f'<div style="font-size:11px;color:{MUTED2};margin-bottom:14px">'
            f'Score {p.get("overall_score",0):.1f}  ·  Conviction {p.get("conviction","—")}  ·  '
            f'{p.get("pass_count",0)} pass / {p.get("warn_count",0)} warn / {p.get("fail_count",0)} fail'
            f'</div>',
            unsafe_allow_html=True,
        )
        gates    = p.get("gates", [])
        statuses = p.get("gate_statuses", [])
        gate_cols = st.columns(7)
        for i, (col, gname, score2, status) in enumerate(
                zip(gate_cols, GATE_NAMES, gates, statuses)):
            sc_col = GREEN if status == "pass" else AMBER if status == "warn" else RED
            bg_col = "#ECFDF5" if status == "pass" else "#FFFBEB" if status == "warn" else "#FEF2F2"
            with col:
                st.markdown(
                    f'<div style="background:{bg_col};border:1px solid {BORDER};'
                    f'border-top:4px solid {sc_col};border-radius:10px;'
                    f'padding:14px 8px;text-align:center">'
                    f'  <div style="font-size:9px;font-weight:700;text-transform:uppercase;'
                    f'       letter-spacing:.08em;color:{MUTED2};margin-bottom:8px">{gname}</div>'
                    f'  <div style="font-size:26px;font-weight:900;color:{sc_col};line-height:1">'
                    f'    {float(score2):.0f}</div>'
                    f'  <div style="font-size:9px;font-weight:800;text-transform:uppercase;'
                    f'       color:{sc_col};margin-top:6px;letter-spacing:.06em">{status.upper()}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — STOCK LOOKUP (any ticker, fetch on demand)
# ─────────────────────────────────────────────────────────────────────────────
def tab_stock_lookup(universe_data, valuation, risk, protocol, rf_rate):
    st.markdown(
        shdr("Stock Lookup",
             "Search any ticker — fetches fresh data on demand, runs full analysis"),
        unsafe_allow_html=True,
    )

    col_inp, col_per, col_btn = st.columns([3, 1, 1])
    with col_inp:
        ticker_input = st.text_input("Ticker Symbol", placeholder="AAPL, TSLA, BRK-B …",
                                     label_visibility="collapsed")
    with col_per:
        period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"],
                              index=3, label_visibility="collapsed")
    with col_btn:
        fetch_btn = st.button("Look Up", type="primary", use_container_width=True)

    ticker_input = ticker_input.strip().upper()

    # Cache lookup result
    if "lookup_ticker" not in st.session_state:
        st.session_state.lookup_ticker  = None
        st.session_state.lookup_period  = "1y"
        st.session_state.lookup_data    = {}
        st.session_state.lookup_val     = {}
        st.session_state.lookup_risk    = {}
        st.session_state.lookup_proto   = []

    if fetch_btn and ticker_input:
        st.session_state.lookup_ticker = ticker_input
        st.session_state.lookup_period = period

        # Use cached data if available, otherwise fetch + run full quant analysis
        if ticker_input in universe_data:
            ticker_data = {ticker_input: universe_data[ticker_input]}
        else:
            with st.spinner(f"Fetching {ticker_input}…"):
                fetcher = DataFetcher("2y")
                fresh   = fetcher.fetch_universe([ticker_input])
            if not fresh:
                st.error(f"Could not fetch data for {ticker_input}. Check the ticker symbol.")
                st.session_state.lookup_ticker = None
                ticker_data = {}
            else:
                st.session_state.lookup_data = fresh
                ticker_data = fresh

        if ticker_data:
            ticker_df = pd.DataFrame([{
                "ticker":          ticker_input,
                "sector":          ticker_data[ticker_input].get("sector", "Unknown"),
                "composite_score": 0,
            }])
            with st.spinner("Running full quantitative analysis…"):
                lv = ValuationEngine(rf_rate).analyze_all(ticker_df, ticker_data)
                lr = RiskEngine().analyze_all(ticker_df, ticker_data, rf_rate)
                lp = ProtocolAnalyzer().analyze_all(ticker_df, ticker_data, lv)
            st.session_state.lookup_val   = lv
            st.session_state.lookup_risk  = lr
            st.session_state.lookup_proto = lp

    if st.session_state.lookup_ticker:
        t   = st.session_state.lookup_ticker
        per = st.session_state.lookup_period
        combined_data  = {**universe_data, **st.session_state.lookup_data}
        combined_val   = {**valuation,     **st.session_state.lookup_val}
        combined_risk  = {**risk,          **st.session_state.lookup_risk}
        existing_proto_tickers = {p["ticker"] for p in protocol}
        combined_proto = list(protocol) + [
            p for p in st.session_state.lookup_proto
            if p["ticker"] not in existing_proto_tickers
        ]
        _render_stock_detail(t, combined_data, combined_val, combined_risk,
                             combined_proto, rf_rate, period=per, fetch_fresh=False)
    else:
        st.markdown(
            f'<div style="text-align:center;padding:60px 20px;color:{MUTED2}">'
            f'Enter a ticker symbol above and click Look Up</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def _run_backtest_simulation(hist: "pd.DataFrame", entry_low: float, target: float, stop: float) -> list:
    """
    Simulate the valuation entry strategy on historical prices.
    Entry : price <= entry_low (in the buy zone)
    Exit  : price >= target (take profit) OR price <= stop (stop loss)
    Returns list of trade dicts.
    """
    trades = []
    in_trade = False
    entry_price = None
    entry_date  = None

    prices = hist["Close"].dropna()
    for date, price in prices.items():
        price = float(price)
        if not in_trade:
            if price <= entry_low:
                in_trade    = True
                entry_price = price
                entry_date  = date
        else:
            if price >= target:
                ret = (price - entry_price) / entry_price * 100
                trades.append({
                    "entry_date": str(entry_date)[:10],
                    "entry":      round(entry_price, 2),
                    "exit_date":  str(date)[:10],
                    "exit":       round(price, 2),
                    "return_pct": round(ret, 2),
                    "reason":     "Target hit",
                    "won":        True,
                })
                in_trade = False
            elif price <= stop:
                ret = (price - entry_price) / entry_price * 100
                trades.append({
                    "entry_date": str(entry_date)[:10],
                    "entry":      round(entry_price, 2),
                    "exit_date":  str(date)[:10],
                    "exit":       round(price, 2),
                    "return_pct": round(ret, 2),
                    "reason":     "Stop loss",
                    "won":        False,
                })
                in_trade = False

    # Open position — mark-to-market
    if in_trade:
        last_price = float(prices.iloc[-1])
        last_date  = prices.index[-1]
        ret = (last_price - entry_price) / entry_price * 100
        trades.append({
            "entry_date": str(entry_date)[:10],
            "entry":      round(entry_price, 2),
            "exit_date":  str(last_date)[:10],
            "exit":       round(last_price, 2),
            "return_pct": round(ret, 2),
            "reason":     "Open position",
            "won":        ret >= 0,
        })

    return trades


def tab_backtest(universe_data: dict, valuation: dict, risk: dict, rf_rate: float):
    st.markdown(
        shdr("Backtest", "Simulate our valuation entry strategy on historical price data"),
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    all_tickers = sorted(set(list(universe_data.keys()) + list(valuation.keys())))
    col_t, col_p, col_run = st.columns([3, 1, 1])
    with col_t:
        bt_ticker = st.selectbox("Select ticker to backtest", all_tickers,
                                 label_visibility="collapsed",
                                 placeholder="Choose a ticker…")
    with col_p:
        bt_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1,
                                 label_visibility="collapsed")
    with col_run:
        go_btn = st.button("Run Backtest", type="primary", use_container_width=True)

    if "bt_result" not in st.session_state:
        st.session_state.bt_result  = None
        st.session_state.bt_ticker  = None
        st.session_state.bt_period  = None

    if go_btn and bt_ticker:
        st.session_state.bt_ticker = bt_ticker
        st.session_state.bt_period = bt_period

        # Get or fetch historical data
        if bt_ticker in universe_data and universe_data[bt_ticker].get("history") is not None:
            hist_raw = universe_data[bt_ticker]["history"]
        else:
            with st.spinner(f"Fetching {bt_ticker} price history…"):
                import yfinance as yf
                hist_raw = yf.download(bt_ticker, period=bt_period, progress=False, auto_adjust=True)

        # Get valuation for entry/target/stop
        val_r = valuation.get(bt_ticker, {})
        if not val_r:
            # Run valuation on the fly
            with st.spinner("Running valuation…"):
                d = universe_data.get(bt_ticker, {})
                if d:
                    df_tmp = pd.DataFrame([{"ticker": bt_ticker,
                                            "sector": d.get("sector","Unknown"),
                                            "composite_score": 0}])
                    val_r = ValuationEngine(rf_rate).analyze_all(df_tmp, universe_data).get(bt_ticker, {})

        entry_low = val_r.get("entry_low")
        target    = val_r.get("target_price")
        stop_loss = val_r.get("stop_loss")
        fair_val  = val_r.get("fair_value")

        if hist_raw is not None and not hist_raw.empty and entry_low and target and stop_loss:
            trades = _run_backtest_simulation(hist_raw, entry_low, target, stop_loss)

            # Fetch S&P500 for comparison
            try:
                import yfinance as yf
                sp_hist = yf.download("^GSPC", period=bt_period, progress=False, auto_adjust=True)
                sp500_ret = (float(sp_hist["Close"].iloc[-1]) / float(sp_hist["Close"].iloc[0]) - 1) * 100 if not sp_hist.empty else None
            except Exception:
                sp500_ret = None

            bah_ret = (float(hist_raw["Close"].iloc[-1]) / float(hist_raw["Close"].iloc[0]) - 1) * 100

            st.session_state.bt_result = {
                "trades":      trades,
                "hist":        hist_raw,
                "val":         val_r,
                "entry_low":   entry_low,
                "target":      target,
                "stop_loss":   stop_loss,
                "fair_value":  fair_val,
                "bah_ret":     bah_ret,
                "sp500_ret":   sp500_ret,
            }
        else:
            st.session_state.bt_result = {"error": "Insufficient valuation data or price history."}

    # ── Results ───────────────────────────────────────────────────────────────
    bt = st.session_state.bt_result
    if not bt:
        st.markdown(
            f'<div style="text-align:center;padding:60px 20px;color:{MUTED2};font-size:14px">'
            f'Select a ticker and click Run Backtest to simulate our valuation entry strategy.</div>',
            unsafe_allow_html=True,
        )
        return

    if bt.get("error"):
        st.error(bt["error"])
        return

    trades    = bt["trades"]
    hist      = bt["hist"]
    entry_low = bt["entry_low"]
    target    = bt["target"]
    stop_loss = bt["stop_loss"]
    fair_val  = bt["fair_value"]
    bah_ret   = bt["bah_ret"]
    sp500_ret = bt.get("sp500_ret")
    t_name    = st.session_state.bt_ticker

    # ── Strategy metrics ──────────────────────────────────────────────────────
    n_trades  = len(trades)
    n_wins    = sum(1 for t in trades if t["won"])
    win_rate  = (n_wins / n_trades * 100) if n_trades > 0 else 0
    open_pos  = next((t for t in trades if t["reason"] == "Open position"), None)
    closed    = [t for t in trades if t["reason"] != "Open position"]

    # Compound return across closed trades
    compound = 1.0
    for t in closed:
        compound *= (1 + t["return_pct"] / 100)
    total_return = (compound - 1) * 100 if closed else (open_pos["return_pct"] if open_pos else 0)

    # ── Summary tiles ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cols_m = st.columns(5)
    def _bt_tile(label, value, sub, color):
        return (
            f'<div style="background:{GRAY_LT};border:1px solid {BORDER};'
            f'border-top:3px solid {color};border-radius:10px;padding:14px 16px">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.09em;color:{MUTED2};margin-bottom:6px">{label}</div>'
            f'<div style="font-size:22px;font-weight:800;color:{color};'
            f'font-variant-numeric:tabular-nums">{value}</div>'
            f'<div style="font-size:11px;color:{MUTED2};margin-top:4px">{sub}</div>'
            f'</div>'
        )

    ret_color  = GREEN if total_return > 0 else RED
    wr_color   = GREEN if win_rate >= 60 else RED if win_rate < 40 else AMBER
    bah_color  = GREEN if bah_ret > 0 else RED
    sp_color   = GREEN if (sp500_ret or 0) > 0 else RED
    alpha_val  = total_return - (sp500_ret or 0)
    alpha_clr  = GREEN if alpha_val > 0 else RED

    with cols_m[0]: st.markdown(_bt_tile("Strategy Return", f"{total_return:+.1f}%", f"{len(closed)} closed trades", ret_color), unsafe_allow_html=True)
    with cols_m[1]: st.markdown(_bt_tile("Win Rate", f"{win_rate:.0f}%", f"{n_wins}W / {n_trades-n_wins}L", wr_color), unsafe_allow_html=True)
    with cols_m[2]: st.markdown(_bt_tile("Buy & Hold", f"{bah_ret:+.1f}%", "same period", bah_color), unsafe_allow_html=True)
    with cols_m[3]: st.markdown(_bt_tile("S&P 500", f"{sp500_ret:+.1f}%" if sp500_ret is not None else "—", "same period", sp_color), unsafe_allow_html=True)
    with cols_m[4]: st.markdown(_bt_tile("Alpha vs S&P", f"{alpha_val:+.1f}%" if sp500_ret is not None else "—", "strategy − benchmark", alpha_clr), unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(shdr("Price History with Entry / Exit Points"), unsafe_allow_html=True)

    try:
        hist_plot = hist.copy()
        if hasattr(hist_plot.index, "tz") and hist_plot.index.tz is not None:
            hist_plot.index = hist_plot.index.tz_localize(None)

        closes = hist_plot["Close"].squeeze()

        fig = go.Figure()
        # Price line
        fig.add_trace(go.Scatter(x=closes.index, y=closes.values,
                                 name=t_name, line=dict(color=BLUE, width=2),
                                 hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"))

        # Shaded zones
        fig.add_hrect(y0=0, y1=entry_low, fillcolor=GREEN, opacity=0.05,
                      layer="below", line_width=0, annotation_text="Buy zone",
                      annotation_position="left")
        if fair_val:
            fig.add_hrect(y0=entry_low, y1=fair_val, fillcolor=AMBER, opacity=0.04,
                          layer="below", line_width=0)
        fig.add_hline(y=target,    line_color=GREEN, line_dash="dash", line_width=1.5,
                      annotation_text=f"Target ${target:,.2f}", annotation_position="right")
        fig.add_hline(y=entry_low, line_color=BLUE,  line_dash="dot",  line_width=1.5,
                      annotation_text=f"Entry ${entry_low:,.2f}", annotation_position="right")
        fig.add_hline(y=stop_loss, line_color=RED,   line_dash="dash", line_width=1.5,
                      annotation_text=f"Stop ${stop_loss:,.2f}", annotation_position="right")
        if fair_val:
            fig.add_hline(y=fair_val, line_color=AMBER, line_dash="dot", line_width=1,
                          annotation_text=f"Fair Value ${fair_val:,.2f}", annotation_position="right")

        # Entry / exit markers
        for tr in trades:
            try:
                fig.add_trace(go.Scatter(
                    x=[tr["entry_date"]], y=[tr["entry"]],
                    mode="markers", marker=dict(symbol="triangle-up", size=12, color=GREEN),
                    name="Entry", showlegend=False,
                    hovertemplate=f"BUY {tr['entry_date']}<br>${tr['entry']:,.2f}<extra></extra>",
                ))
                exit_color = GREEN if tr["won"] else RED
                exit_sym   = "circle" if tr["reason"] != "Open position" else "circle-open"
                fig.add_trace(go.Scatter(
                    x=[tr["exit_date"]], y=[tr["exit"]],
                    mode="markers", marker=dict(symbol=exit_sym, size=10, color=exit_color),
                    name=tr["reason"], showlegend=False,
                    hovertemplate=f"{tr['reason']} {tr['exit_date']}<br>${tr['exit']:,.2f} ({tr['return_pct']:+.1f}%)<extra></extra>",
                ))
            except Exception:
                pass

        fig.update_layout(
            **_plotly_base(),
            height=420,
            showlegend=False,
            margin=dict(l=40, r=120, t=30, b=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False, tickprefix="$"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart error: {e}")

    # ── Trade log ─────────────────────────────────────────────────────────────
    if trades:
        st.markdown(shdr("Trade Log"), unsafe_allow_html=True)
        trade_rows = ""
        for tr in trades:
            ret_c  = GREEN if tr["won"] else RED
            rsnbg  = "#ECFDF5" if tr["reason"] == "Target hit" else "#FEF2F2" if tr["reason"] == "Stop loss" else AMBER_LT
            rsnc   = GREEN if tr["reason"] == "Target hit" else RED if tr["reason"] == "Stop loss" else AMBER
            trade_rows += (
                f'<tr>'
                f'<td style="padding:8px 14px;font-size:12.5px">{tr["entry_date"]}</td>'
                f'<td style="font-family:monospace;padding:8px 14px">${tr["entry"]:,.2f}</td>'
                f'<td style="padding:8px 14px;font-size:12.5px">{tr["exit_date"]}</td>'
                f'<td style="font-family:monospace;padding:8px 14px">${tr["exit"]:,.2f}</td>'
                f'<td style="font-family:monospace;font-weight:700;color:{ret_c};padding:8px 14px">{tr["return_pct"]:+.2f}%</td>'
                f'<td style="padding:8px 14px"><span class="badge" style="color:{rsnc};background:{rsnbg}">{tr["reason"]}</span></td>'
                f'</tr>'
            )
        st.markdown(
            f'<table class="qt"><thead><tr>'
            f'<th>Entry Date</th><th>Entry $</th><th>Exit Date</th><th>Exit $</th>'
            f'<th>Return</th><th>Reason</th>'
            f'</tr></thead><tbody>{trade_rows}</tbody></table>',
            unsafe_allow_html=True,
        )

    # ── Strategy info ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:20px;padding:14px 18px;background:{GRAY_LT};'
        f'border-radius:8px;border:1px solid {BORDER};font-size:12px;color:{MUTED}">'
        f'<b>Strategy rules:</b> Enter when price ≤ entry zone (${entry_low:,.2f} — '
        f'fair value × 0.80 with 20% margin of safety). '
        f'Take profit at target (${target:,.2f} — fair value × 1.20). '
        f'Cut loss at stop (${stop_loss:,.2f} — entry × 0.92). '
        f'Based on current fundamental valuation — historical fundamentals vary.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — PAST SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
def tab_history():
    """Render the Past Sessions history view (works with or without analysis)."""
    import json as _json
    from datetime import datetime, timezone

    mem_path = os.path.join("memory", "history.json")

    st.markdown(shdr("Past Sessions", "Every run is saved automatically — performance is evaluated after 30 days"), unsafe_allow_html=True)

    if not os.path.exists(mem_path):
        st.info("No history yet. Run the analysis to start building session history.")
        return

    try:
        with open(mem_path) as _f:
            _data = _json.load(_f)
        sessions = _data.get("sessions", [])
    except Exception as _e:
        st.error(f"Could not load history: {_e}")
        return

    if not sessions:
        st.info("No past sessions found. Run the analysis to start building session history.")
        return

    sessions = sorted(sessions, key=lambda s: s.get("timestamp", ""), reverse=True)
    evald    = [s for s in sessions if s.get("evaluated")]
    pending  = [s for s in sessions if not s.get("evaluated")]

    # ── Summary metrics ────────────────────────────────────────────────────
    alphas  = [(s["evaluation"] or {}).get("alpha") for s in evald
               if (s.get("evaluation") or {}).get("alpha") is not None]
    avg_returns = [(s["evaluation"] or {}).get("avg_pick_return") for s in evald
                   if (s.get("evaluation") or {}).get("avg_pick_return") is not None]
    avg_alpha  = sum(alphas)  / len(alphas)  if alphas  else None
    avg_return = sum(avg_returns) / len(avg_returns) if avg_returns else None
    beats_sp   = sum(1 for a in alphas if a > 0)
    win_rate   = beats_sp / len(alphas) * 100 if alphas else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(mtile("Total Sessions", str(len(sessions)), "all time", TEXT, BLUE), unsafe_allow_html=True)
    with c2:
        wr_str = f"{win_rate:.0f}%" if win_rate is not None else "—"
        wr_c   = GREEN if (win_rate or 0) >= 50 else RED
        st.markdown(mtile("Win Rate vs S&P", wr_str, f"{beats_sp} of {len(alphas)} beat benchmark", wr_c, wr_c), unsafe_allow_html=True)
    with c3:
        ar_str = f"{avg_return*100:+.1f}%" if avg_return is not None else "—"
        ar_c   = GREEN if (avg_return or 0) >= 0 else RED
        st.markdown(mtile("Avg Pick Return", ar_str, "across evaluated sessions", ar_c, ar_c), unsafe_allow_html=True)
    with c4:
        al_str = f"{avg_alpha*100:+.1f}%" if avg_alpha is not None else "—"
        al_c   = GREEN if (avg_alpha or 0) >= 0 else RED
        st.markdown(mtile("Avg Alpha", al_str, "pick avg minus S&P 500", al_c, al_c), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Session cards ──────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)

    for session in sessions:
        ts      = session.get("timestamp", "")[:10]
        sid     = session.get("session_id", "?")
        prof    = session.get("profile", {})
        risk    = prof.get("risk_level", "?")
        horizon = prof.get("time_horizon", "?")
        goal    = prof.get("goal", "?")
        picks   = session.get("picks", [])
        is_eval = session.get("evaluated", False)

        tickers_short = "  ·  ".join(p["ticker"] for p in picks[:6])
        if len(picks) > 6:
            tickers_short += f"  +{len(picks)-6}"

        if is_eval:
            ev      = session.get("evaluation") or {}
            avg_ret = ev.get("avg_pick_return", 0) or 0
            alpha   = ev.get("alpha", 0) or 0
            ret_sign = "+" if avg_ret >= 0 else ""
            al_sign  = "+" if alpha   >= 0 else ""
            label = (
                f"**{ts}**  ·  #{sid}  ·  Risk {risk} / {horizon} / {goal}"
                f"  —  Return: {ret_sign}{avg_ret*100:.1f}%"
                f"  ·  Alpha: {al_sign}{alpha*100:.1f}%"
            )
        else:
            try:
                ts_dt    = datetime.fromisoformat(session["timestamp"])
                days_old = (now - ts_dt).days
                days_left = max(0, 30 - days_old)
            except Exception:
                days_old  = 0
                days_left = 30
            label = (
                f"**{ts}**  ·  #{sid}  ·  Risk {risk} / {horizon} / {goal}"
                f"  —  Pending ({days_old} days old, evaluates in ~{days_left} more)"
            )

        with st.expander(label, expanded=False):
            # Profile info row
            st.markdown(
                f'<div style="display:flex;gap:28px;margin-bottom:16px;flex-wrap:wrap">'
                f'<div><div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;letter-spacing:.08em">Risk Level</div>'
                f'<div style="font-weight:700;font-size:15px">{risk}</div></div>'
                f'<div><div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;letter-spacing:.08em">Horizon</div>'
                f'<div style="font-weight:700;font-size:15px">{horizon}</div></div>'
                f'<div><div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;letter-spacing:.08em">Goal</div>'
                f'<div style="font-weight:700;font-size:15px">{goal}</div></div>'
                f'<div><div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;letter-spacing:.08em">Picks</div>'
                f'<div style="font-weight:700;font-size:15px">{len(picks)}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if is_eval:
                ev      = session.get("evaluation") or {}
                avg_ret = ev.get("avg_pick_return", 0) or 0
                sp_ret  = ev.get("sp500_return")
                alpha   = ev.get("alpha", 0) or 0
                eval_dt = (ev.get("evaluation_date") or "")[:10]

                ret_c   = GREEN if avg_ret >= 0 else RED
                alpha_c = GREEN if alpha   >= 0 else RED

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(mtile("Avg Pick Return", f"{avg_ret*100:+.1f}%", f"evaluated {eval_dt}", ret_c, ret_c), unsafe_allow_html=True)
                with mc2:
                    sp_str = f"{sp_ret*100:+.1f}%" if sp_ret is not None else "—"
                    sp_c   = GREEN if (sp_ret or 0) >= 0 else RED
                    st.markdown(mtile("S&P 500 Return", sp_str, "over same period", sp_c, sp_c), unsafe_allow_html=True)
                with mc3:
                    st.markdown(mtile("Alpha", f"{alpha*100:+.1f}%", "picks vs benchmark", alpha_c, alpha_c), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                ev_picks = ev.get("picks", [])
                if ev_picks:
                    entry_map = {p["ticker"]: p.get("price_entry") for p in picks}
                    rows = []
                    for ep in ev_picks:
                        r = ep.get("return", 0) or 0
                        rows.append({
                            "Ticker":      ep["ticker"],
                            "Entry":       f"${entry_map.get(ep['ticker'], 0):.2f}" if entry_map.get(ep['ticker']) else "—",
                            "Exit":        f"${ep.get('price_exit', 0):.2f}",
                            "Return":      f"{r*100:+.1f}%",
                            "Score @ Rec": f"{ep.get('composite_at_rec', 0):.1f}",
                        })

                    def _color_ret(val):
                        try:
                            v = float(str(val).replace("%","").replace("+",""))
                            return f"color: {'#16a34a' if v >= 0 else '#dc2626'}; font-weight: 700"
                        except Exception:
                            return ""

                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df.style.map(_color_ret, subset=["Return"]),
                        use_container_width=True, hide_index=True,
                    )
            else:
                try:
                    ts_dt    = datetime.fromisoformat(session["timestamp"])
                    days_old = (now - ts_dt).days
                    days_left = max(0, 30 - days_old)
                except Exception:
                    days_old  = 0
                    days_left = 30

                st.info(
                    f"Evaluation pending — {days_old} days old. "
                    f"Will auto-evaluate in ~{days_left} more days next time you run the tool."
                )

                if picks:
                    rows = [
                        {"Ticker": p["ticker"],
                         "Entry Price": f"${p.get('price_entry', 0):.2f}",
                         "Score": f"{p.get('composite_score', 0):.1f}"}
                        for p in picks
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    profile, run_btn, hist_btn, bt_btn = render_sidebar()

    if "results" not in st.session_state:
        st.session_state.results           = None
        st.session_state.profile           = None
        st.session_state.show_history      = False
        st.session_state.show_backtest     = False
        st.session_state.rankings_selected = None
        st.session_state.bt_result         = None
        st.session_state.bt_ticker         = None
        st.session_state.bt_period         = None

    if run_btn:
        st.session_state.show_history  = False
        st.session_state.show_backtest = False
        with st.spinner("Running analysis…"):
            st.session_state.results = run_analysis(profile)
            st.session_state.profile = profile
        st.rerun()

    if hist_btn:
        st.session_state.show_history = not st.session_state.get("show_history", False)
        st.rerun()

    if bt_btn:
        st.session_state.show_backtest = not st.session_state.get("show_backtest", False)
        st.session_state.show_history  = False
        st.rerun()

    # ── Past Sessions view ─────────────────────────────────────────────────
    if st.session_state.get("show_history", False):
        st.markdown(
            f'<div class="ptitle">Stock Ranking Advisor</div>'
            f'<div class="psub">Past Sessions  ·  Click "Past Sessions" in the sidebar to toggle</div>',
            unsafe_allow_html=True,
        )
        tab_history()
        return

    # ── Backtest view (works before and after analysis) ────────────────────
    if st.session_state.get("show_backtest", False):
        st.markdown(
            f'<div class="ptitle">Stock Ranking Advisor</div>'
            f'<div class="psub">Backtest  ·  Simulate valuation entry strategy  ·  Click "Backtest" to close</div>',
            unsafe_allow_html=True,
        )
        uni_bt  = (st.session_state.results or {}).get("universe_data", {})
        val_bt  = (st.session_state.results or {}).get("valuation", {})
        risk_bt = (st.session_state.results or {}).get("risk", {})
        rf_bt   = (st.session_state.results or {}).get("rf_rate", 0.045)
        tab_backtest(uni_bt, val_bt, risk_bt, rf_bt)
        return

    if st.session_state.results is None:
        render_welcome()
        return

    res     = st.session_state.results
    profile = st.session_state.profile
    top10   = res["top10"]
    macro   = res["macro_data"]
    val     = res["valuation"]
    risk    = res["risk"]
    proto   = res["protocol"]
    uni     = res["universe_data"]
    sp500   = res["sp500_hist"]
    rf      = res["rf_rate"]

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="ptitle">Stock Ranking Advisor</div>'
        f'<div class="psub">v3  ·  Pure Quantitative  ·  No AI APIs  ·  '
        f'{profile.risk_label}  ·  {HORIZON_LABELS[profile.time_horizon]}  ·  '
        f'${profile.portfolio_size:,.0f}  ·  {profile.goal_label}</div>',
        unsafe_allow_html=True,
    )

    render_macro_strip(macro, top10, rf)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Rankings",
        "Valuation",
        "Risk & Quality",
        "Protocol Gates",
        "Portfolio",
        "Macro & Performance",
        "Stock Lookup",
        "History",
        "Backtest",
    ])

    with tab1: tab_rankings(top10, profile, val, proto, risk)
    with tab2: tab_valuation(top10, val)
    with tab3: tab_risk(top10, risk)
    with tab4: tab_protocol(top10, proto)
    with tab5: tab_portfolio(top10, profile)
    with tab6: tab_macro(top10, macro, uni, sp500, profile)
    with tab7: tab_stock_lookup(uni, val, risk, proto, rf)
    with tab8: tab_history()
    with tab9: tab_backtest(uni, val, risk, rf)


if __name__ == "__main__":
    main()
