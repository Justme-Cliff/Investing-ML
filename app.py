"""
app.py — Streamlit dashboard for Stock Ranking Advisor v3

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import streamlit as st

st.set_page_config(
    page_title="Stock Advisor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    STOCK_UNIVERSE, WEIGHT_MATRIX, FACTOR_NAMES,
    RISK_LABELS, GOAL_LABELS, HORIZON_LABELS,
    DYNAMIC_UNIVERSE, UNIVERSE_MIN_MARKET_CAP, UNIVERSE_MAX_TICKERS, PORTFOLIO_N,
    FRESH_PICKS_PENALTY, VERSION,
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
BLUE       = "#3B82F6"
BLUE_LT    = "#1E3A5F"
GREEN      = "#10B981"
GREEN_LT   = "#064E3B"
AMBER      = "#F59E0B"
AMBER_LT   = "#451A03"
RED        = "#EF4444"
RED_LT     = "#450A0A"
ORANGE     = "#F97316"
ORANGE_LT  = "#431407"
TEXT       = "#E2E8F0"
MUTED      = "#94A3B8"
MUTED2     = "#64748B"
BORDER     = "#1E2535"
GRAY_LT    = "#0D1117"

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
# SETTINGS — persistence + theme injection
# ─────────────────────────────────────────────────────────────────────────────
SETTINGS_FILE = os.path.join("memory", "settings.json")

THEMES = {
    "Terminal": {"bg": "#060A10", "sidebar": "#0A0D14", "accent": "#3B82F6"},
    "Dark":     {"bg": "#0F172A", "sidebar": "#1E293B", "accent": "#818CF8"},
    "Light":    {"bg": "#FFFFFF", "sidebar": "#F8FAFC", "accent": "#2563EB"},
    "Warm":     {"bg": "#FFFBF5", "sidebar": "#FFF3E0", "accent": "#D97706"},
}

DEFAULT_SETTINGS = {
    "theme":          "Terminal",
    "fresh_penalty":  FRESH_PICKS_PENALTY,
    "n_sessions":     2,
    "learning_rate":  0.04,
    "signal_mode":    "Balanced",
}


_SETTINGS_SCHEMA = {
    "theme":         (str,   lambda v: v in THEMES,                            "Terminal"),
    "fresh_penalty": (float, lambda v: 0 <= v <= 50,                           FRESH_PICKS_PENALTY),
    "n_sessions":    (int,   lambda v: 1 <= v <= 10,                            2),
    "learning_rate": (float, lambda v: 0.001 <= v <= 0.5,                       0.04),
    "signal_mode":   (str,   lambda v: v in ("Balanced", "Value", "Momentum"),  "Balanced"),
}


def _load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                raw = json.load(f)
            out = {}
            for key, (typ, validator, default) in _SETTINGS_SCHEMA.items():
                val = raw.get(key, DEFAULT_SETTINGS.get(key, default))
                try:
                    val = typ(val)
                    if not validator(val):
                        val = default
                except Exception:
                    val = default
                out[key] = val
            return out
    except Exception:
        pass
    return dict(DEFAULT_SETTINGS)


def _save_settings(s: dict):
    os.makedirs("memory", exist_ok=True)
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass


def _apply_theme_css():
    """Inject per-session CSS overrides based on active theme."""
    s          = st.session_state.get("settings", DEFAULT_SETTINGS)
    theme_name = s.get("theme", "Terminal")
    t          = THEMES.get(theme_name, THEMES["Terminal"])
    bg     = t["bg"]
    sb     = t["sidebar"]
    accent = t["accent"]
    is_dark = theme_name in ("Terminal", "Dark")

    if is_dark:
        extra = (
            f".mtile{{background:#0F1520!important;border-color:#1E2535!important}}"
            f".pick-card{{background:#0A0D14!important;border-color:#1E2535!important}}"
            f".feature-card{{background:#0A0D14!important;border-color:#1E2535!important}}"
            f".qt th{{background:#0F1520!important;color:#94A3B8!important;"
            f"border-bottom:2px solid #1E2535!important}}"
            f".qt td{{color:#E2E8F0!important;border-bottom:1px solid #1E2535!important}}"
            f".qt tbody tr:hover td{{background:#1E2535!important}}"
            f".sbar-wrap{{background:#1E2535!important}}"
            f".shdr{{color:#E2E8F0!important}}"
            f".ssub{{color:#64748B!important}}"
            f".sb-brand{{color:#E2E8F0!important}}"
            f".fc-title{{color:#E2E8F0!important}}"
            f".fc-desc{{color:#94A3B8!important}}"
            f".hero-title{{color:#E2E8F0!important}}"
            f".hero-sub{{color:#94A3B8!important}}"
            f".ptitle{{color:#E2E8F0!important}}"
            f".stTabs [data-baseweb='tab-list']{{border-bottom:2px solid #1E2535!important}}"
            f".stTabs [data-baseweb='tab']{{color:#64748B!important}}"
            f".stTabs [aria-selected='true']{{color:#E2E8F0!important}}"
            f"[data-testid='stSidebar'] label{{color:#94A3B8!important}}"
            f"[data-testid='stSidebar'] p{{color:#E2E8F0!important}}"
            f"[data-testid='stSidebar'] .stSelectbox label{{color:#94A3B8!important}}"
            f"[data-testid='stMetricLabel']{{color:#94A3B8!important}}"
            f"[data-testid='stMetricValue']{{color:#E2E8F0!important}}"
        )
    else:
        extra = (
            f".mtile{{background:#FFFFFF!important;border-color:#E5E7EB!important}}"
            f".pick-card{{background:#FFFFFF!important;border-color:#E5E7EB!important}}"
            f".feature-card{{background:#FFFFFF!important;border-color:#E5E7EB!important}}"
            f".qt th{{background:#F9FAFB!important;color:#374151!important;"
            f"border-bottom:2px solid #E5E7EB!important}}"
            f".qt td{{color:#111827!important;border-bottom:1px solid #F3F4F6!important}}"
            f".qt tbody tr:hover td{{background:#F0F7FF!important}}"
            f".sbar-wrap{{background:#F3F4F6!important}}"
            f".shdr{{color:#111827!important}}"
            f".ssub{{color:#9CA3AF!important}}"
            f".sb-brand{{color:#111827!important}}"
            f".fc-title{{color:#111827!important}}"
            f".fc-desc{{color:#6B7280!important}}"
            f".hero-title{{color:#111827!important}}"
            f".hero-sub{{color:#6B7280!important}}"
            f".ptitle{{color:#111827!important}}"
            f".stTabs [data-baseweb='tab-list']{{border-bottom:2px solid #E5E7EB!important}}"
            f".stTabs [data-baseweb='tab']{{color:#9CA3AF!important}}"
            f".stTabs [aria-selected='true']{{color:#111827!important}}"
            f"[data-testid='stSidebar'] label{{color:#374151!important}}"
        )

    st.markdown(
        f"<style>"
        f".main,.main .block-container{{background-color:{bg}!important}}"
        f"[data-testid='stSidebar']{{background:{sb}!important}}"
        f".stTabs [aria-selected='true']{{border-bottom-color:{accent}!important}}"
        f".stButton>button[kind='primary']{{background:{accent}!important;"
        f"border-color:{accent}!important}}"
        f".stButton>button[kind='primary']:hover{{opacity:.88}}"
        f"{extra}"
        f"</style>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

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
    background: #0A0D14;
    border-right: 1px solid #1E2535;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
[data-testid="stSidebar"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    letter-spacing: .01em;
}

/* ── Tabs — underline style ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 2px solid #1E2535;
    padding: 0;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 10px 24px;
    font-size: 13px;
    font-weight: 500;
    color: #64748B;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: transparent !important;
    transition: color .15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #94A3B8 !important; }
.stTabs [aria-selected="true"] {
    color: #E2E8F0 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #3B82F6 !important;
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
@keyframes terminalBlink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}

/* ── Metric tile — with left accent border ───────────────────────────── */
.mtile {
    background: #0F1520;
    border: 1px solid #1E2535;
    border-left: 3px solid var(--accent, #1E2535);
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
    transition: box-shadow .2s ease, transform .2s ease;
}
.mtile:hover {
    box-shadow: 0 4px 18px rgba(59,130,246,.12);
    transform: translateY(-1px);
}
.mtile-lbl {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #64748B;
    margin-bottom: 8px;
}
.mtile-val {
    font-size: 22px;
    font-weight: 800;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
    font-family: "JetBrains Mono", monospace;
}
.mtile-sub {
    font-size: 11px;
    color: #64748B;
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
    background: #0F1520;
    color: #94A3B8;
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .08em;
    padding: 10px 14px;
    border-bottom: 2px solid #1E2535;
    text-align: left;
    white-space: nowrap;
}
.qt td {
    padding: 10px 14px;
    border-bottom: 1px solid #1E2535;
    color: #E2E8F0;
    vertical-align: middle;
}
.qt tr:last-child td { border-bottom: none; }
.qt tbody tr { transition: background .1s ease; }
.qt tbody tr:hover td { background: #1E2535; }

/* ── Rank chip ───────────────────────────────────────────────────────── */
.rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 6px;
    background: #3B82F6;
    color: #fff;
    font-size: 10.5px;
    font-weight: 800;
}

/* ── Animated score bar ──────────────────────────────────────────────── */
.sbar-wrap {
    background: #1E2535;
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
    color: #E2E8F0;
    letter-spacing: -.01em;
    margin-bottom: 2px;
}
.ssub { font-size: 11.5px; color: #64748B; margin-bottom: 14px; }

/* ── Pick card — left border colored by signal ───────────────────────── */
.pick-card {
    background: #0A0D14;
    border: 1px solid #1E2535;
    border-left: 4px solid var(--signal-color, #1E2535);
    border-radius: 10px;
    padding: 22px;
    height: 100%;
    animation: fadeUp .4s ease both;
    transition: transform .2s ease, box-shadow .2s ease;
}
.pick-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(59,130,246,.15);
}

/* ── Feature card (welcome page) ─────────────────────────────────────── */
.feature-card {
    background: #0A0D14;
    border: 1px solid #1E2535;
    border-top: 3px solid var(--fc, #3B82F6);
    border-radius: 10px;
    padding: 22px;
    animation: fadeUp .5s ease both;
    transition: transform .18s ease, box-shadow .18s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(59,130,246,.12);
}
.fc-num {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--fc, #3B82F6);
    margin-bottom: 10px;
}
.fc-title {
    font-size: 14px;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 7px;
}
.fc-desc { font-size: 12px; color: #94A3B8; line-height: 1.65; }

/* ── Hero (welcome) ──────────────────────────────────────────────────── */
.hero { text-align: center; padding: 72px 20px 56px; }
.hero-eyebrow {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: #3B82F6;
    margin-bottom: 18px;
}
.hero-title {
    font-size: 52px;
    font-weight: 900;
    color: #E2E8F0;
    letter-spacing: -.045em;
    line-height: 1;
    margin-bottom: 20px;
}
.hero-sub {
    font-size: 17px;
    color: #94A3B8;
    max-width: 480px;
    margin: 0 auto 36px;
    line-height: 1.7;
    font-weight: 400;
}
.hero-note {
    font-size: 12px;
    color: #64748B;
    border: 1px solid #1E2535;
    border-radius: 8px;
    display: inline-block;
    padding: 8px 20px;
}

/* ── Page title ──────────────────────────────────────────────────────── */
.ptitle {
    font-size: 22px;
    font-weight: 800;
    color: #E2E8F0;
    letter-spacing: -.03em;
}
.psub { font-size: 12px; color: #64748B; margin-top: 3px; margin-bottom: 20px; }

/* ── Sidebar brand ───────────────────────────────────────────────────── */
.sb-brand { font-size: 15px; font-weight: 800; color: #E2E8F0; letter-spacing: -.02em; }
.sb-brand-sub { font-size: 11px; color: #64748B; margin-top: 2px; margin-bottom: 16px; }
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
    """Common Plotly layout kwargs — dark terminal theme."""
    return dict(
        template="plotly_dark",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#94A3B8"),
        plot_bgcolor="#0F1520",
        paper_bgcolor="#0A0D14",
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


# ─────────────────────────────────────────────────────────────────────────────
# ANTI-THESIS HTML RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def _anti_thesis_html(flags: list) -> str:
    """
    Render Anti-Thesis Engine red flags as styled HTML cards.
    Each flag: {flag, severity, detail}
    """
    if not flags:
        return ""
    SEV_CLR    = {"HIGH": RED,       "MEDIUM": AMBER,  "LOW": MUTED}
    SEV_BG     = {"HIGH": "#1A0606", "MEDIUM": "#1A1100", "LOW": "#0D0F14"}
    SEV_BORDER = {"HIGH": RED,       "MEDIUM": AMBER,  "LOW": MUTED}
    items = ""
    for f in flags:
        sev = f.get("severity", "MEDIUM")
        clr = SEV_CLR.get(sev, MUTED)
        bg  = SEV_BG.get(sev, "#0D0F14")
        brd = SEV_BORDER.get(sev, MUTED)
        items += (
            f'<div style="background:{bg};border:1px solid {brd}44;border-left:3px solid {brd};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:8px">'
            f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
            f'    <span style="font-size:10px;font-weight:800;text-transform:uppercase;'
            f'letter-spacing:.09em;color:{clr};background:{brd}22;padding:2px 8px;border-radius:3px">{sev}</span>'
            f'    <span style="font-size:13px;font-weight:700;color:{TEXT}">{f["flag"]}</span>'
            f'  </div>'
            f'  <div style="font-size:12px;color:{MUTED};line-height:1.65">{f["detail"]}</div>'
            f'</div>'
        )
    return items


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
    prog = st.progress(1, text="Building stock universe…")

    if DYNAMIC_UNIVERSE:
        from advisor.universe import fetch_us_universe
        all_tickers = fetch_us_universe(
            min_market_cap=UNIVERSE_MIN_MARKET_CAP,
            max_tickers=UNIVERSE_MAX_TICKERS,
            verbose=False,
        )
        all_tickers = [t for t in all_tickers if t not in profile.existing_tickers]
    else:
        all_tickers = [
            t for sector, tlist in STOCK_UNIVERSE.items()
            for t in tlist
            if sector not in profile.excluded_sectors
            and t not in profile.existing_tickers
        ]
        all_tickers = list(dict.fromkeys(all_tickers))

    # ── Per-ticker progress callback: maps 0→100% fetch to 2→30% on the bar ──
    _total_tickers = max(len(all_tickers), 1)

    def _on_fetch_tick(done: int, total: int):
        pct = 2 + int(done / total * 28)   # 2 % → 30 %
        prog.progress(
            pct,
            text=f"Fetching stock data… {done} / {total}  ({pct - 2:.0f}%)",
        )

    fetcher              = DataFetcher(profile.yf_period)
    res["universe_data"] = fetcher.fetch_universe(all_tickers,
                                                   progress_callback=_on_fetch_tick)

    # Post-fetch sector exclusion (applies in both modes)
    if profile.excluded_sectors and res["universe_data"]:
        res["universe_data"] = {
            t: d for t, d in res["universe_data"].items()
            if d.get("sector", "Unknown") not in profile.excluded_sectors
        }

    prog.progress(31, text="Fetching S&P 500 benchmark…")
    res["sp500_hist"]    = fetcher.fetch_sp500()
    prog.progress(33, text="Fetching macro data (VIX · 10Y yield · sector ETFs)…")

    res["macro_data"] = MacroFetcher().fetch()
    rf_rate           = (res["macro_data"].get("yield_10y") or 4.5) / 100
    res["rf_rate"]    = rf_rate
    prog.progress(40, text="Scoring stocks on 7 factors…")

    memory  = SessionMemory(); memory.load()
    adapted = memory.get_adapted_weights(profile.risk_level, profile.time_horizon)
    scorer  = MultiFactorScorer(profile, res["macro_data"], adapted)
    ranked_df = scorer.score_all(res["universe_data"], sp500_hist=res.get("sp500_hist"))

    # ── Apply learned intelligence (pattern + sector + regime) ────────────────
    regime = res["macro_data"].get("regime", "neutral")

    # Layer 6: dynamic sector tilts from observed sector performance
    dynamic_tilts = memory.get_dynamic_sector_tilts(regime)
    for sector, tilt in dynamic_tilts.items():
        mask = ranked_df["sector"] == sector
        ranked_df.loc[mask, "composite_score"] = (
            ranked_df.loc[mask, "composite_score"] + tilt
        ).clip(0, 100)

    # Layer 2: sector-specific factor weight adjustments
    from config import WEIGHT_MATRIX as _WM
    factor_names_local = ["momentum", "volatility", "value", "quality",
                           "technical", "sentiment", "dividend"]
    for sector in ranked_df["sector"].unique():
        adj = memory.get_sector_weight_adjustments(sector)
        if not adj:
            continue
        base_w = list(adapted or _WM.get(
            (profile.risk_level, profile.time_horizon), [1/7]*7
        ))
        adjusted_w = [base_w[i] * adj.get(factor_names_local[i], 1.0)
                      for i in range(len(base_w))]
        total = sum(adjusted_w) or 1.0
        adjusted_w = [w / total for w in adjusted_w]
        mask       = ranked_df["sector"] == sector
        score_cols = [f"{f}_score" for f in factor_names_local]
        if all(c in ranked_df.columns for c in score_cols):
            adj_composite = sum(
                adjusted_w[i] * ranked_df.loc[mask, score_cols[i]]
                for i in range(len(factor_names_local))
            )
            ranked_df.loc[mask, "composite_score"] = (
                ranked_df.loc[mask, "composite_score"] * 0.6 + adj_composite * 0.4
            ).clip(0, 100)

    # Layer 4: pattern bonuses (similarity to historical winners/losers)
    pattern_bonuses = memory.get_all_pattern_bonuses(res["universe_data"], ranked_df, regime)
    for ticker, bonus in pattern_bonuses.items():
        mask = ranked_df["ticker"] == ticker
        if mask.any():
            ranked_df.loc[mask, "composite_score"] = (
                ranked_df.loc[mask, "composite_score"] + bonus
            ).clip(0, 100)

    ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = range(1, len(ranked_df) + 1)

    # ── Tier 2 enrichment: options flow + Google Trends + Reddit ──────────────
    prog.progress(50, text="Enriching top 30 with options flow + retail sentiment…")
    try:
        from advisor.alternative_data import enrich_top_n
        ranked_df = enrich_top_n(ranked_df, res["universe_data"], res["macro_data"], n=30)
    except Exception:
        pass   # enrichment is best-effort; pipeline continues regardless

    # Portfolio continuity bonus — applied AFTER enrichment so enrichment scores
    # don't overwrite the bonus. Disabled in fresh-picks mode.
    if not profile.avoid_recent:
        _cont_tickers = memory.get_recent_tickers(n_sessions=1)
        if _cont_tickers:
            _cont_mask = ranked_df["ticker"].isin(_cont_tickers)
            ranked_df.loc[_cont_mask, "composite_score"] = (
                ranked_df.loc[_cont_mask, "composite_score"] + 3.0
            ).clip(upper=100)
            ranked_df = ranked_df.sort_values(
                "composite_score", ascending=False
            ).reset_index(drop=True)
            ranked_df["rank"] = range(1, len(ranked_df) + 1)

    # Fresh picks mode: penalise tickers from recent sessions
    if profile.avoid_recent:
        _cfg           = st.session_state.get("settings", DEFAULT_SETTINGS)
        _n_sess        = int(_cfg.get("n_sessions", 2))
        _penalty       = float(_cfg.get("fresh_penalty", FRESH_PICKS_PENALTY))
        recent_tickers = memory.get_recent_tickers(n_sessions=_n_sess)
        if not recent_tickers:
            st.info("Fresh picks mode: no prior session picks found — all stocks eligible.")
        if recent_tickers:
            PENALTY = _penalty
            mask = ranked_df["ticker"].isin(recent_tickers)
            ranked_df.loc[mask, "composite_score"] = (
                ranked_df.loc[mask, "composite_score"] - PENALTY
            ).clip(lower=0)
            ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    res["ranked_df"] = ranked_df
    prog.progress(62, text="Building portfolio (correlation-aware selection)…")

    constructor  = PortfolioConstructor(n=PORTFOLIO_N)
    top10        = constructor.select(res["ranked_df"], res["universe_data"], profile.risk_level)
    res["top10"] = constructor.size_positions(top10, profile.portfolio_size,
                                               macro_data=res.get("macro_data"),
                                               universe_data=res.get("universe_data"))
    prog.progress(72, text="Running valuation + risk in parallel…")

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TOut
    from config import PIPELINE_CONFIG as _PC
    _timeout = _PC.get("engine_timeout", 120)
    _engine_v = ValuationEngine(rf_rate)
    _engine_r = RiskEngine()
    with ThreadPoolExecutor(max_workers=2) as _exe:
        _fv = _exe.submit(_engine_v.analyze_all, res["top10"], res["universe_data"])
        _fr = _exe.submit(_engine_r.analyze_all, res["top10"], res["universe_data"], rf_rate)
        try:
            res["valuation"] = _fv.result(timeout=_timeout)
        except _TOut:
            res["valuation"] = {}
        try:
            res["risk"] = _fr.result(timeout=_timeout)
        except _TOut:
            res["risk"] = {}
    prog.progress(88, text="Running 7-gate investment protocol…")

    res["protocol"] = ProtocolAnalyzer().analyze_all(
        res["top10"], res["universe_data"], res["valuation"]
    )
    prog.progress(100, text="Done.")
    prog.empty()

    # ── Persist session to history ─────────────────────────────────────────
    try:
        sp500_h = res.get("sp500_hist")
        if sp500_h is not None and len(sp500_h) > 0:
            if isinstance(sp500_h.columns, pd.MultiIndex):
                sp500_h = sp500_h.copy()
                sp500_h.columns = sp500_h.columns.get_level_values(0)
            sp500_price = float(sp500_h["Close"].iloc[-1])
        else:
            sp500_price = 0.0
        memory.save_session(
            profile, res["top10"], sp500_price,
            macro_data=res.get("macro_data"),
            valuation_results=res.get("valuation"),
            universe_data=res.get("universe_data"),
            risk_results=res.get("risk"),
            protocol_results=res.get("protocol"),
        )
        memory.save()
    except Exception as _mem_err:
        pass   # never block the UI over a save failure

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
            min_value=1_000, max_value=1_000_000_000,
            value=50_000, step=10_000, format="%d",
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
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            hist_btn = st.button("History",  type="secondary", use_container_width=True)
        with col_b2:
            bt_btn   = st.button("Backtest", type="secondary", use_container_width=True)
        with col_b3:
            cal_btn  = st.button("Calendar", type="secondary", use_container_width=True)
        settings_btn = st.button("⚙️  Settings", type="secondary", use_container_width=True)
        st.markdown(
            f'<div style="text-align:center;font-size:11px;color:#4B5563;margin-top:12px">'
            f'v{VERSION}</div>',
            unsafe_allow_html=True,
        )

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
    return profile, run_btn, hist_btn, bt_btn, cal_btn, settings_btn


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
        ("#2563EB", "01", "7-Factor Scoring",   "12-1 momentum · EV/EBITDA value · Novy-Marx quality · technicals"),
        ("#059669", "02", "4-Method Valuation",  "DCF (2-stage) · Graham Number · EV/EBITDA target · FCF yield"),
        ("#D97706", "03", "Full Risk Suite",     "Altman Z · Sharpe · Sortino · Max DD · VaR 95% · ROIC/WACC"),
        ("#7C3AED", "04", "7-Gate Protocol",     "Warren Buffett quality screen — every stock must earn its place"),
    ]
    cards_html = '<div style="display:flex;gap:16px;align-items:stretch;">'
    for i, (fc, num, title, desc) in enumerate(features):
        cards_html += (
            f'<div class="feature-card" style="--fc:{fc};animation-delay:{i*0.08}s;flex:1;">'
            f'<div class="fc-num">{num}</div>'
            f'<div class="fc-title">{title}</div>'
            f'<div class="fc-desc">{desc}</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


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
        (c3, "Regime",        regime,                             reasons[:40] if reasons else "—", r_color, r_color),
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
            earn_bg = RED_LT if days_away <= 7 else AMBER_LT if days_away <= 14 else GRAY_LT
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
                f'<div style="height:1px;background:#1E2535;margin:16px 0"></div>'

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
                   gridcolor="#1E2535"),
        yaxis=dict(tickfont=dict(size=12, family="monospace"), gridcolor="#1E2535"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Factor radar + score distribution ─────────────────────────────────────
    col_rad, col_hist = st.columns([1, 1])
    with col_rad:
        st.markdown(shdr("Factor Fingerprint", "Top-5 picks — 7-factor radar chart"),
                    unsafe_allow_html=True)
        ranked_df = st.session_state.results.get("ranked_df", pd.DataFrame())
        st.plotly_chart(_factor_radar_chart(top10), use_container_width=True)
    with col_hist:
        st.markdown(shdr("Score Distribution", "Portfolio picks vs full universe"),
                    unsafe_allow_html=True)
        ranked_df = st.session_state.results.get("ranked_df", pd.DataFrame())
        if not ranked_df.empty:
            st.plotly_chart(_score_distribution_chart(ranked_df, top10), use_container_width=True)
        else:
            st.caption("Score distribution not available.")

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
        fig.add_vline(x=0, line_dash="dash", line_color="#1E2535", line_width=1.2)
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
                       zeroline=True, zerolinecolor="#1E2535", gridcolor="#1E2535"),
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
                            line=dict(width=1.5, color="#0A0D14")),
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
            yaxis=dict(title="Price ($)", tickformat="$,.0f", gridcolor="#1E2535"),
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
            row_bg = "#1A1500" if sname == "Base" else "#0A0D14"
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

    # ── DCF Scenario Waterfall ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(shdr("DCF Scenario Comparison", "Bear / Base / Bull fair values vs current price (dashed)"),
                unsafe_allow_html=True)
    st.plotly_chart(_dcf_waterfall_chart(top10, valuation), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RISK & QUALITY
# ─────────────────────────────────────────────────────────────────────────────
def tab_risk(top10, risk, universe_data=None):
    st.markdown(
        shdr("Risk & Quality Metrics",
             "Altman Z · Sharpe · Sortino · Max DD · VaR 95% · CVaR 95% · ROIC/WACC · Piotroski"),
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
        z_badge  = zone_badge(az.get("zone", "—")) if az.get("zone") else "—"
        rt_data  = r.get("roic_trend", {})
        rt_trend = rt_data.get("trend", "—")
        rt_clr   = GREEN if rt_trend in ("EXPANDING", "IMPROVING") else RED if rt_trend in ("NEGATIVE", "CONTRACTING") else MUTED
        beneish  = r.get("beneish")
        bm_score = beneish.get("m_score") if beneish else None
        bm_clr   = RED if (bm_score is not None and bm_score > -1.78) else GREEN
        bm_str   = f"{bm_score:.2f}" if bm_score is not None else "—"
        rows += (
            f'<tr>'
            f'<td><b>{t}</b></td>'
            f'<td style="font-family:monospace">{fmt_2(az.get("score"))}</td>'
            f'<td>{z_badge}</td>'
            f'<td style="font-family:monospace;color:{sh_clr};font-weight:600">{fmt_2(sh)}</td>'
            f'<td style="font-family:monospace">{fmt_2(r.get("sortino"))}</td>'
            f'<td style="font-family:monospace;color:{RED}">{fmt_pct(r.get("max_drawdown_pct"), False)}</td>'
            f'<td style="font-family:monospace;color:{AMBER}">{fmt_pct(r.get("var_95_pct"), False)}</td>'
            f'<td style="font-family:monospace;color:{RED};font-weight:600">{fmt_pct(r.get("cvar_95_pct"), False)}</td>'
            f'<td style="font-family:monospace;color:{rw_clr};font-weight:700">{fmt_pct(rw_sp)}</td>'
            f'<td style="font-size:11px;color:{MUTED}">{rw.get("verdict","—")}</td>'
            f'<td style="font-weight:700;color:{pf_clr}">'
            f'{pf_sc if pf_sc is not None else "—"}/9</td>'
            f'<td style="font-family:monospace;color:{MUTED}">{fmt_2(r.get("accruals"))}</td>'
            f'<td style="font-size:11px;font-weight:700;color:{rt_clr}">{rt_trend}</td>'
            f'<td style="font-family:monospace;color:{bm_clr};font-weight:600">{bm_str}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="qt"><thead><tr>'
        f'<th>Ticker</th><th>Altman Z</th><th>Zone</th>'
        f'<th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>VaR 95%</th><th>CVaR 95%</th>'
        f'<th>ROIC/WACC</th><th>Verdict</th><th>ROIC Trend</th><th>Piotroski</th><th>Accruals</th><th>Beneish M</th>'
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
                        line=dict(width=1.5, color="#0A0D14")),
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
            xaxis=dict(title="Sharpe Ratio", gridcolor="#1E2535"),
            yaxis=dict(title="ROIC / WACC Spread (%)", gridcolor="#1E2535"),
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
            yaxis=dict(range=[0, 11], title="Score / 9", gridcolor="#1E2535"),
            xaxis=dict(tickfont=dict(size=11, family="monospace")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    if universe_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(shdr("Monte Carlo Portfolio Simulation",
                         "200 paths · 252 trading days · Equal-weighted top-10"),
                    unsafe_allow_html=True)
        st.plotly_chart(_monte_carlo_chart(top10, universe_data), use_container_width=True)

    # ── Anti-Thesis Overview ──────────────────────────────────────────────────
    if universe_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            shdr("Anti-Thesis Overview",
                 "Structural red flags challenging each Buy signal — HIGH = thesis-breaking risk"),
            unsafe_allow_html=True,
        )
        from advisor.risk import RiskEngine as _REAT
        _re_at = _REAT()
        at_rows = ""
        for _, row in top10.iterrows():
            t    = row["ticker"]
            data = universe_data.get(t, {})
            if not data:
                continue
            r_t    = risk.get(t, {})
            flags  = _re_at.anti_thesis(t, data.get("info", {}), data.get("history"), r_t, data)
            high_c = sum(1 for f in flags if f["severity"] == "HIGH")
            med_c  = sum(1 for f in flags if f["severity"] == "MEDIUM")
            top3   = " · ".join(f['flag'] for f in flags[:3]) if flags else "No red flags"
            flag_clr = RED if high_c > 0 else AMBER if med_c > 0 else GREEN
            at_rows += (
                f'<tr>'
                f'<td><b>{t}</b></td>'
                f'<td style="font-weight:700;color:{RED}">{high_c}</td>'
                f'<td style="font-weight:700;color:{AMBER}">{med_c}</td>'
                f'<td style="font-size:12px;color:{MUTED}">{top3}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<table class="qt"><thead><tr>'
            f'<th>Ticker</th><th style="color:{RED}">HIGH flags</th>'
            f'<th style="color:{AMBER}">MEDIUM flags</th><th>Top flags</th>'
            f'</tr></thead><tbody>{at_rows}</tbody></table>',
            unsafe_allow_html=True,
        )

    # ── Tail-Risk Stress Testing ──────────────────────────────────────────────
    if universe_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            shdr("Tail-Risk Stress Testing",
                 "CVaR = expected loss in worst 5% · Rate shock = +200bps beta-scaled · Recession = worst 2% hist"),
            unsafe_allow_html=True,
        )
        from advisor.risk import RiskEngine as _RETail
        tail = _RETail().portfolio_tail_risk(top10["ticker"].tolist(), universe_data)

        # Portfolio-level CVaR vs VaR tiles
        p_cvar = tail.get("portfolio_cvar")
        p_var  = tail.get("portfolio_var")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(
                mtile("Portfolio CVaR 95%",
                      f"{p_cvar:.1f}%" if p_cvar is not None else "n/a",
                      "Expected loss in worst 5% of months", RED, RED),
                unsafe_allow_html=True,
            )
        with tc2:
            st.markdown(
                mtile("Portfolio VaR 95%",
                      f"{p_var:.1f}%" if p_var is not None else "n/a",
                      "Threshold loss at 95% confidence", AMBER, AMBER),
                unsafe_allow_html=True,
            )
        with tc3:
            tail_ratio = None
            if p_cvar is not None and p_var is not None and p_var != 0:
                tail_ratio = round(abs(p_cvar) / abs(p_var), 2)
            st.markdown(
                mtile("CVaR / VaR Ratio",
                      f"{tail_ratio:.2f}×" if tail_ratio is not None else "n/a",
                      "Tail severity multiplier (>1.3 = fat tails)", MUTED, BLUE),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Stress scenario table
        stress = tail.get("stress_scenarios", {})
        worst_avg = tail.get("worst_day_avg", {})
        s_rows = ""
        for _, row in top10.iterrows():
            t   = row["ticker"]
            sc  = stress.get(t, {})
            rs  = sc.get("rate_shock_pct")
            rec = sc.get("recession_pct")
            liq = sc.get("liquidity_pct")
            bt  = sc.get("beta")
            wd  = worst_avg.get(t)
            cvar_t = risk.get(t, {}).get("cvar_95_pct")
            def _c(v): return RED if v is not None and v < -15 else AMBER if v is not None and v < -8 else MUTED
            s_rows += (
                f'<tr>'
                f'<td><b>{t}</b></td>'
                f'<td style="font-family:monospace;color:{MUTED}">{bt if bt is not None else "—"}</td>'
                f'<td style="font-family:monospace;color:{RED if cvar_t and cvar_t < -10 else AMBER}">'
                f'{fmt_pct(cvar_t, False)}</td>'
                f'<td style="font-family:monospace;color:{_c(rs)}">{fmt_pct(rs, False) if rs is not None else "—"}</td>'
                f'<td style="font-family:monospace;color:{_c(rec)}">{fmt_pct(rec, False) if rec is not None else "—"}</td>'
                f'<td style="font-family:monospace;color:{_c(liq)}">{fmt_pct(liq, False) if liq is not None else "—"}</td>'
                f'<td style="font-family:monospace;color:{RED if wd and wd < -2 else MUTED}">'
                f'{f"{wd:+.2f}%" if wd is not None else "—"}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<table class="qt"><thead><tr>'
            f'<th>Ticker</th><th>Beta</th><th>CVaR 95%</th>'
            f'<th>Rate Shock</th><th>Recession</th><th>Liq. Crunch</th><th>Worst-Day Avg</th>'
            f'</tr></thead><tbody>{s_rows}</tbody></table>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:11px;color:{MUTED2};margin-top:8px">'
            f'Rate Shock = +200bps, beta-scaled. Recession = worst 2% historical monthly return. '
            f'Liq. Crunch = VaR × 1.5 stress multiplier. Worst-Day Avg = avg return on portfolio\'s worst 10% days.'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Per-stock CVaR bar chart
        tickers_list = [row["ticker"] for _, row in top10.iterrows()]
        cvar_vals    = [risk.get(t, {}).get("cvar_95_pct") for t in tickers_list]
        var_vals     = [risk.get(t, {}).get("var_95_pct")  for t in tickers_list]
        if any(v is not None for v in cvar_vals):
            st.markdown("<br>", unsafe_allow_html=True)
            fig_tail = go.Figure()
            fig_tail.add_trace(go.Bar(
                name="VaR 95%", x=tickers_list,
                y=[v if v is not None else 0 for v in var_vals],
                marker_color=AMBER, opacity=0.75,
                hovertemplate="<b>%{x}</b><br>VaR 95%: %{y:.1f}%<extra></extra>",
            ))
            fig_tail.add_trace(go.Bar(
                name="CVaR 95%", x=tickers_list,
                y=[v if v is not None else 0 for v in cvar_vals],
                marker_color=RED, opacity=0.85,
                hovertemplate="<b>%{x}</b><br>CVaR 95%: %{y:.1f}%<extra></extra>",
            ))
            fig_tail.update_layout(
                **_plotly_base(), barmode="group", height=300,
                margin=dict(l=0, r=0, t=6, b=40),
                legend=dict(orientation="h", y=1.08, font=dict(size=11)),
                yaxis=dict(title="Monthly Loss (%)", gridcolor="#1E2535"),
                xaxis=dict(tickfont=dict(size=11, family="monospace")),
            )
            st.plotly_chart(fig_tail, use_container_width=True)


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
        [0.00, "#450A0A"], [0.35, "#451A03"],
        [0.60, "#064E3B"], [1.00, "#10B981"],
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

        # Gate 4 disagreement badge — shown when ValuationEngine and traditional
        # metrics diverge by more than 30 points (signals conflicting evidence)
        g4d = p.get("gate4_detail", {})
        disagree_badge = (
            f' <span style="background:{AMBER};color:#000;font-size:10px;'
            f'font-weight:700;padding:1px 5px;border-radius:3px">SPLIT</span>'
            if g4d.get("disagreement") else ""
        )

        rows += (
            f'<tr>'
            f'<td><b>{t}</b>{disagree_badge}</td>'
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
                f'<div style="background:#1E2535;border-radius:3px;height:5px;width:100%">'
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
                       gridcolor="#1E2535"),
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
                    "  ·  ".join(macro.get("regime_reasons", []))[:40], rc, rc),
            unsafe_allow_html=True,
        )

        # ── Regime signal breakdown ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(shdr("Regime Signal Breakdown"), unsafe_allow_html=True)
        _vix_v  = macro.get("vix") or 0
        _y10    = macro.get("yield_10y") or 0
        _y3m    = macro.get("yield_3m") or 0
        _spread = round(_y10 - _y3m, 2) if _y10 and _y3m else None
        _hy     = macro.get("hy_spread")
        _rec    = macro.get("recession_prob")
        _cs     = macro.get("consumer_sentiment")
        _regime_rows = [
            ("VIX",               f"{_vix_v:.1f}" if _vix_v else "N/A",  "&gt;25 = risk-off",    RED if _vix_v > 25 else AMBER if _vix_v > 18 else GREEN),
            ("10Y Yield",         f"{_y10:.2f}%"  if _y10  else "N/A",   "&gt;5% = restrictive", RED if _y10 > 5 else AMBER if _y10 > 4 else GREEN),
            ("Yield Spread",      f"{_spread:+.2f}%" if _spread is not None else "N/A", "&lt;0 = inverted", RED if (_spread or 0) < 0 else GREEN),
            ("HY Spread",         f"{_hy:.2f}%"   if _hy   else "N/A",   "&gt;5% = stress",      RED if (_hy or 0) > 5 else AMBER if (_hy or 0) > 3.5 else GREEN),
            ("Recession Prob",    f"{_rec:.1f}%"  if _rec  else "N/A",   "&gt;20% = caution",    RED if (_rec or 0) > 20 else AMBER if (_rec or 0) > 10 else GREEN),
            ("Consumer Sentiment",f"{_cs:.1f}"    if _cs   else "N/A",   "&lt;70 = weak demand", RED if (_cs or 100) < 70 else AMBER if (_cs or 100) < 80 else GREEN),
        ]
        _rg_html = '<table style="width:100%;border-collapse:collapse;font-size:12px">'
        _rg_html += '<tr><th style="text-align:left;padding:3px 6px;color:#6B7280">Signal</th><th style="text-align:right;padding:3px 6px;color:#6B7280">Value</th><th style="text-align:left;padding:3px 6px;color:#6B7280">Threshold</th></tr>'
        for _label, _val, _thresh, _col in _regime_rows:
            _dot = f'<span style="color:{_col};font-size:14px">&#9679;</span>'
            _rg_html += f'<tr><td style="padding:3px 6px">{_dot} {_label}</td><td style="text-align:right;font-family:monospace;font-weight:600;padding:3px 6px">{_val}</td><td style="padding:3px 6px;color:#6B7280;font-size:11px">{_thresh}</td></tr>'
        _rg_html += '</table>'
        st.markdown(_rg_html, unsafe_allow_html=True)

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
                xaxis=dict(zeroline=True, zerolinecolor="#1E2535", gridcolor="#1E2535"),
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
            xaxis=dict(title="Date", gridcolor="#1E2535"),
            yaxis=dict(title="Normalised (Base = 100)", gridcolor="#1E2535"),
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
            corr_mat = ret_df.corr(method="spearman")
            tks      = corr_mat.columns.tolist()
            txt_m    = [[f"{corr_mat.iloc[r, c]:.2f}" for c in range(len(tks))]
                        for r in range(len(tks))]
            fig_corr = go.Figure(go.Heatmap(
                z=corr_mat.values, x=tks, y=tks,
                text=txt_m, texttemplate="%{text}",
                colorscale=[[0, "#0F1520"], [0.5, "#1E3A5F"], [1, "#3B82F6"]],
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

        # ── Yield Curve ────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(shdr("Yield Curve", "3M · 2Y · 10Y  ·  Red fill = inverted curve"),
                    unsafe_allow_html=True)
        st.plotly_chart(_yield_curve_chart(macro), use_container_width=True)


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
                     gridcolor="#1E2535", tickformat="$,.2f")
    fig.update_yaxes(title_text="Volume",   row=2, col=1,
                     gridcolor="#1E2535", showticklabels=False)
    fig.update_yaxes(title_text="RSI",      row=3, col=1,
                     gridcolor="#1E2535", range=[0, 100])
    fig.update_xaxes(gridcolor="#1E2535", row=3, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# QUANT CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _factor_radar_chart(top10: pd.DataFrame) -> go.Figure:
    """Spider/radar chart showing each pick's 7-factor fingerprint."""
    factor_cols = [f"{f}_score" for f in FACTOR_NAMES]
    factor_labels = [f.capitalize() for f in FACTOR_NAMES]
    neon = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#F97316"]
    fig = go.Figure()
    for i, (_, row) in enumerate(top10.head(5).iterrows()):
        vals = [float(row.get(c, 50) or 50) for c in factor_cols]
        vals_closed = vals + [vals[0]]
        labels_closed = factor_labels + [factor_labels[0]]
        color = neon[i % len(neon)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            name=row["ticker"],
            line=dict(color=color, width=2),
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
            opacity=0.85,
            hovertemplate="<b>" + row["ticker"] + "</b> — %{theta}<br>Score: %{r:.1f}<extra></extra>",
        ))
    fig.update_layout(
        **_plotly_base(),
        height=420,
        polar=dict(
            bgcolor="#0F1520",
            radialaxis=dict(visible=True, range=[0, 100], color="#64748B",
                            gridcolor="#1E2535", tickfont=dict(size=9, color="#64748B")),
            angularaxis=dict(color="#94A3B8", gridcolor="#1E2535",
                             tickfont=dict(size=11, color="#94A3B8")),
        ),
        legend=dict(orientation="v", x=1.05, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=120, t=40, b=40),
    )
    return fig


def _score_distribution_chart(ranked_df: pd.DataFrame, top10: pd.DataFrame) -> go.Figure:
    """Histogram of composite scores across all ranked stocks; marks portfolio picks."""
    if "composite_score" not in ranked_df.columns:
        return go.Figure()
    all_scores = ranked_df["composite_score"].dropna().tolist()
    top_scores = top10["composite_score"].dropna().tolist() if "composite_score" in top10.columns else []
    top_tickers = top10["ticker"].tolist() if not top10.empty else []

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=all_scores, nbinsx=40, name="All Stocks",
        marker=dict(color="#1E3A5F", line=dict(color="#3B82F6", width=0.5)),
        opacity=0.8,
        hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    for sc, tk in zip(top_scores, top_tickers):
        fig.add_vline(
            x=sc, line_dash="dash", line_color="#10B981", line_width=1.5,
            annotation=dict(text=tk, font=dict(size=9, color="#10B981"),
                            bgcolor="#064E3B", borderpad=2),
            annotation_position="top",
        )
    fig.update_layout(
        **_plotly_base(),
        height=300,
        bargap=0.05,
        xaxis=dict(title="Composite Score (0–100)", range=[0, 100], gridcolor="#1E2535"),
        yaxis=dict(title="# Stocks", gridcolor="#1E2535"),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=40),
    )
    return fig


def _monte_carlo_chart(top10: pd.DataFrame, universe_data: dict, n_sims: int = 200) -> go.Figure:
    """200-path Monte Carlo simulation of equal-weighted portfolio over 252 days."""
    np.random.seed(42)
    daily_stats = []
    for _, row in top10.iterrows():
        t = row["ticker"]
        if t not in universe_data:
            continue
        hist = universe_data[t].get("history", pd.DataFrame())
        if hist.empty or "Close" not in hist.columns:
            continue
        close = DataFetcher.strip_tz(hist["Close"].dropna())
        if len(close) < 30:
            continue
        ret = close.pct_change().dropna()
        daily_stats.append((float(ret.mean()), float(ret.std())))
    if not daily_stats:
        return go.Figure()

    days = 252
    paths = np.ones((n_sims, days + 1))
    for mu, sigma in daily_stats:
        draws = np.random.normal(mu, sigma, (n_sims, days))
        paths[:, 1:] *= (1 + draws)
    paths = paths ** (1 / len(daily_stats))  # geometric mean per stock
    x = list(range(days + 1))

    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()

    # Dim paths
    for i in range(min(n_sims, 80)):
        fig.add_trace(go.Scatter(
            x=x, y=paths[i], mode="lines",
            line=dict(color="#1E2535", width=0.5), showlegend=False, opacity=0.4,
            hoverinfo="skip",
        ))

    band_colors = ["#064E3B", "#065F46", "#059669", "#047857", "#064E3B"]
    band_labels = ["P5–P25 (Worst Quartile)", None, None, "P75–P95 (Best Quartile)", None]
    for lo, hi, clr, lbl in [
        (0, 1, "#450A0A", "P5 (Worst 5%)"),
        (1, 2, "#064E3B", "P25–P75 (Median Range)"),
        (3, 4, "#1E3A5F", "P95 (Best 5%)"),
    ]:
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(pcts[hi]) + list(pcts[lo])[::-1],
            fill="toself", mode="none",
            fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.33)", name=lbl,
            hoverinfo="skip",
        ))

    # Median
    fig.add_trace(go.Scatter(
        x=x, y=pcts[2], mode="lines", name="Median",
        line=dict(color="#10B981", width=2.5),
        hovertemplate="Day %{x}<br>Portfolio: %{y:.2f}×<extra></extra>",
    ))
    # Breakeven
    fig.add_hline(y=1.0, line_dash="dot", line_color="#EF4444", line_width=1,
                  annotation_text="Breakeven", annotation_font=dict(color="#EF4444", size=10))

    # Annotations
    med_final = float(pcts[2, -1])
    p5_final  = float(pcts[0, -1])
    p95_final = float(pcts[4, -1])
    fig.add_annotation(x=days, y=med_final, text=f"Median: {med_final:.2f}×",
                       showarrow=False, xanchor="left", font=dict(color="#10B981", size=10))
    fig.add_annotation(x=days, y=p5_final, text=f"P5: {p5_final:.2f}×",
                       showarrow=False, xanchor="left", font=dict(color="#EF4444", size=9))
    fig.add_annotation(x=days, y=p95_final, text=f"P95: {p95_final:.2f}×",
                       showarrow=False, xanchor="left", font=dict(color="#3B82F6", size=9))

    fig.update_layout(
        **_plotly_base(),
        height=420,
        xaxis=dict(title="Trading Days", gridcolor="#1E2535"),
        yaxis=dict(title="Portfolio Multiplier (1.0 = start)", gridcolor="#1E2535"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=80, t=50, b=40),
    )
    return fig


def _dcf_waterfall_chart(top10: pd.DataFrame, valuation: dict) -> go.Figure:
    """Bear / Base / Bull DCF fair value grouped bar chart with current price overlay."""
    tickers = top10["ticker"].tolist()
    scenarios = ["Bear", "Base", "Bull"]
    scenario_colors = [RED, AMBER, GREEN]

    fig = go.Figure()
    for sname, scolor in zip(scenarios, scenario_colors):
        fvs = []
        for t in tickers:
            sens = valuation.get(t, {}).get("sensitivity", {})
            sv = sens.get(sname, {})
            fvs.append(sv.get("fair_value"))
        fig.add_trace(go.Bar(
            x=tickers, y=fvs, name=sname,
            marker_color=scolor, opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b> — {sname}<br>FV: $%{{y:,.2f}}<extra></extra>",
        ))

    # Current price lines per ticker
    for i, t in enumerate(tickers):
        cur = valuation.get(t, {}).get("current_price")
        if cur:
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=i - 0.4, x1=i + 0.4, y0=cur, y1=cur,
                line=dict(color="#E2E8F0", width=2, dash="dot"),
            )

    fig.update_layout(
        **_plotly_base(),
        barmode="group",
        height=380,
        xaxis=dict(tickfont=dict(size=11, family="monospace"), gridcolor="#1E2535"),
        yaxis=dict(title="Fair Value ($)", tickformat="$,.0f", gridcolor="#1E2535"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def _yield_curve_chart(macro: dict) -> go.Figure:
    """Yield curve visualization using available macro data."""
    y3m  = macro.get("yield_3m")
    y10  = macro.get("yield_10y")
    spread = macro.get("yield_10_2_spread")

    # Build best-effort curve from available fields
    maturities = ["3M", "10Y"]
    yields_raw = [y3m, y10]
    labels = ["3M", "10Y"]

    # Estimate 2Y from spread (if available)
    y2 = None
    if y10 is not None and spread is not None:
        y2 = y10 - spread

    if y2 is not None:
        maturities = ["3M", "2Y", "10Y"]
        yields_raw = [y3m, y2, y10]
        labels     = ["3M", "2Y", "10Y"]

    maturities_clean = [m for m, y in zip(maturities, yields_raw) if y is not None]
    yields_clean     = [y for y in yields_raw if y is not None]

    if len(yields_clean) < 2:
        fig = go.Figure()
        fig.update_layout(**_plotly_base(), height=260,
                          title=dict(text="Yield Curve (insufficient data)", font=dict(size=13, color="#94A3B8")))
        return fig

    inverted = yields_clean[-1] < yields_clean[0]
    curve_color = RED if inverted else GREEN
    fill_color  = "rgba(239,68,68,0.10)" if inverted else "rgba(16,185,129,0.10)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities_clean, y=yields_clean,
        mode="lines+markers",
        name="Yield Curve",
        line=dict(color=curve_color, width=2.5),
        fill="tozeroy",
        fillcolor=fill_color,
        marker=dict(size=8, color=curve_color),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    if inverted:
        fig.add_annotation(
            x=maturities_clean[-1], y=yields_clean[-1],
            text="⚠ INVERTED",
            showarrow=False, xanchor="left",
            font=dict(color=RED, size=12, family="JetBrains Mono"),
            bgcolor="#450A0A", borderpad=4,
        )
    fig.update_layout(
        **_plotly_base(),
        height=280,
        xaxis=dict(title="Maturity", gridcolor="#1E2535"),
        yaxis=dict(title="Yield (%)", gridcolor="#1E2535", tickformat=".2f"),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=40),
    )
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
        f'<div style="background:linear-gradient(135deg,{GRAY_LT} 0%,#0F1520 100%);'
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
            earn_c, earn_bg = RED, RED_LT
            earn_icon = "⚠"
            earn_msg = f"Earnings in {days_away} days ({edate}) — high event risk, consider waiting for results"
        elif days_away <= 14:
            earn_c, earn_bg = AMBER, AMBER_LT
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
            f'<div style="background:#0A1525;border:1px solid #1E3A5F;border-left:4px solid {sig_accent};'
            f'border-radius:10px;padding:16px 20px;margin-bottom:18px">'
            f'<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
            f'letter-spacing:.10em;color:{BLUE};margin-bottom:8px">Quant Thesis</div>'
            f'<div style="font-size:13.5px;line-height:1.8;color:{TEXT}">{thesis_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── ANTI-THESIS ───────────────────────────────────────────────────────────
    from advisor.risk import RiskEngine as _RiskEngineAT
    _at_flags = _RiskEngineAT().anti_thesis(ticker, info, hist, r, data)
    if _at_flags:
        _high_count = sum(1 for f in _at_flags if f["severity"] == "HIGH")
        _expander_label = (
            f"Anti-Thesis — {len(_at_flags)} risk flag{'s' if len(_at_flags) != 1 else ''} "
            f"({_high_count} HIGH)" if _high_count else
            f"Anti-Thesis — {len(_at_flags)} risk flag{'s' if len(_at_flags) != 1 else ''}"
        )
        with st.expander(_expander_label, expanded=_high_count > 0):
            st.markdown(
                f'<div style="background:#0A0D14;border:1px solid {RED}33;border-left:4px solid {RED};'
                f'border-radius:10px;padding:14px 18px 6px;margin-bottom:4px">'
                f'<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
                f'letter-spacing:.10em;color:{RED};margin-bottom:10px">Bear Case / Risk Flags</div>'
                f'{_anti_thesis_html(_at_flags)}'
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
        f'<div style="background:#0F1520;border:1px solid {BORDER};border-radius:8px;'
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
            row_bg = "#1A1500" if sname == "Base" else "#0A0D14"
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

    # ── REVERSE DCF ───────────────────────────────────────────────────────────
    _rdcf = val.get("reverse_dcf")
    if _rdcf:
        _ig   = _rdcf.get("implied_growth")
        _rg   = _rdcf.get("realistic_growth")
        _gap  = _rdcf.get("gap_pct")
        _verd = _rdcf.get("verdict", "")
        _rdcf_clr = {
            "OVERPRICED":        RED,
            "STRETCHED":         AMBER,
            "FAIR":              MUTED,
            "ATTRACTIVE":        GREEN,
            "DEEPLY_UNDERVALUED": "#10B981",
        }.get(_verd, MUTED)
        st.markdown(
            f'<div style="background:#0A0D14;border:1px solid {_rdcf_clr}44;'
            f'border-left:4px solid {_rdcf_clr};border-radius:10px;'
            f'padding:14px 20px;margin-bottom:18px">'
            f'<div style="font-size:10px;font-weight:800;text-transform:uppercase;'
            f'letter-spacing:.10em;color:{_rdcf_clr};margin-bottom:10px">Reverse DCF — Implied Growth Analysis</div>'
            f'<div style="display:flex;gap:32px;flex-wrap:wrap">'
            f'  <div><div style="font-size:11px;color:{MUTED2}">Market-Implied Growth</div>'
            f'     <div style="font-size:22px;font-weight:800;color:{_rdcf_clr}">'
            f'     {f"{_ig:+.1f}%" if _ig is not None else "n/a"}</div></div>'
            f'  <div><div style="font-size:11px;color:{MUTED2}">Realistic / Analyst Est.</div>'
            f'     <div style="font-size:22px;font-weight:800;color:{TEXT}">'
            f'     {f"{_rg:+.1f}%" if _rg is not None else "n/a"}</div></div>'
            f'  <div><div style="font-size:11px;color:{MUTED2}">Gap (Implied − Realistic)</div>'
            f'     <div style="font-size:22px;font-weight:800;color:{_rdcf_clr}">'
            f'     {f"{_gap:+.1f}%" if _gap is not None else "n/a"}</div></div>'
            f'  <div><div style="font-size:11px;color:{MUTED2}">Verdict</div>'
            f'     <div style="font-size:14px;font-weight:800;color:{_rdcf_clr};'
            f'     padding:4px 10px;background:{_rdcf_clr}22;border-radius:4px;margin-top:6px">'
            f'     {_verd.replace("_"," ")}</div></div>'
            f'</div>'
            f'<div style="font-size:11px;color:{MUTED2};margin-top:10px">'
            f'The market is pricing in <b style="color:{_rdcf_clr}">{f"{_ig:+.1f}%" if _ig is not None else "n/a"}</b> annual FCF growth. '
            f'If the company actually grows at <b>{f"{_rg:+.1f}%" if _rg is not None else "n/a"}</b>, '
            f'{"the stock is overvalued by the implied growth gap." if (_gap or 0) > 0 else "the stock is undervalued — growth is priced in cheaply."}'
            f'</div>'
            f'</div>',
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
        sc_bg    = GREEN_LT if score >= 60 else RED_LT if score < 40 else AMBER_LT
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
            bg_col = GREEN_LT if status == "pass" else AMBER_LT if status == "warn" else RED_LT
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
def _normalize_yf(df) -> "pd.DataFrame":
    """
    yfinance ≥0.2.31 returns MultiIndex columns from yf.download()
    e.g. ('Close', 'AAPL').  Flatten to single-level so code using
    df['Close'] keeps working.  No-op if already single-level.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the ticker level — keep only the field name
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        # If duplicate column names remain (multi-ticker download), de-dup
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _run_backtest_simulation(hist: "pd.DataFrame", entry_low: float, target: float, stop_loss: float) -> list:
    """
    Valuation-based backtest using the system's own DCF/fair-value levels.

    Entry  : price ≤ entry_low  (FV × 0.80 — 20% margin of safety)
    Exit   : price ≥ target     (FV × 1.20 — take profit at 20% above FV)
          OR price ≤ stop_loss  (entry_low × 0.92 — 8% hard stop below entry)

    NOTE: Uses today's fair value as a static reference applied to historical
    prices. This is illustrative — it shows how the current valuation level
    has anchored the stock's price action over the selected period.
    """
    hist   = _normalize_yf(hist)
    closes = hist["Close"].squeeze().dropna()
    if len(closes) < 10:
        return []

    trades      = []
    in_trade    = False
    entry_price = None
    entry_date  = None

    for i, date in enumerate(closes.index):
        price = float(closes.iloc[i])

        if not in_trade:
            if price <= entry_low:
                in_trade    = True
                entry_price = price
                entry_date  = date
        else:
            ret = (price - entry_price) / entry_price * 100

            exit_reason = None
            if price >= target:
                exit_reason = "Target hit"
            elif price <= stop_loss:
                exit_reason = "Stop loss"

            if exit_reason:
                trades.append({
                    "entry_date": str(entry_date)[:10],
                    "entry":      round(entry_price, 2),
                    "exit_date":  str(date)[:10],
                    "exit":       round(price, 2),
                    "return_pct": round(ret, 2),
                    "reason":     exit_reason,
                    "won":        ret > 0,
                })
                in_trade = False

    # Mark-to-market any open position
    if in_trade:
        last_price = float(closes.iloc[-1])
        last_date  = closes.index[-1]
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


def tab_backtest(top10, universe_data: dict, valuation: dict, risk: dict, rf_rate: float):
    """Portfolio-wide backtest: honest technical entry strategy vs S&P 500."""
    import yfinance as yf

    st.markdown(
        shdr("Portfolio Backtest", "Valuation-based entry strategy — buy the margin of safety, sell at target"),
        unsafe_allow_html=True,
    )

    # ── Methodology + bias warnings ───────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{AMBER_LT};border-left:4px solid {AMBER};border-radius:8px;
                padding:14px 18px;margin-bottom:18px;font-size:13px;color:{TEXT}">
      <b>📊 How this backtest works</b><br><br>
      <b>Entry signal:</b> Price drops to or below <b>Entry Low</b> (Fair Value × 0.80) — the 20% margin
      of safety level calculated by the DCF/Graham/EV engine.<br>
      <b>Exit:</b> Price reaches <b>Target</b> (Fair Value × 1.20) — take profit · OR price falls to
      <b>Stop Loss</b> (Entry Low × 0.92) — 8% hard stop below entry.<br><br>
      <b>⚠️ Known limitations (be honest with yourself):</b><br>
      • <b>Static fair value</b> — uses <i>today's</i> DCF fair value applied to historical prices.
        Actual fundamentals changed over the period, so treat results as illustrative of how the
        valuation level has anchored price action, not a true historical simulation.<br>
      • <b>Survivorship bias</b> — only tests stocks that still exist today. Delistings are excluded,
        so results look better than real-world.<br>
      • <b>Slippage &amp; liquidity</b> — assumes perfect execution at close prices. Real fills are worse.<br>
      • <b>Past ≠ future</b> — a strategy that worked historically can still fail going forward.
    </div>
    """, unsafe_allow_html=True)

    if top10 is None or (hasattr(top10, "__len__") and len(top10) == 0):
        st.markdown(
            f'<div style="text-align:center;padding:80px 20px;color:{MUTED2};font-size:15px">'
            f'Run the analysis first — the backtest will simulate <b>all 10 picks</b> automatically.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Period selector + Run button ─────────────────────────────────────────
    col_p, col_run = st.columns([2, 1])
    with col_p:
        bt_period = st.selectbox(
            "Backtest Period", ["1y", "2y", "3y", "5y"], index=1,
            help="How far back to simulate the strategy",
        )
    with col_run:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("Run Portfolio Backtest", type="primary", use_container_width=True)

    if "bt_result" not in st.session_state:
        st.session_state.bt_result = None
        st.session_state.bt_period = None

    if go_btn:
        st.session_state.bt_period = bt_period
        st.session_state.bt_errors = {}
        tickers = list(top10["ticker"]) if hasattr(top10, "iterrows") else list(top10)

        prog = st.progress(0, text="Fetching S&P 500 benchmark…")

        # Fetch S&P 500
        try:
            sp_hist   = _normalize_yf(yf.download("^GSPC", period=bt_period, progress=False, auto_adjust=True))
            sp_closes = sp_hist["Close"].squeeze()
            if hasattr(sp_closes.index, "tz") and sp_closes.index.tz is not None:
                sp_closes.index = sp_closes.index.tz_localize(None)
            sp500_total_ret = (float(sp_closes.iloc[-1]) / float(sp_closes.iloc[0]) - 1) * 100
            sp_equity = (sp_closes / float(sp_closes.iloc[0]) * 100).reset_index()
            sp_equity.columns = ["Date", "Equity"]
        except Exception:
            sp_hist = None; sp500_total_ret = None; sp_equity = None

        stock_results = {}
        equity_curves = {}   # ticker -> pd.Series (indexed by date, value = equity %)

        for i, t in enumerate(tickers):
            prog.progress((i + 1) / len(tickers), text=f"Backtesting {t}…")
            try:
                # Get historical data
                d = universe_data.get(t, {})
                if d.get("history") is not None and not d["history"].empty and bt_period == "1y":
                    # Use cached 1-year history directly (already single-level columns)
                    hist_raw = d["history"]
                else:
                    # Download extended history; normalize MultiIndex columns
                    hist_raw = _normalize_yf(
                        yf.download(t, period=bt_period, progress=False, auto_adjust=True)
                    )

                if hist_raw is None or hist_raw.empty:
                    continue

                hist_plot = _normalize_yf(hist_raw.copy())
                if hasattr(hist_plot.index, "tz") and hist_plot.index.tz is not None:
                    hist_plot.index = hist_plot.index.tz_localize(None)
                closes = hist_plot["Close"].squeeze().dropna()

                # Valuation levels
                val_r = valuation.get(t, {})
                entry_low = val_r.get("entry_low")
                target    = val_r.get("target_price")
                stop_loss = val_r.get("stop_loss")
                fair_val  = val_r.get("fair_value")

                bah_ret = (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100

                if entry_low and target and stop_loss:
                    trades = _run_backtest_simulation(hist_raw, entry_low, target, stop_loss)
                    closed = [tr for tr in trades if tr["reason"] != "Open position"]
                    open_p = next((tr for tr in trades if tr["reason"] == "Open position"), None)
                    n_wins = sum(1 for tr in trades if tr["won"])
                    compound = 1.0
                    for tr in closed:
                        compound *= (1 + tr["return_pct"] / 100)
                    strat_ret = (compound - 1) * 100 if closed else (open_p["return_pct"] if open_p else 0.0)
                    win_rate = (n_wins / len(trades) * 100) if trades else 0

                    # Build strategy equity curve (cash=100, positions track price)
                    eq_cash = 100.0
                    eq_in_trade = False
                    eq_entry_p = None
                    eq_vals = []
                    for _p in closes:
                        _p = float(_p)
                        if not eq_in_trade:
                            if _p <= entry_low:
                                eq_in_trade = True
                                eq_entry_p = _p
                            eq_vals.append(eq_cash)
                        else:
                            cur_eq = eq_cash * (_p / eq_entry_p)
                            eq_vals.append(cur_eq)
                            if _p >= target or _p <= stop_loss:
                                eq_cash = cur_eq
                                eq_in_trade = False
                    equity_curves[t] = pd.Series(eq_vals, index=closes.index)
                else:
                    trades = []; strat_ret = 0.0; win_rate = 0
                    entry_low = target = stop_loss = fair_val = None
                    # No valuation data — fall back to buy-and-hold curve
                    equity_curves[t] = (closes / float(closes.iloc[0]) * 100)

                stock_results[t] = {
                    "ticker":    t,
                    "trades":    trades,
                    "strat_ret": strat_ret,
                    "bah_ret":   bah_ret,
                    "win_rate":  win_rate,
                    "n_trades":  len(trades),
                    "entry_low": entry_low,
                    "target":    target,
                    "stop_loss": stop_loss,
                    "fair_val":  fair_val,
                    "hist":      hist_raw,
                }
            except Exception as _bt_err:
                # Store error for debugging but keep going with remaining tickers
                if "bt_errors" not in st.session_state:
                    st.session_state.bt_errors = {}
                st.session_state.bt_errors[t] = str(_bt_err)
                continue

        prog.empty()

        if not stock_results:
            err_detail = ""
            if st.session_state.get("bt_errors"):
                sample = list(st.session_state.bt_errors.items())[:3]
                err_detail = " Errors: " + "; ".join(f"{k}: {v}" for k, v in sample)
            st.session_state.bt_result = {"error": f"Could not fetch price data for any stock.{err_detail}"}
        else:
            # Portfolio aggregate: equal-weight strategy returns
            strat_rets  = [v["strat_ret"]  for v in stock_results.values()]
            bah_rets    = [v["bah_ret"]    for v in stock_results.values()]
            port_return = sum(strat_rets) / len(strat_rets)
            port_bah    = sum(bah_rets)   / len(bah_rets)
            total_trades = sum(v["n_trades"] for v in stock_results.values())
            total_wins   = sum(
                sum(1 for tr in v["trades"] if tr["won"]) for v in stock_results.values()
            )
            port_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

            # Alpha
            alpha = port_return - (sp500_total_ret or port_bah)

            st.session_state.bt_result = {
                "stock_results":  stock_results,
                "equity_curves":  equity_curves,
                "port_return":    port_return,
                "port_bah":       port_bah,
                "port_win_rate":  port_win_rate,
                "total_trades":   total_trades,
                "total_wins":     total_wins,
                "alpha":          alpha,
                "sp500_ret":      sp500_total_ret,
                "sp_equity":      sp_equity,
                "period":         bt_period,
            }

    # ── Results ───────────────────────────────────────────────────────────────
    bt = st.session_state.bt_result
    if not bt:
        st.markdown(
            f'<div style="text-align:center;padding:60px 20px;color:{MUTED2};font-size:14px">'
            f'Select a period and click <b>Run Portfolio Backtest</b> to simulate all 10 picks.</div>',
            unsafe_allow_html=True,
        )
        return

    if bt.get("error"):
        st.error(bt["error"])
        return

    port_return   = bt["port_return"]
    port_bah      = bt["port_bah"]
    port_win_rate = bt["port_win_rate"]
    total_trades  = bt["total_trades"]
    total_wins    = bt["total_wins"]
    alpha         = bt["alpha"]
    sp500_ret     = bt.get("sp500_ret")
    equity_curves = bt["equity_curves"]
    sp_equity     = bt.get("sp_equity")
    stock_results = bt["stock_results"]
    period        = bt.get("period", bt_period)

    # ── Summary tiles ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

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

    cols_m = st.columns(5)
    ret_c  = GREEN if port_return > 0 else RED
    wr_c   = GREEN if port_win_rate >= 60 else RED if port_win_rate < 40 else AMBER
    bah_c  = GREEN if port_bah > 0 else RED
    sp_c   = GREEN if (sp500_ret or 0) > 0 else RED
    al_c   = GREEN if alpha > 0 else RED
    sp_str = f"{sp500_ret:+.1f}%" if sp500_ret is not None else "n/a"

    with cols_m[0]: st.markdown(_bt_tile("Portfolio Return", f"{port_return:+.1f}%", f"equal-weight · {period}", ret_c), unsafe_allow_html=True)
    with cols_m[1]: st.markdown(_bt_tile("Win Rate", f"{port_win_rate:.0f}%", f"{total_wins}W / {total_trades-total_wins}L trades", wr_c), unsafe_allow_html=True)
    with cols_m[2]: st.markdown(_bt_tile("Buy & Hold", f"{port_bah:+.1f}%", "equal-weight same stocks", bah_c), unsafe_allow_html=True)
    with cols_m[3]: st.markdown(_bt_tile("S&P 500", sp_str, "benchmark same period", sp_c), unsafe_allow_html=True)
    with cols_m[4]: st.markdown(_bt_tile("Alpha vs S&P", f"{alpha:+.1f}%", "portfolio − S&P 500", al_c), unsafe_allow_html=True)

    # ── Portfolio Equity Curve vs S&P 500 ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(shdr("Portfolio Equity Curve vs S&P 500", "Strategy equity (entry at MoS, exit at target/stop) vs benchmark — base 100"), unsafe_allow_html=True)

    try:
        fig = go.Figure()

        # Individual stock equity curves (faint)
        _STOCK_COLORS = [
            "#3B82F6","#10B981","#F59E0B","#EC4899","#8B5CF6",
            "#06B6D4","#EF4444","#34D399","#FBBF24","#60A5FA",
        ]
        for idx, (t, eq) in enumerate(equity_curves.items()):
            fig.add_trace(go.Scatter(
                x=list(eq.index), y=list(eq.values),
                name=t, mode="lines",
                line=dict(color=_STOCK_COLORS[idx % len(_STOCK_COLORS)], width=1),
                opacity=0.45,
                hovertemplate=f"{t} %{{x}}<br>%{{y:.1f}}<extra></extra>",
            ))

        # Portfolio average (thick green)
        if equity_curves:
            all_eq = pd.concat(list(equity_curves.values()), axis=1)
            all_eq.columns = list(equity_curves.keys())
            all_eq = all_eq.ffill().dropna(how="all")
            port_eq = all_eq.mean(axis=1)
            fig.add_trace(go.Scatter(
                x=list(port_eq.index), y=list(port_eq.values),
                name="Portfolio (equal-weight)", mode="lines",
                line=dict(color=GREEN, width=3),
                hovertemplate="Portfolio %{x}<br>%{y:.1f}<extra></extra>",
            ))

        # S&P 500 (blue)
        if sp_equity is not None:
            fig.add_trace(go.Scatter(
                x=list(sp_equity["Date"]), y=list(sp_equity["Equity"]),
                name="S&P 500", mode="lines",
                line=dict(color=BLUE, width=2, dash="dot"),
                hovertemplate="S&P 500 %{x}<br>%{y:.1f}<extra></extra>",
            ))

        fig.add_hline(y=100, line_color=BORDER, line_dash="solid", line_width=1)
        fig.update_layout(
            **_plotly_base(),
            height=460,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                       title="Equity (base 100)", ticksuffix=""),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as _e:
        st.warning(f"Chart error: {_e}")

    # ── Per-stock breakdown table ──────────────────────────────────────────────
    st.markdown(shdr("Per-Stock Breakdown"), unsafe_allow_html=True)

    tbl_rows = ""
    for t, sr in sorted(stock_results.items(), key=lambda x: x[1]["strat_ret"], reverse=True):
        rc  = GREEN if sr["strat_ret"] > 0 else RED
        bc  = GREEN if sr["bah_ret"]   > 0 else RED
        wrc = GREEN if sr["win_rate"] >= 60 else RED if sr["win_rate"] < 40 else AMBER
        al2 = sr["strat_ret"] - (sp500_ret or sr["bah_ret"])
        alc = GREEN if al2 > 0 else RED
        tbl_rows += (
            f'<tr>'
            f'<td style="padding:9px 14px;font-weight:700">{t}</td>'
            f'<td style="font-family:monospace;color:{rc};font-weight:700;padding:9px 14px">{sr["strat_ret"]:+.1f}%</td>'
            f'<td style="font-family:monospace;color:{bc};padding:9px 14px">{sr["bah_ret"]:+.1f}%</td>'
            f'<td style="font-family:monospace;color:{alc};padding:9px 14px">{al2:+.1f}%</td>'
            f'<td style="font-family:monospace;color:{wrc};padding:9px 14px">{sr["win_rate"]:.0f}%</td>'
            f'<td style="font-family:monospace;padding:9px 14px">{sr["n_trades"]}</td>'
            f'<td style="font-family:monospace;padding:9px 14px">{fmt_price(sr["entry_low"])}</td>'
            f'<td style="font-family:monospace;padding:9px 14px">{fmt_price(sr["target"])}</td>'
            f'<td style="font-family:monospace;padding:9px 14px">{fmt_price(sr["stop_loss"])}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table class="qt"><thead><tr>'
        f'<th>Ticker</th><th>Strategy Ret</th><th>Buy&Hold</th><th>Alpha</th>'
        f'<th>Win Rate</th><th>Trades</th><th>Entry Zone</th><th>Target</th><th>Stop</th>'
        f'</tr></thead><tbody>{tbl_rows}</tbody></table>',
        unsafe_allow_html=True,
    )

    # ── Strategy rules note ───────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:20px;padding:14px 18px;background:{GRAY_LT};'
        f'border-radius:8px;border:1px solid {BORDER};font-size:12px;color:{MUTED}">'
        f'<b>Strategy rules:</b> Enter when price ≤ entry zone (fair value × 0.80, 20% margin of safety). '
        f'Take profit at target (fair value × 1.20). Cut loss at stop (entry × 0.92). '
        f'Portfolio is equal-weighted across all {len(stock_results)} picks. '
        f'Based on current fundamental valuation — historical fundamentals vary. Treat as illustrative.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 10 — EARNINGS CALENDAR
# ─────────────────────────────────────────────────────────────────────────────
def _safe_dict(v):
    """Return v if it's a dict, else {}. Guards against floats/None stored where dicts are expected."""
    return v if isinstance(v, dict) else {}


def _calendar_quant_analysis(t, val_r, risk_r, info, tech, days_to_earnings, score):
    """
    Generate Wall Street-style directional analysis from pure quant data.
    Returns a dict with recommendation, price targets, entry, risk rating, thesis.
    """
    # Normalize all inputs — any could be a float/None if yfinance data is incomplete
    val_r  = _safe_dict(val_r)
    risk_r = _safe_dict(risk_r)
    info   = _safe_dict(info)
    tech   = _safe_dict(tech)

    price  = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    fv     = val_r.get("fair_value")
    entry  = val_r.get("entry_low")
    target = val_r.get("target_price")
    stop   = val_r.get("stop_loss")
    signal = val_r.get("signal", "")
    premium= val_r.get("premium_pct") or (((price / fv - 1) * 100) if fv and price else 0)
    upside = val_r.get("upside_pct") or (((target / price - 1) * 100) if target and price else 0)
    rr     = val_r.get("rr_ratio")

    # ── Bear / Base / Bull targets from DCF sensitivity ──────────────────────
    sens    = _safe_dict(val_r.get("sensitivity"))
    bear_fv = _safe_dict(sens.get("Bear")).get("fair_value")
    base_fv = _safe_dict(sens.get("Base")).get("fair_value") or fv
    bull_fv = _safe_dict(sens.get("Bull")).get("fair_value")

    # If no sensitivity, derive from ±25% / ±50% of fair value
    if not bear_fv: bear_fv = fv * 0.75 if fv else (price * 0.80 if price else None)
    if not bull_fv: bull_fv = fv * 1.50 if fv else (price * 1.25 if price else None)

    bear_target = bear_fv * 0.80 if bear_fv else (price * 0.82 if price else None)
    base_target = target or (base_fv * 1.20 if base_fv else (price * 1.10 if price else None))
    bull_target = bull_fv * 1.20 if bull_fv else (price * 1.30 if price else None)

    bear_pct = ((bear_target / price - 1) * 100) if bear_target and price else None
    base_pct = ((base_target / price - 1) * 100) if base_target and price else None
    bull_pct = ((bull_target / price - 1) * 100) if bull_target and price else None

    # ── Quality & risk inputs ─────────────────────────────────────────────────
    piotroski   = _safe_dict(risk_r.get("piotroski")).get("score", 5)     # 0–9
    altman_z    = _safe_dict(risk_r.get("altman_z")).get("zone", "GRAY")
    sharpe      = risk_r.get("sharpe") or 0
    sortino     = risk_r.get("sortino") or 0
    roic_spread = _safe_dict(risk_r.get("roic_wacc")).get("spread") or 0
    max_dd      = abs(risk_r.get("max_drawdown_pct") or 0)

    # RSI from technical dict
    rsi_val = tech.get("rsi")

    # ── Composite recommendation score (0–10) ─────────────────────────────────
    # Valuation signal → 0–4
    sig_pts = {"STRONG_BUY": 4, "BUY": 3.2, "HOLD_WATCH": 2, "WAIT": 1, "AVOID_PEAK": 0}.get(signal, 2)
    # Quality → 0–2
    quality_pts = (piotroski / 9) * 2
    # Financial safety → 0–1
    safety_pts = 1.0 if altman_z == "SAFE" else 0.4 if altman_z == "GRAY" else 0.0
    # Momentum → 0–1.5
    mom_pts = min(max(sharpe, 0) * 0.75, 1.5)
    # ROIC/WACC → 0–1
    roic_pts = min(max(roic_spread * 10, 0), 1.0) if roic_spread else 0
    # RSI mean-reversion bonus (oversold = good, overbought = bad)
    rsi_pts = 0.3 if rsi_val and rsi_val < 35 else (-0.5 if rsi_val and rsi_val > 70 else 0)
    # Earnings risk penalty
    earn_pen = 0.8 if days_to_earnings and days_to_earnings <= 7 else (
               0.3 if days_to_earnings and days_to_earnings <= 14 else 0)

    composite = sig_pts + quality_pts + safety_pts + mom_pts + roic_pts + rsi_pts - earn_pen

    # ── Map to recommendation ─────────────────────────────────────────────────
    if composite >= 7.5:
        rec = "STRONG BUY"; rec_c = GREEN; rec_bg = GREEN_LT
    elif composite >= 5.8:
        rec = "BUY";         rec_c = GREEN; rec_bg = GREEN_LT
    elif composite >= 4.2:
        rec = "ACCUMULATE";  rec_c = BLUE;  rec_bg = BLUE_LT
    elif composite >= 2.8:
        rec = "HOLD";        rec_c = AMBER; rec_bg = AMBER_LT
    elif composite >= 1.5:
        rec = "REDUCE";      rec_c = AMBER; rec_bg = AMBER_LT
    else:
        rec = "AVOID";       rec_c = RED;   rec_bg = RED_LT

    # ── Risk rating ───────────────────────────────────────────────────────────
    risk_score = 0
    if altman_z == "DISTRESS": risk_score += 3
    elif altman_z == "GRAY":   risk_score += 1
    if max_dd > 45: risk_score += 2
    elif max_dd > 30: risk_score += 1
    if days_to_earnings and days_to_earnings <= 7: risk_score += 2
    elif days_to_earnings and days_to_earnings <= 14: risk_score += 1
    if rsi_val and rsi_val > 72: risk_score += 1
    if piotroski < 4: risk_score += 1

    if risk_score >= 5:
        risk_label = "VERY HIGH"; risk_c = RED
    elif risk_score >= 3:
        risk_label = "HIGH"; risk_c = AMBER
    elif risk_score >= 1:
        risk_label = "MODERATE"; risk_c = BLUE
    else:
        risk_label = "LOW"; risk_c = GREEN

    # ── One-line thesis ───────────────────────────────────────────────────────
    parts = []
    if signal == "STRONG_BUY":
        parts.append(f"trading at a deep discount — {abs(premium):.0f}% below fair value")
    elif signal == "BUY":
        parts.append(f"within buy zone ({abs(premium):.0f}% below FV)")
    elif signal == "AVOID_PEAK":
        parts.append(f"overvalued by {premium:.0f}% vs intrinsic value — wait for pullback")
    else:
        parts.append(f"near fair value ({premium:+.0f}% premium)")
    if piotroski >= 7:
        parts.append(f"Piotroski {piotroski}/9 signals strong fundamentals")
    elif piotroski <= 3:
        parts.append(f"Piotroski {piotroski}/9 flags fundamental weakness")
    if sharpe > 1.0:
        parts.append(f"Sharpe {sharpe:.2f} — excellent risk-adjusted returns")
    elif sharpe < 0:
        parts.append(f"negative Sharpe — underperforming risk-free rate")
    if roic_spread and roic_spread > 0.05:
        parts.append(f"ROIC exceeds WACC by {roic_spread*100:.1f}pp — value creation confirmed")
    if days_to_earnings and days_to_earnings <= 14:
        parts.append(f"earnings in {days_to_earnings}d — binary risk event approaching")
    thesis = ". ".join(p.capitalize() for p in parts[:3]) + "."

    return {
        "rec": rec, "rec_c": rec_c, "rec_bg": rec_bg,
        "composite": composite,
        "bear_target": bear_target, "base_target": base_target, "bull_target": bull_target,
        "bear_pct": bear_pct, "base_pct": base_pct, "bull_pct": bull_pct,
        "entry": entry, "stop": stop, "target": target,
        "upside": upside, "rr": rr, "premium": premium,
        "risk_label": risk_label, "risk_c": risk_c,
        "piotroski": piotroski, "sharpe": sharpe, "altman_z": altman_z,
        "rsi": rsi_val, "thesis": thesis,
    }


def tab_calendar(top10, universe_data: dict, valuation: dict = None, risk: dict = None):
    """Wall Street-style earnings calendar + quant analysis for the top-10 portfolio."""
    st.markdown(
        shdr("Market Intelligence — Earnings Calendar & Outlook",
             "Quantitative analysis + price targets + trade roadmap for every portfolio stock"),
        unsafe_allow_html=True,
    )

    if top10 is None or (hasattr(top10, "__len__") and len(top10) == 0):
        st.info("Run the analysis first to populate the earnings calendar for your picks.")
        return

    val  = valuation or {}
    rsk  = risk or {}

    # ── Build enriched rows ───────────────────────────────────────────────────
    rows = []
    for _, row in top10.iterrows():
        t     = row["ticker"]
        d     = universe_data.get(t, {})
        info  = d.get("info", {})
        tech  = d.get("technical", {})
        name  = info.get("shortName", t)
        sector= d.get("sector", "—")
        days  = d.get("earnings_days_away")
        edate = d.get("earnings_date", "") or ""
        score = float(row.get("composite_score", 0))
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)

        val_r = val.get(t, {})
        risk_r= rsk.get(t, {})

        analysis = _calendar_quant_analysis(t, val_r, risk_r, info, tech, days, score)

        rows.append({
            "ticker": t, "name": name, "sector": sector,
            "days": days, "edate": edate, "score": score, "price": price,
            "signal": val_r.get("signal", ""),
            "analysis": analysis,
        })

    has_date = sorted([r for r in rows if r["days"] is not None], key=lambda r: r["days"])
    no_date  = [r for r in rows if r["days"] is None]
    ordered  = has_date + no_date

    # ── Portfolio Sentiment Overview ──────────────────────────────────────────
    all_recs = [r["analysis"]["rec"] for r in rows]
    buys  = sum(1 for r in all_recs if r in ("STRONG BUY","BUY","ACCUMULATE"))
    holds = sum(1 for r in all_recs if r == "HOLD")
    sells = sum(1 for r in all_recs if r in ("REDUCE","AVOID"))
    avg_base_pct = sum(
        r["analysis"]["base_pct"] for r in rows if r["analysis"]["base_pct"] is not None
    ) / max(sum(1 for r in rows if r["analysis"]["base_pct"] is not None), 1)
    urgents = [r["ticker"] for r in rows if r["days"] is not None and r["days"] <= 14]

    overview_c = GREEN if buys > holds + sells else RED if sells > buys else AMBER
    st.markdown(
        f'<div style="background:{GRAY_LT};border:1px solid {BORDER};border-radius:12px;'
        f'padding:18px 24px;margin-bottom:24px;display:flex;gap:32px;flex-wrap:wrap;align-items:center">'
        f'<div><div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:4px">Portfolio Consensus</div>'
        f'<div style="font-size:24px;font-weight:900;color:{overview_c}">'
        f'{buys} BUY&nbsp; · &nbsp;{holds} HOLD&nbsp; · &nbsp;{sells} REDUCE</div></div>'
        f'<div><div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:4px">Avg Base-Case Upside</div>'
        f'<div style="font-size:24px;font-weight:900;color:{GREEN if avg_base_pct>0 else RED}">'
        f'{avg_base_pct:+.1f}%</div></div>'
        f'<div><div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;'
        f'letter-spacing:.08em;margin-bottom:4px">Earnings Risk (≤14d)</div>'
        f'<div style="font-size:20px;font-weight:900;color:{RED if urgents else GREEN}">'
        f'{", ".join(urgents) if urgents else "None"}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Next-up hero banner ───────────────────────────────────────────────────
    if has_date:
        n  = has_date[0]
        d2 = n["days"]
        nb_c = RED if d2<=7 else AMBER if d2<=14 else BLUE if d2<=30 else MUTED
        nb_bg= RED_LT if d2<=7 else AMBER_LT if d2<=14 else BLUE_LT if d2<=30 else GRAY_LT
        urgency = ("URGENT — wait for results" if d2<=7 else
                   "SOON — reduce size" if d2<=14 else
                   "THIS MONTH — monitor" if d2<=30 else f"{d2}d out")
        st.markdown(
            f'<div style="background:{nb_bg};border:1px solid {nb_c}44;border-left:5px solid {nb_c};'
            f'border-radius:12px;padding:18px 24px;margin-bottom:24px;'
            f'display:flex;align-items:center;gap:24px;flex-wrap:wrap">'
            f'<div style="font-size:44px;font-weight:900;color:{nb_c};line-height:1">'
            f'{d2}<span style="font-size:18px">d</span></div>'
            f'<div><div style="font-weight:800;font-size:16px;color:{TEXT}">'
            f'{n["ticker"]} — Next Earnings In Portfolio</div>'
            f'<div style="font-size:13px;color:{MUTED};margin-top:3px">'
            f'{n["name"]}  ·  {n["edate"]}  ·  {urgency}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Full Wall Street Analysis Cards ───────────────────────────────────────
    st.markdown(shdr("Stock-by-Stock Analysis", "Quant-driven recommendation · price targets · entry roadmap"), unsafe_allow_html=True)

    for r in ordered:
        a   = r["analysis"]
        d2  = r["days"]
        t   = r["ticker"]

        # Earnings urgency color
        e_c = RED if d2 and d2<=7 else AMBER if d2 and d2<=14 else BLUE if d2 and d2<=30 else MUTED2
        e_bg= RED_LT if d2 and d2<=7 else AMBER_LT if d2 and d2<=14 else BLUE_LT if d2 and d2<=30 else GRAY_LT
        earn_txt = f"{d2}d — {r['edate']}" if d2 is not None else "No date"

        def _pct(v):
            if v is None: return "n/a"
            return f"{v:+.1f}%"
        def _px(v):
            if v is None: return "n/a"
            return f"${v:,.2f}"

        bear_c = GREEN if (a["bear_pct"] or 0) > 0 else RED
        base_c = GREEN if (a["base_pct"] or 0) > 0 else RED
        bull_c = GREEN

        # Build bear/base/bull bars (width = |pct| capped at 50%)
        def _scenario_bar(pct, color, label, target_px):
            if pct is None: return ""
            w = min(abs(pct), 50) * 2   # 0-100%
            arrow = "▲" if pct > 0 else "▼"
            return (
                f'<div style="margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:11px;font-weight:700;margin-bottom:3px">'
                f'<span style="color:{MUTED2};text-transform:uppercase;letter-spacing:.05em">{label}</span>'
                f'<span style="color:{color}">{arrow} {_pct(pct)} &nbsp; {_px(target_px)}</span>'
                f'</div>'
                f'<div style="background:{BORDER};border-radius:3px;height:5px">'
                f'<div style="width:{w:.0f}%;background:{color};border-radius:3px;height:100%"></div>'
                f'</div></div>'
            )

        bear_bar = _scenario_bar(a["bear_pct"], RED,   "Bear Case", a["bear_target"])
        base_bar = _scenario_bar(a["base_pct"], base_c,"Base Case", a["base_target"])
        bull_bar = _scenario_bar(a["bull_pct"], GREEN, "Bull Case", a["bull_target"])

        # RSI indicator
        rsi_txt = f'RSI {a["rsi"]:.0f}' if a["rsi"] else ""
        rsi_c   = RED if a["rsi"] and a["rsi"]>70 else GREEN if a["rsi"] and a["rsi"]<30 else MUTED2
        rsi_lbl = " — Overbought" if a["rsi"] and a["rsi"]>70 else " — Oversold" if a["rsi"] and a["rsi"]<30 else ""

        # Entry checklist
        entry_ok  = r["price"] and a["entry"] and r["price"] <= a["entry"]
        entry_msg = ("✅ IN BUY ZONE — entry confirmed" if entry_ok else
                     f"⏳ Wait for pullback to {_px(a['entry'])}")
        entry_mc  = GREEN if entry_ok else AMBER

        with st.expander(
            f"{'🔴' if d2 and d2<=7 else '🟡' if d2 and d2<=14 else '🔵' if d2 and d2<=30 else '⚪'}  "
            f"{t} — {r['name']}  ·  {a['rec']}  ·  Earnings: {earn_txt}",
            expanded=(d2 is not None and d2 <= 14),
        ):
            col_left, col_right = st.columns([3, 2])

            with col_left:
                # Header: price + recommendation badge
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;flex-wrap:wrap">'
                    f'<div style="font-size:32px;font-weight:900;color:{TEXT};'
                    f'font-variant-numeric:tabular-nums">{_px(r["price"])}</div>'
                    f'<div style="background:{a["rec_bg"]};color:{a["rec_c"]};'
                    f'font-weight:800;font-size:14px;padding:6px 16px;border-radius:24px;'
                    f'border:1.5px solid {a["rec_c"]}44;letter-spacing:.04em">{a["rec"]}</div>'
                    f'{signal_badge(r["signal"])}'
                    f'<div style="font-size:11px;color:{a["risk_c"]};font-weight:700;'
                    f'background:{GRAY_LT};padding:4px 10px;border-radius:20px;'
                    f'border:1px solid {a["risk_c"]}44">Risk: {a["risk_label"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Thesis
                st.markdown(
                    f'<div style="background:#0A1525;border:1px solid {BLUE}44;border-left:3px solid {BLUE};'
                    f'border-radius:8px;padding:12px 14px;font-size:13px;color:{TEXT};'
                    f'line-height:1.6;margin-bottom:16px"><b>Quant Thesis:</b> {a["thesis"]}</div>',
                    unsafe_allow_html=True,
                )

                # Scenarios
                st.markdown(
                    f'<div style="background:{GRAY_LT};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:14px 16px;margin-bottom:14px">'
                    f'<div style="font-size:10px;font-weight:700;color:{MUTED2};'
                    f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">Price Scenarios (12-Month)</div>'
                    f'{bear_bar}{base_bar}{bull_bar}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Entry road map
                st.markdown(
                    f'<div style="background:{GRAY_LT};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:14px 16px">'
                    f'<div style="font-size:10px;font-weight:700;color:{MUTED2};'
                    f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">Trade Road Map</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px">'
                    f'<div><div style="font-size:9px;color:{MUTED2};text-transform:uppercase;margin-bottom:3px">Entry Zone</div>'
                    f'<div style="font-weight:800;font-size:15px;color:{GREEN}">{_px(a["entry"])}</div></div>'
                    f'<div><div style="font-size:9px;color:{MUTED2};text-transform:uppercase;margin-bottom:3px">Target</div>'
                    f'<div style="font-weight:800;font-size:15px;color:{GREEN}">{_px(a["target"])}</div></div>'
                    f'<div><div style="font-size:9px;color:{MUTED2};text-transform:uppercase;margin-bottom:3px">Stop Loss</div>'
                    f'<div style="font-weight:800;font-size:15px;color:{RED}">{_px(a["stop"])}</div></div>'
                    f'<div><div style="font-size:9px;color:{MUTED2};text-transform:uppercase;margin-bottom:3px">R/R Ratio</div>'
                    f'<div style="font-weight:800;font-size:15px;color:{BLUE}">{f"{a[chr(114)+chr(114)]:.2f}x" if a["rr"] else "n/a"}</div></div>'
                    f'</div>'
                    f'<div style="margin-top:10px;font-size:12px;color:{entry_mc};font-weight:700">{entry_msg}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_right:
                # Earnings tile
                st.markdown(
                    f'<div style="background:{e_bg};border:1px solid {e_c}44;border-radius:10px;'
                    f'padding:14px 16px;margin-bottom:14px;text-align:center">'
                    f'<div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;'
                    f'letter-spacing:.08em;margin-bottom:6px">Next Earnings</div>'
                    f'<div style="font-size:28px;font-weight:900;color:{e_c};line-height:1">'
                    f'{"—" if d2 is None else str(d2)}'
                    f'{"" if d2 is None else "<span style=font-size:14px> days</span>"}</div>'
                    f'<div style="font-size:12px;color:{MUTED};margin-top:4px">{r["edate"] or "No date"}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Key quant metrics
                az_c = GREEN if a["altman_z"]=="SAFE" else RED if a["altman_z"]=="DISTRESS" else AMBER
                sh_c = GREEN if a["sharpe"] and a["sharpe"]>0.5 else RED if a["sharpe"] and a["sharpe"]<0 else MUTED2
                pi_c = GREEN if a["piotroski"]>=7 else RED if a["piotroski"]<=3 else AMBER
                up_c = GREEN if a["upside"] and a["upside"]>0 else RED

                st.markdown(
                    f'<div style="background:{GRAY_LT};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:14px 16px;margin-bottom:14px">'
                    f'<div style="font-size:10px;font-weight:700;color:{MUTED2};text-transform:uppercase;'
                    f'letter-spacing:.08em;margin-bottom:10px">Quant Metrics</div>'
                    f'<div style="display:flex;flex-direction:column;gap:8px">'
                    + "".join([
                        f'<div style="display:flex;justify-content:space-between;'
                        f'border-bottom:1px solid {BORDER};padding-bottom:6px">'
                        f'<span style="font-size:12px;color:{MUTED}">{lbl}</span>'
                        f'<span style="font-size:12px;font-weight:700;color:{vc}">{val_str}</span></div>'
                        for lbl, val_str, vc in [
                            ("Composite Score",  f"{r['score']:.0f} / 100",      BLUE),
                            ("Valuation Signal", r["signal"] or "—",              GREEN if "BUY" in r["signal"] else RED if r["signal"]=="AVOID_PEAK" else AMBER),
                            ("Piotroski Score",  f"{a['piotroski']} / 9",         pi_c),
                            ("Altman Z",         a["altman_z"],                   az_c),
                            ("Sharpe Ratio",     f"{a['sharpe']:.2f}" if a["sharpe"] else "n/a", sh_c),
                            ("Upside to Target", _pct(a["upside"]),               up_c),
                            ("RSI (14d)",        f"{a['rsi']:.0f}{rsi_lbl}" if a["rsi"] else "n/a", rsi_c),
                        ]
                    ])
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Earnings action guidance
                if d2 is not None:
                    if d2 <= 7:
                        guidance = "Do NOT enter new positions. Wait for results and market reaction. Binary risk — gap risk both ways."
                        g_c = RED
                    elif d2 <= 14:
                        guidance = "Reduce size to 50% of target. Set limit orders below entry zone in case of post-earnings dip."
                        g_c = AMBER
                    elif d2 <= 30:
                        guidance = "Acceptable to enter at or below entry zone. Keep stop loss active heading into earnings."
                        g_c = BLUE
                    else:
                        guidance = "Earnings far enough out — enter freely if price is in the buy zone with full position sizing."
                        g_c = GREEN
                    st.markdown(
                        f'<div style="background:{GRAY_LT};border:1px solid {g_c}44;'
                        f'border-left:3px solid {g_c};border-radius:8px;'
                        f'padding:11px 14px;font-size:12px;color:{TEXT};line-height:1.5">'
                        f'<b style="color:{g_c}">Action guidance:</b> {guidance}</div>',
                        unsafe_allow_html=True,
                    )

    # ── Visual timeline chart ─────────────────────────────────────────────────
    if has_date:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(shdr("Earnings Timeline"), unsafe_allow_html=True)

        tickers_p = [r["ticker"] for r in has_date]
        days_p    = [r["days"]   for r in has_date]
        colors_p  = [RED if d2<=7 else AMBER if d2<=14 else BLUE if d2<=30 else MUTED2 for d2 in days_p]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=days_p, y=tickers_p, orientation="h",
            marker_color=colors_p,
            text=[f"{d2}d" for d2 in days_p], textposition="outside",
            hovertemplate="%{y}: %{x} days until earnings<extra></extra>",
        ))
        fig.add_vline(x=7,  line_color=RED,   line_dash="dot", annotation_text="7d")
        fig.add_vline(x=14, line_color=AMBER,  line_dash="dot", annotation_text="14d")
        fig.add_vline(x=30, line_color=BLUE,   line_dash="dot", annotation_text="30d")
        fig.update_layout(
            **_plotly_base(),
            height=max(280, len(has_date) * 42),
            margin=dict(l=60, r=80, t=30, b=40),
            xaxis=dict(title="Days Until Earnings", showgrid=True, gridcolor=BORDER, range=[0, max(days_p) * 1.18]),
            yaxis=dict(showgrid=False, autorange="reversed"),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Methodology note ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:16px;padding:13px 18px;background:{GRAY_LT};'
        f'border-radius:8px;border:1px solid {BORDER};font-size:12px;color:{MUTED}">'
        f'<b>Methodology:</b> Recommendations are generated from pure quantitative inputs — '
        f'valuation signal (DCF/Graham/EV-EBITDA/FCF yield), Piotroski F-Score, Altman Z-Score, '
        f'Sharpe ratio, ROIC/WACC spread, RSI, and earnings proximity risk. '
        f'Price scenarios derive from DCF sensitivity (Bear 50% / Base 100% / Bull 150% of growth estimate). '
        f'No external AI or analyst opinions — all outputs are mathematical. '
        f'This is not financial advice — always conduct your own due diligence.'
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
    _FACTOR_LABELS = {
        "momentum":   "Momentum",
        "volatility": "Volatility",
        "value":      "Value",
        "quality":    "Quality",
        "technical":  "Technical",
        "sentiment":  "Sentiment",
        "dividend":   "Dividend",
    }

    for session in sessions:
        ts      = session.get("timestamp", "")
        ts_date = ts[:10]
        ts_time = ts[11:16] if len(ts) > 15 else ""
        sid     = session.get("session_id", "?")
        prof    = session.get("profile", {})
        risk    = prof.get("risk_level", "?")
        horizon = prof.get("time_horizon", "?")
        goal    = prof.get("goal", "?")
        picks   = session.get("picks", [])
        is_eval = session.get("evaluated", False)

        if is_eval:
            ev      = session.get("evaluation") or {}
            avg_ret = ev.get("avg_pick_return", 0) or 0
            alpha   = ev.get("alpha", 0) or 0
            ret_sign = "+" if avg_ret >= 0 else ""
            al_sign  = "+" if alpha   >= 0 else ""
            label = (
                f"**{ts_date}** {ts_time}  ·  Session #{sid}  ·  "
                f"Risk {risk} / {horizon} / {goal}"
                f"  —  Return: {ret_sign}{avg_ret*100:.1f}%  ·  Alpha: {al_sign}{alpha*100:.1f}%"
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
                f"**{ts_date}** {ts_time}  ·  Session #{sid}  ·  "
                f"Risk {risk} / {horizon} / {goal}"
                f"  —  Pending evaluation ({days_old}d old)"
            )

        with st.expander(label, expanded=False):
            # ── Session profile header ──────────────────────────────────────
            st.markdown(
                f'<div style="background:{GRAY_LT};border:1px solid {BORDER};border-radius:10px;'
                f'padding:14px 20px;margin-bottom:20px;display:flex;gap:32px;flex-wrap:wrap">'
                f'<div><div style="font-size:9.5px;font-weight:700;color:{MUTED2};'
                f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Session</div>'
                f'<div style="font-weight:700;font-size:14px;color:{TEXT}">#{sid}</div></div>'
                f'<div><div style="font-size:9.5px;font-weight:700;color:{MUTED2};'
                f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Date &amp; Time</div>'
                f'<div style="font-weight:700;font-size:14px;color:{TEXT}">{ts_date} {ts_time} UTC</div></div>'
                f'<div><div style="font-size:9.5px;font-weight:700;color:{MUTED2};'
                f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Risk Level</div>'
                f'<div style="font-weight:700;font-size:14px;color:{TEXT}">{risk}</div></div>'
                f'<div><div style="font-size:9.5px;font-weight:700;color:{MUTED2};'
                f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Horizon</div>'
                f'<div style="font-weight:700;font-size:14px;color:{TEXT}">{horizon}</div></div>'
                f'<div><div style="font-size:9.5px;font-weight:700;color:{MUTED2};'
                f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Goal</div>'
                f'<div style="font-weight:700;font-size:14px;color:{TEXT}">{goal}</div></div>'
                f'<div><div style="font-size:9.5px;font-weight:700;color:{MUTED2};'
                f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Picks</div>'
                f'<div style="font-weight:700;font-size:14px;color:{BLUE}">{len(picks)}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Performance summary (evaluated sessions) ────────────────────
            if is_eval:
                ev      = session.get("evaluation") or {}
                avg_ret = ev.get("avg_pick_return", 0) or 0
                sp_ret  = ev.get("sp500_return")
                alpha2  = ev.get("alpha", 0) or 0
                eval_dt = (ev.get("evaluation_date") or "")[:10]

                ret_c   = GREEN if avg_ret >= 0 else RED
                alpha_c = GREEN if alpha2  >= 0 else RED
                sp_c    = GREEN if (sp_ret or 0) >= 0 else RED

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(mtile("Avg Pick Return", f"{avg_ret*100:+.1f}%",
                                      f"evaluated {eval_dt}", ret_c, ret_c), unsafe_allow_html=True)
                with mc2:
                    sp_str2 = f"{sp_ret*100:+.1f}%" if sp_ret is not None else "n/a"
                    st.markdown(mtile("S&P 500 Return", sp_str2,
                                      "over same period", sp_c, sp_c), unsafe_allow_html=True)
                with mc3:
                    st.markdown(mtile("Alpha", f"{alpha2*100:+.1f}%",
                                      "picks vs benchmark", alpha_c, alpha_c), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Pending banner ──────────────────────────────────────────────
            else:
                try:
                    ts_dt2    = datetime.fromisoformat(session["timestamp"])
                    days_old2 = (now - ts_dt2).days
                    days_left2 = max(0, 30 - days_old2)
                except Exception:
                    days_old2  = 0
                    days_left2 = 30

                st.markdown(
                    f'<div style="background:{AMBER_LT};border:1px solid {AMBER}44;'
                    f'border-left:4px solid {AMBER};border-radius:8px;padding:12px 16px;'
                    f'margin-bottom:16px;font-size:13px;color:{TEXT}">'
                    f'<b>Evaluation pending</b> — {days_old2} days old. '
                    f'Auto-evaluates in ~{days_left2} more days when you next run the tool.</div>',
                    unsafe_allow_html=True,
                )

            # ── Full pick cards ─────────────────────────────────────────────
            if picks:
                st.markdown(
                    f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:{MUTED2};margin-bottom:12px">Picks — as generated</div>',
                    unsafe_allow_html=True,
                )

                # Get evaluated exit prices if available
                ev_picks_map = {}
                if is_eval:
                    for ep in (session.get("evaluation") or {}).get("picks", []):
                        ev_picks_map[ep["ticker"]] = ep

                for pick in picks:
                    pt     = pick["ticker"]
                    ep_    = pick.get("price_entry", 0)
                    sc_    = pick.get("composite_score", 0)
                    factors= pick.get("factors", {})
                    ev_p   = ev_picks_map.get(pt, {})
                    exit_p = ev_p.get("price_exit")
                    ret_p  = ev_p.get("return")

                    sc_color = score_color(sc_)
                    ret_html = ""
                    if ret_p is not None:
                        rc2 = GREEN if ret_p >= 0 else RED
                        ret_html = (
                            f'<div style="font-size:11px;color:{MUTED2};margin-top:2px">'
                            f'Exit: <b style="color:{TEXT}">${exit_p:.2f}</b>'
                            f'  →  <b style="color:{rc2}">{ret_p*100:+.1f}%</b></div>'
                        )

                    # Factor score bars
                    factor_bars = ""
                    if factors:
                        for fname, flabel in _FACTOR_LABELS.items():
                            fkey = f"{fname}_score"
                            fval = factors.get(fkey)
                            if fval is not None:
                                fw   = min(float(fval), 100)
                                fc2  = score_color(fw)
                                factor_bars += (
                                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'
                                    f'<div style="width:68px;font-size:10px;color:{MUTED};text-align:right">{flabel}</div>'
                                    f'<div style="flex:1;background:{BORDER};border-radius:3px;height:6px">'
                                    f'<div style="width:{fw:.0f}%;background:{fc2};border-radius:3px;height:100%"></div></div>'
                                    f'<div style="width:30px;font-size:10px;font-weight:700;color:{fc2}">{fw:.0f}</div>'
                                    f'</div>'
                                )

                    st.markdown(
                        f'<div style="background:#0A0D14;border:1px solid {BORDER};border-left:4px solid {sc_color};'
                        f'border-radius:10px;padding:14px 18px;margin-bottom:10px;'
                        f'display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">'
                        # Ticker block
                        f'<div style="min-width:80px">'
                        f'<div style="font-size:20px;font-weight:900;color:{TEXT}">{pt}</div>'
                        f'<div style="font-size:11px;color:{MUTED};margin-top:2px">Score: '
                        f'<b style="color:{sc_color}">{sc_:.0f}</b></div>'
                        f'</div>'
                        # Score bar
                        f'<div style="flex:1;min-width:120px;padding-top:4px">'
                        f'<div style="background:{BORDER};border-radius:4px;height:8px;margin-bottom:6px">'
                        f'<div style="width:{min(sc_,100):.0f}%;background:{sc_color};border-radius:4px;height:100%"></div></div>'
                        # Entry / exit
                        f'<div style="font-size:12px;color:{MUTED}">'
                        f'Entry: <b style="color:{TEXT}">${ep_:.2f}</b></div>'
                        f'{ret_html}'
                        f'</div>'
                        # Factor bars
                        f'<div style="min-width:240px;flex:2">{factor_bars}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Evaluated full table ────────────────────────────────────────
            if is_eval and ev_picks_map:
                st.markdown("<br>", unsafe_allow_html=True)
                def _color_ret2(val):
                    try:
                        v = float(str(val).replace("%","").replace("+",""))
                        return f"color: {'#16a34a' if v >= 0 else '#dc2626'}; font-weight: 700"
                    except Exception:
                        return ""
                ev_rows = []
                for pick in picks:
                    pt2  = pick["ticker"]
                    ep2  = ev_picks_map.get(pt2, {})
                    r2   = ep2.get("return") or 0
                    ev_rows.append({
                        "Ticker":      pt2,
                        "Entry $":     f"${pick.get('price_entry',0):.2f}",
                        "Exit $":      f"${ep2.get('price_exit',0):.2f}" if ep2.get("price_exit") else "n/a",
                        "Return":      f"{r2*100:+.1f}%",
                        "Score @ Rec": f"{pick.get('composite_score',0):.1f}",
                    })
                df_ev = pd.DataFrame(ev_rows)
                st.dataframe(
                    df_ev.style.map(_color_ret2, subset=["Return"]),
                    use_container_width=True, hide_index=True,
                )

            # ── Time-machine button ─────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                f"📊 Open Full Analysis — {ts_date}",
                key=f"detail_{sid}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.show_session_detail = session
                st.session_state.show_history        = False
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION TIME-MACHINE — full historical analysis reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def render_session_detail(session: dict):
    """
    Reconstruct the full analysis page for a past session.
    Uses stored scores + factors + entry prices, then fetches live prices for P&L.
    """
    import yfinance as yf

    sid     = session.get("session_id", "?")
    ts      = session.get("timestamp", "")
    ts_date = ts[:10]
    ts_time = ts[11:16] if len(ts) > 15 else ""
    prof    = session.get("profile", {})
    picks   = session.get("picks", [])
    sp500_e = session.get("sp500_entry")
    is_eval = session.get("evaluated", False)
    ev      = session.get("evaluation") or {}

    risk_lv  = prof.get("risk_level", "?")
    horizon  = prof.get("time_horizon", "?")
    goal     = prof.get("goal", "?")

    _FLABELS = {
        "momentum": "Momentum", "volatility": "Volatility", "value": "Value",
        "quality": "Quality",   "technical": "Technical",   "sentiment": "Sentiment",
        "dividend": "Dividend",
    }

    # ── Back button ───────────────────────────────────────────────────────────
    if st.button("← Back to History", type="secondary"):
        st.session_state.show_session_detail = None
        st.rerun()

    # ── Session hero header ───────────────────────────────────────────────────
    eval_badge = ""
    if is_eval:
        avg_ret = (ev.get("avg_pick_return") or 0) * 100
        alpha   = (ev.get("alpha") or 0) * 100
        r_c     = "#22c55e" if avg_ret >= 0 else "#ef4444"
        a_c     = "#22c55e" if alpha   >= 0 else "#ef4444"
        eval_badge = (
            f'<span style="background:rgba(255,255,255,0.15);color:#fff;'
            f'font-weight:700;padding:5px 14px;border-radius:20px;font-size:13px;margin-left:12px">'
            f'Return: <span style="color:{r_c}">{avg_ret:+.1f}%</span>'
            f'  ·  Alpha: <span style="color:{a_c}">{alpha:+.1f}%</span></span>'
        )

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563EB 100%);'
        f'border-radius:14px;padding:24px 30px;margin-bottom:24px;color:#fff">'
        f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.10em;color:rgba(255,255,255,.6);margin-bottom:6px">'
        f'Session #{sid}  ·  Historical Analysis</div>'
        f'<div style="font-size:28px;font-weight:900;margin-bottom:4px">'
        f'{ts_date} &nbsp;<span style="font-size:16px;opacity:.7">{ts_time} UTC</span>'
        f'{eval_badge}</div>'
        f'<div style="font-size:14px;opacity:.8;margin-top:4px">'
        f'Risk Level {risk_lv}  ·  {horizon.capitalize()} horizon  ·  Goal: {goal}'
        f'{"  ·  S&P 500 at $" + f"{sp500_e:,.0f}" if sp500_e else ""}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Performance overview (evaluated sessions) ─────────────────────────────
    if is_eval:
        avg_ret = (ev.get("avg_pick_return") or 0) * 100
        sp_ret  = (ev.get("sp500_return") or 0) * 100
        alpha   = (ev.get("alpha") or 0) * 100
        eval_dt = (ev.get("evaluation_date") or "")[:10]
        ret_c   = GREEN if avg_ret >= 0 else RED
        sp_c    = GREEN if sp_ret  >= 0 else RED
        al_c    = GREEN if alpha   >= 0 else RED

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(mtile("Avg Pick Return", f"{avg_ret:+.1f}%", f"evaluated {eval_dt}", ret_c, ret_c), unsafe_allow_html=True)
        with c2: st.markdown(mtile("S&P 500 Return",  f"{sp_ret:+.1f}%",  "same period", sp_c, sp_c), unsafe_allow_html=True)
        with c3: st.markdown(mtile("Alpha",            f"{alpha:+.1f}%",   "picks vs benchmark", al_c, al_c), unsafe_allow_html=True)
        with c4: st.markdown(mtile("Picks",            str(len(picks)),    "in this session", BLUE, BLUE), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Fetch live prices ─────────────────────────────────────────────────────
    tickers_list = [p["ticker"] for p in picks]
    prices_now   = {}
    ev_picks_map = {ep["ticker"]: ep for ep in ev.get("picks", [])} if is_eval else {}

    with st.spinner("Fetching live prices for P&L…"):
        for t in tickers_list:
            try:
                df = _normalize_yf(yf.download(t, period="5d", progress=False, auto_adjust=True))
                if not df.empty:
                    prices_now[t] = float(df["Close"].dropna().squeeze().iloc[-1])
            except Exception:
                pass

    # ── Score rankings chart ──────────────────────────────────────────────────
    if picks:
        st.markdown(shdr("Composite Score Rankings", "As generated at time of analysis"), unsafe_allow_html=True)
        sorted_picks = sorted(picks, key=lambda p: p.get("composite_score", 0), reverse=True)
        bar_html = ""
        for rank, p in enumerate(sorted_picks, 1):
            sc  = p.get("composite_score", 0)
            sc_c= score_color(sc)
            w   = min(sc, 100)
            t   = p["ticker"]
            ep  = p.get("price_entry", 0)
            lp  = prices_now.get(t)
            ret = ((lp / ep - 1) * 100) if lp and ep else None
            ret_str = f'<span style="color:{GREEN if (ret or 0)>=0 else RED};font-weight:700;font-size:11px">{ret:+.1f}%</span>' if ret is not None else ""
            bar_html += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                f'<div style="width:14px;font-size:10px;color:{MUTED2};text-align:right">{rank}</div>'
                f'<div style="width:52px;font-weight:800;font-size:14px;color:{TEXT}">{t}</div>'
                f'<div style="flex:1;background:{BORDER};border-radius:4px;height:10px">'
                f'<div style="width:{w:.0f}%;background:{sc_c};border-radius:4px;height:100%"></div></div>'
                f'<div style="width:34px;font-weight:700;font-size:12px;color:{sc_c}">{sc:.0f}</div>'
                f'{ret_str}'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:{GRAY_LT};border:1px solid {BORDER};border-radius:12px;'
            f'padding:18px 22px;margin-bottom:20px">{bar_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Returns comparison chart (Plotly) ─────────────────────────────────────
    ret_data = []
    for p in picks:
        t  = p["ticker"]
        ep = p.get("price_entry", 0)
        lp = prices_now.get(t)
        if lp and ep:
            ret_data.append({"ticker": t, "ret": (lp / ep - 1) * 100})

    if ret_data or is_eval:
        st.markdown(shdr("Portfolio Returns", "Live P&L since entry date"), unsafe_allow_html=True)

        # Use evaluated final returns if available, else live
        chart_data = []
        for p in picks:
            t  = p["ticker"]
            ep_ev = ev_picks_map.get(t, {})
            final = (ep_ev.get("return") or 0) * 100 if is_eval and ep_ev.get("return") is not None else None
            live  = next((r["ret"] for r in ret_data if r["ticker"] == t), None)
            ret_use = final if final is not None else live
            if ret_use is not None:
                chart_data.append({"ticker": t, "ret": ret_use, "source": "Evaluated" if final is not None else "Live"})

        if chart_data:
            chart_data.sort(key=lambda x: x["ret"], reverse=True)
            bar_colors  = [GREEN if d["ret"] >= 0 else RED for d in chart_data]
            fig = go.Figure(go.Bar(
                x=[d["ticker"] for d in chart_data],
                y=[d["ret"] for d in chart_data],
                marker_color=bar_colors,
                text=[f"{d['ret']:+.1f}%" for d in chart_data],
                textposition="outside",
                hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
            ))
            # S&P 500 reference line
            if sp500_e and prices_now:
                try:
                    sp_df   = _normalize_yf(yf.download("^GSPC", period="1d", progress=False, auto_adjust=True))
                    sp_now  = float(sp_df["Close"].dropna().iloc[-1]) if not sp_df.empty else None
                    sp_live = ((sp_now / sp500_e - 1) * 100) if sp_now and sp500_e else None
                except Exception:
                    sp_live = None
            else:
                sp_live = (ev.get("sp500_return") or 0) * 100 if is_eval else None

            if sp_live is not None:
                fig.add_hline(y=sp_live, line_color=BLUE, line_dash="dash", line_width=2,
                              annotation_text=f"S&P 500: {sp_live:+.1f}%",
                              annotation_position="right")
            fig.add_hline(y=0, line_color=BORDER, line_width=1)
            fig.update_layout(
                **_plotly_base(),
                height=360,
                showlegend=False,
                margin=dict(l=20, r=100, t=40, b=40),
                yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False, ticksuffix="%"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Full pick cards (with valuation, risk, protocol) ─────────────────────
    st.markdown(shdr("Full Portfolio — Picks As Generated",
                     "Factor scores · Valuation · Risk · Protocol — exactly as generated"),
                unsafe_allow_html=True)

    # Macro snapshot banner
    macro_snap = session.get("macro_snapshot", {})
    if macro_snap:
        regime  = macro_snap.get("regime", "neutral")
        vix_v   = macro_snap.get("vix")
        y10_v   = macro_snap.get("yield_10y")
        crash_s = macro_snap.get("crash_signals")
        regime_colors = {
            "crisis": RED, "pre_crisis": AMBER, "risk_off": AMBER,
            "risk_on": GREEN, "rising_rate": AMBER, "falling_rate": GREEN, "neutral": BLUE,
        }
        rc = regime_colors.get(regime, BLUE)
        snap_html = (
            f'<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;'
            f'margin-bottom:20px;padding:12px 18px;background:{GRAY_LT};'
            f'border:1px solid {BORDER};border-radius:10px;font-size:13px">'
            f'<span style="font-weight:700;color:{MUTED}">Macro at time of analysis:</span>'
            f'<span style="background:{rc};color:#fff;padding:3px 12px;border-radius:12px;'
            f'font-weight:700;font-size:12px">{regime.upper()}</span>'
        )
        if vix_v is not None:
            snap_html += f'<span style="color:{TEXT}">VIX <b>{vix_v:.1f}</b></span>'
        if y10_v is not None:
            snap_html += f'<span style="color:{TEXT}">10Y Yield <b>{y10_v:.2f}%</b></span>'
        if crash_s is not None:
            cs_c = RED if crash_s >= 3 else AMBER if crash_s >= 2 else TEXT
            snap_html += (
                f'<span style="color:{cs_c}">Crash signals <b>{crash_s}/5</b></span>'
            )
        snap_html += '</div>'
        st.markdown(snap_html, unsafe_allow_html=True)

    def _kv(label, val_html):
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:4px 0;border-bottom:1px solid {BORDER};font-size:12px">'
            f'<span style="color:{MUTED}">{label}</span>'
            f'<span style="font-weight:700">{val_html}</span>'
            f'</div>'
        )

    sorted_picks = sorted(picks, key=lambda p: p.get("composite_score", 0), reverse=True)

    for pi, pick in enumerate(sorted_picks):
        t       = pick["ticker"]
        ep      = pick.get("price_entry", 0)
        sc      = pick.get("composite_score", 0)
        factors = pick.get("factors", {})
        lp      = prices_now.get(t)
        ev_p    = ev_picks_map.get(t, {})
        exit_p  = ev_p.get("price_exit")
        final_r = ev_p.get("return")

        sc_c     = score_color(sc)
        live_r   = ((lp / ep - 1) * 100) if lp and ep else None
        ret_show = (final_r * 100) if final_r is not None else live_r
        ret_lbl  = "Final Return" if final_r is not None else "Live P&L"

        if ret_show is not None:
            ret_str = f"{ret_show:+.1f}% {ret_lbl}"
        else:
            ret_str = "P&L pending"

        label_str = f"#{pi+1}  {t}  ·  Score: {sc:.0f}  ·  Entry: ${ep:,.2f}  ·  {ret_str}"

        with st.expander(label_str, expanded=(pi == 0)):
            val_data   = pick.get("valuation") or {}
            risk_data  = pick.get("risk") or {}
            proto_data = pick.get("protocol") or {}

            c1, c2, c3 = st.columns(3)

            # ── Col 1: Valuation ─────────────────────────────────────────────
            with c1:
                st.markdown(
                    f'<div style="background:#0A1525;border:1px solid #1E3A5F;'
                    f'border-radius:10px;padding:14px 16px;height:100%">'
                    f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:{BLUE};margin-bottom:10px">Valuation</div>',
                    unsafe_allow_html=True,
                )
                if val_data:
                    fv     = val_data.get("fair_value")
                    sig    = val_data.get("signal", "—")
                    ent_lo = val_data.get("entry_low")
                    ent_hi = val_data.get("entry_high")
                    tgt    = val_data.get("target")
                    stp    = val_data.get("stop_loss")
                    rr     = val_data.get("rr_ratio")
                    up     = val_data.get("upside_pct")
                    prem   = val_data.get("premium_pct")
                    ests   = val_data.get("estimates", {})
                    sens   = val_data.get("sensitivity", {})

                    sig_colors = {
                        "STRONG_BUY": GREEN, "BUY": "#16a34a",
                        "HOLD_WATCH": AMBER, "WAIT": "#d97706", "AVOID_PEAK": RED,
                    }
                    sc_color_val = sig_colors.get(sig, MUTED)

                    kv_rows = ""
                    if fv:
                        kv_rows += _kv("Fair Value", f'<span style="color:{BLUE}">${fv:,.2f}</span>')
                    if ent_lo and ent_hi:
                        kv_rows += _kv("Entry Zone", f'${ent_lo:,.2f} – ${ent_hi:,.2f}')
                    if tgt:
                        kv_rows += _kv("Target", f'${tgt:,.2f}')
                    if stp:
                        kv_rows += _kv("Stop Loss", f'<span style="color:{RED}">${stp:,.2f}</span>')
                    if rr is not None:
                        kv_rows += _kv("Risk / Reward", f'{rr:.2f}x')
                    if up is not None:
                        up_c = GREEN if up > 0 else RED
                        kv_rows += _kv("Upside", f'<span style="color:{up_c}">{up:+.1f}%</span>')
                    if prem is not None:
                        prem_c = RED if prem > 0 else GREEN
                        kv_rows += _kv("Premium/Discount", f'<span style="color:{prem_c}">{prem:+.1f}%</span>')

                    est_rows = ""
                    method_labels = {
                        "dcf": "DCF", "graham": "Graham",
                        "ev_ebitda": "EV/EBITDA", "fcf_yield": "FCF Yield",
                    }
                    for k, lbl in method_labels.items():
                        v = ests.get(k)
                        if v:
                            est_rows += _kv(lbl, f'${v:,.2f}')

                    sens_html = ""
                    if sens:
                        sens_html = (
                            f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                            f'color:{MUTED};margin-bottom:4px">DCF Sensitivity</div>'
                            f'<table style="width:100%;font-size:11px;border-collapse:collapse">'
                            f'<tr style="color:{MUTED}">'
                            f'<th style="text-align:left;padding:2px 4px">Scenario</th>'
                            f'<th style="text-align:right;padding:2px 4px">Fair Value</th>'
                            f'<th style="text-align:right;padding:2px 4px">Signal</th>'
                            f'</tr>'
                        )
                        scenario_colors = {"Bear": RED, "Base": BLUE, "Bull": GREEN}
                        for scenario in ["Bear", "Base", "Bull"]:
                            sd  = sens.get(scenario, {})
                            sfv = sd.get("fair_value")
                            ssig= sd.get("signal", "—")
                            sc_s= scenario_colors.get(scenario, MUTED)
                            if sfv:
                                sens_html += (
                                    f'<tr>'
                                    f'<td style="padding:3px 4px;color:{sc_s};font-weight:700">{scenario}</td>'
                                    f'<td style="text-align:right;padding:3px 4px">${sfv:,.2f}</td>'
                                    f'<td style="text-align:right;padding:3px 4px;color:{MUTED};font-size:10px">{ssig}</td>'
                                    f'</tr>'
                                )
                        sens_html += '</table>'

                    st.markdown(
                        f'<div style="background:{sc_color_val};color:#fff;font-weight:800;'
                        f'font-size:13px;padding:5px 10px;border-radius:6px;'
                        f'text-align:center;margin-bottom:10px">{sig.replace("_", " ")}</div>'
                        + kv_rows
                        + (f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                           f'color:{MUTED};margin-bottom:4px">Method Estimates</div>' + est_rows
                           if est_rows else "")
                        + sens_html
                        + '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="color:{MUTED};font-size:12px;padding:8px 0">'
                        f'Valuation not stored (older session format).</div></div>',
                        unsafe_allow_html=True,
                    )

            # ── Col 2: Risk ──────────────────────────────────────────────────
            with c2:
                st.markdown(
                    f'<div style="background:#041A12;border:1px solid #064E3B;'
                    f'border-radius:10px;padding:14px 16px;height:100%">'
                    f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:{GREEN};margin-bottom:10px">Risk & Quality</div>',
                    unsafe_allow_html=True,
                )
                if risk_data:
                    az   = risk_data.get("altman_z", {})
                    rw   = risk_data.get("roic_wacc", {})
                    pio  = risk_data.get("piotroski", {})

                    az_score  = az.get("score")
                    az_zone   = az.get("zone", "—")
                    az_c = GREEN if "Safe" in az_zone else RED if "Distress" in az_zone else AMBER

                    sharpe = risk_data.get("sharpe")
                    sortino= risk_data.get("sortino")
                    maxdd  = risk_data.get("max_drawdown_pct")
                    var95  = risk_data.get("var_95_pct")

                    roic    = rw.get("roic")
                    wacc    = rw.get("wacc")
                    spread  = rw.get("spread")
                    rw_verd = rw.get("verdict", "—")
                    rw_c    = GREEN if "Creating" in rw_verd else RED

                    pio_sc  = pio.get("score")
                    pio_int = pio.get("interpretation", "—")
                    pio_c   = GREEN if (pio_sc or 0) >= 7 else RED if (pio_sc or 0) <= 3 else AMBER

                    kv_rows = ""
                    if az_score is not None:
                        kv_rows += _kv("Altman Z-Score",
                                       f'<span style="color:{az_c}">{az_score:.2f} ({az_zone})</span>')
                    if sharpe is not None:
                        sh_c = GREEN if sharpe >= 1 else RED if sharpe < 0 else AMBER
                        kv_rows += _kv("Sharpe Ratio",
                                       f'<span style="color:{sh_c}">{sharpe:.2f}</span>')
                    if sortino is not None:
                        so_c = GREEN if sortino >= 1 else RED if sortino < 0 else AMBER
                        kv_rows += _kv("Sortino Ratio",
                                       f'<span style="color:{so_c}">{sortino:.2f}</span>')
                    if maxdd is not None:
                        kv_rows += _kv("Max Drawdown",
                                       f'<span style="color:{RED}">{maxdd:.1f}%</span>')
                    if var95 is not None:
                        kv_rows += _kv("VaR 95%",
                                       f'<span style="color:{RED}">{var95:.1f}%</span>')
                    if roic is not None:
                        kv_rows += _kv("ROIC", f'{roic:.1f}%')
                    if wacc is not None:
                        kv_rows += _kv("WACC", f'{wacc:.1f}%')
                    if spread is not None:
                        kv_rows += _kv("ROIC – WACC",
                                       f'<span style="color:{rw_c}">{spread:+.1f}%</span>')
                    if pio_sc is not None:
                        kv_rows += _kv("Piotroski F-Score",
                                       f'<span style="color:{pio_c}">{pio_sc}/9 ({pio_int})</span>')

                    st.markdown(f'{kv_rows}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="color:{MUTED};font-size:12px;padding:8px 0">'
                        f'Risk data not stored (older session format).</div></div>',
                        unsafe_allow_html=True,
                    )

            # ── Col 3: Protocol + Factor Scores ──────────────────────────────
            with c3:
                # Protocol gates
                st.markdown(
                    f'<div style="background:#1A1500;border:1px solid #451A03;'
                    f'border-radius:10px;padding:14px 16px;margin-bottom:12px">'
                    f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:{AMBER};margin-bottom:10px">Protocol Gates</div>',
                    unsafe_allow_html=True,
                )
                if proto_data:
                    gates      = proto_data.get("gates", [])
                    gate_stats = proto_data.get("gate_statuses", [])
                    pass_ct    = proto_data.get("pass_count", 0)
                    conv       = proto_data.get("conviction", "—")
                    ov_score   = proto_data.get("overall_score", 0)

                    conv_colors = {
                        "STRONG BUY": GREEN, "BUY": "#16a34a",
                        "WATCH": AMBER, "AVOID": RED,
                    }
                    cc = conv_colors.get(conv, MUTED)

                    gate_html = (
                        f'<div style="background:{cc};color:#fff;font-weight:800;'
                        f'font-size:12px;padding:4px 10px;border-radius:6px;'
                        f'text-align:center;margin-bottom:8px">'
                        f'{conv}  ·  {pass_ct}/7 passed  ·  {ov_score:.0f}pts</div>'
                    )
                    status_icons = {"PASS": (GREEN, "✓"), "WARN": (AMBER, "~"), "FAIL": (RED, "✗")}
                    for gi, gate_st in enumerate(gate_stats):
                        gate_name = GATE_NAMES[gi] if gi < len(GATE_NAMES) else f"Gate {gi+1}"
                        st_key = gate_st.upper() if isinstance(gate_st, str) else "FAIL"
                        color, sym = status_icons.get(st_key, (MUTED, "?"))
                        short_name = gate_name[:28] + "…" if len(gate_name) > 28 else gate_name
                        gate_html += (
                            f'<div style="display:flex;align-items:center;gap:5px;'
                            f'padding:2px 0;font-size:11px">'
                            f'<span style="color:{color};font-weight:800;width:12px">{sym}</span>'
                            f'<span style="color:{color};font-weight:600;width:36px">G{gi+1}</span>'
                            f'<span style="color:{MUTED};flex:1;overflow:hidden;'
                            f'text-overflow:ellipsis;white-space:nowrap">{short_name}</span>'
                            f'</div>'
                        )
                    st.markdown(f'{gate_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="color:{MUTED};font-size:12px;padding:8px 0">'
                        f'Protocol not stored (older session format).</div></div>',
                        unsafe_allow_html=True,
                    )

                # Factor score bars
                st.markdown(
                    f'<div style="background:{GRAY_LT};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:14px 16px">'
                    f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.08em;color:{MUTED};margin-bottom:10px">Factor Scores</div>',
                    unsafe_allow_html=True,
                )
                fbar = ""
                for fname, flabel in _FLABELS.items():
                    fval = factors.get(fname)  # stored WITHOUT _score suffix
                    if fval is not None:
                        fw = min(float(fval), 100)
                        fc = score_color(fw)
                        fbar += (
                            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">'
                            f'<div style="width:66px;font-size:10px;color:{MUTED};text-align:right">{flabel}</div>'
                            f'<div style="flex:1;background:{BORDER};border-radius:3px;height:7px">'
                            f'<div style="width:{fw:.0f}%;background:{fc};border-radius:3px;height:100%"></div></div>'
                            f'<div style="width:28px;font-size:10px;font-weight:700;color:{fc}">{fw:.0f}</div>'
                            f'</div>'
                        )
                if fbar:
                    st.markdown(f'{fbar}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="color:{MUTED};font-size:12px">'
                        f'Factor scores not available.</div></div>',
                        unsafe_allow_html=True,
                    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:24px;padding:13px 18px;background:{GRAY_LT};'
        f'border-radius:8px;border:1px solid {BORDER};font-size:12px;color:{MUTED}">'
        f'Session #{sid} generated on {ts_date}. '
        f'Factor scores and composite rankings reflect the model output at time of generation. '
        f'Live P&L uses the most recent closing price fetched today. '
        f'Evaluated return uses the official 30-day evaluation recorded by the memory system.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to History  ", type="secondary"):
        st.session_state.show_session_detail = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def tab_settings():
    s = st.session_state.get("settings", dict(DEFAULT_SETTINGS))

    st.markdown(shdr("⚙️  Settings", "Appearance · Data Export · Analysis Behaviour"), unsafe_allow_html=True)

    # ── Section 1: Appearance ─────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:{TEXT};'
        f'letter-spacing:.05em;text-transform:uppercase;margin:24px 0 12px">Appearance</div>',
        unsafe_allow_html=True,
    )

    theme_cols = st.columns(len(THEMES))
    current_theme = s.get("theme", "Terminal")
    new_theme = current_theme
    for i, (tname, tmeta) in enumerate(THEMES.items()):
        with theme_cols[i]:
            selected = current_theme == tname
            border = f"3px solid {tmeta['accent']}" if selected else f"2px solid {BORDER}"
            check  = "✓ " if selected else ""
            st.markdown(
                f'<div style="border:{border};border-radius:12px;padding:14px 10px;'
                f'text-align:center;background:{tmeta["bg"]};cursor:pointer;'
                f'box-shadow:{"0 0 0 3px " + tmeta["accent"] + "33" if selected else "none"}">'
                f'<div style="width:28px;height:28px;border-radius:50%;background:{tmeta["accent"]};'
                f'margin:0 auto 8px"></div>'
                f'<div style="font-size:12px;font-weight:700;color:{"#E2E8F0" if tname in ("Terminal","Dark") else "#111827"}">{check}{tname}</div>'
                f'<div style="font-size:10px;color:{"#64748B" if tname in ("Terminal","Dark") else "#6B7280"};margin-top:2px">bg · {tmeta["bg"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Select {tname}", key=f"theme_btn_{tname}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                new_theme = tname

    if new_theme != current_theme:
        s["theme"] = new_theme
        st.session_state.settings = s
        _save_settings(s)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Data Export ────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:{TEXT};'
        f'letter-spacing:.05em;text-transform:uppercase;margin:8px 0 12px">Data Export</div>',
        unsafe_allow_html=True,
    )

    res = st.session_state.get("results")
    if res is None:
        st.markdown(
            f'<div style="background:{GRAY_LT};border:1px solid {BORDER};border-radius:10px;'
            f'padding:16px 20px;color:{MUTED};font-size:13px">'
            f'Run an analysis first to enable data export.</div>',
            unsafe_allow_html=True,
        )
    else:
        exp_c1, exp_c2, exp_c3 = st.columns(3)
        top10 = res.get("top10")
        val   = res.get("valuation", {})
        risk  = res.get("risk", {})

        with exp_c1:
            if top10 is not None:
                csv_bytes = top10.to_csv(index=False).encode()
                st.download_button(
                    "⬇  Rankings CSV",
                    data=csv_bytes,
                    file_name="stock_rankings.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with exp_c2:
            # Build valuation summary CSV
            rows = []
            for t, vr in val.items():
                if not isinstance(vr, dict):
                    continue
                est = vr.get("estimates", {})
                rows.append({
                    "ticker":        t,
                    "fair_value":    vr.get("fair_value"),
                    "entry_low":     vr.get("entry_low"),
                    "target_price":  vr.get("target_price"),
                    "signal":        vr.get("signal"),
                    "upside_pct":    vr.get("upside_pct"),
                    "dcf":           est.get("dcf"),
                    "graham":        est.get("graham"),
                    "ev_ebitda":     est.get("ev_ebitda"),
                    "fcf_yield":     est.get("fcf_yield"),
                })
            if rows:
                val_csv = pd.DataFrame(rows).to_csv(index=False).encode()
                st.download_button(
                    "⬇  Valuation CSV",
                    data=val_csv,
                    file_name="valuation_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with exp_c3:
            # Session history JSON
            try:
                if os.path.exists("memory/history.json"):
                    with open("memory/history.json", "rb") as f:
                        hist_bytes = f.read()
                    st.download_button(
                        "⬇  Session History JSON",
                        data=hist_bytes,
                        file_name="session_history.json",
                        mime="application/json",
                        use_container_width=True,
                    )
            except Exception:
                st.caption("History file unavailable.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Analysis Behaviour ────────────────────────────────────
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:{TEXT};'
        f'letter-spacing:.05em;text-transform:uppercase;margin:8px 0 12px">Analysis Behaviour</div>',
        unsafe_allow_html=True,
    )

    b_c1, b_c2 = st.columns(2)
    with b_c1:
        new_penalty = st.slider(
            "Fresh Picks Penalty (pts)",
            min_value=0, max_value=50, value=int(s.get("fresh_penalty", 22)), step=1,
            help="How many points to subtract from tickers seen in recent sessions when Fresh Picks Mode is on. Higher = more aggressive rotation.",
        )
        new_n = st.slider(
            "Sessions to Remember",
            min_value=1, max_value=5, value=int(s.get("n_sessions", 2)), step=1,
            help="Number of past sessions checked for the fresh-picks rotation penalty.",
        )
    with b_c2:
        new_lr = st.slider(
            "Adaptive Learning Rate",
            min_value=0.01, max_value=0.15, value=float(s.get("learning_rate", 0.04)),
            step=0.01, format="%.2f",
            help="How aggressively factor weights adapt based on which factors predicted returns well. Takes effect on the next analysis run.",
        )
        new_signal = st.selectbox(
            "Signal Mode",
            options=["Conservative", "Balanced", "Aggressive"],
            index=["Conservative", "Balanced", "Aggressive"].index(s.get("signal_mode", "Balanced")),
            help=(
                "Conservative: only STRONG BUY shown as actionable.  "
                "Balanced: STRONG BUY + BUY (default).  "
                "Aggressive: STRONG BUY + BUY + HOLD/WATCH all treated as buys."
            ),
        )

    st.markdown("<br>", unsafe_allow_html=True)

    save_col, reset_col, _ = st.columns([1, 1, 4])
    with save_col:
        if st.button("💾  Save Behaviour", type="primary", use_container_width=True):
            s["fresh_penalty"] = new_penalty
            s["n_sessions"]    = new_n
            s["learning_rate"] = new_lr
            s["signal_mode"]   = new_signal
            st.session_state.settings = s
            _save_settings(s)
            st.success("Settings saved.", icon="✅")

    with reset_col:
        if st.button("↺  Reset Defaults", type="secondary", use_container_width=True):
            st.session_state.settings = dict(DEFAULT_SETTINGS)
            _save_settings(dict(DEFAULT_SETTINGS))
            st.rerun()

    # ── Info card ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sig_desc = {
        "Conservative": "Only <b>STRONG BUY</b> signals treated as actionable. Best for cautious investors.",
        "Balanced":     "<b>STRONG BUY</b> and <b>BUY</b> signals both actionable. Recommended default.",
        "Aggressive":   "<b>STRONG BUY</b>, <b>BUY</b>, and <b>HOLD/WATCH</b> all considered. Higher turnover.",
    }
    cur_sig = s.get("signal_mode", "Balanced")
    st.markdown(
        f'<div style="background:#0A1525;border:1px solid #1E3A5F;border-left:4px solid {BLUE};'
        f'border-radius:10px;padding:14px 18px;font-size:13px;color:{TEXT}">'
        f'<b>Active configuration</b> — '
        f'Theme: <b>{s.get("theme","Light")}</b> · '
        f'Penalty: <b>{s.get("fresh_penalty",22)} pts</b> · '
        f'Memory: <b>{s.get("n_sessions",2)} sessions</b> · '
        f'LR: <b>{s.get("learning_rate",0.04):.2f}</b> · '
        f'Signal: <b>{cur_sig}</b><br><br>'
        f'{sig_desc.get(cur_sig,"")}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load persisted settings once per session
    if "settings" not in st.session_state:
        st.session_state.settings = _load_settings()

    _apply_theme_css()

    profile, run_btn, hist_btn, bt_btn, cal_btn, settings_btn = render_sidebar()

    if "results" not in st.session_state:
        st.session_state.results             = None
        st.session_state.profile             = None
        st.session_state.show_history        = False
        st.session_state.show_backtest       = False
        st.session_state.show_calendar       = False
        st.session_state.show_settings       = False
        st.session_state.show_session_detail = None
        st.session_state.rankings_selected   = None
        st.session_state.bt_result           = None
        st.session_state.bt_period           = None

    if run_btn:
        st.session_state.show_history        = False
        st.session_state.show_backtest       = False
        st.session_state.show_calendar       = False
        st.session_state.show_settings       = False
        st.session_state.show_session_detail = None
        st.session_state.bt_result           = None
        with st.spinner("Running analysis…"):
            st.session_state.results = run_analysis(profile)
            st.session_state.profile = profile
        st.rerun()

    if hist_btn:
        st.session_state.show_history        = not st.session_state.get("show_history", False)
        st.session_state.show_backtest       = False
        st.session_state.show_calendar       = False
        st.session_state.show_settings       = False
        st.session_state.show_session_detail = None
        st.rerun()

    if bt_btn:
        st.session_state.show_backtest       = not st.session_state.get("show_backtest", False)
        st.session_state.show_history        = False
        st.session_state.show_calendar       = False
        st.session_state.show_settings       = False
        st.session_state.show_session_detail = None
        st.rerun()

    if cal_btn:
        st.session_state.show_calendar       = not st.session_state.get("show_calendar", False)
        st.session_state.show_history        = False
        st.session_state.show_backtest       = False
        st.session_state.show_settings       = False
        st.session_state.show_session_detail = None
        st.rerun()

    if settings_btn:
        st.session_state.show_settings       = not st.session_state.get("show_settings", False)
        st.session_state.show_history        = False
        st.session_state.show_backtest       = False
        st.session_state.show_calendar       = False
        st.session_state.show_session_detail = None
        st.rerun()

    # ── Settings view ─────────────────────────────────────────────────────
    if st.session_state.get("show_settings", False):
        st.markdown(
            f'<div class="ptitle">Stock Ranking Advisor</div>'
            f'<div class="psub">Settings  ·  Appearance · Export · Behaviour  ·  Click "⚙️ Settings" to close</div>',
            unsafe_allow_html=True,
        )
        tab_settings()
        return

    # ── Session time-machine (full historical analysis) ────────────────────
    if st.session_state.get("show_session_detail") is not None:
        st.markdown(
            f'<div class="ptitle">Stock Ranking Advisor</div>'
            f'<div class="psub">Session Replay  ·  Reconstructing historical analysis</div>',
            unsafe_allow_html=True,
        )
        render_session_detail(st.session_state.show_session_detail)
        return

    # ── Past Sessions view ─────────────────────────────────────────────────
    if st.session_state.get("show_history", False):
        st.markdown(
            f'<div class="ptitle">Stock Ranking Advisor</div>'
            f'<div class="psub">Past Sessions  ·  Click "History" in the sidebar to toggle</div>',
            unsafe_allow_html=True,
        )
        tab_history()
        return

    # ── Backtest view ──────────────────────────────────────────────────────
    if st.session_state.get("show_backtest", False):
        st.markdown(
            f'<div class="ptitle">Stock Ranking Advisor</div>'
            f'<div class="psub">Portfolio Backtest  ·  Full 10-stock basket vs S&amp;P 500  ·  Click "Backtest" to close</div>',
            unsafe_allow_html=True,
        )
        _res   = st.session_state.results or {}
        top_bt = _res.get("top10")
        uni_bt = _res.get("universe_data", {})
        val_bt = _res.get("valuation", {})
        rsk_bt = _res.get("risk", {})
        rf_bt  = _res.get("rf_rate", 0.045)
        tab_backtest(top_bt, uni_bt, val_bt, rsk_bt, rf_bt)
        return

    # ── Calendar view ──────────────────────────────────────────────────────
    if st.session_state.get("show_calendar", False):
        st.markdown(
            f'<div class="ptitle">Stock Ranking Advisor</div>'
            f'<div class="psub">Earnings Calendar  ·  Upcoming events for your portfolio  ·  Click "Calendar" to close</div>',
            unsafe_allow_html=True,
        )
        _res    = st.session_state.results or {}
        top_cal = _res.get("top10")
        uni_cal = _res.get("universe_data", {})
        val_cal = _res.get("valuation", {})
        rsk_cal = _res.get("risk", {})
        tab_calendar(top_cal, uni_cal, val_cal, rsk_cal)
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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Rankings",
        "Valuation",
        "Risk & Quality",
        "Protocol Gates",
        "Portfolio",
        "Macro & Performance",
        "Stock Lookup",
        "History",
        "Backtest",
        "Calendar",
    ])

    with tab1:  tab_rankings(top10, profile, val, proto, risk)
    with tab2:  tab_valuation(top10, val)
    with tab3:  tab_risk(top10, risk, uni)
    with tab4:  tab_protocol(top10, proto)
    with tab5:  tab_portfolio(top10, profile)
    with tab6:  tab_macro(top10, macro, uni, sp500, profile)
    with tab7:  tab_stock_lookup(uni, val, risk, proto, rf)
    with tab8:  tab_history()
    with tab9:  tab_backtest(top10, uni, val, risk, rf)
    with tab10: tab_calendar(top10, uni, val, risk)


if __name__ == "__main__":
    main()
