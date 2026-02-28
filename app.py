"""
app.py — Streamlit dashboard for Stock Ranking Advisor v3

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ── Must be the very first Streamlit call ─────────────────────────────────────
st.set_page_config(
    page_title="Stock Advisor",
    page_icon="📈",
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
from advisor.collector  import UserProfile
from advisor.fetcher    import DataFetcher, MacroFetcher
from advisor.scorer     import MultiFactorScorer
from advisor.portfolio  import PortfolioConstructor
from advisor.learner    import SessionMemory
from advisor.protocol   import ProtocolAnalyzer, GATE_SHORT
from advisor.valuation  import ValuationEngine
from advisor.risk       import RiskEngine


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BLUE       = "#2563EB"
BLUE_LT    = "#EFF6FF"
GREEN      = "#16a34a"
GREEN_LT   = "#f0fdf4"
AMBER      = "#d97706"
AMBER_LT   = "#fffbeb"
RED        = "#dc2626"
RED_LT     = "#fef2f2"
TEXT       = "#111827"
MUTED      = "#6B7280"
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
    "HOLD_WATCH":        (AMBER,  AMBER_LT,  "HOLD/WATCH"),
    "WAIT":              ("#ea580c", "#fff7ed", "WAIT"),
    "AVOID_PEAK":        (RED,    RED_LT,    "AVOID PEAK"),
    "INSUFFICIENT_DATA": (MUTED,  GRAY_LT,   "NO DATA"),
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
/* ── Base ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
}
.main .block-container {
    padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1440px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F9FAFB; border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #F3F4F6; padding: 4px;
    border-radius: 10px; margin-bottom: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 6px 20px;
    font-size: 13px; font-weight: 500; color: #6B7280;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #111827 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.10), 0 1px 2px rgba(0,0,0,0.06);
}

/* ── Metric tile ── */
.mtile {
    background: #fff; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 16px 20px; height: 100%;
}
.mtile-lbl {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #6B7280; margin-bottom: 6px;
}
.mtile-val {
    font-size: 22px; font-weight: 800; color: #111827; line-height: 1.2;
}
.mtile-sub { font-size: 11px; color: #9CA3AF; margin-top: 4px; }

/* ── Badge ── */
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 10.5px; font-weight: 700; letter-spacing: .03em;
}

/* ── Q-Table ── */
.qt { width: 100%; border-collapse: collapse; font-size: 13px; }
.qt th {
    background: #F9FAFB; color: #374151; font-weight: 600;
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .05em;
    padding: 9px 14px; border-bottom: 2px solid #E5E7EB;
    text-align: left; white-space: nowrap;
}
.qt td {
    padding: 9px 14px; border-bottom: 1px solid #F3F4F6;
    color: #111827; vertical-align: middle;
}
.qt tr:last-child td { border-bottom: none; }
.qt tr:hover td { background: #FAFAFA; }

/* ── Rank circle ── */
.rank {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: #1D4ED8; color: #fff;
    font-size: 10.5px; font-weight: 800;
}

/* ── Score mini-bar ── */
.sbar-wrap { background: #F3F4F6; border-radius: 3px; height: 5px; }
.sbar       { height: 5px; border-radius: 3px; }

/* ── Hero ── */
.hero { text-align: center; padding: 70px 20px 50px; }
.hero-title {
    font-size: 44px; font-weight: 900; color: #111827;
    letter-spacing: -.025em; margin-bottom: 14px;
}
.hero-sub {
    font-size: 17px; color: #6B7280; max-width: 560px;
    margin: 0 auto 32px; line-height: 1.65;
}

/* ── Section header ── */
.shdr { font-size: 15px; font-weight: 700; color: #111827; margin-bottom: 2px; }
.ssub { font-size: 11.5px; color: #9CA3AF; margin-bottom: 14px; }

/* ── Top-pick card ── */
.pick-card {
    background: #fff; border: 1px solid #E5E7EB; border-radius: 14px;
    padding: 20px; height: 100%;
    transition: box-shadow .15s;
}
.pick-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.08); }
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

def fmt_price(v):
    if v is None: return "—"
    try:    return f"${float(v):,.2f}"
    except: return "—"

def fmt_pct(v, plus=True):
    if v is None: return "—"
    try:    return f"{float(v):+.1f}%" if plus else f"{float(v):.1f}%"
    except: return "—"

def fmt_2(v):
    if v is None: return "—"
    try:    return f"{float(v):.2f}"
    except: return "—"

def score_color(s):
    s = s or 0
    if s >= 70: return GREEN
    if s >= 45: return AMBER
    return RED

def sbar(s, color=None):
    c = color or score_color(s or 0)
    return (f'<div class="sbar-wrap"><div class="sbar" '
            f'style="width:{min(s or 0, 100):.0f}%;background:{c}"></div></div>')

def mtile(label, value, sub="", color=TEXT):
    return (f'<div class="mtile">'
            f'<div class="mtile-lbl">{label}</div>'
            f'<div class="mtile-val" style="color:{color}">{value}</div>'
            f'{"<div class=mtile-sub>" + sub + "</div>" if sub else ""}'
            f'</div>')

def shdr(title, sub=""):
    out = f'<div class="shdr">{title}</div>'
    if sub: out += f'<div class="ssub">{sub}</div>'
    return out


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
    res["ranked_df"] = scorer.score_all(res["universe_data"])
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
    prog.progress(100, text="Done!")
    prog.empty()
    return res


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-size:19px;font-weight:800;color:#111827;margin-bottom:2px">📈 Stock Advisor</div>'
            '<div style="font-size:11.5px;color:#9CA3AF;margin-bottom:18px">'
            'Hedge-fund grade quant · Free data only</div>',
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

        st.divider()
        run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

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
    )
    return profile, run_btn


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME
# ─────────────────────────────────────────────────────────────────────────────
def render_welcome():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Stock Ranking Advisor</div>
      <div class="hero-sub">
        Hedge-fund grade quantitative analysis entirely on free data.
        DCF · Graham · EV/EBITDA · Altman Z · Piotroski · ROIC/WACC.
        No subscriptions. No API keys. Just math.
      </div>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("🔢", "7-Factor Scoring",    "12-1 momentum · EV/EBITDA value · Novy-Marx quality · 5-factor technicals"),
        ("💰", "4-Method Valuation",  "DCF (2-stage) · Graham Number · EV/EBITDA target · FCF yield"),
        ("🛡️", "Full Risk Suite",      "Altman Z · Sharpe · Sortino · Max DD · VaR 95% · ROIC/WACC"),
        ("🚪", "7-Gate Protocol",      "Warren Buffett quality screen — every stock must justify its place"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            with st.container(border=True):
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Configure your profile in the sidebar, then click **▶ Run Analysis**.", icon="👈")


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

    r_color = (GREEN if "RISK ON" in regime else RED if "RISK OFF" in regime else AMBER)
    vix_color = (RED if (vix or 0) > 25 else AMBER if (vix or 0) > 18 else GREEN)

    c1, c2, c3, c4, c5 = st.columns(5)
    tiles = [
        (c1, "VIX",           f"{vix:.1f}" if vix else "N/A",    "Elevated" if (vix or 0) > 25 else "Normal range", vix_color),
        (c2, "10-Year Yield", f"{y10:.2f}%" if y10 else "N/A",   f"Risk-free rate  ·  rf = {rf_rate*100:.2f}%", TEXT),
        (c3, "Regime",        regime,                             reasons[:48] if reasons else "—", r_color),
        (c4, "In Portfolio",  str(n),                             "Stocks selected", BLUE),
        (c5, "Avg Score",     f"{avg_sc:.1f}",                    "Portfolio composite / 100", score_color(avg_sc)),
    ]
    for col, lbl, val, sub, color in tiles:
        with col:
            st.markdown(mtile(lbl, val, sub, color), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
def tab_rankings(top10, profile, valuation, protocol):
    proto_map = {p["ticker"]: p for p in protocol}

    # ── Top-3 highlight cards ──────────────────────────────────────────────
    cols = st.columns(3)
    for col, (_, row) in zip(cols, top10.head(3).iterrows()):
        t   = row["ticker"]
        val = valuation.get(t, {})
        sig = val.get("signal", "INSUFFICIENT_DATA")
        sc  = float(row["composite_score"])
        c_sig, bg_sig, lbl_sig = SIGNAL_META.get(sig, (MUTED, GRAY_LT, sig))
        conv = proto_map.get(t, {}).get("conviction", "—")
        c_cv, bg_cv = CONV_META.get(conv, (MUTED, GRAY_LT))
        fv   = val.get("fair_value")
        prem = val.get("premium_pct")
        prem_str = f"{prem:+.1f}%" if prem is not None else "—"

        with col:
            st.markdown(
                f'<div class="pick-card">'
                f'  <div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'    <div>'
                f'      <div style="font-size:22px;font-weight:800;color:{TEXT}">{t}</div>'
                f'      <div style="font-size:12px;color:{MUTED};margin-top:1px">{row["sector"]}</div>'
                f'    </div>'
                f'    <div style="text-align:right">'
                f'      <span class="badge" style="color:{c_sig};background:{bg_sig}">{lbl_sig}</span><br>'
                f'      <span class="badge" style="color:{c_cv};background:{bg_cv};margin-top:4px">{conv}</span>'
                f'    </div>'
                f'  </div>'
                f'  <div style="margin:14px 0 4px">'
                f'    <div style="font-size:32px;font-weight:900;color:{score_color(sc)}">{sc:.1f}'
                f'      <span style="font-size:14px;color:{MUTED};font-weight:500">/100</span>'
                f'    </div>'
                f'  </div>'
                f'  {sbar(sc)}'
                f'  <div style="margin-top:12px;display:flex;gap:20px">'
                f'    <div><div style="font-size:10px;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.06em">Fair Value</div>'
                f'         <div style="font-size:14px;font-weight:700">{fmt_price(fv)}</div></div>'
                f'    <div><div style="font-size:10px;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.06em">Premium</div>'
                f'         <div style="font-size:14px;font-weight:700;color:{RED if (prem or 0) > 5 else GREEN}">{prem_str}</div></div>'
                f'    <div><div style="font-size:10px;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.06em">Entry</div>'
                f'         <div style="font-size:14px;font-weight:700;color:{GREEN}">{fmt_price(val.get("entry_low"))}</div></div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

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
        rows += (
            f'<tr>'
            f'<td><span class="rank">{int(row["rank"])}</span></td>'
            f'<td><b style="font-size:14px">{t}</b></td>'
            f'<td style="color:{MUTED}">{row["sector"]}</td>'
            f'<td style="min-width:130px">'
            f'  <div style="display:flex;align-items:center;gap:8px">'
            f'    <span style="font-weight:700;color:{score_color(sc)};width:34px">{sc:.1f}</span>'
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
            f'</tr>'
        )
    st.markdown(
        f'<table class="qt"><thead><tr>'
        f'<th>#</th><th>Ticker</th><th>Sector</th><th>Score</th>'
        f'<th>Signal</th><th>Conviction</th><th>Price</th>'
        f'<th>Fair Value</th><th>Premium</th><th>Entry (−20%)</th><th>Stop Loss</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Factor breakdown chart ─────────────────────────────────────────────
    st.markdown(shdr("Factor Score Breakdown", "Weighted contribution of each factor to composite score"),
                unsafe_allow_html=True)

    weights    = WEIGHT_MATRIX.get((profile.risk_level, profile.time_horizon), [1/7]*7)
    tickers_r  = top10["ticker"].tolist()[::-1]
    scores_r   = top10["composite_score"].tolist()[::-1]

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
        barmode="stack", template="plotly_white", height=390,
        margin=dict(l=0, r=70, t=6, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(title="Weighted Score Contribution (0–100)", range=[0, 108]),
        yaxis=dict(tickfont=dict(size=12, family="monospace")),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
    )
    st.plotly_chart(fig, use_container_width=True)


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
        t   = row["ticker"]
        val = valuation.get(t, {})
        est = val.get("estimates", {})
        sig = val.get("signal", "INSUFFICIENT_DATA")
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

    # ── Entry price positioning + method spread ────────────────────────────
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
        for x0, x1, color, label, lx in [
            (-42, -20, f"rgba(22,163,74,.07)",   "STRONG BUY", -31),
            (-20,   0, f"rgba(37,99,235,.07)",   "BUY",         -10),
            (  0,  10, f"rgba(217,119,6,.07)",   "WATCH",         5),
            ( 10,  50, f"rgba(220,38,38,.05)",   "EXPENSIVE",    28),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0)
            fig.add_annotation(x=lx, y=len(tickers)-0.3, text=label,
                               showarrow=False, font=dict(size=9, color="#9CA3AF"), yref="y")
        fig.add_vline(x=0, line_dash="dash", line_color="#9CA3AF", line_width=1.2)
        fig.add_vline(x=-20, line_dash="dot", line_color=GREEN, line_width=1)
        fig.add_trace(go.Bar(
            x=prems, y=tickers, orientation="h",
            marker_color=colors, opacity=0.82,
            text=[f"{p:+.1f}%" for p in prems], textposition="outside",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=htexts,
        ))
        fig.update_layout(
            template="plotly_white", height=370,
            margin=dict(l=0, r=80, t=6, b=36),
            xaxis=dict(title="% vs Fair Value", range=[-47, 58],
                       zeroline=True, zerolinecolor="#9CA3AF"),
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
            template="plotly_white", height=370,
            margin=dict(l=0, r=0, t=6, b=36),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            yaxis=dict(title="Price ($)", tickformat="$,.0f"),
            xaxis=dict(tickfont=dict(size=10, family="monospace")),
        )
        st.plotly_chart(fig2, use_container_width=True)


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
        t   = row["ticker"]
        r   = risk.get(t, {})
        az  = r.get("altman_z", {})
        rw  = r.get("roic_wacc", {})
        pf  = r.get("piotroski", {})
        pf_sc   = pf.get("score")
        pf_clr  = GREEN if (pf_sc or 0) >= 7 else AMBER if (pf_sc or 0) >= 4 else RED
        rw_sp   = rw.get("spread")
        rw_clr  = GREEN if (rw_sp or 0) > 5 else AMBER if (rw_sp or 0) > 0 else RED
        sh      = r.get("sharpe")
        sh_clr  = GREEN if (sh or 0) > 1 else AMBER if (sh or 0) > 0 else RED
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
            shdr("Risk / Return Scatter",
                 "X = Sharpe · Y = ROIC/WACC spread · Color = Altman Z zone"),
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
            szs.append(max(14, min(40, abs(spread) * 2 + 14)))
            dot_clrs.append(ZONE_META.get(zone, (MUTED, GRAY_LT))[0])
            texts.append(t)

        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=texts, textposition="top center",
            textfont=dict(size=10, family="monospace"),
            marker=dict(size=szs, color=dot_clrs, opacity=0.88,
                        line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{text}</b><br>Sharpe: %{x:.2f}<br>ROIC/WACC: %{y:+.1f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#D1D5DB", line_width=1)
        fig.add_vline(x=0, line_dash="dash", line_color="#D1D5DB", line_width=1)
        # Quadrant labels
        fig.add_annotation(x=max(xs or [0]) * 0.9, y=max(ys or [0]) * 0.9,
                           text="IDEAL", showarrow=False,
                           font=dict(size=10, color=GREEN), opacity=0.6)
        fig.update_layout(
            template="plotly_white", height=340,
            margin=dict(l=0, r=0, t=6, b=40),
            xaxis=dict(title="Sharpe Ratio"),
            yaxis=dict(title="ROIC / WACC Spread (%)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(
            shdr("Piotroski F-Score",
                 "9-point quality score · ≥ 7 = Strong · 4–6 = Average · ≤ 3 = Weak"),
            unsafe_allow_html=True,
        )
        tickers = top10["ticker"].tolist()
        scores, clrs = [], []
        for t in tickers:
            sc = risk.get(t, {}).get("piotroski", {}).get("score")
            scores.append(sc if sc is not None else 0)
            clrs.append(GREEN if (sc or 0) >= 7 else AMBER if (sc or 0) >= 4 else RED)

        fig2 = go.Figure(go.Bar(
            x=tickers, y=scores, marker_color=clrs, opacity=0.88,
            text=scores, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Piotroski: %{y}/9<extra></extra>",
        ))
        fig2.add_hline(y=7, line_dash="dot", line_color=GREEN,
                       annotation_text="Strong ≥7", annotation_position="top right",
                       annotation_font=dict(color=GREEN, size=10))
        fig2.add_hline(y=3, line_dash="dot", line_color=RED,
                       annotation_text="Weak ≤3", annotation_position="top right",
                       annotation_font=dict(color=RED, size=10))
        fig2.update_layout(
            template="plotly_white", height=340,
            margin=dict(l=0, r=70, t=6, b=40),
            yaxis=dict(range=[0, 11], title="Score / 9"),
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

    # Heatmap
    gate_matrix = []
    text_matrix = []
    for t in tickers:
        gates = proto_map.get(t, {}).get("gates", [50] * 7)
        gate_matrix.append([float(g) for g in gates[:7]])
        text_matrix.append([f"{float(g):.0f}" for g in gates[:7]])

    colorscale = [
        [0.00, "#FEF2F2"], [0.35, "#FEF3C7"],
        [0.60, "#F0FDF4"], [1.00, "#15803D"],
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
        template="plotly_white", height=370,
        margin=dict(l=0, r=90, t=6, b=6),
        xaxis=dict(side="top", tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12, family="monospace"), autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Protocol summary table
    st.markdown(shdr("Protocol Summary"), unsafe_allow_html=True)
    rows = ""
    for _, row in top10.iterrows():
        t     = row["ticker"]
        p     = proto_map.get(t, {})
        ea    = p.get("entry_analysis", {})
        conv  = p.get("conviction", "—")
        over  = p.get("overall_score", 0)

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
            labels=[f"{t}" for t in tickers],
            values=weights, hole=0.60,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value:.1%}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>${profile.portfolio_size:,.0f}</b>",
            x=0.5, y=0.5, font=dict(size=14, color=TEXT), showarrow=False,
        )
        fig.update_layout(
            template="plotly_white", height=360,
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
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
        rows = ""
        total = profile.portfolio_size
        for _, row in top10.iterrows():
            t   = row["ticker"]
            w   = float(row.get("weight", 0))
            amt = float(row.get("dollar_amount", w * total))
            sh  = row.get("approx_shares", "—")
            px  = float(row.get("current_price", 0))
            sec = row.get("sector", "—")
            sc  = SECTOR_COLORS.get(sec, "#9CA3AF")
            bar = (f'<div style="background:#F3F4F6;border-radius:3px;height:5px;width:100%">'
                   f'<div style="background:{sc};height:5px;border-radius:3px;width:{w*100:.0f}%"></div></div>')
            rows += (
                f'<tr>'
                f'<td><b>{t}</b><br><span style="font-size:11px;color:{MUTED}">{sec}</span></td>'
                f'<td style="min-width:130px">'
                f'  <div style="display:flex;align-items:center;gap:8px">'
                f'    <span style="font-weight:700;color:{sc};width:38px">{w*100:.1f}%</span>'
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

        # Weight bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(shdr("Position Weights"), unsafe_allow_html=True)
        clrs = [SECTOR_COLORS.get(s, "#9CA3AF") for s in top10["sector"].tolist()]
        fig2 = go.Figure(go.Bar(
            x=top10["ticker"].tolist(),
            y=(top10["weight"] * 100).tolist() if "weight" in top10.columns else [],
            marker_color=clrs, opacity=0.88,
            text=[f"{w*100:.1f}%" for w in top10.get("weight", [])],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Weight: %{y:.1f}%<extra></extra>",
        ))
        fig2.add_hline(y=10, line_dash="dot", line_color="#9CA3AF", line_width=1,
                       annotation_text="Equal Weight 10%",
                       annotation_font=dict(size=9, color=MUTED))
        fig2.update_layout(
            template="plotly_white", height=240,
            margin=dict(l=0, r=0, t=6, b=36),
            yaxis=dict(title="Weight (%)", range=[0, max((top10["weight"]*100).tolist() or [20]) * 1.25]),
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
        rc = (GREEN if "RISK ON" in regime else RED if "RISK OFF" in regime else AMBER)
        vc = (RED if (vix or 0) > 25 else AMBER if (vix or 0) > 18 else GREEN)
        st.markdown(
            mtile("VIX", f"{vix:.1f}" if vix else "N/A",
                  "Elevated — caution" if (vix or 0) > 25 else "Normal range", vc)
            + "<br>"
            + mtile("10-Year Yield", f"{y10:.2f}%" if y10 else "N/A",
                    "Risk-free rate basis", TEXT)
            + "<br>"
            + mtile("Regime", regime,
                    "  ·  ".join(macro.get("regime_reasons", []))[:50], rc),
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
                text=[f"{r:+.1f}%" for _, r in sorted_etf],
                textposition="outside",
                hovertemplate="<b>%{y}</b>: %{x:+.1f}%<extra></extra>",
            ))
            fig_etf.update_layout(
                template="plotly_white", height=280,
                margin=dict(l=0, r=60, t=0, b=30),
                xaxis=dict(zeroline=True, zerolinecolor="#9CA3AF"),
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
            sp     = DataFetcher.strip_tz(sp500_hist["Close"].dropna())
            start_date = sp.index[0]
            sp_norm = sp / sp.iloc[0] * 100
            sp_ret  = (sp.iloc[-1] / sp.iloc[0] - 1) * 100
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
            template="plotly_white", height=360,
            margin=dict(l=0, r=0, t=6, b=40),
            legend=dict(font=dict(size=10), orientation="v",
                        yanchor="top", y=1, xanchor="left", x=1.01),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Normalised (Base = 100)"),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # Correlation matrix
        st.markdown(
            shdr("Return Correlation Matrix",
                 "Daily returns · Lower = better diversification"),
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
                colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1D4ED8"]],
                zmin=0, zmax=1,
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
                colorbar=dict(thickness=14, len=0.85),
            ))
            fig_corr.update_layout(
                template="plotly_white", height=280,
                margin=dict(l=0, r=60, t=6, b=0),
                xaxis=dict(tickfont=dict(size=10, family="monospace")),
                yaxis=dict(tickfont=dict(size=10, family="monospace"), autorange="reversed"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    profile, run_btn = render_sidebar()

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.profile = None

    if run_btn:
        with st.spinner("Running analysis…"):
            st.session_state.results = run_analysis(profile)
            st.session_state.profile = profile
        st.rerun()

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

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
        f'<span style="font-size:26px;font-weight:900;color:{TEXT}">Stock Ranking Advisor</span>'
        f'<span style="font-size:12.5px;color:{MUTED};font-weight:500">'
        'v3  ·  Pure Quantitative  ·  No AI APIs</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:12.5px;color:{MUTED};margin-bottom:20px">'
        f'{profile.risk_label}  ·  {HORIZON_LABELS[profile.time_horizon]}  ·  '
        f'${profile.portfolio_size:,.0f}  ·  {profile.goal_label}'
        f'</div>',
        unsafe_allow_html=True,
    )

    render_macro_strip(macro, top10, rf)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊  Rankings",
        "💰  Valuation",
        "🛡️  Risk & Quality",
        "🚪  Protocol Gates",
        "🗂️  Portfolio",
        "🌐  Macro & Performance",
    ])

    with tab1: tab_rankings(top10, profile, val, proto)
    with tab2: tab_valuation(top10, val)
    with tab3: tab_risk(top10, risk)
    with tab4: tab_protocol(top10, proto)
    with tab5: tab_portfolio(top10, profile)
    with tab6: tab_macro(top10, macro, uni, sp500, profile)


if __name__ == "__main__":
    main()
