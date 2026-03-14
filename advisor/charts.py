# advisor/charts.py — 6 dark-theme matplotlib charts

import os
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from config import FACTOR_NAMES
from advisor.collector import UserProfile, HORIZON_LABELS
from advisor.fetcher import DataFetcher


# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BORDER    = "#30363d"
TXT_DIM   = "#8b949e"
TXT_WHITE = "#e6edf3"

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

FACTOR_COLORS = {
    "momentum":   "#58a6ff",
    "volatility": "#3fb950",
    "value":      "#e3b341",
    "quality":    "#f78166",
    "technical":  "#d2a8ff",
    "sentiment":  "#79c0ff",
    "dividend":   "#56d364",
}


def _dark_axes(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TXT_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


class ChartEngine:

    def __init__(self, profile: UserProfile, sp500: Optional[pd.DataFrame],
                 macro_data: dict):
        self.profile    = profile
        self.sp500      = sp500
        self.macro_data = macro_data

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 1 — Stacked factor contribution bar chart
    # ─────────────────────────────────────────────────────────────────────────
    def score_breakdown(self, top10: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(13, 7))
        fig.patch.set_facecolor(BG_DARK)
        _dark_axes(ax)

        tickers = top10["ticker"].tolist()[::-1]
        n       = len(tickers)
        y_pos   = np.arange(n)

        from config import WEIGHT_MATRIX
        weights = WEIGHT_MATRIX.get(
            (self.profile.risk_level, self.profile.time_horizon),
            [1/7]*7
        )

        left = np.zeros(n)
        handles = []
        for i, factor in enumerate(FACTOR_NAMES):
            col = f"{factor}_score"
            if col not in top10.columns:
                continue
            vals = (top10[col] * weights[i]).tolist()[::-1]
            color = FACTOR_COLORS.get(factor, "#aaa")
            bars  = ax.barh(y_pos, vals, left=left, color=color,
                            edgecolor=BG_DARK, height=0.62, label=factor.capitalize())
            left += np.array(vals)
            handles.append(mpatches.Patch(color=color, label=factor.capitalize()))

        # Composite score label at bar end
        scores = top10["composite_score"].tolist()[::-1]
        for yi, score in zip(y_pos, scores):
            ax.text(min(score + 0.8, 98), yi, f"{score:.1f}",
                    va="center", ha="left", fontsize=9,
                    color=TXT_WHITE, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tickers, color=TXT_WHITE, fontsize=10)
        ax.set_xlim(0, 108)
        ax.set_xlabel("Weighted Score Contribution  (0–100)", color=TXT_DIM, fontsize=10)
        ax.set_title(
            f"Top 10 — Factor Breakdown   ({self.profile.risk_label}  ·  "
            f"{HORIZON_LABELS[self.profile.time_horizon]})",
            color=TXT_WHITE, fontsize=12, fontweight="bold", pad=12,
        )
        ax.legend(handles=handles, loc="lower right", facecolor="#1c2128",
                  labelcolor=TXT_WHITE, edgecolor=BORDER, fontsize=8,
                  ncol=4, framealpha=0.9)
        ax.grid(axis="x", alpha=0.12, color="white")
        ax.set_axisbelow(True)
        fig.tight_layout(pad=1.5)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 2 — Normalised performance vs S&P 500
    # ─────────────────────────────────────────────────────────────────────────
    def performance(self, top10: pd.DataFrame, universe_data: Dict) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(BG_DARK)
        _dark_axes(ax)

        cmap       = plt.colormaps["tab10"]
        start_date = None

        if self.sp500 is not None and len(self.sp500) > 0:
            sp_close = DataFetcher.strip_tz(self.sp500["Close"].dropna())
            start_date = sp_close.index[0]
            sp_norm    = sp_close / sp_close.iloc[0] * 100
            sp_ret     = (sp_close.iloc[-1] / sp_close.iloc[0] - 1) * 100
            ax.plot(sp_norm.index, sp_norm.values, color="white",
                    linewidth=2.8, zorder=10, label=f"S&P 500  ({sp_ret:+.1f}%)")

        ax.axhline(100, color="#444", linestyle="--", linewidth=1, alpha=0.7)

        for i, (_, row) in enumerate(top10.iterrows()):
            t = row["ticker"]
            if t not in universe_data:
                continue
            close = DataFetcher.strip_tz(universe_data[t]["history"]["Close"].dropna())
            if start_date is not None:
                close = close[close.index >= start_date]
            if len(close) < 10:
                continue

            norm    = close / close.iloc[0] * 100
            tot_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
            color   = cmap(i / 10)

            ax.plot(norm.index, norm.values, color=color, linewidth=1.6,
                    alpha=0.88, label=f"{t}  ({tot_ret:+.1f}%)")
            ax.annotate(t, xy=(norm.index[-1], float(norm.values[-1])),
                        xytext=(5, 0), textcoords="offset points",
                        color=color, fontsize=8, va="center")

        ax.set_title(
            f"Historical Performance vs S&P 500  —  {HORIZON_LABELS[self.profile.time_horizon]}",
            color=TXT_WHITE, fontsize=12, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Date", color=TXT_DIM)
        ax.set_ylabel("Normalised Price  (Base = 100)", color=TXT_DIM)
        ax.legend(loc="upper left", facecolor="#1c2128", labelcolor=TXT_WHITE,
                  edgecolor=BORDER, fontsize=8, ncol=2)
        ax.grid(alpha=0.10, color="white")
        fig.tight_layout(pad=1.5)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 3 — Factor score heatmap (10 × 7)
    # ─────────────────────────────────────────────────────────────────────────
    def factor_heatmap(self, top10: pd.DataFrame) -> plt.Figure:
        score_cols = [f"{f}_score" for f in FACTOR_NAMES]
        available  = [c for c in score_cols if c in top10.columns]
        labels     = [c.replace("_score", "").capitalize() for c in available]

        data = top10[available].values       # (10, 7)
        tickers = top10["ticker"].tolist()

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_PANEL)

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg", ["#da3633", "#e3b341", "#3fb950"], N=256
        )
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, color=TXT_WHITE, fontsize=10)
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(
            [f"#{i+1}  {t}" for i, t in enumerate(tickers)],
            color=TXT_WHITE, fontsize=9
        )
        ax.tick_params(axis="x", top=True, bottom=False,
                       labeltop=True, labelbottom=False, colors=TXT_WHITE)
        ax.tick_params(axis="y", colors=TXT_WHITE)

        # Annotate cells
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                v = data[r, c]
                txt_color = "black" if v > 55 else TXT_WHITE
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Score (0–100)", shrink=0.8,
                     pad=0.02).ax.yaxis.label.set_color(TXT_DIM)
        ax.set_title("Factor Score Heatmap — Top 10 Stocks",
                     color=TXT_WHITE, fontsize=12, fontweight="bold", pad=14)
        fig.tight_layout(pad=1.5)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 4 — Macro dashboard (2×2 grid)
    # ─────────────────────────────────────────────────────────────────────────
    def macro_dashboard(self, top10: pd.DataFrame, universe_data: Dict) -> plt.Figure:
        fig = plt.figure(figsize=(14, 9))
        fig.patch.set_facecolor(BG_DARK)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])   # VIX trend
        ax2 = fig.add_subplot(gs[0, 1])   # 10Y yield trend
        ax3 = fig.add_subplot(gs[1, 0])   # Sector ETF 3-month returns
        ax4 = fig.add_subplot(gs[1, 1])   # Top-10 pairwise correlation

        for ax in (ax1, ax2, ax3, ax4):
            _dark_axes(ax)

        # ── VIX ───────────────────────────────────────────────────────────────
        vix_hist = self.macro_data.get("vix_hist")
        if vix_hist is not None and len(vix_hist) > 0:
            vix_s = DataFetcher.strip_tz(vix_hist["Close"])
            color = "#da3633" if (self.macro_data.get("vix") or 20) > 25 else "#3fb950"
            ax1.plot(vix_s.index, vix_s.values, color=color, linewidth=1.5)
            ax1.axhline(20, color="#555", linestyle="--", linewidth=0.8)
            ax1.set_title("VIX (Fear Index)", color=TXT_WHITE, fontsize=10)
            ax1.set_ylabel("VIX Level", color=TXT_DIM, fontsize=8)
        else:
            ax1.text(0.5, 0.5, "VIX data\nunavailable",
                     ha="center", va="center", color=TXT_DIM, transform=ax1.transAxes)

        # ── 10Y Yield ─────────────────────────────────────────────────────────
        y_hist = self.macro_data.get("yield_hist")
        if y_hist is not None and len(y_hist) > 0:
            y_s   = DataFetcher.strip_tz(y_hist["Close"])
            dy    = float(y_s.iloc[-1]) - float(y_s.iloc[0])
            color = "#da3633" if dy > 0 else "#3fb950"
            ax2.plot(y_s.index, y_s.values, color=color, linewidth=1.5)
            ax2.set_title("10-Year Treasury Yield", color=TXT_WHITE, fontsize=10)
            ax2.set_ylabel("Yield (%)", color=TXT_DIM, fontsize=8)
        else:
            ax2.text(0.5, 0.5, "Yield data\nunavailable",
                     ha="center", va="center", color=TXT_DIM, transform=ax2.transAxes)

        # ── Sector ETF returns ────────────────────────────────────────────────
        etf_perf = self.macro_data.get("sector_etf", {})
        if etf_perf:
            sorted_etf = sorted(etf_perf.items(), key=lambda x: x[1], reverse=True)
            sectors    = [s for s, _ in sorted_etf]
            returns    = [r for _, r in sorted_etf]
            colors     = ["#3fb950" if r >= 0 else "#da3633" for r in returns]
            y_pos      = np.arange(len(sectors))
            ax3.barh(y_pos, returns, color=colors, edgecolor=BORDER, height=0.6)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([s[:3].upper() + " " + s[:10]
                                 for s in sectors], color=TXT_WHITE, fontsize=7)
            ax3.axvline(0, color="#555", linewidth=0.8)
            ax3.set_title("Sector ETF 3-Month Returns", color=TXT_WHITE, fontsize=10)
            ax3.set_xlabel("Return (%)", color=TXT_DIM, fontsize=8)
        else:
            ax3.text(0.5, 0.5, "Sector data\nunavailable",
                     ha="center", va="center", color=TXT_DIM, transform=ax3.transAxes)

        # ── Correlation heatmap of selected stocks ────────────────────────────
        returns_dict = {}
        for _, row in top10.iterrows():
            t = row["ticker"]
            if t in universe_data:
                close = DataFetcher.strip_tz(
                    universe_data[t]["history"]["Close"].dropna()
                )
                returns_dict[t] = close.pct_change().dropna()

        if len(returns_dict) >= 3:
            ret_df   = pd.DataFrame(returns_dict).dropna()
            corr_mat = ret_df.corr()
            tickers  = corr_mat.columns.tolist()

            cmap2 = mcolors.LinearSegmentedColormap.from_list(
                "corr", ["#3fb950", "#ffffff", "#da3633"], N=256
            )
            im = ax4.imshow(corr_mat.values, cmap=cmap2, vmin=-1, vmax=1, aspect="auto")
            ax4.set_xticks(range(len(tickers)))
            ax4.set_yticks(range(len(tickers)))
            ax4.set_xticklabels(tickers, color=TXT_WHITE, fontsize=7, rotation=45, ha="right")
            ax4.set_yticklabels(tickers, color=TXT_WHITE, fontsize=7)
            ax4.set_title("Top-10 Return Correlation", color=TXT_WHITE, fontsize=10)
        else:
            ax4.text(0.5, 0.5, "Insufficient data\nfor correlation",
                     ha="center", va="center", color=TXT_DIM, transform=ax4.transAxes)

        regime   = self.macro_data.get("regime", "neutral").upper()
        reasons  = "  |  ".join(self.macro_data.get("regime_reasons", []))
        fig.suptitle(f"Macro Dashboard  —  Regime: {regime}   {reasons}",
                     color=TXT_WHITE, fontsize=11, fontweight="bold", y=0.98)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Chart 5 — Quantitative protocol report (gates + entry + quant conviction)
    # ─────────────────────────────────────────────────────────────────────────
    def thought_process(self, top10: pd.DataFrame,
                        protocol_results: list,
                        valuation_results: dict = None,
                        risk_results: dict = None) -> plt.Figure:
        """
        Three-panel quantitative protocol report:
          ┌─────────────────────┬────────────────────┐
          │  Protocol Gate      │  Entry Price       │
          │  Scorecard          │  Positioning       │
          │  (7 gates × stocks) │  (% vs fair value) │
          ├─────────────────────┴────────────────────┤
          │  Conviction + entry prices + quant thesis │
          └───────────────────────────────────────────┘
        """
        from advisor.protocol import GATE_SHORT, PASS_THRESHOLD, WARN_THRESHOLD

        tickers   = top10["ticker"].tolist()
        n_stocks  = len(tickers)
        n_gates   = 7
        proto_map = {p["ticker"]: p for p in protocol_results}
        val_map   = valuation_results or {}
        risk_map  = risk_results or {}

        fig = plt.figure(figsize=(17, 12))
        fig.patch.set_facecolor(BG_DARK)

        # Outer grid: top 62% + bottom 38%
        gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                                     height_ratios=[0.62, 0.38], hspace=0.06)
        # Top: gate grid (left 57%) + entry chart (right 43%)
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_outer[0], wspace=0.04, width_ratios=[0.57, 0.43]
        )
        ax_gates   = fig.add_subplot(gs_top[0])
        ax_entry   = fig.add_subplot(gs_top[1])
        ax_verdict = fig.add_subplot(gs_outer[1])

        for ax in (ax_gates, ax_entry):
            _dark_axes(ax)

        # ── Panel A: Protocol gate scorecard ─────────────────────────────────
        gate_matrix = np.full((n_stocks, n_gates), 50.0)
        for i, t in enumerate(tickers):
            gates = proto_map.get(t, {}).get("gates", [])
            for j, g in enumerate(gates[:n_gates]):
                gate_matrix[i, j] = float(g)

        cmap_proto = mcolors.LinearSegmentedColormap.from_list(
            "protocol", ["#da3633", "#e3b341", "#3fb950"], N=256
        )
        im = ax_gates.imshow(gate_matrix, cmap=cmap_proto,
                             vmin=0, vmax=100, aspect="auto")

        for ri in range(n_stocks):
            for ci in range(n_gates):
                v         = gate_matrix[ri, ci]
                txt_color = "#000000" if 40 <= v <= 80 else TXT_WHITE
                ax_gates.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                              fontsize=8, color=txt_color, fontweight="bold")

        # Pass/Fail summary at right edge of each row
        for i, t in enumerate(tickers):
            p = proto_map.get(t, {})
            pc, fc = p.get("pass_count", 0), p.get("fail_count", 0)
            col = "#3fb950" if fc == 0 else "#e3b341" if fc <= 2 else "#da3633"
            ax_gates.text(n_gates + 0.05, i, f"{pc}P {fc}F",
                          ha="left", va="center", fontsize=7.5, color=col,
                          transform=ax_gates.get_xaxis_transform())

        ax_gates.set_xticks(range(n_gates))
        ax_gates.set_xticklabels(GATE_SHORT, color=TXT_WHITE, fontsize=9,
                                 rotation=30, ha="right")
        ax_gates.tick_params(axis="x", top=True, bottom=False,
                             labeltop=True, labelbottom=False, colors=TXT_WHITE)
        ax_gates.set_yticks(range(n_stocks))
        ax_gates.set_yticklabels(
            [f"#{i+1}  {t}" for i, t in enumerate(tickers)],
            color=TXT_WHITE, fontsize=9
        )
        ax_gates.set_title("PROTOCOL GATE SCORECARD  (7 Gates · 0–100)",
                           color=TXT_WHITE, fontsize=11, fontweight="bold", pad=12)

        # Colorbar legend
        cbar = plt.colorbar(im, ax=ax_gates, shrink=0.55, pad=0.16, aspect=18)
        cbar.set_label("Gate Score", color=TXT_DIM, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=TXT_DIM, labelcolor=TXT_DIM, labelsize=7)
        for marker in [35, 60]:
            cbar.ax.axhline(marker, color="#555", linewidth=0.8)

        # ── Panel B: Entry price positioning ─────────────────────────────────
        SIGNAL_COLORS = {
            "STRONG_BUY":        "#3fb950",
            "BUY":               "#79c0ff",
            "HOLD_WATCH":        "#e3b341",
            "WAIT":              "#f78166",
            "AVOID_PEAK":        "#da3633",
            "INSUFFICIENT_DATA": "#8b949e",
        }

        # Shaded zones (% vs fair value on x-axis)
        ax_entry.axvline(0,   color=TXT_WHITE,  linewidth=1.2, alpha=0.5, linestyle="--")
        ax_entry.axvline(-20, color="#3fb950", linewidth=0.8, alpha=0.4, linestyle=":")
        ax_entry.axvspan(-40, -20, alpha=0.07, color="#3fb950")  # strong buy
        ax_entry.axvspan(-20,   0, alpha=0.07, color="#79c0ff")  # buy
        ax_entry.axvspan(  0,  10, alpha=0.07, color="#e3b341")  # watch
        ax_entry.axvspan( 10,  45, alpha=0.05, color="#da3633")  # expensive

        for i, t in enumerate(tickers):
            # Prefer ValuationEngine entry_analysis, fall back to protocol
            ea = val_map.get(t, {})
            if not ea.get("fair_value"):
                ea = proto_map.get(t, {}).get("entry_analysis", {})
            else:
                # Map ValuationEngine fields to entry_analysis format
                ea = {
                    "fair_value":    ea.get("fair_value"),
                    "entry_target":  ea.get("entry_low"),
                    "current_price": ea.get("current_price"),
                    "signal":        ea.get("signal", "INSUFFICIENT_DATA"),
                    "premium_pct":   ea.get("premium_pct"),
                }

            prem   = ea.get("premium_pct")
            signal = ea.get("signal", "INSUFFICIENT_DATA")
            color  = SIGNAL_COLORS.get(signal, "#8b949e")
            y      = n_stocks - 1 - i        # flip so rank 1 is at top

            if prem is not None:
                bar_l = min(-20.0, prem)
                bar_r = prem
                ax_entry.barh(y, bar_r - bar_l, left=bar_l,
                              color=color, alpha=0.30, height=0.6)
                ax_entry.scatter([prem], [y], color=color, s=70, zorder=5)
                lbl = f"${ea.get('current_price','?')} ({prem:+.0f}%)"
                ax_entry.text(prem + 0.8, y, lbl, va="center", ha="left",
                              color=color, fontsize=7.5)
            else:
                ax_entry.scatter([0], [y], color="#8b949e", s=50, marker="x")
                ax_entry.text(1, y, "No fair value data", va="center",
                              color="#8b949e", fontsize=7.5)

        ax_entry.set_yticks(range(n_stocks))
        ax_entry.set_yticklabels([t for t in tickers[::-1]],
                                 color=TXT_WHITE, fontsize=9)
        ax_entry.set_xlim(-42, 50)
        ax_entry.set_xlabel("% vs Fair Value  (0 = Fair Value  |  negative = discount)",
                            color=TXT_DIM, fontsize=9)
        ax_entry.set_title("ENTRY PRICE POSITIONING\n(DCF · Graham · EV/EBITDA · FCF Yield median)",
                           color=TXT_WHITE, fontsize=11, fontweight="bold", pad=8)

        # Zone annotations at top
        top_y = n_stocks - 0.45
        for x, lbl, col in [(-30, "STRONG BUY", "#3fb950"), (-10, "BUY", "#79c0ff"),
                             (5, "WATCH", "#e3b341"), (27, "EXPENSIVE", "#da3633")]:
            ax_entry.text(x, top_y, lbl, ha="center", va="center",
                          fontsize=7, color=col, alpha=0.85)
        ax_entry.grid(axis="x", alpha=0.10, color="white")

        # ── Panel C: Quantitative conviction + entry prices + thesis ──────────
        ax_verdict.set_facecolor(BG_PANEL)
        for sp in ax_verdict.spines.values():
            sp.set_edgecolor(BORDER)
        ax_verdict.set_xticks([])
        ax_verdict.set_yticks([])

        CONV_COLORS = {
            "HIGH":   "#3fb950",
            "MEDIUM": "#e3b341",
            "LOW":    "#f78166",
            "AVOID":  "#da3633",
        }

        verdict_ax = ax_verdict.transAxes
        n_rows = n_stocks
        row_h  = 0.88 / n_rows
        y_top  = 0.94

        for i, t in enumerate(tickers):
            proto = proto_map.get(t, {})
            val   = val_map.get(t, {})
            risk  = risk_map.get(t, {})
            conv  = proto.get("conviction", "MEDIUM")
            color = CONV_COLORS.get(conv, TXT_DIM)

            # Prefer ValuationEngine signal over protocol signal
            ea  = proto.get("entry_analysis", {})
            sig = val.get("signal") or ea.get("signal", "")
            sig_c = SIGNAL_COLORS.get(sig, TXT_DIM)

            y_c = y_top - i * row_h - row_h * 0.5

            # Conviction badge
            badge = mpatches.FancyBboxPatch(
                (0.002, y_c - row_h * 0.38), 0.065, row_h * 0.76,
                boxstyle="round,pad=0.005", transform=verdict_ax,
                facecolor=color, edgecolor="none", alpha=0.85, clip_on=False
            )
            ax_verdict.add_patch(badge)
            ax_verdict.text(0.035, y_c, conv[:4], ha="center", va="center",
                           fontsize=7, fontweight="bold", color="black",
                           transform=verdict_ax)

            # Ticker + valuation signal
            ax_verdict.text(0.075, y_c, f"#{i+1} {t}",
                           ha="left", va="center", fontsize=9,
                           fontweight="bold", color=color, transform=verdict_ax)
            ax_verdict.text(0.175, y_c, f"[{sig}]",
                           ha="left", va="center", fontsize=8,
                           color=sig_c, transform=verdict_ax)

            # Entry price (prefer ValuationEngine)
            fv  = val.get("fair_value") or ea.get("fair_value")
            ent = val.get("entry_low")  or ea.get("entry_target")
            cur = val.get("current_price") or ea.get("current_price")
            if fv and ent and cur:
                price_info = f"FV ${fv:,.0f}  →  Entry ${ent:,.0f}  (now ${cur:,.0f})"
            elif fv and cur:
                price_info = f"FV ${fv:,.0f}  (now ${cur:,.0f})"
            else:
                price_info = "Entry: insufficient valuation data"
            ax_verdict.text(0.345, y_c, price_info,
                           ha="left", va="center", fontsize=7.5,
                           color=TXT_DIM, transform=verdict_ax)

            # Auto-generated quantitative thesis from risk/valuation data
            thesis_parts = []
            if val.get("methods_count", 0) >= 2:
                prem = val.get("premium_pct", 0) or 0
                thesis_parts.append(
                    f"{abs(prem):.0f}% {'above' if prem > 0 else 'below'} FV"
                    f" ({val.get('methods_count',0)}-method median)"
                )
            rw = risk.get("roic_wacc", {})
            if rw.get("spread") is not None:
                thesis_parts.append(f"ROIC/WACC {rw['spread']:+.1f}% [{rw.get('verdict','')}]")
            piotr = risk.get("piotroski", {})
            if piotr.get("score") is not None:
                thesis_parts.append(f"Piotroski {piotr['score']}/9")
            az = risk.get("altman_z", {})
            if az.get("zone"):
                thesis_parts.append(f"Altman Z [{az['zone']}]")
            thesis = "  ·  ".join(thesis_parts) if thesis_parts else "Quantitative data pending"
            if len(thesis) > 100:
                thesis = thesis[:97] + "..."
            ax_verdict.text(0.62, y_c, thesis,
                           ha="left", va="center", fontsize=8,
                           color=TXT_DIM, transform=verdict_ax)

        # Column headers
        for x, lbl in [(0.035, "CONV"), (0.120, "TICKER / SIGNAL"),
                       (0.345, "ENTRY ANALYSIS"), (0.620, "QUANTITATIVE THESIS")]:
            ax_verdict.text(x, 0.97, lbl, ha="left", va="center",
                           fontsize=7, color=TXT_DIM, fontweight="bold",
                           transform=verdict_ax)

        fig.suptitle(
            "QUANTITATIVE INVESTMENT PROTOCOL REPORT",
            color=TXT_WHITE, fontsize=14, fontweight="bold", y=0.993
        )
        ax_verdict.set_title(
            "7-Gate Protocol  ·  DCF · Graham · EV/EBITDA · FCF Yield  ·  "
            "ROIC/WACC · Piotroski · Altman Z  ·  No AI APIs required",
            color=TXT_DIM, fontsize=8, pad=6
        )

        fig.tight_layout(rect=[0, 0, 1, 0.992], pad=1.0)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Save all
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Chart 6 — Candlestick chart (dark theme, with volume + SMA + RSI)
    # ─────────────────────────────────────────────────────────────────────────
    def candlestick(self, ticker: str, history: pd.DataFrame,
                    period: str = "6mo") -> plt.Figure:
        """
        Dark-theme OHLCV candlestick chart with:
          - SMA 20 (amber), SMA 50 (blue), SMA 200 (red dashed)
          - Volume panel (green/red bars)
          - RSI(14) panel with overbought/oversold bands
        Uses mplfinance if available, falls back to pure matplotlib.
        """
        if history is None or history.empty:
            fig, ax = plt.subplots()
            fig.patch.set_facecolor(BG_DARK)
            ax.set_facecolor(BG_PANEL)
            ax.text(0.5, 0.5, "No data available", color=TXT_DIM,
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        # Trim to requested period
        period_days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252,
                       "2y": 504, "5y": 1260}
        days  = period_days.get(period, 126)
        df    = history.tail(days).copy()
        df    = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # Indicators
        df["sma20"]  = df["Close"].rolling(20).mean()
        df["sma50"]  = df["Close"].rolling(50).mean()
        df["sma200"] = df["Close"].rolling(200).mean()
        delta        = df["Close"].diff()
        gain         = delta.clip(lower=0).rolling(14).mean()
        loss         = (-delta.clip(upper=0)).rolling(14).mean()
        rs           = gain / loss
        df["rsi"]    = 100 - (100 / (1 + rs))

        try:
            import mplfinance as mpf

            mc    = mpf.make_marketcolors(
                up="#3fb950", down="#da3633",
                wick={"up": "#3fb950", "down": "#da3633"},
                volume={"up": "#3fb950", "down": "#da3633"},
                edge={"up": "#3fb950", "down": "#da3633"},
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                facecolor=BG_PANEL, figcolor=BG_DARK,
                gridcolor=BORDER, gridstyle="--",
                y_on_right=True,
                rc={
                    "axes.labelcolor":  TXT_DIM,
                    "axes.edgecolor":   BORDER,
                    "xtick.color":      TXT_DIM,
                    "ytick.color":      TXT_DIM,
                    "figure.titlesize": 13,
                },
            )
            add_plots = [
                mpf.make_addplot(df["sma20"],  color="#e3b341", width=1.2,
                                 label="SMA 20"),
                mpf.make_addplot(df["sma50"],  color="#79c0ff", width=1.2,
                                 label="SMA 50"),
                mpf.make_addplot(df["sma200"], color="#f78166", width=1.2,
                                 linestyle="--", label="SMA 200"),
                mpf.make_addplot(df["rsi"], panel=2, color="#d2a8ff",
                                 width=1.5, ylabel="RSI(14)"),
            ]
            fig, axes = mpf.plot(
                df, type="candle", style=style,
                volume=True, addplot=add_plots,
                returnfig=True, figsize=(14, 9),
                panel_ratios=(3, 1, 1),
                title=f"\n{ticker} — Candlestick  ({period})",
            )
            fig.patch.set_facecolor(BG_DARK)
            return fig

        except ImportError:
            # Fallback: pure matplotlib candlestick
            return self._candlestick_fallback(ticker, df, period)

    def _candlestick_fallback(self, ticker: str, df: pd.DataFrame,
                               period: str) -> plt.Figure:
        """Pure-matplotlib fallback when mplfinance is not installed."""
        fig = plt.figure(figsize=(14, 9), facecolor=BG_DARK)
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.06,
                                height_ratios=[3, 1, 1])
        ax1 = fig.add_subplot(gs[0])   # price
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # volume
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # RSI

        for ax in (ax1, ax2, ax3):
            _dark_axes(ax)
            ax.tick_params(labelbottom=False)
        ax3.tick_params(labelbottom=True)

        x = np.arange(len(df))
        w = 0.6
        for i, (_, row) in enumerate(df.iterrows()):
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            color = "#3fb950" if c >= o else "#da3633"
            ax1.plot([x[i], x[i]], [l, h], color=color, linewidth=0.8)
            ax1.bar(x[i], abs(c - o), w, bottom=min(o, c), color=color)

        for col, style, lbl in [
            ("sma20",  {"color": "#e3b341", "lw": 1.2}, "SMA 20"),
            ("sma50",  {"color": "#79c0ff", "lw": 1.2}, "SMA 50"),
            ("sma200", {"color": "#f78166", "lw": 1.2, "ls": "--"}, "SMA 200"),
        ]:
            vals = df[col].values
            ax1.plot(x, vals, label=lbl, **style)

        ax1.legend(loc="upper left", fontsize=8, facecolor=BG_PANEL,
                   labelcolor=TXT_DIM, framealpha=0.7)
        ax1.set_title(f"{ticker} — Candlestick  ({period})",
                      color=TXT_WHITE, fontsize=12, pad=8)
        ax1.set_ylabel("Price ($)", color=TXT_DIM, fontsize=9)

        vol_colors = ["#3fb950" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                      else "#da3633" for i in range(len(df))]
        ax2.bar(x, df["Volume"].values, width=w, color=vol_colors, alpha=0.7)
        ax2.set_ylabel("Volume", color=TXT_DIM, fontsize=8)

        rsi_vals = df["rsi"].values
        ax3.plot(x, rsi_vals, color="#d2a8ff", linewidth=1.5)
        ax3.axhline(70, color="#da3633", linewidth=0.8, linestyle="--")
        ax3.axhline(30, color="#3fb950", linewidth=0.8, linestyle="--")
        ax3.fill_between(x, rsi_vals, 70, where=(rsi_vals >= 70),
                         alpha=0.15, color="#da3633")
        ax3.fill_between(x, rsi_vals, 30, where=(rsi_vals <= 30),
                         alpha=0.15, color="#3fb950")
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI(14)", color=TXT_DIM, fontsize=8)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        fig.tight_layout()
        return fig

    def save_all(self, figs: list):
        names = [
            "chart1_score_breakdown.png",
            "chart2_performance.png",
            "chart3_factor_heatmap.png",
            "chart4_macro_dashboard.png",
            "chart5_quant_protocol.png",
        ]
        for fig, name in zip(figs, names):
            try:
                fig.savefig(name, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            except Exception as e:
                print(f"  Warning: could not save {name} — {e}")
        saved = [n for n in names if os.path.exists(n)]
        print(f"  Charts saved: {' | '.join(saved)}")
