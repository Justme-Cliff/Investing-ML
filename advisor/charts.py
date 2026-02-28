# advisor/charts.py — 4 dark-theme matplotlib charts

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

        cmap       = plt.cm.get_cmap("tab10")
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
            ax3.set_yticklabels([SECTOR_COLORS.get(s, s)[:3] + " " + s[:10]
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
    # Save all
    # ─────────────────────────────────────────────────────────────────────────
    def save_all(self, figs: list):
        names = [
            "chart1_score_breakdown.png",
            "chart2_performance.png",
            "chart3_factor_heatmap.png",
            "chart4_macro_dashboard.png",
        ]
        for fig, name in zip(figs, names):
            try:
                fig.savefig(name, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            except Exception as e:
                print(f"  Warning: could not save {name} — {e}")
        saved = [n for n in names if os.path.exists(n)]
        print(f"  Charts saved: {' | '.join(saved)}")
