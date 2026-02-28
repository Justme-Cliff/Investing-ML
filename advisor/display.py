# advisor/display.py — All terminal output (rich-enhanced, plain fallback)

from typing import List, Optional
import pandas as pd

from config import FACTOR_NAMES, HORIZON_LABELS
from advisor.collector import UserProfile

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box as rich_box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None


def _p(msg: str = ""):
    print(msg)


class TerminalDisplay:

    # ── Welcome + track record ────────────────────────────────────────────────
    def show_welcome(self, track_record: List[dict]):
        _p("""
╔══════════════════════════════════════════════════════════════════╗
║         PERSONALIZED  STOCK  RANKING  ADVISOR  v2              ║
║  7-factor model · macro regime · adaptive learning · Excel     ║
╚══════════════════════════════════════════════════════════════════╝""")

        if not track_record:
            _p("  No previous sessions on record yet — this is your first run!\n")
            return

        _p(f"\n  TRACK RECORD  ({len(track_record)} evaluated session(s))")
        _p("  " + "─" * 60)
        wins = 0
        for s in track_record[:5]:
            ev      = s.get("evaluation", {})
            date    = s["timestamp"][:10]
            picks   = ", ".join(p["ticker"] for p in s["picks"][:5])
            avg_ret = ev.get("avg_pick_return")
            alpha   = ev.get("alpha")
            if avg_ret is not None:
                flag = "✓ BEAT S&P" if (alpha or 0) > 0 else "✗ lagged"
                if (alpha or 0) > 0:
                    wins += 1
                _p(f"  {date}  |  {picks}...  |  avg {avg_ret*100:+.1f}%  |  alpha {alpha*100:+.1f}%  {flag}")
        _p(f"\n  Win rate: {wins}/{min(len(track_record),5)} sessions beat the S&P 500\n")

    # ── Macro regime ──────────────────────────────────────────────────────────
    def show_macro(self, macro_data: dict):
        regime  = macro_data.get("regime", "neutral").upper()
        reasons = macro_data.get("regime_reasons", [])
        vix     = macro_data.get("vix")
        y10     = macro_data.get("yield_10y")

        _p("─" * 66)
        _p(f"  CURRENT MACRO REGIME:  {regime}")
        _p("─" * 66)
        if vix:   _p(f"  VIX              :  {vix:.1f}")
        if y10:   _p(f"  10-Year Yield    :  {y10:.2f}%")
        for r in reasons:
            _p(f"  Signal           :  {r}")
        etf = macro_data.get("sector_etf", {})
        if etf:
            top3 = sorted(etf.items(), key=lambda x: x[1], reverse=True)[:3]
            _p(f"  Leading sectors  :  " + "  ·  ".join(f"{s} ({r:+.1f}%)" for s, r in top3))
        _p("─" * 66 + "\n")

    # ── Results table ─────────────────────────────────────────────────────────
    def show_results(self, top10: pd.DataFrame):
        W = 96
        _p("\n" + "═" * W)
        _p("  TOP 10 STOCKS FOR YOUR PROFILE")
        _p("═" * W)
        hdr = (f"  {'#':<4} {'Ticker':<7} {'Sector':<14} {'Score':>6}"
               f" {'Mom':>6} {'Vol':>6} {'Val':>6} {'Qual':>6}"
               f" {'Tech':>6} {'Sent':>6} {'Div%':>6}  {'Price':>8}")
        _p(hdr)
        _p("─" * W)

        factor_cols = [f"{f}_score" for f in FACTOR_NAMES]
        for _, row in top10.iterrows():
            scores = []
            for fc in factor_cols:
                scores.append(f"{row.get(fc, 0):>6.0f}")
            _p(
                f"  {int(row['rank']):<4}"
                f" {row['ticker']:<7}"
                f" {row['sector']:<14}"
                f" {row['composite_score']:>6.1f}"
                + "".join(scores) +
                f"  {row.get('div_pct', 0):>5.1f}%"
                f"  ${row['current_price']:>8,.2f}"
            )
        _p("═" * W)
        _p("  Columns: Mom=Momentum  Vol=Volatility  Val=Value  Qual=Quality"
           "  Tech=Technical  Sent=Sentiment  Div=Dividend\n")

    # ── Allocation table ──────────────────────────────────────────────────────
    def show_allocation(self, top10: pd.DataFrame, portfolio_size: float):
        _p("─" * 70)
        _p(f"  PORTFOLIO ALLOCATION   (${portfolio_size:,.0f} total  ·  Half-Kelly sizing)")
        _p("─" * 70)
        for _, row in top10.iterrows():
            w       = float(row.get("weight", 0))
            dollars = float(row.get("dollar_amount", 0))
            shares  = row.get("approx_shares", "?")
            price   = float(row["current_price"])
            _p(
                f"  #{int(row['rank']):<2}  {row['ticker']:<6}  │"
                f"  {w*100:>5.1f}%  │"
                f"  ${dollars:>9,.0f}  │"
                f"  ~{shares} shares  @  ${price:,.2f}"
            )
        _p("─" * 70 + "\n")

    # ── Adapted weights notice ────────────────────────────────────────────────
    def show_weight_adaptation(self, adapted: Optional[list]):
        if adapted is None:
            return
        from config import FACTOR_NAMES
        _p("  [Learning] Weights adapted from past session data:")
        for name, w in zip(FACTOR_NAMES, adapted):
            bar = "█" * int(w * 100 / 5)
            _p(f"    {name:<12} {w*100:>5.1f}%  {bar}")
        _p()

    # ── Disclaimer ────────────────────────────────────────────────────────────
    def show_disclaimer(self):
        _p("\n" + "─" * 66)
        _p("  DISCLAIMER")
        _p("─" * 66)
        _p("  This tool is for educational & informational purposes only.")
        _p("  Past performance does not guarantee future results.")
        _p("  All rankings are quantitative — not financial advice.")
        _p("  Always conduct your own due diligence before investing.")
        _p("─" * 66 + "\n")
