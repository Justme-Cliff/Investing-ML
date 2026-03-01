# advisor/display.py — Terminal output: results, protocol, deep quant analysis

from typing import List, Optional
import pandas as pd

from config import FACTOR_NAMES, HORIZON_LABELS
from advisor.collector import UserProfile


def _p(msg: str = ""):
    print(msg)


class TerminalDisplay:

    # ── Welcome + track record ────────────────────────────────────────────────
    def show_welcome(self, track_record: List[dict]):
        _p("""
╔══════════════════════════════════════════════════════════════════╗
║      QUANTITATIVE STOCK ADVISOR  —  HEDGE-FUND GRADE MATH       ║
║  DCF · Graham · EV/EBITDA · FCF Yield · Altman Z · ROIC/WACC   ║
║  Accruals · Gross Profitability · Piotroski 9pt · Sortino/VaR   ║
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
        _p(f"\n  Win rate: {wins}/{min(len(track_record), 5)} sessions beat the S&P 500\n")

    # ── Macro regime ──────────────────────────────────────────────────────────
    def show_macro(self, macro_data: dict):
        regime  = macro_data.get("regime", "neutral").upper()
        reasons = macro_data.get("regime_reasons", [])
        vix     = macro_data.get("vix")
        y10     = macro_data.get("yield_10y")

        _p("─" * 70)
        _p(f"  CURRENT MACRO REGIME:  {regime}")
        _p("─" * 70)
        if vix: _p(f"  VIX              :  {vix:.1f}")
        if y10: _p(f"  10-Year Yield    :  {y10:.2f}%")
        for r in reasons:
            _p(f"  Signal           :  {r}")
        etf = macro_data.get("sector_etf", {})
        if etf:
            top3 = sorted(etf.items(), key=lambda x: x[1], reverse=True)[:3]
            _p(f"  Leading sectors  :  " + "  ·  ".join(f"{s} ({r:+.1f}%)" for s, r in top3))
        _p("─" * 70 + "\n")

    # ── Quick results table ───────────────────────────────────────────────────
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
            scores = [f"{row.get(fc, 0):>6.0f}" for fc in factor_cols]
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
        _p("  Mom=Momentum  Vol=Volatility  Val=Value  Qual=Quality"
           "  Tech=Technical  Sent=Sentiment\n")

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

    # ── Deep quantitative analysis ────────────────────────────────────────────
    def show_deep_analysis(self, top10: pd.DataFrame,
                           valuation_results: dict,
                           risk_results: dict,
                           universe_data: dict = None):
        W = 100
        _p("\n" + "█" * W)
        _p("  DEEP QUANTITATIVE ANALYSIS  (Hedge-Fund Grade Math)")
        _p("█" * W)

        for _, row in top10.iterrows():
            t      = row["ticker"]
            sector = row.get("sector", "?")
            score  = row.get("composite_score", 0)
            name   = row.get("longName", "")

            val  = valuation_results.get(t, {})
            risk = risk_results.get(t, {})
            udata = (universe_data or {}).get(t, {})

            _p(f"\n  {'─'*98}")
            label = f"#{int(row['rank'])}  {t}"
            if name:
                label += f"  —  {name[:45]}"
            label += f"  |  {sector}  |  Composite: {score:.1f}/100"

            # Earnings warning
            days_away = udata.get("earnings_days_away")
            edate     = udata.get("earnings_date", "")
            if days_away is not None:
                if   days_away <= 7:  warn = f"  ⚠  EARNINGS IN {days_away} DAYS ({edate})  ⚠"
                elif days_away <= 14: warn = f"  !  Earnings in {days_away} days ({edate})"
                elif days_away <= 30: warn = f"  Earnings in {days_away} days ({edate})"
                else:                 warn = ""
                if warn:
                    label += warn

            _p(f"  {label}")
            _p(f"  {'─'*98}")

            # ── Valuation matrix ──────────────────────────────────────────────
            estimates = val.get("estimates", {})
            fv        = val.get("fair_value")
            entry_lo  = val.get("entry_low")
            entry_hi  = val.get("entry_high")
            target    = val.get("target_price")
            stop      = val.get("stop_loss")
            cur       = val.get("current_price", row["current_price"])
            prem      = val.get("premium_pct")
            rr        = val.get("rr_ratio")
            sig       = val.get("signal", "INSUFFICIENT_DATA")
            n_methods = val.get("methods_count", 0)

            _EST_LABELS = {
                "dcf":       "DCF (2-stage)",
                "graham":    "Graham Number",
                "ev_ebitda": "EV/EBITDA",
                "fcf_yield": "FCF Yield @4.5%",
            }
            if estimates:
                _p(f"  ┌── VALUATION  ({n_methods} independent method{'s' if n_methods != 1 else ''})")
                for key, est in estimates.items():
                    method   = _EST_LABELS.get(key, key)
                    diff_pct = (cur - est) / est * 100
                    arrow    = "premium" if diff_pct > 0 else "discount"
                    _p(f"  │  {method:<22}  ${est:>8,.2f}  │  current {diff_pct:+.1f}% {arrow}")
                _p(f"  │  {'─'*60}")
                _p(f"  │  Median Fair Value  :  ${fv:>8,.2f}  │  Entry Zone: ${entry_lo:,.2f} – ${entry_hi:,.2f}")
                if target and stop:
                    upside = val.get("upside_pct", 0) or 0
                    rr_str = f"  │  R/R Ratio: {rr:.1f}:1" if rr else ""
                    _p(f"  │  Target Price      :  ${target:>8,.2f}  │  Stop Loss: ${stop:,.2f}"
                       f"  │  Upside: {upside:+.1f}%{rr_str}")
                sig_icon = {"STRONG_BUY": "✓✓", "BUY": "✓", "HOLD_WATCH": "~",
                            "WAIT": "⏳", "AVOID_PEAK": "✗"}.get(sig, "?")
                _p(f"  └── Signal: {sig_icon} {sig}  "
                   f"(current ${cur:,.2f} is {abs(prem or 0):.1f}% "
                   f"{'above' if (prem or 0) > 0 else 'below'} fair value)")
            else:
                _p(f"  VALUATION: Insufficient data (no EPS/FCF/EBITDA available)")

            # ── DCF Sensitivity ───────────────────────────────────────────────
            sens = val.get("sensitivity", {})
            if sens:
                _p(f"  ┌── DCF SENSITIVITY  (Bear / Base / Bull growth scenarios)")
                for sname, sv in sens.items():
                    fv_s  = f"${sv['fair_value']:,.2f}" if sv.get("fair_value") else "N/A"
                    gr_s  = f"{sv['growth_rate']:+.1f}% growth"
                    sig_s = sv.get("signal", "—")
                    prem_s = f"{sv['premium_pct']:+.1f}% vs FV" if sv.get("premium_pct") is not None else ""
                    _p(f"  │  {sname:<5}  FV {fv_s:<12}  {gr_s:<18}  {sig_s:<14}  {prem_s}")
                _p(f"  └──")

            # ── ROIC / WACC / Quality ─────────────────────────────────────────
            rw     = risk.get("roic_wacc", {})
            piotr  = risk.get("piotroski", {})
            acc    = risk.get("accruals")
            gp     = risk.get("gross_prof")

            roic   = rw.get("roic")
            wacc   = rw.get("wacc")
            spread = rw.get("spread")
            verd   = rw.get("verdict", "")

            piotr_s = piotr.get("score")
            piotr_i = piotr.get("interpretation", "")

            acc_lbl = ""
            if acc is not None:
                if acc < -0.02:
                    acc_lbl = "CLEAN (cash > accounting)"
                elif acc < 0.02:
                    acc_lbl = "NEUTRAL"
                else:
                    acc_lbl = "WARNING (accounting > cash)"

            parts = []
            if roic is not None and spread is not None:
                parts.append(f"ROIC {roic:.1f}%  WACC ~{wacc:.1f}%  Spread {spread:+.1f}%  [{verd}]")
            if piotr_s is not None:
                parts.append(f"Piotroski {piotr_s}/9 [{piotr_i}]")
            if acc is not None:
                parts.append(f"Accruals {acc:+.3f} {acc_lbl}")
            if gp is not None:
                parts.append(f"Gross/Assets {gp:.2f}")
            if parts:
                _p(f"  ┌── QUALITY & VALUE CREATION")
                for part in parts:
                    _p(f"  │  {part}")
                _p(f"  └──")

            # ── Risk profile ──────────────────────────────────────────────────
            az     = risk.get("altman_z", {})
            sharpe = risk.get("sharpe")
            sort   = risk.get("sortino")
            mdd    = risk.get("max_drawdown_pct")
            var95  = risk.get("var_95_pct")
            beta   = float(row.get("beta", 1.0))
            vol_ann = float(row.get("vol", 0)) * 100

            az_score = az.get("score")
            az_zone  = az.get("zone", "UNKNOWN")
            az_icon  = {"SAFE": "✓", "GRAY": "~", "DISTRESS": "✗"}.get(az_zone, "?")

            risk_parts = []
            if az_score is not None:
                risk_parts.append(f"Altman Z {az_score:.1f} [{az_icon} {az_zone}]")
            if sharpe is not None:
                risk_parts.append(f"Sharpe {sharpe:.2f}")
            if sort is not None:
                risk_parts.append(f"Sortino {sort:.2f}")
            if mdd is not None:
                risk_parts.append(f"Max DD {mdd:.1f}%")
            if var95 is not None:
                risk_parts.append(f"VaR(95% 1mo) {var95:.1f}%")
            risk_parts.append(f"Beta {beta:.2f}")
            risk_parts.append(f"Ann. Vol {vol_ann:.1f}%")

            if risk_parts:
                _p(f"  ┌── RISK PROFILE")
                # Print in 2 per line
                for i in range(0, len(risk_parts), 3):
                    _p(f"  │  " + "  ·  ".join(risk_parts[i:i+3]))
                _p(f"  └──")

            # ── Protocol summary ──────────────────────────────────────────────
            _p("")   # breathing room between stocks

        _p("█" * W + "\n")

    # ── Protocol + entry price table ─────────────────────────────────────────
    def show_protocol(self, protocol_results: list):
        W = 100
        _p("\n" + "═" * W)
        _p("  INVESTMENT PROTOCOL  (7 Gates · Margin of Safety · Entry Prices)")
        _p("═" * W)
        _p(f"  {'#':<3} {'Ticker':<7} {'Score':>6} {'Gates':<10} {'Conv':<8}"
           f" {'Signal':<14} {'Fair Value':>11} {'Entry Zone':>20} {'Current':>9}")
        _p("─" * W)

        CONV_ICON = {"HIGH": "●●●", "MEDIUM": "●● ", "LOW": "●  ", "AVOID": "✗  "}

        for i, p in enumerate(protocol_results, 1):
            ea    = p.get("entry_analysis", {})
            gates = f"{p['pass_count']}P {p['warn_count']}W {p['fail_count']}F"
            sig   = ea.get("signal", "N/A")
            conv  = p.get("conviction", "?")
            icon  = CONV_ICON.get(conv, "   ")
            fv    = f"${ea['fair_value']:,.2f}"   if ea.get("fair_value")   else "  N/A"
            lo    = f"${ea['entry_target']:,.2f}" if ea.get("entry_target") else "N/A"
            cur   = f"${ea['current_price']:,.2f}" if ea.get("current_price") else "N/A"
            zone  = f"{lo} – {fv}"
            _p(
                f"  {i:<3} {p['ticker']:<7} {p['overall_score']:>6.1f} {gates:<10}"
                f" {icon} {conv:<4} {sig:<14} {fv:>11} {zone:>20} {cur:>9}"
            )

        _p("═" * W)
        _p("  P=Pass(≥60)  W=Warn(35-59)  F=Fail(<35)"
           "  ·  Fair value via DCF/Graham/EV·EBITDA/FCF-yield median\n")

    # ── Adapted weights notice ────────────────────────────────────────────────
    def show_weight_adaptation(self, adapted: Optional[list]):
        if adapted is None:
            return
        from config import FACTOR_NAMES
        _p("  [Learning] Weights adapted from past session performance:")
        for name, w in zip(FACTOR_NAMES, adapted):
            bar = "█" * int(w * 100 / 5)
            _p(f"    {name:<12} {w*100:>5.1f}%  {bar}")
        _p()

    # ── Disclaimer ────────────────────────────────────────────────────────────
    def show_disclaimer(self):
        _p("\n" + "─" * 70)
        _p("  DISCLAIMER")
        _p("─" * 70)
        _p("  This tool produces quantitative analysis — not financial advice.")
        _p("  Past performance does not guarantee future results.")
        _p("  All valuations are models — reality may differ significantly.")
        _p("  Always conduct your own due diligence before investing.")
        _p("  DCF estimates are highly sensitive to growth-rate assumptions.")
        _p("─" * 70 + "\n")
