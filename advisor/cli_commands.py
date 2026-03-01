"""
advisor/cli_commands.py — Interactive slash command handler for the CLI

After the main analysis, enters a persistent REPL where the user can run:

  /help                    — list all commands
  /stock  AAPL             — full deep-dive on any ticker
  /news   AAPL  [n=10]     — latest multi-source news
  /chart  AAPL  [period]   — candlestick chart  (1mo 3mo 6mo 1y 2y 5y)
  /compare T1 T2           — side-by-side comparison of two tickers
  /add    AAPL MSFT …      — save tickers to watchlist
  /remove AAPL             — remove from watchlist
  /watchlist               — show current watchlist with live prices
  /macro                   — macro environment snapshot
  /exit  (or /quit)        — leave interactive mode

Usage:
    from advisor.cli_commands import CommandHandler
    handler = CommandHandler(
        universe_data, valuation_results, risk_results,
        protocol_results, rf_rate, macro_data,
    )
    handler.run()
"""

import json
import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich         import box

from advisor.news_fetcher import NewsFetcher
from advisor.valuation    import ValuationEngine
from advisor.risk         import RiskEngine
from advisor.fetcher      import DataFetcher


_CONSOLE = Console()

_WATCHLIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "memory", "watchlist.json"
)

_VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}

_HELP_TEXT = """
[bold cyan]Stock Advisor — Interactive Commands[/bold cyan]

  [green]/stock  AAPL[/green]          Full deep-dive on any ticker
  [green]/news   AAPL [n][/green]      Latest multi-source news (default n=10)
  [green]/chart  AAPL [period][/green] Candlestick chart  (1mo 3mo 6mo 1y 2y 5y)
  [green]/compare T1 T2[/green]        Side-by-side comparison of two tickers
  [green]/add    AAPL MSFT …[/green]   Add tickers to watchlist
  [green]/remove AAPL[/green]          Remove ticker from watchlist
  [green]/watchlist[/green]            Show watchlist with live prices
  [green]/macro[/green]                Macro environment snapshot
  [green]/history [n][/green]          Past session results (default n=10)
  [green]/exit[/green]                 Leave interactive mode

[dim]Tip: any ticker works, even ones outside the universe.[/dim]
"""


class CommandHandler:

    def __init__(
        self,
        universe_data:     Dict,
        valuation_results: Dict,
        risk_results:      Dict,
        protocol_results:  List,
        rf_rate:           float,
        macro_data:        Dict,
    ):
        self._uni   = universe_data
        self._val   = valuation_results
        self._risk  = risk_results
        self._proto = {p["ticker"]: p for p in protocol_results}
        self._rf    = rf_rate
        self._macro = macro_data
        self._news  = NewsFetcher()

    # ── Main REPL ─────────────────────────────────────────────────────────────

    def run(self):
        _CONSOLE.print(
            Panel(
                "[bold]Interactive mode active[/bold] — type [green]/help[/green] "
                "for commands or [red]/exit[/red] to quit.",
                border_style="dim",
            )
        )
        while True:
            try:
                raw = input("\n[advisor] ").strip()
            except (EOFError, KeyboardInterrupt):
                _CONSOLE.print("\n[dim]Goodbye![/dim]")
                break

            if not raw:
                continue

            parts = raw.split()
            cmd   = parts[0].lower()
            args  = parts[1:]

            if cmd in ("/exit", "/quit"):
                _CONSOLE.print("[dim]Leaving interactive mode.[/dim]")
                break
            elif cmd == "/help":
                _CONSOLE.print(_HELP_TEXT)
            elif cmd == "/stock":
                self._cmd_stock(args)
            elif cmd == "/news":
                self._cmd_news(args)
            elif cmd == "/chart":
                self._cmd_chart(args)
            elif cmd == "/compare":
                self._cmd_compare(args)
            elif cmd == "/add":
                self._cmd_add(args)
            elif cmd == "/remove":
                self._cmd_remove(args)
            elif cmd == "/watchlist":
                self._cmd_watchlist()
            elif cmd == "/macro":
                self._cmd_macro()
            elif cmd == "/history":
                self._cmd_history(args)
            else:
                _CONSOLE.print(
                    f"[red]Unknown command:[/red] {cmd}  "
                    "(type [green]/help[/green] for list)"
                )

    # ── /stock ────────────────────────────────────────────────────────────────

    def _cmd_stock(self, args: List[str]):
        if not args:
            _CONSOLE.print("[red]Usage:[/red] /stock TICKER")
            return
        ticker = args[0].upper()
        _CONSOLE.print(f"\n[dim]Fetching data for [bold]{ticker}[/bold]…[/dim]")

        # Use cached data if available, else fetch fresh
        if ticker in self._uni:
            data = self._uni[ticker]
        else:
            fetcher = DataFetcher("1y")
            fresh   = fetcher.fetch_universe([ticker])
            if not fresh:
                _CONSOLE.print(f"[red]Could not fetch data for {ticker}[/red]")
                return
            data = fresh[ticker]

        info = data.get("info", {})
        self._print_stock_header(ticker, info, data.get("sector", "—"), data)
        self._print_quant_thesis(ticker)
        self._print_valuation_block(ticker, info)
        self._print_risk_block(ticker, data)
        self._print_protocol_block(ticker)
        self._print_key_financials(info)
        self._print_analyst_targets(info)
        self._print_technical_summary(info, data.get("history"))

    def _print_stock_header(self, ticker: str, info: dict, sector: str, data: dict = None):
        name  = info.get("longName", ticker)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        mktcap = info.get("marketCap", 0)
        mktcap_str = (f"${mktcap/1e12:.2f}T" if mktcap >= 1e12
                      else f"${mktcap/1e9:.1f}B" if mktcap >= 1e9
                      else "—")
        desc = (info.get("longBusinessSummary") or "")[:220]

        # Earnings warning
        days_away = (data or {}).get("earnings_days_away")
        edate     = (data or {}).get("earnings_date", "")
        earn_line = ""
        if days_away is not None and days_away <= 30:
            if   days_away <= 7:  earn_line = f"\n[bold red]⚠  EARNINGS IN {days_away} DAYS ({edate}) — high event risk[/bold red]"
            elif days_away <= 14: earn_line = f"\n[yellow]!  Earnings in {days_away} days ({edate})[/yellow]"
            else:                  earn_line = f"\n[dim]Earnings in {days_away} days ({edate})[/dim]"

        _CONSOLE.print(
            Panel(
                f"[bold white]{name}[/bold white]  [dim]({ticker})[/dim]\n"
                f"[dim]{sector}[/dim]  ·  "
                f"Price: [bold cyan]${price:,.2f}[/bold cyan]  ·  "
                f"Market Cap: [cyan]{mktcap_str}[/cyan]"
                f"{earn_line}\n\n"
                f"[dim]{desc}{'…' if len(desc)==220 else ''}[/dim]",
                title=f"[bold]{ticker}[/bold]",
                border_style="cyan",
            )
        )

    def _print_quant_thesis(self, ticker: str):
        """Auto-generate a Citadel-style quant thesis using cached val/risk/proto."""
        val = self._val.get(ticker, {})
        r   = self._risk.get(ticker, {})
        p   = self._proto.get(ticker, {})

        parts = []

        # Valuation
        sig   = val.get("signal", "")
        fv    = val.get("fair_value")
        price = val.get("current_price")
        prem  = val.get("premium_pct")
        rr    = val.get("rr_ratio")
        methods = val.get("methods_count", 0)
        upside  = val.get("upside_pct")
        tgt     = val.get("target_price")

        if fv and price:
            direction = "below" if (prem or 0) < 0 else "above"
            abs_pct   = abs(prem or 0)
            sig_map = {
                "STRONG_BUY": "firmly in the STRONG BUY zone",
                "BUY":        "in the BUY zone",
                "HOLD_WATCH": "at HOLD/WATCH",
                "WAIT":       "above FV — WAIT for pullback",
                "AVOID_PEAK": "at AVOID PEAK — not a clean entry",
            }
            sig_text = sig_map.get(sig, "")
            rr_text  = f"  R/R {rr:.1f}:1." if rr else ""
            up_text  = f"  Target ${tgt:,.2f} → {upside:+.1f}% upside." if (tgt and upside) else ""
            parts.append(
                f"[bold white]{ticker}[/bold white] at [cyan]${price:,.2f}[/cyan] — "
                f"[bold]{abs_pct:.1f}%[/bold] {direction} {methods}-method FV "
                f"[green]${fv:,.2f}[/green], {sig_text}.{rr_text}{up_text}"
            )

        # ROIC/WACC
        rw = r.get("roic_wacc", {})
        if rw.get("spread") is not None:
            spread  = rw["spread"]
            verdict = rw.get("verdict", "—")
            c       = "green" if spread > 8 else "red" if spread < 0 else "yellow"
            parts.append(
                f"ROIC/WACC [bold {c}]{spread:+.1f}%[/bold {c}] — {verdict}."
            )

        # Piotroski
        pf = r.get("piotroski", {})
        if pf.get("score") is not None:
            sc  = pf["score"]
            it  = pf.get("interpretation", "")
            pc  = "green" if sc >= 7 else "red" if sc <= 3 else "yellow"
            parts.append(f"Piotroski [{pc}]{sc}/9[/{pc}] — {it}.")

        # Altman Z
        az = r.get("altman_z", {})
        if az.get("score") is not None:
            azs = az["score"]
            azz = az.get("zone", "—")
            azc = "green" if azz == "SAFE" else "red" if azz == "DISTRESS" else "yellow"
            parts.append(f"Altman Z [cyan]{azs:.2f}[/cyan] [{azc}]{azz}[/{azc}].")

        # Sharpe
        sharpe = r.get("sharpe")
        if sharpe is not None:
            sc = "green" if sharpe > 1.2 else "red" if sharpe < 0.5 else "yellow"
            parts.append(f"Sharpe [{sc}]{sharpe:.2f}[/{sc}].")

        # Protocol
        if p:
            pconv = p.get("conviction", "—")
            passc = p.get("pass_count", 0)
            warnc = p.get("warn_count", 0)
            failc = p.get("fail_count", 0)
            overall = p.get("overall_score", 0)
            cc = "green" if pconv == "HIGH" else "red" if pconv == "LOW" else "yellow"
            parts.append(
                f"Protocol: [{cc}]{passc}P/{warnc}W/{failc}F[/{cc}] "
                f"score {overall:.0f} · [{cc}]{pconv}[/{cc}] conviction."
            )

        if not parts:
            return

        body = "  " + "\n  ".join(parts)
        _CONSOLE.print(
            Panel(
                body,
                title="[bold yellow]Quant Thesis[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    def _print_analyst_targets(self, info: dict):
        """Print analyst price target distribution."""
        mean_t = info.get("targetMeanPrice")
        high_t = info.get("targetHighPrice")
        low_t  = info.get("targetLowPrice")
        n_ana  = info.get("numberOfAnalystOpinions")
        rec_key = info.get("recommendationKey", "")

        if not mean_t:
            return

        cur    = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        upside = ((float(mean_t) / float(cur)) - 1) * 100 if cur and mean_t else None

        rec_lbl = rec_key.upper().replace("_", " ") if rec_key else "—"
        rec_map  = {"STRONG BUY": "green", "BUY": "cyan", "HOLD": "yellow",
                    "UNDERPERFORM": "orange1", "SELL": "red"}
        rec_color = rec_map.get(rec_lbl, "white")

        t = Table(title="Analyst Targets", box=box.SIMPLE, show_header=False)
        t.add_column("Label", style="dim", min_width=18)
        t.add_column("Value", justify="right")
        t.add_row("Consensus",   f"[{rec_color}]{rec_lbl}[/{rec_color}]")
        t.add_row("# Analysts",  str(int(n_ana)) if n_ana else "—")
        t.add_row("Mean Target", f"[cyan]{_price(mean_t)}[/cyan]" + (
            f"  [green]+{upside:.1f}%[/green]" if (upside or 0) > 0 else
            f"  [red]{upside:.1f}%[/red]" if upside else ""
        ))
        t.add_row("High Target", _price(high_t))
        t.add_row("Low Target",  _price(low_t))
        _CONSOLE.print(t)

    def _print_technical_summary(self, info: dict, hist=None):
        """Print plain-English technical status: SMA, 52W position, momentum."""
        import numpy as np

        sma50  = info.get("fiftyDayAverage")
        sma200 = info.get("twoHundredDayAverage")
        price  = info.get("currentPrice") or info.get("regularMarketPrice")
        low52  = info.get("fiftyTwoWeekLow")
        high52 = info.get("fiftyTwoWeekHigh")

        rows = []

        if price and sma200:
            price, sma200 = float(price), float(sma200)
            trend = "above" if price > sma200 else "below"
            col   = "green" if price > sma200 else "red"
            vs200 = ((price / sma200) - 1) * 100
            rows.append(("SMA 200", f"[{col}]{trend}[/{col}] ({vs200:+.1f}%)"))
        if price and sma50:
            price, sma50 = float(price), float(sma50)
            trend = "above" if price > sma50 else "below"
            col   = "green" if price > sma50 else "red"
            vs50  = ((price / sma50) - 1) * 100
            rows.append(("SMA 50",  f"[{col}]{trend}[/{col}] ({vs50:+.1f}%)"))

        if price and low52 and high52:
            rng = float(high52) - float(low52)
            if rng > 0:
                pos = (float(price) - float(low52)) / rng * 100
                pc  = "red" if pos > 85 else "green" if pos < 30 else "yellow"
                rows.append(("52W Range", f"[{pc}]{pos:.0f}%[/{pc}] of range  "
                             f"[dim]${float(low52):,.2f}–${float(high52):,.2f}[/dim]"))

        if hist is not None and hasattr(hist, '__len__') and len(hist) >= 15:
            try:
                closes = hist["Close"].dropna().values.astype(float)
                deltas = np.diff(closes[-15:])
                gains  = deltas[deltas > 0].mean() if (deltas > 0).any() else 0
                losses = (-deltas[deltas < 0]).mean() if (deltas < 0).any() else 0.001
                rs     = gains / losses
                rsi    = 100 - (100 / (1 + rs))
                rsi_lbl = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                rsi_col = "red" if rsi > 70 else "green" if rsi < 30 else "yellow"
                rows.append(("RSI (14)", f"[{rsi_col}]{rsi:.1f} — {rsi_lbl}[/{rsi_col}]"))
            except Exception:
                pass

        if hist is not None and hasattr(hist, '__len__') and len(hist) >= 63:
            try:
                closes  = hist["Close"].dropna().values.astype(float)
                mom_3m  = (closes[-1] / closes[-63] - 1) * 100
                mc      = "green" if mom_3m > 5 else "red" if mom_3m < -5 else "yellow"
                rows.append(("3M Momentum", f"[{mc}]{mom_3m:+.1f}%[/{mc}]"))
            except Exception:
                pass

        if not rows:
            return

        t = Table(title="Technical Status", box=box.SIMPLE, show_header=False)
        t.add_column("Indicator", style="dim", min_width=18)
        t.add_column("Status", justify="right")
        for k, v in rows:
            t.add_row(k, v)
        _CONSOLE.print(t)

    def _print_valuation_block(self, ticker: str, info: dict):
        val = self._val.get(ticker)
        if not val:
            _CONSOLE.print("[dim]  Valuation not in top-10 session — running now…[/dim]")
            try:
                row = pd.DataFrame([{
                    "ticker": ticker,
                    "sector": self._uni.get(ticker, {}).get("sector", "Unknown"),
                    "composite_score": 0,
                }])
                val = ValuationEngine(self._rf).analyze_all(row, self._uni)
                val = val.get(ticker, {})
            except Exception:
                val = {}

        if not val:
            return

        est = val.get("estimates", {})
        t = Table(title="Valuation  (4 methods)", box=box.SIMPLE_HEAVY,
                  show_header=True, header_style="bold dim")
        t.add_column("Method",     style="dim",        min_width=14)
        t.add_column("Estimate",   style="cyan",       justify="right")
        t.add_column("vs Current", justify="right")

        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        methods = [
            ("DCF (2-stage)",   est.get("dcf")),
            ("Graham Number",   est.get("graham")),
            ("EV/EBITDA target",est.get("ev_ebitda")),
            ("FCF Yield @4.5%", est.get("fcf_yield")),
        ]
        for method, est_price in methods:
            if est_price is None:
                t.add_row(method, "—", "—")
                continue
            diff = ((price / est_price) - 1) * 100 if est_price else 0
            diff_str = f"{diff:+.1f}%"
            diff_style = "red" if diff > 5 else "green" if diff < -10 else "yellow"
            t.add_row(method, f"${est_price:,.2f}", f"[{diff_style}]{diff_str}[/{diff_style}]")

        _CONSOLE.print(t)
        sig = val.get("signal", "")
        fv  = val.get("fair_value")
        el  = val.get("entry_low")
        rr  = val.get("rr_ratio")
        _CONSOLE.print(
            f"  Fair Value [bold]${fv:,.2f}[/bold]  ·  "
            f"Entry Zone  [green]${el:,.2f}[/green]  ·  "
            f"R/R  [cyan]{rr:.1f}:1[/cyan]  ·  "
            f"Signal  [bold]{sig}[/bold]"
            if fv and el and rr else ""
        )

        # DCF sensitivity
        sens = val.get("sensitivity", {})
        if sens:
            t_sens = Table(title="DCF Sensitivity", box=box.SIMPLE, show_header=True,
                           header_style="bold dim")
            t_sens.add_column("Scenario", style="dim",  min_width=6)
            t_sens.add_column("Growth",   style="dim",  justify="right")
            t_sens.add_column("Fair Value", style="cyan", justify="right")
            t_sens.add_column("Premium",  justify="right")
            t_sens.add_column("Signal",   style="bold")
            for sname, sv in sens.items():
                sfv   = sv.get("fair_value")
                sgr   = sv.get("growth_rate")
                sprem = sv.get("premium_pct")
                ssig  = sv.get("signal", "—")
                pstyle = "red" if (sprem or 0) > 5 else "green" if (sprem or 0) < -10 else "yellow"
                t_sens.add_row(
                    sname,
                    f"{sgr:+.1f}%" if sgr is not None else "—",
                    f"${sfv:,.2f}" if sfv else "N/A",
                    f"[{pstyle}]{sprem:+.1f}%[/{pstyle}]" if sprem is not None else "—",
                    ssig,
                )
            _CONSOLE.print(t_sens)

    def _print_risk_block(self, ticker: str, data: dict):
        r = self._risk.get(ticker)
        if not r:
            _CONSOLE.print("[dim]  Risk not cached — running now…[/dim]")
            try:
                row = pd.DataFrame([{"ticker": ticker,
                                     "composite_score": 0,
                                     "sector": data.get("sector", "Unknown")}])
                r = RiskEngine().analyze_all(row, {ticker: data}, self._rf)
                r = r.get(ticker, {})
            except Exception:
                r = {}

        if not r:
            return

        az = r.get("altman_z", {})
        rw = r.get("roic_wacc", {})
        pf = r.get("piotroski", {})

        _CONSOLE.print(
            f"\n  [bold]Risk Profile[/bold]\n"
            f"  Altman Z [cyan]{_fmt(az.get('score'),'2f')}[/cyan]"
            f" [{_zone_color(az.get('zone',''))}]{az.get('zone','—')}[/{_zone_color(az.get('zone',''))}]"
            f"  ·  Sharpe [cyan]{_fmt(r.get('sharpe'),'2f')}[/cyan]"
            f"  ·  Sortino [cyan]{_fmt(r.get('sortino'),'2f')}[/cyan]\n"
            f"  Max DD [red]{_fmt(r.get('max_drawdown_pct'),'1f')}%[/red]"
            f"  ·  VaR(95%) [yellow]{_fmt(r.get('var_95_pct'),'1f')}%[/yellow]"
            f"  ·  ROIC/WACC [cyan]{_fmt(rw.get('spread'),'+.1f')}%[/cyan]"
            f" [dim]{rw.get('verdict','—')}[/dim]\n"
            f"  Piotroski [bold]{pf.get('score','—')}/9[/bold]"
            f" [dim]{pf.get('interpretation','—')}[/dim]"
        )

    def _print_protocol_block(self, ticker: str):
        p = self._proto.get(ticker)
        if not p:
            return
        gates    = p.get("gates", [])
        statuses = p.get("gate_statuses", [])
        gate_names = ["Biz Quality","Moat","Health","Valuation",
                      "Technical","Sentiment","Trend"]
        dots = ""
        for st in statuses:
            if st == "pass":   dots += "[green]●[/green] "
            elif st == "warn": dots += "[yellow]◐[/yellow] "
            else:              dots += "[red]○[/red] "

        _CONSOLE.print(
            f"\n  [bold]7-Gate Protocol[/bold]  "
            f"Score [bold]{p.get('overall_score',0):.1f}[/bold]  ·  "
            f"Conviction [bold]{p.get('conviction','—')}[/bold]\n"
            f"  {dots}"
        )

    def _print_key_financials(self, info: dict):
        rows = [
            ("P/E (TTM)",          _ifmt(info.get("trailingPE"),  ".1f")),
            ("Forward P/E",        _ifmt(info.get("forwardPE"),   ".1f")),
            ("PEG Ratio",          _ifmt(info.get("pegRatio"),    ".2f")),
            ("EV/EBITDA",          _ifmt(info.get("enterpriseToEbitda"), ".1f")),
            ("Revenue Growth",     _pct(info.get("revenueGrowth"))),
            ("Gross Margin",       _pct(info.get("grossMargins"))),
            ("Operating Margin",   _pct(info.get("operatingMargins"))),
            ("Net Margin",         _pct(info.get("profitMargins"))),
            ("ROE",                _pct(info.get("returnOnEquity"))),
            ("ROA",                _pct(info.get("returnOnAssets"))),
            ("Debt / Equity",      _ifmt(info.get("debtToEquity"), ".2f")),
            ("Current Ratio",      _ifmt(info.get("currentRatio"), ".2f")),
            ("Dividend Yield",     _pct(info.get("dividendYield"))),
            ("Beta",               _ifmt(info.get("beta"),         ".2f")),
            ("52W High",           _price(info.get("fiftyTwoWeekHigh"))),
            ("52W Low",            _price(info.get("fiftyTwoWeekLow"))),
        ]
        t = Table(title="Key Financials", box=box.SIMPLE, show_header=False)
        t.add_column("Metric", style="dim", min_width=18)
        t.add_column("Value",  style="cyan", justify="right")
        for k, v in rows:
            t.add_row(k, v)
        _CONSOLE.print(t)

    # ── /news ─────────────────────────────────────────────────────────────────

    def _cmd_news(self, args: List[str]):
        if not args:
            _CONSOLE.print("[red]Usage:[/red] /news TICKER [n]")
            return
        ticker = args[0].upper()
        n      = int(args[1]) if len(args) > 1 and args[1].isdigit() else 10

        _CONSOLE.print(f"\n[dim]Fetching news for [bold]{ticker}[/bold]…[/dim]")
        articles = self._news.fetch_ticker_news(ticker, n)
        if not articles:
            _CONSOLE.print(f"[yellow]No news found for {ticker}[/yellow]")
            return

        score = self._news.score_sentiment([a["title"] for a in articles])
        sc_color = "green" if score >= 60 else "red" if score < 40 else "yellow"
        sentiment_label = "POSITIVE" if score >= 60 else "NEGATIVE" if score < 40 else "NEUTRAL"

        lines = [f"Sentiment: [{sc_color}]{score:.1f}/100  {sentiment_label}[/{sc_color}]\n"]
        for i, a in enumerate(articles, 1):
            hint  = a.get("sentiment_hint", "neutral")
            hcol  = "green" if hint == "positive" else "red" if hint == "negative" else "dim"
            lines.append(
                f"  [{hcol}]{i:2d}.[/{hcol}] {a['title']}\n"
                f"      [dim]{a.get('source','—')}  ·  {a.get('published','—')}[/dim]"
            )

        _CONSOLE.print(
            Panel(
                "\n".join(lines),
                title=f"[bold]News — {ticker}[/bold]  ({len(articles)} articles)",
                border_style="dim",
            )
        )

    # ── /chart ────────────────────────────────────────────────────────────────

    def _cmd_chart(self, args: List[str]):
        if not args:
            _CONSOLE.print("[red]Usage:[/red] /chart TICKER [period]  (1mo 3mo 6mo 1y 2y 5y)")
            return
        ticker = args[0].upper()
        period = args[1].lower() if len(args) > 1 else "6mo"
        if period not in _VALID_PERIODS:
            period = "6mo"

        _CONSOLE.print(f"[dim]Building candlestick for {ticker} ({period})…[/dim]")
        # Get history
        if ticker in self._uni and self._uni[ticker].get("history") is not None:
            hist = self._uni[ticker]["history"]
        else:
            try:
                hist = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
            except Exception:
                _CONSOLE.print("[red]Could not fetch price data.[/red]")
                return

        if hist is None or hist.empty:
            _CONSOLE.print(f"[red]No price data for {ticker}[/red]")
            return

        from advisor.charts import ChartEngine
        from advisor.collector import UserProfile
        engine = ChartEngine(
            UserProfile.__new__(UserProfile), None, self._macro
        )
        fig = engine.candlestick(ticker, hist, period)

        import matplotlib.pyplot as plt
        try:
            plt.show()
        except Exception:
            fname = f"chart_{ticker}_{period}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            _CONSOLE.print(f"[dim]Chart saved as [cyan]{fname}[/cyan][/dim]")

    # ── /compare ──────────────────────────────────────────────────────────────

    def _cmd_compare(self, args: List[str]):
        if len(args) < 2:
            _CONSOLE.print("[red]Usage:[/red] /compare TICKER1 TICKER2")
            return
        t1, t2 = args[0].upper(), args[1].upper()

        def _info(ticker):
            if ticker in self._uni:
                return self._uni[ticker].get("info", {})
            try:
                return yf.Ticker(ticker).info or {}
            except Exception:
                return {}

        i1, i2 = _info(t1), _info(t2)
        v1 = self._val.get(t1, {})
        v2 = self._val.get(t2, {})
        r1 = self._risk.get(t1, {})
        r2 = self._risk.get(t2, {})

        t = Table(title=f"[bold]{t1}  vs  {t2}[/bold]",
                  box=box.SIMPLE_HEAVY, show_header=True,
                  header_style="bold")
        t.add_column("Metric",     style="dim",   min_width=20)
        t.add_column(t1,           style="cyan",  justify="right")
        t.add_column(t2,           style="green", justify="right")

        rows = [
            ("Price",              _price(i1.get("currentPrice")), _price(i2.get("currentPrice"))),
            ("Market Cap",         _mktcap(i1.get("marketCap")),  _mktcap(i2.get("marketCap"))),
            ("P/E (TTM)",          _ifmt(i1.get("trailingPE"),".1f"), _ifmt(i2.get("trailingPE"),".1f")),
            ("EV/EBITDA",          _ifmt(i1.get("enterpriseToEbitda"),".1f"), _ifmt(i2.get("enterpriseToEbitda"),".1f")),
            ("Revenue Growth",     _pct(i1.get("revenueGrowth")), _pct(i2.get("revenueGrowth"))),
            ("Gross Margin",       _pct(i1.get("grossMargins")),  _pct(i2.get("grossMargins"))),
            ("ROE",                _pct(i1.get("returnOnEquity")),_pct(i2.get("returnOnEquity"))),
            ("Debt/Equity",        _ifmt(i1.get("debtToEquity"),".2f"), _ifmt(i2.get("debtToEquity"),".2f")),
            ("Beta",               _ifmt(i1.get("beta"),".2f"),   _ifmt(i2.get("beta"),".2f")),
            ("--- Valuation ---",  "",            ""),
            ("Fair Value",         _price(v1.get("fair_value")),  _price(v2.get("fair_value"))),
            ("Signal",             v1.get("signal","—"),          v2.get("signal","—")),
            ("Premium vs FV",      _pct_v(v1.get("premium_pct")), _pct_v(v2.get("premium_pct"))),
            ("R/R Ratio",          _ifmt(v1.get("rr_ratio"),".1f"), _ifmt(v2.get("rr_ratio"),".1f")),
            ("--- Risk ---",       "",            ""),
            ("Altman Z",           _ifmt(r1.get("altman_z",{}).get("score"),".2f"),
                                   _ifmt(r2.get("altman_z",{}).get("score"),".2f")),
            ("Sharpe",             _ifmt(r1.get("sharpe"),".2f"), _ifmt(r2.get("sharpe"),".2f")),
            ("Piotroski",          _piof(r1), _piof(r2)),
        ]
        for row in rows:
            t.add_row(*row)

        _CONSOLE.print(t)

    # ── /add ──────────────────────────────────────────────────────────────────

    def _cmd_add(self, args: List[str]):
        if not args:
            _CONSOLE.print("[red]Usage:[/red] /add AAPL MSFT …")
            return
        wl = self._load_watchlist()
        added = []
        for t in args:
            t = t.upper()
            if t not in wl:
                wl.append(t)
                added.append(t)
        self._save_watchlist(wl)
        _CONSOLE.print(f"[green]Added:[/green] {', '.join(added) or 'none (already in list)'}")

    # ── /remove ───────────────────────────────────────────────────────────────

    def _cmd_remove(self, args: List[str]):
        if not args:
            _CONSOLE.print("[red]Usage:[/red] /remove TICKER")
            return
        ticker = args[0].upper()
        wl = self._load_watchlist()
        if ticker in wl:
            wl.remove(ticker)
            self._save_watchlist(wl)
            _CONSOLE.print(f"[green]Removed {ticker} from watchlist.[/green]")
        else:
            _CONSOLE.print(f"[yellow]{ticker} was not in watchlist.[/yellow]")

    # ── /watchlist ────────────────────────────────────────────────────────────

    def _cmd_watchlist(self):
        wl = self._load_watchlist()
        if not wl:
            _CONSOLE.print("[dim]Watchlist is empty. Use /add TICKER to add stocks.[/dim]")
            return
        _CONSOLE.print(f"[dim]Fetching live prices for {len(wl)} stocks…[/dim]")
        t = Table(title="Watchlist", box=box.SIMPLE_HEAVY, show_header=True,
                  header_style="bold dim")
        t.add_column("Ticker",   style="bold", min_width=8)
        t.add_column("Price",    style="cyan",  justify="right")
        t.add_column("Change",   justify="right")
        t.add_column("52W High", justify="right", style="dim")
        t.add_column("Signal",   justify="left")

        for ticker in wl:
            try:
                info   = yf.Ticker(ticker).info or {}
                price  = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                prev   = info.get("previousClose") or price
                chg    = ((price / prev) - 1) * 100 if prev else 0
                w52h   = info.get("fiftyTwoWeekHigh")
                chg_s  = f"[green]+{chg:.2f}%[/green]" if chg >= 0 else f"[red]{chg:.2f}%[/red]"
                val    = self._val.get(ticker, {})
                sig    = val.get("signal", "—")
                t.add_row(
                    ticker,
                    f"${price:,.2f}",
                    chg_s,
                    _price(w52h),
                    sig,
                )
            except Exception:
                t.add_row(ticker, "—", "—", "—", "—")

        _CONSOLE.print(t)

    # ── /macro ────────────────────────────────────────────────────────────────

    def _cmd_macro(self):
        m = self._macro
        vix    = m.get("vix")
        y10    = m.get("yield_10y")
        regime = m.get("regime", "neutral").upper().replace("_", " ")
        reasons = "  ·  ".join(m.get("regime_reasons", []))

        vix_c = "red" if (vix or 0) > 25 else "yellow" if (vix or 0) > 18 else "green"
        reg_c = "green" if "RISK ON" in regime else "red" if "RISK OFF" in regime else "yellow"

        lines = [
            f"VIX              [{vix_c}]{vix:.1f}[/{vix_c}]" if vix else "VIX              —",
            f"10-Year Yield    [cyan]{y10:.2f}%[/cyan]" if y10 else "10-Year Yield    —",
            f"Regime           [{reg_c}]{regime}[/{reg_c}]",
            f"Drivers          [dim]{reasons}[/dim]" if reasons else "",
        ]
        etf = m.get("sector_etf", {})
        if etf:
            lines.append("\nSector ETF 3-Month Returns:")
            for sector, ret in sorted(etf.items(), key=lambda x: -x[1]):
                c = "green" if ret >= 0 else "red"
                lines.append(f"  {sector:<18} [{c}]{ret:+.1f}%[/{c}]")

        _CONSOLE.print(
            Panel("\n".join(l for l in lines if l),
                  title="[bold]Macro Environment[/bold]",
                  border_style="dim")
        )

    # ── /history ──────────────────────────────────────────────────────────────

    def _cmd_history(self, args: List[str]):
        from datetime import datetime, timezone

        _history_path = os.path.join(
            os.path.dirname(__file__), "..", "memory", "history.json"
        )
        try:
            with open(_history_path) as f:
                data = json.load(f)
            sessions = data.get("sessions", [])
        except FileNotFoundError:
            _CONSOLE.print("[dim]No history yet. Run the analysis to start building session history.[/dim]")
            return
        except Exception as e:
            _CONSOLE.print(f"[red]Could not load history: {e}[/red]")
            return

        if not sessions:
            _CONSOLE.print("[dim]No past sessions found.[/dim]")
            return

        sessions = sorted(sessions, key=lambda s: s.get("timestamp", ""), reverse=True)
        n_show   = int(args[0]) if args and args[0].isdigit() else 10

        evald   = [s for s in sessions if s.get("evaluated")]
        pending = [s for s in sessions if not s.get("evaluated")]

        # Compute summary stats
        alphas   = [s["evaluation"]["alpha"] for s in evald
                    if (s.get("evaluation") or {}).get("alpha") is not None]
        avg_alpha  = sum(alphas) / len(alphas) if alphas else None
        beats_sp   = sum(1 for a in alphas if a > 0)
        win_rate   = beats_sp / len(alphas) * 100 if alphas else None

        alpha_str = ""
        if avg_alpha is not None:
            ac = "green" if avg_alpha >= 0 else "red"
            alpha_str = (
                f"  ·  Avg Alpha [{ac}]{avg_alpha*100:+.1f}%[/{ac}]"
                f"  ·  Win Rate [cyan]{win_rate:.0f}%[/cyan]"
            )

        _CONSOLE.print(
            Panel(
                f"Total Sessions: [bold]{len(sessions)}[/bold]"
                f"  ·  Evaluated: [cyan]{len(evald)}[/cyan]"
                f"  ·  Pending: [dim]{len(pending)}[/dim]"
                + alpha_str,
                title="[bold]Session History[/bold]",
                border_style="cyan",
            )
        )

        now = datetime.now(timezone.utc)

        for session in sessions[:n_show]:
            ts      = session.get("timestamp", "")[:10]
            sid     = session.get("session_id", "?")
            prof    = session.get("profile", {})
            risk    = prof.get("risk_level", "?")
            horizon = prof.get("time_horizon", "?")
            goal    = prof.get("goal", "?")
            picks   = session.get("picks", [])

            ticker_parts = [
                f"[cyan]{p['ticker']}[/cyan] @ ${p.get('price_entry', 0):.0f}"
                for p in picks
            ]
            tickers_str = "  ".join(ticker_parts)

            if session.get("evaluated"):
                ev      = session["evaluation"]
                avg_ret = ev.get("avg_pick_return", 0) or 0
                sp_ret  = ev.get("sp500_return")
                alpha   = ev.get("alpha", 0) or 0
                eval_dt = ev.get("evaluation_date", "")[:10]

                ret_c   = "green" if avg_ret >= 0 else "red"
                alpha_c = "green" if alpha   >= 0 else "red"
                sp_str  = f"{sp_ret*100:+.1f}%" if sp_ret is not None else "—"

                # Per-pick return lines
                pick_lines = []
                for ep in ev.get("picks", []):
                    r  = ep.get("return", 0) or 0
                    rc = "green" if r >= 0 else "red"
                    pick_lines.append(
                        f"  [bold]{ep['ticker']:<6}[/bold] [{rc}]{r*100:+.1f}%[/{rc}]"
                        f"  [dim]exit ${ep.get('price_exit', 0):.2f}[/dim]"
                    )

                body = (
                    f"[dim]{ts}  ·  #{sid}  ·  Risk {risk}  ·  {horizon}  ·  {goal}[/dim]\n"
                    f"{tickers_str}\n\n"
                    f"Avg Return [{ret_c}]{avg_ret*100:+.1f}%[/{ret_c}]"
                    f"  ·  S&P {sp_str}"
                    f"  ·  Alpha [{alpha_c}]{alpha*100:+.1f}%[/{alpha_c}]"
                    f"  [dim](evaluated {eval_dt})[/dim]\n"
                    + "\n".join(pick_lines)
                )
                border = "green" if alpha >= 0 else "red"

            else:
                try:
                    ts_dt    = datetime.fromisoformat(session["timestamp"])
                    days_old = (now - ts_dt).days
                    days_left = max(0, 30 - days_old)
                except Exception:
                    days_old  = 0
                    days_left = 30
                body = (
                    f"[dim]{ts}  ·  #{sid}  ·  Risk {risk}  ·  {horizon}  ·  {goal}[/dim]\n"
                    f"{tickers_str}\n\n"
                    f"[dim]Pending evaluation — {days_old} days old, "
                    f"evaluates in ~{days_left} more days[/dim]"
                )
                border = "dim"

            _CONSOLE.print(Panel(body, border_style=border))

        if len(sessions) > n_show:
            _CONSOLE.print(
                f"[dim]Showing {n_show} of {len(sessions)} sessions. "
                f"Use /history {len(sessions)} to see all.[/dim]"
            )

    # ── Watchlist persistence ──────────────────────────────────────────────────

    def _load_watchlist(self) -> List[str]:
        try:
            with open(_WATCHLIST_PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def _save_watchlist(self, wl: List[str]):
        os.makedirs(os.path.dirname(_WATCHLIST_PATH), exist_ok=True)
        with open(_WATCHLIST_PATH, "w") as f:
            json.dump(wl, f, indent=2)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt(v, spec):
    if v is None: return "—"
    try:    return format(float(v), spec)
    except: return "—"

def _ifmt(v, spec):
    if v is None: return "—"
    try:    return format(float(v), spec)
    except: return "—"

def _pct(v):
    if v is None: return "—"
    try:    return f"{float(v)*100:.1f}%"
    except: return "—"

def _pct_v(v):
    if v is None: return "—"
    try:    return f"{float(v):+.1f}%"
    except: return "—"

def _price(v):
    if v is None: return "—"
    try:    return f"${float(v):,.2f}"
    except: return "—"

def _mktcap(v):
    if v is None: return "—"
    try:
        v = float(v)
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9:  return f"${v/1e9:.1f}B"
        if v >= 1e6:  return f"${v/1e6:.0f}M"
        return f"${v:,.0f}"
    except: return "—"

def _piof(r: dict) -> str:
    pf = r.get("piotroski", {})
    sc = pf.get("score")
    return f"{sc}/9" if sc is not None else "—"

def _zone_color(zone: str) -> str:
    return {"SAFE": "green", "GRAY": "yellow", "DISTRESS": "red"}.get(zone, "dim")
