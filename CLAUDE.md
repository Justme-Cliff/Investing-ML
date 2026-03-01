# Stock Ranking Advisor — Project Context for Claude

This file is auto-loaded by Claude Code at the start of every session.
Read this before making any changes to the codebase.

---

## What This Project Is

A fully local, quantitative stock ranking + deep analysis tool — no paid AI APIs required.
Users enter a portfolio size, risk level, time horizon, and optional fresh-picks mode.
The system fetches ~110 stocks from Yahoo Finance, scores them on 7 factors,
runs a 7-gate investment protocol, performs hedge-fund grade valuation
(DCF, Graham Number, EV/EBITDA, FCF yield + DCF sensitivity) and risk analysis
(Altman Z, Sharpe, Sortino, ROIC/WACC, Piotroski 9pt, VaR), and can backtest
the valuation entry strategy on historical price data.

There are **two interfaces**:
- `main.py` — CLI pipeline with rich terminal output + 5 charts + Excel export
- `app.py` — Streamlit web dashboard (browser-based, localhost:8501)

Real money goes in based on this output — quality and precision matter.

---

## Project Structure

```
portfolio/
├── main.py                  ← 16-step CLI pipeline + interactive command loop
├── app.py                   ← Streamlit web dashboard (9 tabs)
├── config.py                ← STOCK_UNIVERSE, WEIGHT_MATRIX, SECTOR_MEDIAN_PE, SECTOR_EV_EBITDA,
│                               MACRO_TILTS, + optional API keys (FINNHUB_KEY, NEWSAPI_KEY, FRED_KEY)
├── requirements.txt         ← all dependencies incl. mplfinance, feedparser, requests
├── CLAUDE.md                ← this file
├── Book1.xlsx               ← auto-generated Excel output (6 sheets, CLI only)
├── chart1_score_breakdown.png
├── chart2_performance.png
├── chart3_factor_heatmap.png
├── chart4_macro_dashboard.png
├── chart5_quant_protocol.png  ← quantitative protocol report chart (CLI only)
├── .streamlit/
│   └── config.toml          ← Streamlit theme (white bg, black text, blue accent)
├── memory/
│   ├── history.json         ← session log (auto-created)
│   └── watchlist.json       ← CLI /add watchlist (auto-created)
└── advisor/
    ├── __init__.py
    ├── collector.py         ← UserProfile dataclass + 8-question CLI collector (incl. avoid_recent)
    ├── fetcher.py           ← DataFetcher (yfinance), MacroFetcher, technicals, sentiment, earnings calendar
    ├── scorer.py            ← 7-factor MultiFactorScorer with macro regime tilts
    ├── portfolio.py         ← Greedy correlation-aware selection + half-Kelly sizing
    ├── protocol.py          ← 7-gate Warren Buffett protocol + entry price analysis
    ├── valuation.py         ← ValuationEngine: DCF, Graham, EV/EBITDA, FCF yield + DCF sensitivity
    ├── risk.py              ← RiskEngine: Altman Z, Sharpe, Sortino, ROIC/WACC, Piotroski
    ├── news_fetcher.py      ← NewsFetcher: yfinance + RSS + Finnhub + NewsAPI + FRED
    ├── cli_commands.py      ← Interactive slash command REPL (post-analysis)
    ├── charts.py            ← 5 dark-theme charts + candlestick() method (CLI only)
    ├── display.py           ← Rich terminal output (results, protocol, deep analysis)
    ├── exporter.py          ← Excel export (6 sheets incl. Deep Analysis)
    └── learner.py           ← Persistent session memory + adaptive weight learning + get_recent_tickers()
```

---

## Pipeline (main.py — 16 steps)

1. Load memory (`memory/history.json`) + evaluate past sessions
2. Show track record (win rate vs S&P 500)
3. Collect user profile (8 CLI questions — last one is fresh picks mode)
4. Get adapted weights from learning history
5. Fetch ~110 stocks from Yahoo Finance (batched, with retry)
6. Fetch S&P 500 benchmark
7. Fetch macro data (VIX, 10Y yield, sector ETFs) → extract rf_rate
8. Score all stocks (7 factors, normalised 0–100, macro tilt); optionally apply −22pt penalty to recent picks
9. Portfolio construction: top 10 via greedy correlation-aware selection + half-Kelly
10. Display quantitative results in terminal
11. **ValuationEngine** — DCF, Graham, EV/EBITDA, FCF yield → fair value, entry zone, stop loss, DCF sensitivity
12. **RiskEngine** — Altman Z, Sharpe, Sortino, Max DD, VaR, ROIC/WACC, Piotroski
13. **Protocol analysis** — 7-gate Warren Buffett screen using ValuationEngine signal for Gate 4
14. **show_deep_analysis()** — scary-detailed per-stock terminal output with quant thesis
15. Generate 5 charts + Excel export + save session
16. **CommandHandler.run()** — interactive slash command REPL (/stock /news /chart /compare /watchlist /macro /history /exit)

**IMPORTANT:** `plt.show(block=False)` + `plt.pause(0.1)` is used so chart windows are non-blocking and the command loop starts immediately.

---

## UserProfile (advisor/collector.py)

```python
@dataclass
class UserProfile:
    portfolio_size:     float
    time_horizon:       str           # "short" | "medium" | "long"
    time_horizon_years: int
    yf_period:          str           # "1y" | "3y" | "5y"
    risk_label:         str
    risk_level:         int           # 1–4
    goal:               str
    goal_label:         str
    drawdown_ok:        float
    preferred_sectors:  List[str]
    excluded_sectors:   List[str]
    existing_tickers:   List[str]
    avoid_recent:       bool = False  # penalise last 2 sessions' picks by −22pts
```

Q8 in the CLI asks "Fresh picks mode?" (y/n). Web dashboard has a "Fresh picks mode" checkbox in the sidebar.

---

## Fresh Picks Mode

After scoring, if `profile.avoid_recent`:
1. `memory.get_recent_tickers(n_sessions=2)` loads tickers from last 2 sessions
2. Those tickers get `composite_score -= 22` (clipped to 0)
3. `ranked_df` is re-sorted so new stocks bubble up
4. Applied identically in `main.py` (step 8b) and `app.py` `run_analysis()`

---

## Streamlit Dashboard (app.py)

Run with: `streamlit run app.py` → opens at `http://localhost:8501`

**Theme:** white background (#FFFFFF), black text (#111827), blue accent (#2563EB)

**Sidebar:** All 8 profile inputs as widgets + "Run Analysis" button + "Past Sessions" and "Backtest" buttons side-by-side.

**Session state keys:**
- `results` — full pipeline output dict
- `profile` — UserProfile used for last run
- `show_history` — bool toggle for past sessions full-page view
- `show_backtest` — bool toggle for backtest full-page view
- `rankings_selected` — ticker string for detail panel pre-selection
- `lookup_ticker/period/data/val/risk/proto` — stock lookup cache
- `bt_result/bt_ticker/bt_period` — backtest cache

**`run_analysis(profile)`** — runs the same pipeline as main.py, returns dict:
```python
{
    "universe_data": ..., "sp500_hist": ..., "macro_data": ..., "rf_rate": float,
    "ranked_df": ..., "top10": ...,
    "valuation": valuation_results,   # {ticker: {...}}
    "risk":      risk_results,        # {ticker: {...}}
    "protocol":  protocol_results,    # {ticker: {...}}
}
```

**9 Tabs:**
1. **Rankings** — Top-3 pick cards with signal + conviction + earnings badges; full rankings table with earnings column; clickable ticker buttons + "View Full Analysis" per card; per-stock detail panel
2. **Valuation** — Valuation matrix table; entry positioning chart; method spread scatter; DCF sensitivity Bear/Base/Bull table
3. **Risk & Quality** — Risk metrics table; Sharpe vs ROIC/WACC bubble; Piotroski bar chart
4. **Protocol Gates** — Gate heatmap (Plotly); protocol summary with pass/warn/fail
5. **Portfolio** — Donut pie chart; position breakdown table; weight bar chart
6. **Macro & Performance** — Macro metric tiles; sector ETF bar; normalised price history vs S&P 500; correlation heatmap
7. **Stock Lookup** — Search any ticker; runs full ValuationEngine + RiskEngine + ProtocolAnalyzer on fresh fetch; full detail panel
8. **History** — Past sessions summary tiles + expandable session cards with evaluated returns
9. **Backtest** — Ticker/period selector; runs `_run_backtest_simulation()`; 5 summary tiles; annotated Plotly chart; trade log; strategy rules

---

## Key Helper Functions (app.py)

- `badge(text, color, bg)` — inline HTML badge
- `signal_badge(signal)`, `conv_badge(conviction)`, `zone_badge(zone)` — typed badges
- `sbar(score)` — animated score progress bar (CSS `@keyframes barGrow`)
- `mtile(label, value, sub, color, accent)` — metric tile with colored left-border
- `shdr(title, sub)` — section header HTML
- `_stat(label, value)` — inline stat block for pick cards
- `_plotly_base()` — consistent Plotly layout kwargs
- `_candlestick_fig(ticker, hist, period)` — Plotly 3-panel (price+SMAs, volume, RSI)
- `_generate_quant_thesis(ticker, val, risk, proto_dict)` → HTML paragraph auto-generating investment thesis from quant data
- `_analyst_targets_html(info)` → HTML table with mean/high/low/count/consensus
- `_technical_summary_html(info, hist)` → HTML table with SMA 200/50, 52W range, RSI, 3M momentum
- `_factor_bars_html(ticker)` → HTML 7-factor score bars (reads from `st.session_state.results.ranked_df`)
- `_run_backtest_simulation(hist, entry_low, target, stop)` → list of trade dicts
- `_render_stock_detail(ticker, universe_data, valuation, risk, protocol, rf_rate, period, fetch_fresh)` — full per-stock detail panel
- `tab_stock_lookup(...)` — Tab 7 stock search
- `tab_backtest(universe_data, valuation, risk, rf_rate)` — Tab 9 backtest
- `tab_history()` — Tab 8 past sessions
- `run_analysis(profile)` — full pipeline with `st.progress()` live updates

---

## `_render_stock_detail` layout (current design)

1. **Hero header** — gradient card, left accent border color-coded to signal, 36px price with day-change %, domain link
2. **Earnings banner** — shows up to 90 days out: ≤7d RED, ≤14d AMBER, ≤30d BLUE, ≤90d GRAY, None = "not available"
3. **Quant thesis** — blue-tinted card with auto-generated investment narrative
4. **Business description** — collapsed into st.expander
5. **Key metrics pill strip** — P/E, Forward P/E, EV/EBITDA, Gross Margin, ROE, Beta, Piotroski, Altman Z, Sharpe (color-coded)
6. **Factor score bars** — collapsed into st.expander
7. **Candlestick chart** — full width Plotly 3-panel
8. **Valuation deep dive** — full-width card with large method estimates + 7-metric summary bar (FV, entry zone, target, stop, R/R, upside, premium)
9. **DCF sensitivity** — Bear/Base/Bull table (styled card)
10. **3 columns**: Risk & Quality | Analyst Targets + Key Financials | Technical Status
11. **News feed** — full width, 12 articles, sentiment badge
12. **Protocol gates** — 7 cards with score + status, color-coded

---

## Key Modules to Know

### advisor/valuation.py — Multi-method Valuation

**IMPORTANT: estimate dict keys are SHORT lowercase:**
```python
estimates = {
    "dcf":       float,   # DCF (2-stage)
    "graham":    float,   # Graham Number
    "ev_ebitda": float,   # EV/EBITDA target
    "fcf_yield": float,   # FCF Yield @4.5%
}
```
Always use `est.get("dcf")`, `est.get("graham")`, etc. — NOT the long human-readable names.

`analyze()` return dict also includes:
```python
{
    "sensitivity": {
        "Bear": {"fair_value": float, "growth_rate": float, "signal": str, "premium_pct": float},
        "Base": {...},
        "Bull": {...},
    }
}
```

**Signals:** STRONG_BUY (≤entry_low) | BUY (≤entry_high) | HOLD_WATCH (≤FV) | WAIT (≤FV×1.10) | AVOID_PEAK (>FV×1.10)

### advisor/risk.py — Risk & Quality Metrics
Each result: `{ altman_z{score, zone}, sharpe, sortino, max_drawdown_pct, var_95_pct, roic_wacc{roic, wacc, spread, verdict}, accruals, gross_prof, piotroski{score, out_of, interpretation, passed, failed} }`

### advisor/protocol.py — Investment Protocol
`analyze_all(top10, universe_data, valuation_results)` → `[{ticker, gates[7], gate_statuses[7], pass_count, warn_count, fail_count, overall_score, conviction, entry_analysis}]`

### advisor/learner.py — Session Memory

```python
memory = SessionMemory()
memory.load()
memory.get_recent_tickers(n_sessions=2)  # tickers from last N sessions for fresh picks mode
memory.get_adapted_weights(risk_level, time_horizon)
memory.save_session(profile, top10_df, sp500_price)
memory.save()
```

Session history JSON structure:
```json
{
  "session_id": "abc12345",
  "timestamp": "...",
  "profile": {"risk_level": 2, "time_horizon": "medium", "goal": "wealth"},
  "sp500_entry": 5200.0,
  "evaluated": false,
  "evaluation": null,
  "picks": [{"ticker": "AAPL", "price_entry": 185.0, "composite_score": 87.4, "factors": {...}}]
}
```

### advisor/fetcher.py — Data

`_fetch_one()` returns:
```python
{
    "info":               dict,   # full yfinance info
    "history":            DataFrame,
    "sector":             str,
    "technical":          dict,
    "piotroski":          dict,
    "sentiment":          float,
    "news_titles":        List[str],
    "earnings_date":      str | None,   # ISO date string e.g. "2025-04-30"
    "earnings_days_away": int | None,   # days until next earnings
}
```

Earnings calendar parsed from `yf.Ticker.calendar` — handles both dict and DataFrame formats.

### advisor/cli_commands.py — Interactive Slash Command REPL

**Available commands:**

| Command | Description |
|---------|-------------|
| `/stock TICKER` | Header + quant thesis + valuation + DCF sensitivity + risk + analyst targets + technical + protocol + key financials |
| `/news TICKER [n]` | Headlines with sentiment score + per-article colour coding |
| `/chart TICKER [period]` | Dark-theme candlestick + SMA + RSI |
| `/compare T1 T2` | Side-by-side rich table |
| `/add TICKER` | Add to watchlist |
| `/remove TICKER` | Remove from watchlist |
| `/watchlist` | Show watchlist |
| `/macro` | Macro snapshot |
| `/history [n]` | Past session results |
| `/exit` | Exit REPL |

New methods on `CommandHandler`:
- `_print_quant_thesis(ticker)` — Rich yellow panel with auto-generated narrative
- `_print_analyst_targets(info)` — Consensus, # analysts, mean/high/low targets
- `_print_technical_summary(info, hist)` — SMA 200/50, 52W range, RSI, 3M momentum

---

## Backtest Strategy (`_run_backtest_simulation`)

```python
trades = _run_backtest_simulation(hist, entry_low, target, stop_loss)
# Each trade: {entry_date, entry, exit_date, exit, return_pct, reason, won}
# reason: "Target hit" | "Stop loss" | "Open position"
```

Entry rule: price ≤ entry_low (FV × 0.80)
Exit rules: price ≥ target (FV × 1.20) OR price ≤ stop_loss (entry_low × 0.92)

Uses **current** fair value as static reference. Historical fundamentals vary — treat as illustrative.

---

## Color Palettes

**CLI charts (charts.py):**
```python
BG_DARK = "#0d1117"   BG_PANEL = "#161b22"   BORDER = "#30363d"
TXT_DIM = "#8b949e"   TXT_WHITE = "#e6edf3"
PASS = "#3fb950"   WARN = "#e3b341"   FAIL = "#da3633"
```

**Streamlit dashboard (app.py):**
```python
BLUE="#2563EB"   BLUE_LT="#EFF6FF"
GREEN="#059669"  GREEN_LT="#ECFDF5"
AMBER="#D97706"  AMBER_LT="#FFFBEB"
RED="#DC2626"    RED_LT="#FEF2F2"
TEXT="#111827"   MUTED="#6B7280"   MUTED2="#9CA3AF"
BORDER="#E5E7EB" GRAY_LT="#F9FAFB"
```

**Empty field display:** `fmt_price()`, `fmt_pct()`, `fmt_2()` return a styled dim `n/a` span — never a raw "—" dash.

---

## Running the Project

```bash
# CLI pipeline
python main.py

# Streamlit dashboard
streamlit run app.py

# Install dependencies
pip install -r requirements.txt
```

Environment: Windows 11, Python 3.10, bash shell.

---

## Design Principles (don't break these)

1. **No paid data APIs** — everything via yfinance (free)
2. **No external AI APIs** — pure math: DCF, Graham, Altman Z, Piotroski, etc.
3. **Real money context** — users invest based on this output; never be vague
4. **Warren Buffett philosophy** — always check valuation, always require margin of safety
5. **CLI charts dark theme** — all matplotlib charts use the dark palette in `charts.py`
6. **Dashboard light theme** — Streamlit uses white bg + black text (`.streamlit/config.toml`)
7. **Graceful degradation** — missing yfinance fields are handled; tool always produces output
8. **Non-blocking charts** — `plt.show(block=False)` + `plt.pause(0.1)` so CLI command loop starts immediately
9. **Estimate key names are SHORT** — `"dcf"`, `"graham"`, `"ev_ebitda"`, `"fcf_yield"` — never the long human-readable versions

---

## Common Modifications

**Add a stock:** Edit `config.py` → `STOCK_UNIVERSE[sector].append("TICKER")`

**Change margin of safety:** Edit `advisor/valuation.py` → `analyze()`:
- `entry_low = fv * 0.80` (20% MoS)
- `entry_high = fv * 0.90` (10% MoS)

**Change fresh picks penalty:** Edit `main.py` step 5b and `app.py` `run_analysis()`:
- `PENALTY = 22.0` (points subtracted from composite_score)

**Change backtest entry/exit rules:** Edit `_run_backtest_simulation()` in `app.py`

**Change protocol gate weights:** Edit `advisor/protocol.py` → `GATE_WEIGHTS` (must sum to 1.0)

**Add a new factor:**
1. Add computation in `advisor/scorer.py` → `_score_one()`
2. Add to `FACTOR_NAMES` in `config.py`
3. Update `WEIGHT_MATRIX` rows
4. Update `advisor/charts.py` factor heatmap
5. Update `app.py` factor chart and `_factor_bars_html()`
