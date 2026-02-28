# Stock Ranking Advisor — Project Context for Claude

This file is auto-loaded by Claude Code at the start of every session.
Read this before making any changes to the codebase.

---

## What This Project Is

A fully local, quantitative stock ranking + deep analysis tool — no paid AI APIs required.
Users enter a portfolio size, risk level, and time horizon.
The system fetches ~110 stocks from Yahoo Finance, scores them on 7 factors,
runs a 7-gate investment protocol, performs hedge-fund grade valuation
(DCF, Graham Number, EV/EBITDA, FCF yield) and risk analysis
(Altman Z, Sharpe, Sortino, ROIC/WACC, Piotroski 9pt, VaR).

There are **two interfaces**:
- `main.py` — CLI pipeline with rich terminal output + 5 charts + Excel export
- `app.py` — Streamlit web dashboard (browser-based, localhost:8501)

Real money goes in based on this output — quality and precision matter.

---

## Project Structure

```
portfolio/
├── main.py                  ← 16-step CLI pipeline + interactive command loop
├── app.py                   ← Streamlit web dashboard (streamlit run app.py)
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
    ├── collector.py         ← UserProfile dataclass + 7-question CLI collector
    ├── fetcher.py           ← DataFetcher (yfinance), MacroFetcher, technicals, sentiment
    ├── scorer.py            ← 7-factor MultiFactorScorer with macro regime tilts
    ├── portfolio.py         ← Greedy correlation-aware selection + half-Kelly sizing
    ├── protocol.py          ← 7-gate Warren Buffett protocol + entry price analysis
    ├── valuation.py         ← ValuationEngine: DCF, Graham, EV/EBITDA, FCF yield
    ├── risk.py              ← RiskEngine: Altman Z, Sharpe, Sortino, ROIC/WACC, Piotroski
    ├── news_fetcher.py      ← NewsFetcher: yfinance + RSS + Finnhub + NewsAPI + FRED
    ├── cli_commands.py      ← Interactive slash command REPL (post-analysis)
    ├── charts.py            ← 5 dark-theme charts + candlestick() method (CLI only)
    ├── display.py           ← Rich terminal output (results, protocol, deep analysis)
    ├── exporter.py          ← Excel export (6 sheets incl. Deep Analysis)
    └── learner.py           ← Persistent session memory + adaptive weight learning
```

---

## Pipeline (main.py — 16 steps)

1. Load memory (`memory/history.json`) + evaluate past sessions
2. Show track record (win rate vs S&P 500)
3. Collect user profile (7 CLI questions)
4. Get adapted weights from learning history
5. Fetch ~110 stocks from Yahoo Finance (batched, with retry)
6. Fetch S&P 500 benchmark
7. Fetch macro data (VIX, 10Y yield, sector ETFs) → extract rf_rate
8. Score all stocks (7 factors, normalised 0–100, macro tilt)
9. Portfolio construction: top 10 via greedy correlation-aware selection + half-Kelly
10. Display quantitative results in terminal
11. **ValuationEngine** — DCF, Graham, EV/EBITDA, FCF yield → fair value, entry zone, stop loss
12. **RiskEngine** — Altman Z, Sharpe, Sortino, Max DD, VaR, ROIC/WACC, Piotroski
13. **Protocol analysis** — 7-gate Warren Buffett screen using ValuationEngine signal for Gate 4
14. **show_deep_analysis()** — scary-detailed per-stock terminal output
15. Generate 5 charts + Excel export + save session
16. **CommandHandler.run()** — interactive slash command REPL (post-analysis)

---

## Streamlit Dashboard (app.py)

Run with: `streamlit run app.py` → opens at `http://localhost:8501`

**Theme:** white background (#FFFFFF), black text (#111827), blue accent (#2563EB)

**Sidebar:** All 7 profile inputs as widgets + "Run Analysis" button.
Results cached in `st.session_state` — rerun without re-fetching.

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

**7 Tabs:**
1. **Rankings** — Top-3 pick cards with signal + conviction badges; full rankings table; stacked factor bar chart (Plotly); per-stock detail panel (candlestick + news + financials) at bottom
2. **Valuation** — Valuation matrix table; horizontal entry positioning chart (entry zone vs current price); method spread scatter
3. **Risk & Quality** — Risk metrics table; Sharpe vs ROIC/WACC bubble scatter; Piotroski bar chart
4. **Protocol Gates** — Gate heatmap (Plotly, red→amber→green); protocol summary table with pass/warn/fail dots
5. **Portfolio** — Donut pie chart (hole=0.60); position breakdown table; weight bar chart
6. **Macro & Performance** — Macro metric tiles; sector ETF returns bar chart; normalised price history vs S&P 500; correlation heatmap
7. **Stock Lookup** — Search any ticker (not just top 10); full detail panel: candlestick chart (Plotly, 3 panels: price+SMAs, volume, RSI), news feed with sentiment, key financials, valuation detail, risk metrics, 7-gate protocol cards

**Color constants in app.py:**
```python
BLUE="#2563EB", GREEN="#16a34a", AMBER="#d97706", RED="#dc2626"
TEXT="#111827", MUTED="#6B7280", BORDER="#E5E7EB"
```

**Key helper functions:**
- `badge(text, color, bg)` — inline HTML badge
- `signal_badge(signal)`, `conv_badge(conviction)`, `zone_badge(zone)` — typed badges
- `sbar(score)` — animated score progress bar HTML (CSS `@keyframes barGrow` + `--w` custom property)
- `mtile(label, value, sub, color, accent)` — metric tile HTML with colored left-border accent
- `_stat(label, value)` — inline stat block used in pick cards
- `_plotly_base()` — consistent Plotly layout kwargs (template, font, bgcolor)
- `_candlestick_fig(ticker, hist, period)` — Plotly 3-panel candlestick (price+SMAs, volume, RSI)
- `_render_stock_detail(ticker, universe_data, valuation, risk, protocol, rf_rate, period, fetch_fresh)` — full per-stock detail panel
- `tab_stock_lookup(universe_data, valuation, risk, protocol, rf_rate)` — Tab 7 stock search
- `run_analysis(profile)` — full pipeline with `st.progress()` live updates

---

## Key Modules to Know

### advisor/valuation.py — Multi-method Valuation
```python
from advisor.valuation import ValuationEngine

rf_rate = macro_data.get("yield_10y", 4.5) / 100
engine  = ValuationEngine(rf_rate)
results = engine.analyze_all(top10_df, universe_data)
# Each result: { estimates{dcf, graham, ev_ebitda, fcf_yield},
#               fair_value, entry_low(FV×0.80), entry_high(FV×0.90),
#               target_price(FV×1.20), stop_loss(entry_low×0.92),
#               premium_pct, upside_pct, rr_ratio, signal, methods_count }
```
**4 valuation methods:**
- DCF (2-stage): 5yr growth + terminal at min(3%, g×0.30), discount = rf + 5.5%
- Graham Number: sqrt(22.5 × EPS × BVPS)
- EV/EBITDA target: sector-median multiple → implied price
- FCF Yield target: FCF/share ÷ 4.5% target yield

**Signals:** STRONG_BUY (≤entry_low) | BUY (≤entry_high) | HOLD_WATCH (≤FV) | WAIT (≤FV×1.10) | AVOID_PEAK (>FV×1.10)

### advisor/risk.py — Risk & Quality Metrics
```python
from advisor.risk import RiskEngine

results = RiskEngine().analyze_all(top10_df, universe_data, rf_rate)
# Each result: { altman_z{score, zone}, sharpe, sortino,
#               max_drawdown_pct, var_95_pct,
#               roic_wacc{roic, wacc, spread, verdict},
#               accruals, gross_prof,
#               piotroski{score, out_of, interpretation, passed, failed} }
```
**Metrics computed:**
- Altman Z-Score (modified): SAFE >2.6 | GRAY 1.1–2.6 | DISTRESS <1.1
- Sharpe Ratio: (return − rf) / vol × √252
- Sortino Ratio: uses downside deviation only
- Max Drawdown: worst peak-to-trough over period
- VaR 95% (1 month): 5th percentile of 21-day rolling returns
- ROIC/WACC spread: EXCEPTIONAL >15% | STRONG >8% | POSITIVE >2% | NEUTRAL >−3% | DESTROYING VALUE
- Accruals Ratio: (net_income − OCF) / total_assets (negative = clean earnings)
- Gross Profitability: gross_profit / total_assets (Novy-Marx quality factor)
- Piotroski 9-point F-Score: full profitability + leverage + efficiency

### advisor/protocol.py — Investment Protocol
```python
from advisor.protocol import ProtocolAnalyzer, GATE_NAMES, GATE_SHORT, GATE_WEIGHTS

results = ProtocolAnalyzer().analyze_all(top10_df, universe_data, valuation_results)
# Each result: { ticker, gates[7], gate_statuses[7], pass_count, warn_count,
#               fail_count, overall_score, conviction, entry_analysis }
```
**7 Gates (weights):**
1. Business Quality (0.20) — ROA, ROE, FCF, profit margins, growth
2. Competitive Moat  (0.15) — gross margins, operating margins, scale
3. Financial Health  (0.15) — debt/equity, current ratio, interest coverage, **Altman Z**
4. Valuation         (0.22) — **ValuationEngine signal** (65%) + P/E vs sector (35%)
5. Technical Entry   (0.10) — 52-week positioning, analyst upside, forward P/E
6. News & Sentiment  (0.08) — sentiment score + analyst recommendation
7. Trend Alignment   (0.10) — SMA200, SMA50, 3-month momentum

**Gate status thresholds:** PASS ≥ 60 | WARN 35–59 | FAIL < 35

**Entry analysis**: Uses ValuationEngine results if available; falls back to 4-method simple calculation.

### advisor/fetcher.py — Data
- `DataFetcher(yf_period).fetch_universe(tickers)` → dict keyed by ticker
- Each value: `{ info, history, sector, technical, piotroski, sentiment, news_titles }`
- `news_titles` = list of up to 12 raw headline strings
- `MacroFetcher().fetch()` → `{ vix, vix_hist, yield_10y, yield_hist, sector_etf, regime, regime_reasons }`
- Technical score: RSI(25%) + MACD(30%) + MA crossover(20%) + Bollinger %B(15%) + OBV trend(10%)

### advisor/scorer.py — 7-Factor Scoring
Key upgrades from baseline:
- **Momentum**: 12-1 skip-month blend (1m×10% + 3m×25% + 6m×35% + 12-1×30%) — avoids reversal
- **Value**: P/E(40%) + EV/EBITDA(35%) + FCF yield(25%) composite
- **Quality**: includes accruals ratio and gross profitability (Novy-Marx)
- **Sentiment**: includes analyst recommendation nudge

### advisor/charts.py — Charts (CLI only)
```python
fig5 = charts.thought_process(top10_sized, protocol_results, valuation_results, risk_results)
```
Three panels:
- **Gate scorecard**: 10 stocks × 7 gates heatmap (green/amber/red)
- **Entry price positioning**: % vs fair value, shaded buy/watch/expensive zones
- **Quant conviction strip**: conviction badge + signal + auto-generated quant thesis
  (e.g. "18% below FV (3-method median) · ROIC/WACC +21.4% [EXCEPTIONAL] · Piotroski 7/9 · Altman Z [SAFE]")

**Candlestick chart (on-demand via CLI `/chart` command):**
```python
fig = charts.candlestick(ticker, history, period="6mo")
```
- Tries `mplfinance` first (custom dark `make_mpf_style`), falls back to pure matplotlib
- 3 panels: price + SMA 20/50/200 overlays, volume bars, RSI(14) with overbought/oversold fill
- Saved as `chart_{ticker}_{period}.png` or displayed via `plt.show()`

### advisor/news_fetcher.py — Multi-source News Aggregation
```python
from advisor.news_fetcher import NewsFetcher, fetch_fred_macro

nf       = NewsFetcher()
articles = nf.fetch_ticker_news("AAPL", n=15)
score    = nf.score_sentiment([a["title"] for a in articles])
market   = nf.fetch_market_news(n=20)
macro    = fetch_fred_macro(fred_key)   # CPI, FEDFUNDS, UNRATE, T10Y2Y, etc.
```
**Four news sources (all free, no subscription required):**
1. `_yf_news()` — Yahoo Finance via `yfinance.Ticker.news` (always available)
2. `_rss_news()` — Yahoo Finance RSS via `feedparser` (no key)
3. `_finnhub_news()` — Finnhub company news API (optional free key, 60 calls/min)
4. `_newsapi_news()` — NewsAPI everything endpoint (optional free key, 100 req/day)

**Article dict format:** `{title, source, url, published, summary, sentiment_hint}`
- `sentiment_hint`: `"positive"` / `"negative"` / `"neutral"` per headline (keyword matching)
- `score_sentiment()`: returns 0–100 float based on POSITIVE_WORDS / NEGATIVE_WORDS from config

**FRED macro helper:** `fetch_fred_macro(fred_key)` → `{cpi, fed_funds, unemployment, yield_curve, hy_spread, m2}`

### advisor/cli_commands.py — Interactive Slash Command REPL
```python
from advisor.cli_commands import CommandHandler

CommandHandler(
    universe_data=universe_data,
    valuation_results=valuation_results,
    risk_results=risk_results,
    protocol_results=protocol_results,
    rf_rate=rf_rate,
    macro_data=macro_data,
).run()
```
**Available commands after analysis completes:**

| Command | Description |
|---------|-------------|
| `/stock TICKER` | Full report: header + valuation + risk + protocol + key financials (rich tables) |
| `/news TICKER [n]` | Last n headlines with sentiment score + per-article colour coding |
| `/chart TICKER [period]` | Dark-theme candlestick + SMA + RSI chart (saves PNG or plt.show()) |
| `/compare T1 T2` | Side-by-side rich table: price, fundamentals, valuation, risk |
| `/add TICKER` | Add ticker to watchlist (`memory/watchlist.json`) |
| `/remove TICKER` | Remove ticker from watchlist |
| `/watchlist` | Display current watchlist with key metrics |
| `/macro` | VIX, 10Y yield, regime, sector ETF returns from cached macro data |
| `/exit` | Exit the command loop |

- Cached tickers (already in `universe_data`) are displayed instantly; uncached tickers trigger a fresh `DataFetcher("2y").fetch_universe([ticker])` call
- Watchlist persisted at `memory/watchlist.json` (relative to `advisor/cli_commands.py`)

### advisor/exporter.py — Excel (CLI only)
```python
ExcelExporter().export(
    top10, macro_data, profile, memory, top10,
    protocol_results=protocol_results,
    valuation_results=valuation_results,
    risk_results=risk_results,
)
```
Sheets: Latest Picks | Allocation | Macro Overview | History | Track Record | Deep Analysis

The **Deep Analysis** sheet has three sections:
1. Gate scorecard (with ValuationEngine-sourced fair value columns)
2. Multi-method valuation detail (all 4 methods + entry zone + stop loss + R/R ratio)
3. Risk & quality metrics (Altman Z zone, Sharpe, Sortino, ROIC/WACC, Piotroski)

---

## config.py Reference

```python
STOCK_UNIVERSE      # 110 stocks across 9 sectors
SECTOR_MEDIAN_PE    # { "Technology": 28, "Healthcare": 22, ... }
SECTOR_EV_EBITDA    # { "Technology": 22, "Healthcare": 16, ... }
WEIGHT_MATRIX       # { (risk_level, time_horizon): [7 floats] }
FACTOR_NAMES        # ["momentum","volatility","value","quality","technical","sentiment","dividend"]
RISK_LABELS         # { 1: "Conservative", 2: "Moderate", 3: "Aggressive", 4: "Speculative" }
GOAL_LABELS         # { "growth": ..., "income": ..., "preserve": ..., "speculative": ... }
HORIZON_LABELS      # { "short": ..., "medium": ..., "long": ... }
MACRO_TILTS         # sector adjustments per regime (risk_on/risk_off/rising_rate/falling_rate)
POSITIVE_WORDS      # news sentiment positive keywords
NEGATIVE_WORDS      # news sentiment negative keywords

# Optional free API keys — leave empty string "" to skip that source
FINNHUB_KEY       = ""   # https://finnhub.io — 60 calls/min free tier
NEWSAPI_KEY       = ""   # https://newsapi.org — 100 req/day free tier
FRED_KEY          = ""   # https://fred.stlouisfed.org — unlimited free
ALPHAVANTAGE_KEY  = ""   # https://alphavantage.co — 25 req/day free tier (reserved)
```

---

## Running the Project

```bash
# CLI pipeline (terminal output + charts + Excel)
python main.py

# Streamlit web dashboard (browser-based)
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

---

## Color Palettes

**CLI charts (charts.py):**
```python
BG_DARK   = "#0d1117"   # figure background
BG_PANEL  = "#161b22"   # axes background
BORDER    = "#30363d"
TXT_DIM   = "#8b949e"
TXT_WHITE = "#e6edf3"
PASS      = "#3fb950"   # green
WARN      = "#e3b341"   # amber
FAIL      = "#da3633"   # red
```

**Streamlit dashboard (app.py):**
```python
BLUE  = "#2563EB"   # primary accent
GREEN = "#16a34a"   # PASS / STRONG BUY
AMBER = "#d97706"   # WARN / HOLD
RED   = "#dc2626"   # FAIL / AVOID
TEXT  = "#111827"   # body text
MUTED = "#6B7280"   # secondary text
```

---

## Common Modifications

**Add a stock to the universe:**
Edit `config.py` → `STOCK_UNIVERSE[sector].append("TICKER")`

**Change protocol gate weights:**
Edit `advisor/protocol.py` → `GATE_WEIGHTS` list (must sum to 1.0)

**Change fair value margin of safety:**
Edit `advisor/valuation.py` → `analyze()`:
- `entry_low  = fv * 0.80`  (currently 20% MoS)
- `entry_high = fv * 0.90`  (currently 10% MoS)

**Change DCF discount rate or growth assumptions:**
Edit `advisor/valuation.py` → `__init__()` and `_dcf()`

**Change Altman Z zone thresholds:**
Edit `advisor/risk.py` → `altman_z()` → SAFE/GRAY/DISTRESS cutoffs

**Add a new factor to the 7-factor model:**
1. Add raw score computation in `advisor/scorer.py` → `_score_one()`
2. Add to `FACTOR_NAMES` in `config.py`
3. Update `WEIGHT_MATRIX` rows to include the new weight
4. Update `advisor/charts.py` factor heatmap if needed
5. Update `app.py` factor chart if needed
