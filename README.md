# Stock Ranking Advisor v3

A fully local, hedge-fund grade quantitative stock analysis tool built entirely on free data.
No paid APIs. No AI subscriptions. Just Yahoo Finance + serious math.

It ranks ~110 stocks across your investment profile, runs a 7-gate Warren Buffett protocol,
computes intrinsic value using 4 independent methods, backtests the strategy on historical prices,
and delivers a level of analysis that would cost thousands per month on a professional terminal — for free.

---

## Two Ways to Run

### CLI Pipeline (terminal + charts + Excel)
```bash
pip install -r requirements.txt
python main.py
```

### Streamlit Web Dashboard (browser-based)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501` — white background, black text, 10 interactive tabs.

---

## What's New (latest session)

- **Portfolio backtest** — backtests the entire 10-stock basket simultaneously; equal-weighted equity curve vs S&P 500 with individual stock lines; 5 aggregate metrics (return, win rate, buy-and-hold, S&P 500, alpha); per-stock breakdown table — no ticker selector needed
- **Earnings Calendar tab** — dedicated tab 10 + sidebar button; shows all 10 picks sorted by urgency (≤7d RED / ≤14d AMBER / ≤30d BLUE / >30d GRAY); hero countdown banner for nearest earnings; horizontal timeline chart; valuation signal + price vs FV on each card; earnings risk guidance
- **History full analysis** — each session now expands into the complete picture: full profile header with exact timestamp, per-pick cards showing all 7 factor score bars, entry price, exit price + P&L (for evaluated sessions); pending sessions show an amber evaluation countdown
- **Backtest engine** — simulate our valuation entry strategy (buy at entry zone, sell at target / stop loss) on any ticker with annotated chart, trade log, strategy vs buy-and-hold vs S&P 500
- **Past Sessions** — `/history` CLI command + dedicated web tab + sidebar button; shows every session with win rates, alpha, and P&L
- **Fresh picks mode** — Q8 in CLI / checkbox in web; applies a −22pt penalty to stocks from your last 2 sessions so the tool recommends new ideas each run
- **Quant thesis auto-generator** — auto-generates a Citadel-style investment thesis per stock (valuation signal, ROIC/WACC, Piotroski, Altman Z, Sharpe, protocol conviction) — web and CLI
- **Analyst targets panel** — consensus recommendation, # analysts, mean/high/low price targets with upside % — web and CLI
- **Technical status panel** — plain-English SMA 200/50 positioning, 52-week range, RSI, 3M momentum — web and CLI
- **7-factor score bars** — visual breakdown of the 7 scoring factors per stock in the detail panel
- **DCF sensitivity** — Bear / Base / Bull scenario table (50% / 100% / 150% of base growth rate)
- **Citadel-level stock detail redesign** — gradient hero header, key metrics pill strip, full-width valuation deep dive card, 3-column risk/analyst/technical section, full-width news feed, larger protocol gates
- **Valuation estimates fix** — all 4 method estimates now display correctly everywhere
- **No more dashes** — empty fields show styled `n/a` instead of raw `—`
- **plt.show() fix** — CLI chart windows are non-blocking; command loop starts immediately

---

## Streamlit Dashboard — 10 Tabs

| Tab | Contents |
|-----|----------|
| **Rankings** | Top-3 pick cards with signal badges + earnings alerts; full rankings table; stacked factor bar chart; clickable per-stock detail panel |
| **Valuation** | 4-method valuation matrix; entry positioning chart; DCF sensitivity Bear/Base/Bull table |
| **Risk & Quality** | Risk metrics table; Sharpe vs ROIC/WACC bubble scatter; Piotroski bar chart |
| **Protocol Gates** | 10×7 gate heatmap; protocol summary with pass/warn/fail indicators |
| **Portfolio** | Donut allocation chart; position breakdown table with weight bars |
| **Macro & Performance** | VIX + 10Y yield tiles; sector ETF returns; normalised price history vs S&P 500; correlation heatmap |
| **Stock Lookup** | Search any ticker — full quant detail: candlestick chart, news, valuation, risk, protocol, analyst targets, technical status |
| **History** | Per-session expanders showing full pick cards with all 7 factor bars, entry price, exit P&L; aggregate win rate / alpha tiles |
| **Backtest** | Portfolio-wide backtest: all 10 picks simultaneously; equal-weighted equity curve vs S&P 500; 5 aggregate tiles; per-stock breakdown table |
| **Calendar** | Earnings timeline for all 10 picks sorted by urgency; hero countdown banner; horizontal bar chart; valuation signal on each card; event-risk guidance |

---

## What It Does (16-Step Pipeline)

| Step | Action |
|------|--------|
| 1–2 | Load session memory, show track record vs S&P 500 |
| 3–4 | Collect 8-question investor profile (incl. fresh picks mode) |
| 5–7 | Fetch ~110 stocks, S&P 500 benchmark, and macro data (VIX, 10Y yield, sector ETFs) |
| 8 | Score all stocks on 7 factors with macro regime tilt; optionally penalise recent picks |
| 9 | Select top 10 via greedy correlation-aware algorithm + half-Kelly position sizing |
| 10 | Display ranked results and allocation table |
| 11 | **ValuationEngine** — 4-method fair value, entry zones, stop loss, risk/reward ratio, DCF sensitivity |
| 12 | **RiskEngine** — Altman Z, Sharpe, Sortino, Max DD, VaR, ROIC/WACC, Piotroski |
| 13 | **7-Gate Protocol** — quality, moat, health, valuation, entry, news, trend |
| 14 | **Deep analysis output** — full per-stock terminal report with quant thesis |
| 15 | 5 charts + Excel export (6 sheets) + save session to memory |
| 16 | **Interactive command loop** — `/stock`, `/news`, `/chart`, `/compare`, `/watchlist`, `/macro`, `/history` |

---

## The 8-Question Investor Profile

| # | Question | Why It Matters |
|---|----------|----------------|
| 1 | Portfolio size | Drives dollar amounts in allocation table |
| 2 | Time horizon (1yr / 3yr / 5yr) | Short-term → momentum-heavy; long-term → value + quality |
| 3 | Risk tolerance (1–4) | Controls beta filters and factor weight distribution |
| 4 | Investment goal | Income goal lifts dividend weight; speculative lifts momentum |
| 5 | Drawdown gut check | Adds volatility penalty if you can't stomach large drops |
| 6 | Sector focus / exclusions | Filters universe; sector score bonus for preferred sectors |
| 7 | Existing holdings | Removes overlap from recommendations |
| 8 | **Fresh picks mode** | Penalises last 2 sessions' picks −22pts to force new ideas |

---

## The 7-Factor Scoring Model

Each stock is scored 0–100 on seven independent factors, then combined using a weight matrix tuned to your risk profile and time horizon.

### Factor 1 — Momentum (12-1 skip-month)
Academic-grade momentum that avoids the 1-month reversal effect:
- 1-month (10%) + 3-month (25%) + 6-month (35%) + **12-1 skip-month (30%)**

### Factor 2 — Volatility
Annualised standard deviation, **inverted** — low volatility scores high.

### Factor 3 — Value (EV/EBITDA composite)
- P/E vs sector median (40%) + EV/EBITDA vs sector median (35%) + FCF yield (25%)

### Factor 4 — Quality (accruals + gross profitability)
- Piotroski-style 8-point checklist (80%) + Accruals ratio (20%)
- Gross Profitability (Novy-Marx 2013) as quality enhancement

### Factor 5 — Technical (5-factor)
- RSI 14-day (25%) + MACD 12/26/9 (30%) + MA crossover (20%) + Bollinger %B (15%) + OBV trend (10%)

### Factor 6 — Sentiment
- Last 12 news headlines scored via 50+ financial keywords, nudged by analyst consensus

### Factor 7 — Dividend
- Raw yield capped at 15%; heavily weighted only for income-focused profiles

---

## Backtest Strategy

The backtest simulates our valuation entry strategy on historical daily closing prices using the **current** fair value estimates as static reference levels:

| Signal | Trigger | Action |
|--------|---------|--------|
| **Entry** | Price ≤ entry_low (FV × 0.80) | Buy — 20% margin of safety |
| **Take profit** | Price ≥ target (FV × 1.20) | Sell — 20% above fair value |
| **Stop loss** | Price ≤ stop (entry_low × 0.92) | Sell — 8% below entry |

**Output:** Equal-weighted portfolio return, win rate, buy-and-hold, S&P 500, alpha vs benchmark, multi-line equity curve chart, per-stock breakdown table, per-trade log per stock.

> Note: uses current fundamental data — historical fundamentals vary, so treat as illustrative.

---

## ValuationEngine — 4 Independent Methods

| Method | Formula | What It Captures |
|--------|---------|-----------------|
| **DCF (2-stage)** | FCF/share × 5yr growth + terminal value, discounted at rf + 5.5% | Future cash generation |
| **Graham Number** | √(22.5 × EPS × Book Value/share) | Graham's classic intrinsic value |
| **EV/EBITDA Target** | EBITDA × sector median multiple → implied price | How the market values peers |
| **FCF Yield Target** | FCF/share ÷ 4.5% target yield | Price at a 4.5% free cash return |

**Output per stock:** Fair value · Entry low (−20%) · Entry high (−10%) · Target (+20%) · Stop loss · R/R ratio · Signal · DCF Bear/Base/Bull sensitivity

**Signals:** `STRONG_BUY` | `BUY` | `HOLD_WATCH` | `WAIT` | `AVOID_PEAK`

---

## RiskEngine — Full Risk & Quality Suite

| Metric | What It Measures |
|--------|-----------------|
| **Altman Z-Score** | Bankruptcy risk — SAFE (>2.6) / GRAY / DISTRESS (<1.1) |
| **Sharpe Ratio** | Risk-adjusted return: (return − rf) / vol × √252 |
| **Sortino Ratio** | Downside-only Sharpe |
| **Max Drawdown** | Worst peak-to-trough over the period |
| **VaR 95% (1mo)** | 5th percentile of 21-day rolling returns |
| **ROIC/WACC Spread** | Value creation spread — EXCEPTIONAL / STRONG / POSITIVE / NEUTRAL / DESTROYING VALUE |
| **Accruals Ratio** | Earnings quality — negative = cash-backed, not accounting fiction |
| **Gross Profitability** | Novy-Marx (2013) quality factor |
| **Piotroski F-Score** | Full 9-point: profitability + leverage + efficiency |

---

## The 7-Gate Investment Protocol

| Gate | Weight | What It Checks |
|------|--------|---------------|
| 1. Business Quality | 20% | ROA, ROE, FCF, profit margins, growth |
| 2. Competitive Moat | 15% | Gross margins, operating margins, scale |
| 3. Financial Health | 15% | Debt/equity, current ratio, interest coverage, Altman Z |
| 4. Valuation | 22% | ValuationEngine signal (65%) + P/E vs sector (35%) |
| 5. Technical Entry | 10% | 52-week positioning, analyst upside, forward P/E |
| 6. News & Sentiment | 8% | Sentiment score + analyst recommendation |
| 7. Trend Alignment | 10% | SMA200, SMA50, 3-month momentum |

**Gate thresholds:** PASS (≥60) | WARN (35–59) | FAIL (<35)
**Conviction:** HIGH (≤1 fail, ≥70 overall, ≥6 pass) | MEDIUM (≤2 fails) | LOW (3+ fails)

---

## CLI Slash Commands

After the analysis runs, the terminal enters an interactive command loop. Top-10 data is instant; other tickers fetch on demand.

| Command | Description |
|---------|-------------|
| `/stock AAPL` | Full report: quant thesis, valuation (4 methods + DCF sensitivity), risk metrics, analyst targets, technical status, 7-gate protocol, key financials |
| `/news AAPL [15]` | Latest headlines with per-article sentiment colour coding |
| `/chart AAPL [6mo]` | Dark-theme candlestick + SMA 20/50/200 + RSI panel |
| `/compare AAPL MSFT` | Side-by-side: price, P/E, EV/EBITDA, Sharpe, Piotroski, signal |
| `/add AAPL` | Add to persistent watchlist (`memory/watchlist.json`) |
| `/remove AAPL` | Remove from watchlist |
| `/watchlist` | Display watchlist with live prices |
| `/macro` | VIX, 10Y yield, regime, sector ETF returns |
| `/history [n]` | Past session results — win rate, alpha, per-pick returns |
| `/exit` | Exit command loop |

---

## Adaptive Learning

Every session is saved to `memory/history.json`. After 30+ days:
1. Current prices are fetched for all past picks
2. Each pick's return vs S&P 500 is calculated
3. Pearson correlation identifies which factors predicted returns
4. Factors with avg r > 0.3 → weight +4%; avg r < 0.0 → weight −4%
5. Weights renormalised and floored at 3%

---

## Macro Regime Detection

| Regime | Trigger | Score Tilt |
|--------|---------|------------|
| Risk-On | VIX < 16 | +4 Tech, +3 Consumer, −5 Utilities |
| Risk-Off | VIX > 27 | +7 Utilities, +5 Healthcare, −4 Tech |
| Rising Rate | 10Y yield up >0.35% in 1mo | +5 Financials, −7 REITs |
| Falling Rate | 10Y yield down >0.30% | +5 REITs, +5 Utilities, +3 Tech |

---

## Portfolio Construction

1. Top 30 candidates by composite score enter the pool
2. Pearson correlation matrix computed on daily returns
3. Greedy diversification: each pick chosen for score × (0.70 + 0.30 × (1 − avg_corr))
4. Position sizing: half-Kelly, capped at 20%, renormalised to 100%

---

## CLI Charts (5 PNGs)

| File | Contents |
|------|----------|
| `chart1_score_breakdown.png` | Stacked horizontal bars — factor contribution per stock |
| `chart2_performance.png` | Normalised price history vs S&P 500 |
| `chart3_factor_heatmap.png` | 10 × 7 colour grid of all factor scores |
| `chart4_macro_dashboard.png` | VIX · 10Y yield · sector ETF returns · correlation matrix |
| `chart5_quant_protocol.png` | Gate scorecard · Entry price positioning · Quant thesis per stock |

---

## Excel Export (Book1.xlsx — 6 Sheets)

| Sheet | Contents |
|-------|----------|
| **Latest Picks** | Top 10 with all 7 factor scores |
| **Allocation** | Weight%, dollar amounts, approx share counts |
| **Macro Overview** | VIX, 10Y yield, regime, sector ETF rankings |
| **History** | All past sessions with tickers and entry prices |
| **Track Record** | Evaluated sessions — avg return, S&P return, alpha |
| **Deep Analysis** | Gate scorecard · 4-method valuation detail · Risk metrics |

---

## File Structure

```
portfolio/
├── main.py                 ← CLI pipeline (16 steps) + interactive command loop
├── app.py                  ← Streamlit dashboard (10 tabs)
├── config.py               ← Universe, weight matrix, sector multiples, optional API keys
├── requirements.txt
├── .streamlit/config.toml  ← Streamlit theme
├── advisor/
│   ├── collector.py        ← 8-question profile builder (incl. fresh picks mode)
│   ├── fetcher.py          ← yfinance + 5-factor technicals + sentiment + earnings calendar
│   ├── scorer.py           ← 7-factor scoring with 12-1 momentum, EV/EBITDA, accruals
│   ├── portfolio.py        ← Correlation-aware selection + half-Kelly sizing
│   ├── valuation.py        ← DCF · Graham · EV/EBITDA · FCF yield + DCF sensitivity
│   ├── risk.py             ← Altman Z · Sharpe · Sortino · ROIC/WACC · Piotroski
│   ├── protocol.py         ← 7-gate Warren Buffett investment protocol
│   ├── learner.py          ← Session memory + adaptive weights + get_recent_tickers()
│   ├── news_fetcher.py     ← Multi-source: yfinance + RSS + Finnhub + NewsAPI + FRED
│   ├── cli_commands.py     ← Interactive REPL (/stock /news /chart /compare /history …)
│   ├── charts.py           ← 5 dark-theme charts + candlestick() (CLI)
│   ├── display.py          ← Terminal output + deep analysis
│   └── exporter.py         ← Excel export (6 sheets)
├── memory/
│   ├── history.json        ← Auto-created session log
│   └── watchlist.json      ← CLI /add watchlist
└── Book1.xlsx              ← Auto-generated on each CLI run
```

---

## Dependencies

```
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
rich>=13.0.0
openpyxl>=3.1.0
streamlit>=1.32.0
plotly>=5.18.0
mplfinance>=0.12.9
feedparser>=6.0.0
requests>=2.31.0
```

No paid APIs required. All optional keys default to empty — the tool works fully without them.

---

## Optional Free API Keys

Add to `config.py` to unlock additional news and macro data sources:

```python
FINNHUB_KEY  = ""   # finnhub.io — 60 calls/min free
NEWSAPI_KEY  = ""   # newsapi.org — 100 req/day free
FRED_KEY     = ""   # fred.stlouisfed.org — unlimited free (CPI, FEDFUNDS, T10Y2Y …)
```

---

## Disclaimer

This tool is for **educational and informational purposes only**.
Past performance does not guarantee future results.
Rankings, valuations, and backtest results are quantitative outputs and are **not financial advice**.
Always conduct your own due diligence before making any investment decisions.
