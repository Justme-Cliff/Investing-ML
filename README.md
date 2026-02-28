# Stock Ranking Advisor v3

A fully local, hedge-fund grade quantitative stock analysis tool built entirely on free data.
No paid APIs. No AI subscriptions. Just Yahoo Finance + serious math.

It ranks ~110 stocks across your investment profile, runs a 7-gate Warren Buffett protocol,
computes intrinsic value using 4 independent methods, and delivers a level of analysis that
would cost thousands per month on a professional terminal — for free.

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
Opens at `http://localhost:8501` — white background, black text, 7 interactive tabs.

---

## What's New in v3

- **Multi-method valuation engine** — DCF, Graham Number, EV/EBITDA target, FCF yield target
- **Full risk suite** — Altman Z-Score, Sharpe, Sortino, Max Drawdown, VaR 95%, ROIC/WACC spread
- **Full Piotroski F-Score** — all 9 points across profitability, leverage, and efficiency
- **12-1 skip-month momentum** — the academic-grade version that avoids short-term reversal
- **EV/EBITDA in value scoring** — sector-median enterprise value multiples
- **Accruals ratio + Gross Profitability** — earnings quality and Novy-Marx (2013) quality factor
- **5-factor technical score** — added Bollinger %B and OBV trend to RSI + MACD + MA
- **7-gate protocol** — Gate 3 now includes Altman Z; Gate 4 driven by the valuation engine
- **Scary-detailed terminal output** — full valuation matrix, ROIC/WACC verdict, Piotroski, risk profile per stock
- **Quantitative protocol chart** — auto-generated quant thesis per stock (no AI)
- **Deep Analysis Excel sheet** — 3 sections: gate scorecard, valuation detail, risk metrics
- **Streamlit dashboard** — fully interactive browser UI with Plotly charts (7 tabs)
- **Multi-source news aggregation** — Yahoo Finance + RSS + optional Finnhub + NewsAPI (all free)
- **CLI slash commands** — interactive REPL after analysis: `/stock`, `/news`, `/chart`, `/compare`, `/watchlist`, `/macro`
- **Candlestick charts** — dark-theme OHLCV + SMA 20/50/200 + RSI panel (CLI) and Plotly interactive (web)
- **Web stock detail view** — click any stock in Rankings or use Stock Lookup tab for full per-stock analysis
- **Optional free API keys** — Finnhub, NewsAPI, FRED for richer news and macro data

---

## Streamlit Dashboard — 7 Tabs

| Tab | Contents |
|-----|----------|
| **Rankings** | Top-3 pick cards; full rankings table; stacked factor bar chart; per-stock detail panel at bottom |
| **Valuation** | 4-method valuation matrix; entry positioning chart (current price vs entry zone); method spread |
| **Risk & Quality** | Risk metrics table; Sharpe vs ROIC/WACC bubble scatter; Piotroski F-Score bar chart |
| **Protocol Gates** | 10×7 gate heatmap (Plotly, red→amber→green); protocol summary with pass/warn/fail indicators |
| **Portfolio** | Donut allocation chart; position breakdown table with weight bars |
| **Macro & Performance** | VIX + 10Y yield tiles; sector ETF returns bar; normalised price history vs S&P 500; correlation heatmap |
| **Stock Lookup** | Search any ticker — full detail: candlestick chart, news feed, key financials, valuation, risk, 7-gate protocol |

---

## What It Does (16-Step Pipeline)

| Step | Action |
|------|--------|
| 1–2 | Load session memory, show track record vs S&P 500 |
| 3–4 | Collect 7-question investor profile |
| 5–7 | Fetch ~110 stocks, S&P 500 benchmark, and macro data (VIX, 10Y yield, sector ETFs) |
| 8 | Score all stocks on 7 factors with macro regime tilt |
| 9 | Select top 10 via greedy correlation-aware algorithm + half-Kelly position sizing |
| 10 | Display ranked results and allocation table |
| 11 | **ValuationEngine** — 4-method fair value, entry zones, stop loss, risk/reward ratio |
| 12 | **RiskEngine** — Altman Z, Sharpe, Sortino, Max DD, VaR, ROIC/WACC, Piotroski |
| 13 | **7-Gate Protocol** — each stock must pass quality, moat, health, valuation, entry, news, trend |
| 14 | **Deep analysis output** — full per-stock terminal report (valuation matrix, quality metrics, risk profile) |
| 15 | 5 charts + Excel export + save session to memory |
| 16 | **Interactive command loop** — `/stock`, `/news`, `/chart`, `/compare`, `/watchlist`, `/macro` |

---

## The 7-Question Investor Profile

| # | Question | Why It Matters |
|---|----------|----------------|
| 1 | Portfolio size | Drives dollar amounts and approx share counts |
| 2 | Time horizon (1yr / 3yr / 5yr) | Short-term → momentum-heavy; long-term → value + quality |
| 3 | Risk tolerance (1–4) | Controls beta filters and factor weight distribution |
| 4 | Investment goal | Income goal lifts dividend weight; speculative lifts momentum |
| 5 | Drawdown gut check | Adds volatility penalty if you can't stomach large drops |
| 6 | Sector focus / exclusions | Filters universe; sector score bonus for preferred sectors |
| 7 | Existing holdings | Removes overlap from recommendations |

---

## The 7-Factor Scoring Model

Each stock is scored 0–100 on seven independent factors, then combined using a weight matrix tuned to your risk profile and time horizon.

### Factor 1 — Momentum (upgraded: 12-1 skip-month)
Academic-grade momentum that avoids the well-documented 1-month reversal effect:
- 1-month (10%) + 3-month (25%) + 6-month (35%) + **12-1 skip-month (30%)**
- Skip-month = return from 252 days ago to 21 days ago (excludes recent reversal window)

### Factor 2 — Volatility
Annualised standard deviation of daily returns, **inverted** — low volatility scores high.
Heavily weighted for conservative profiles; near-zero for speculative.

### Factor 3 — Value (upgraded: EV/EBITDA composite)
Three-signal composite:
- **P/E vs sector median (40%)** — trailing P/E relative to peers
- **EV/EBITDA vs sector median (35%)** — enterprise-level valuation multiple
- **FCF yield (25%)** — free cash flow / market cap (quality-adjusted value)

### Factor 4 — Quality (upgraded: accruals + gross profitability)
Extended quality model:
- **Piotroski-style 8-point checklist** (ROA, OCF, FCF, D/E, CR, margins, rev growth, earnings growth) — 80% weight
- **Accruals ratio** — (net income − OCF) / total assets; negative = earnings backed by cash, not accounting — 20% blend
- **Gross Profitability** (Novy-Marx 2013) — gross profit / total assets; high = durable competitive advantage

### Factor 5 — Technical (upgraded: 5-factor)
All computed from price and volume history — no extra data needed:
- **RSI 14-day (25%)** — ideal zone 40–65; oversold gets contrarian bonus
- **MACD 12/26/9 (30%)** — above signal + positive histogram = bullish
- **MA crossover (20%)** — price > SMA50 > SMA200 = strong uptrend; golden cross bonus
- **Bollinger %B (15%)** — price position within band; 0.20–0.65 = healthy range
- **OBV trend (10%)** — OBV 20-day SMA > OBV 50-day SMA = smart money confirmation

### Factor 6 — Sentiment
Fetches the last 12 news headlines via Yahoo Finance (no extra API).
Scores using 50+ curated financial positive/negative keywords → 0–100.
Nudged by analyst consensus recommendation (strong_buy adds +15, strong_sell −25).

### Factor 7 — Dividend
Raw dividend yield (capped at 15% to prevent outliers dominating).
Heavily weighted only for income-focused profiles.

---

## Weight Matrix (examples)

| Profile | Mom | Vol | Val | Qual | Tech | Sent | Div |
|---------|-----|-----|-----|------|------|------|-----|
| Low risk / Long term | 5% | 18% | 27% | 25% | 5% | 5% | 15% |
| Moderate / Medium | 18% | 14% | 22% | 25% | 11% | 5% | 5% |
| High risk / Short term | 38% | 7% | 12% | 22% | 16% | 5% | 0% |
| Speculative / Short | 45% | 4% | 8% | 22% | 16% | 5% | 0% |

---

## Macro Regime Detection

Fetched live from Yahoo Finance (^VIX, ^TNX, sector ETFs XLK/XLV/XLF/...):

| Regime | Trigger | Score Tilt |
|--------|---------|------------|
| Risk-On | VIX < 16 | +4 Tech, +3 Consumer, −5 Utilities |
| Risk-Off | VIX > 27 | +7 Utilities, +5 Healthcare, −4 Tech |
| Rising Rate | 10Y yield up >0.35% in 1mo | +5 Financials, −7 REITs, −5 Utilities |
| Falling Rate | 10Y yield down >0.30% | +5 REITs, +5 Utilities, +3 Tech |

Top-3 sector ETFs by 3-month performance also receive a +3 bonus.

---

## Portfolio Construction

1. **Top 30 candidates** by composite score enter the selection pool
2. **Pearson correlation matrix** computed on daily returns
3. **Greedy diversification** — each pick chosen to maximise `score × (0.70 + 0.30 × (1 − avg_corr_with_selected))`
4. Final 10 are both high-scoring **and** genuinely uncorrelated

**Position sizing (half-Kelly):**
- `kelly = (score/100) / vol² / 2`
- Capped at 20% per position, renormalised to sum = 100%

---

## ValuationEngine — 4 Independent Methods

Every stock in the top 10 is valued using four fundamentally different approaches.
When multiple methods converge on a similar price, that convergence **is** the conviction signal.

| Method | Formula | What It Captures |
|--------|---------|-----------------|
| **DCF (2-stage)** | FCF/share × 5yr growth + terminal value, discounted at rf + 5.5% | Future cash generation ability |
| **Graham Number** | √(22.5 × EPS × Book Value/share) | Benjamin Graham's classic intrinsic value |
| **EV/EBITDA Target** | EBITDA × sector median multiple → implied price | How the market values peers |
| **FCF Yield Target** | FCF/share ÷ 4.5% target yield | Price at which the stock pays 4.5% in free cash |

**Output per stock:**
- Fair value (median of available methods)
- Entry low (FV × 0.80) — strong buy zone, 20% margin of safety
- Entry high (FV × 0.90) — buy zone, 10% margin of safety
- Target price (FV × 1.20)
- Stop loss (entry_low × 0.92)
- Risk/reward ratio (upside to target ÷ downside to stop)
- Signal: `STRONG_BUY` | `BUY` | `HOLD_WATCH` | `WAIT` | `AVOID_PEAK`

---

## RiskEngine — Full Risk & Quality Suite

| Metric | What It Measures |
|--------|-----------------|
| **Altman Z-Score** | Bankruptcy risk — SAFE (>2.6) / GRAY (1.1–2.6) / DISTRESS (<1.1) |
| **Sharpe Ratio** | Risk-adjusted return: (return − rf) / vol × √252 |
| **Sortino Ratio** | Downside-only Sharpe — penalises losses, not upside volatility |
| **Max Drawdown** | Worst peak-to-trough over the period |
| **VaR 95% (1mo)** | 5th percentile of 21-day rolling returns — worst month in 20 |
| **ROIC/WACC Spread** | ROA as ROIC proxy vs CAPM WACC; positive = value creation |
| **Accruals Ratio** | (Net Income − OCF) / Total Assets; negative = clean, cash-backed earnings |
| **Gross Profitability** | Gross Profit / Total Assets (Novy-Marx 2013 quality factor) |
| **Piotroski F-Score** | Full 9-point: 4 profitability + 3 leverage/liquidity + 2 efficiency signals |

ROIC/WACC verdicts: `EXCEPTIONAL` (>15%) | `STRONG` (>8%) | `POSITIVE` (>2%) | `NEUTRAL` | `DESTROYING VALUE`

---

## The 7-Gate Investment Protocol

Every stock in the top 10 must pass through 7 quality gates before a buy signal is issued.

| Gate | Weight | What It Checks |
|------|--------|---------------|
| 1. Business Quality | 20% | ROA, ROE, FCF, profit margins, revenue & earnings growth |
| 2. Competitive Moat | 15% | Gross margins, operating margins, market cap (scale) |
| 3. Financial Health | 15% | Debt/equity, current ratio, interest coverage, **Altman Z-Score** |
| 4. Valuation | 22% | **ValuationEngine signal (65%)** + P/E vs sector + FCF yield + PEG (35%) |
| 5. Technical Entry | 10% | 52-week high proximity, analyst consensus upside, forward P/E |
| 6. News & Sentiment | 8% | Sentiment score + analyst recommendation |
| 7. Trend Alignment | 10% | SMA200, SMA50, 3-month momentum |

**Gate thresholds:** PASS (≥60) | WARN (35–59) | FAIL (<35)

**Conviction levels:**
- `HIGH` — ≤1 FAIL, overall ≥70, ≥6 gates pass
- `MEDIUM` — ≤2 FAILs, ≥4 gates pass
- `LOW` — 3+ FAILs

---

## Per-Stock Deep Analysis Output (CLI)

For each of the top 10, the terminal prints a full investment brief:

```
#1  AAPL  —  Apple Inc.  |  Technology  |  Composite: 87.4/100
┌── VALUATION (4 independent methods)
│  DCF (2-stage)        $198.50  │  current +17.9% premium
│  Graham Number        $185.30  │  current +26.2% premium
│  EV/EBITDA Target     $241.80  │  current  -3.4% discount
│  FCF Yield @4.5%      $227.50  │  current  +2.8% premium
│  ─────────────────────────────────────────────────────────
│  Median Fair Value  : $213.30  │  Entry Zone: $170.60 – $192.00
│  Target Price       : $256.00  │  Stop Loss: $157.00  │  R/R: 1.8:1
└── Signal: ⏳ WAIT  (9.8% above fair value)

┌── QUALITY & VALUE CREATION
│  ROIC 28.4%  WACC ~9.7%  Spread +18.7%  [EXCEPTIONAL — significant economic value added]
│  Piotroski 7/9 [STRONG]  ·  Accruals -0.040 CLEAN (cash > accounting earnings)
│  Gross Profit / Assets 0.43
└──

┌── RISK PROFILE
│  Altman Z 4.2 [✓ SAFE]  ·  Sharpe 1.82  ·  Sortino 2.41
│  Max DD -22.1%  ·  VaR(95% 1mo) -8.4%  ·  Beta 1.31  ·  Ann. Vol 32.1%
└──
```

---

## CLI Charts (5 PNGs)

| File | Contents |
|------|----------|
| `chart1_score_breakdown.png` | Stacked horizontal bars — factor contribution per stock |
| `chart2_performance.png` | Normalised price history vs S&P 500 |
| `chart3_factor_heatmap.png` | 10 × 7 colour grid of all factor scores |
| `chart4_macro_dashboard.png` | VIX · 10Y yield · sector ETF returns · correlation matrix |
| `chart5_quant_protocol.png` | **Protocol gate scorecard · Entry price positioning · Quant thesis per stock** |

Chart 5 auto-generates a one-line thesis from the data, e.g.:
> *"18% below FV (3-method median)  ·  ROIC/WACC +21.4% [EXCEPTIONAL]  ·  Piotroski 7/9  ·  Altman Z [SAFE]"*

---

## Excel Export (Book1.xlsx — 6 Sheets)

| Sheet | Contents |
|-------|----------|
| **Latest Picks** | Top 10 with all 7 factor scores (green/amber/red colour coding) |
| **Allocation** | Weight%, dollar amounts, approx shares at current price |
| **Macro Overview** | VIX, 10Y yield, regime, sector ETF rankings |
| **History** | All past sessions with tickers picked |
| **Track Record** | Evaluated sessions — avg return, S&P return, alpha |
| **Deep Analysis** | Gate scorecard · Full valuation detail (all 4 methods) · Risk metrics table |

---

## Adaptive Learning

Every session is saved to `memory/history.json`. After 30+ days:

1. Current prices are fetched for all past picks
2. Each pick's return vs S&P 500 is calculated
3. Pearson correlation measures which factors actually predicted returns
4. Factors with avg r > 0.3 → weight increases 4%
5. Factors with avg r < 0.0 → weight decreases 4%
6. Weights renormalised and floored at 3%

The more sessions you run, the smarter the factor weights become.

---

## CLI Slash Commands

After the analysis runs, the terminal enters an interactive command loop.
All data from the session is available — no re-fetching needed for stocks in the top 10.

| Command | Description |
|---------|-------------|
| `/stock AAPL` | Full report: price, valuation (all 4 methods), risk metrics, 7-gate protocol, key financials |
| `/news AAPL [15]` | Latest headlines with sentiment score + positive/negative/neutral colour coding |
| `/chart AAPL [6mo]` | Dark-theme candlestick chart with SMA 20/50/200 overlays + RSI panel |
| `/compare AAPL MSFT` | Side-by-side table: price, P/E, EV/EBITDA, Sharpe, Piotroski, valuation signal |
| `/add AAPL` | Add ticker to persistent watchlist (`memory/watchlist.json`) |
| `/remove AAPL` | Remove ticker from watchlist |
| `/watchlist` | Display all watchlist tickers with key metrics |
| `/macro` | VIX, 10Y yield, regime, sector ETF returns from session data |
| `/exit` | Exit the command loop |

Tickers not in the top 10 are fetched on-demand from Yahoo Finance.

---

## Candlestick Charts

Both the CLI and the web dashboard include full candlestick charts with technical overlays.

**CLI (`/chart TICKER [period]`):**
- Built with `mplfinance` (dark market colours: green up, red down) with `matplotlib` fallback
- 3 panels: price + SMA 20/50/200, volume bars, RSI(14) with overbought/oversold bands
- Saved as `chart_{ticker}_{period}.png` or shown via `plt.show()`

**Web (Stock Lookup tab + Rankings detail panel):**
- Built with Plotly `go.Candlestick` — fully interactive (zoom, hover, range slider)
- 3 rows via `make_subplots`: candlestick + SMAs, volume, RSI(14)
- Period selector: 1mo / 3mo / 6mo / 1y / 2y

---

## Free News API Integrations

News is aggregated from up to four sources — all free:

| Source | Requires Key | Rate Limit | Notes |
|--------|-------------|------------|-------|
| Yahoo Finance (yfinance) | No | None | Always active; last 12 headlines |
| Yahoo Finance RSS (feedparser) | No | None | Always active; last 10 from RSS |
| Finnhub | Optional | 60 calls/min | Get free key at finnhub.io |
| NewsAPI | Optional | 100 req/day | Get free key at newsapi.org |

To enable optional sources, add your key to `config.py`:
```python
FINNHUB_KEY  = "your_key_here"
NEWSAPI_KEY  = "your_key_here"
FRED_KEY     = "your_key_here"   # macro data: CPI, FEDFUNDS, T10Y2Y, etc.
```

All keys are optional. The tool works fully without any of them.

---

## File Structure

```
portfolio/
├── main.py                 ← CLI pipeline (16 steps) + interactive command loop
├── app.py                  ← Streamlit dashboard (streamlit run app.py)
├── config.py               ← Universe, weight matrix, sector multiples, optional API keys
├── requirements.txt        ← All dependencies
├── .streamlit/
│   └── config.toml         ← Streamlit theme settings
├── advisor/
│   ├── collector.py        ← 7-question profile builder
│   ├── fetcher.py          ← yfinance + 5-factor technicals + sentiment
│   ├── scorer.py           ← 7-factor scoring with 12-1 momentum, EV/EBITDA, accruals
│   ├── portfolio.py        ← Correlation-aware selection + half-Kelly sizing
│   ├── valuation.py        ← DCF · Graham · EV/EBITDA · FCF yield engine
│   ├── risk.py             ← Altman Z · Sharpe · Sortino · ROIC/WACC · Piotroski
│   ├── protocol.py         ← 7-gate Warren Buffett investment protocol
│   ├── learner.py          ← Session memory + adaptive weight learning
│   ├── news_fetcher.py     ← Multi-source news: yfinance + RSS + Finnhub + NewsAPI + FRED
│   ├── cli_commands.py     ← Interactive slash command REPL (/stock /news /chart /compare ...)
│   ├── charts.py           ← 5 dark-theme charts + candlestick() on-demand (CLI)
│   ├── display.py          ← Terminal output + deep analysis (CLI)
│   └── exporter.py         ← Excel export — 6 sheets (CLI)
├── memory/
│   ├── history.json        ← Auto-created, grows over time
│   └── watchlist.json      ← CLI /add watchlist (auto-created)
├── Book1.xlsx              ← Auto-generated on each CLI run
├── chart1_score_breakdown.png
├── chart2_performance.png
├── chart3_factor_heatmap.png
├── chart4_macro_dashboard.png
└── chart5_quant_protocol.png
```

---

## Dependencies

```
yfinance>=0.2.36      # free stock data
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0     # CLI charts
rich>=13.0.0          # terminal formatting (CLI)
openpyxl>=3.1.0       # Excel export (CLI)
streamlit>=1.32.0     # web dashboard
plotly>=5.18.0        # interactive dashboard charts
mplfinance>=0.12.9    # candlestick charts (CLI /chart command)
feedparser>=6.0.0     # Yahoo Finance RSS news feed
requests>=2.31.0      # Finnhub + NewsAPI + FRED HTTP calls
```

No paid APIs required. All optional API keys default to empty — the tool works fully without them.

---

## Disclaimer

This tool is for **educational and informational purposes only**.
Past performance does not guarantee future results.
Rankings and valuations are quantitative outputs and are **not financial advice**.
Always conduct your own due diligence before making any investment decisions.
