# Stock Ranking Advisor v7

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Data](https://img.shields.io/badge/Data-100%25%20Free-brightgreen?style=flat-square)
![APIs](https://img.shields.io/badge/Paid%20APIs-None%20Required-brightgreen?style=flat-square)
![Theme](https://img.shields.io/badge/Theme-Wall%20Street%20Terminal-0A0D14?style=flat-square)
![Stocks](https://img.shields.io/badge/Universe-800%2B%20Stocks-blueviolet?style=flat-square)
![Signals](https://img.shields.io/badge/Crash%20Signals-9-red?style=flat-square)

> Hedge-fund grade quantitative stock analysis — entirely on free data.
> No paid APIs. No AI subscriptions. Just math, discipline, and **9 independent data sources**.
> **Dark "Wall Street terminal" UI** by default — feels like Bloomberg without the $24k/year bill.

Scores up to **800 stocks** across your investor profile, runs a **7-gate Warren Buffett protocol**, computes intrinsic value via **4 independent valuation methods**, detects earnings manipulation via **Beneish M-Score**, monitors **9 live crash signals**, simulates **200 Monte Carlo paths**, backtests the strategy on historical prices, and delivers institutional-grade analysis — **for free**.

---

## Dashboard

![Rankings Dashboard](images/photo_2026-03-07_09-35-41.jpg)
*Rankings tab — macro environment tiles, top-3 pick cards with valuation signal + conviction + entry zone, and 10 interactive tabs*

---

## Key Stats

| Metric | Value |
|--------|-------|
| Stocks Scored | Up to **800** per run (dynamic NASDAQ universe) |
| Data Sources | **9 independent** (Tier 1 all-tickers + Tier 2 top-30 + Smart Money) |
| Valuation Methods | 4 (DCF, Graham, EV/EBITDA, FCF Yield) |
| Risk Metrics | 11 (Altman Z, Sharpe, Sortino, VaR, ROIC/WACC, Piotroski, Beneish M…) |
| Protocol Gates | 7 (Warren Buffett-inspired) |
| Crash Signals | **9** (VIX, HYG, yield curve, SPY drawdown, FRED, CFTC COT, BLS JOLTS) |
| Monte Carlo Paths | 200 per run (252 trading days) |
| Portfolio Size | **15 stocks** with sector concentration limit (max 3/sector) |
| Position Cap | 15% per position (half-Kelly, VIX-scaled) |
| Paid APIs Required | **0** |

---

## Two Ways to Run

### CLI Pipeline — terminal + 5 dark charts + Excel export
```bash
pip install -r requirements.txt
python main.py
```

### Streamlit Dashboard — browser-based, 10 interactive tabs
```bash
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Rankings — Full 15-Stock Table

![Full Rankings Table](images/photo_2026-03-07_09-35-28.jpg)
*Every pick shows composite score, valuation signal, conviction level, fair value, entry zone (20% MoS), stop loss, and earnings date*

---

## Architecture Overview

```mermaid
flowchart TD
    A[User Profile\n8 questions] --> B

    subgraph TIER1["Tier 1 — All Tickers (parallelized)"]
        B[yfinance + Extended\nfundamentals · history · earnings]
        B2[Stooq CSV fallback\nprice history]
        B3[Finnhub\ninsider signals · EPS surprise]
        B4[MacroFetcher\nVIX · yields · FRED · CFTC COT · BLS JOLTS]
    end

    B --> C
    B2 --> C
    B3 --> C
    B4 --> C

    C[7-Factor Scorer\nrank-percentile normalisation · GARP · Value×Quality\nBeneish M-Score · data quality penalty\nearnings proximity · price freshness]

    C --> D

    subgraph TIER2["Tier 2 — Top 30 Only"]
        D[Options P/C + IV rank]
        D2[Google Trends + Reddit RSS]
        D3[Alpha Vantage EPS beats]
        D4[FMP analyst revisions]
        D5[SEC EDGAR XBRL revenue]
        D6[Smart Money\nForm 4 cluster · 8-K NLP]
    end

    D --> E
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    D6 --> E

    E[Portfolio Construction\ngreedy correlation-aware · sector limit · beta cap\nVIX-scaled half-Kelly · 15 stocks]

    E --> F1[ValuationEngine\nDCF · Graham · EV/EBITDA · FCF yield\nDCF sensitivity Bear/Base/Bull]
    E --> F2[RiskEngine\nAltman Z · Sharpe · Sortino · ROIC/WACC\nMax DD · VaR · Piotroski]
    E --> F3[Protocol Analyzer\n7-Gate Buffett Screen]

    F1 --> G[Output]
    F2 --> G
    F3 --> G

    G --> H1[CLI Terminal\n+ 5 dark charts\n+ Excel export]
    G --> H2[Streamlit Dashboard\n10 interactive tabs]

    style A fill:#3B82F6,color:#fff
    style E fill:#8B5CF6,color:#fff
    style G fill:#10B981,color:#fff
    style TIER1 fill:#0F172A,color:#94A3B8
    style TIER2 fill:#1E1B4B,color:#A5B4FC
```

---

## The 9 Data Sources

### Tier 1 — All Tickers (runs on every stock, parallelized)

| # | Source | Data Extracted | Key |
|---|--------|---------------|-----|
| 1 | **Yahoo Finance** | Price history, fundamentals, news, insider trades, options IV, earnings calendar | None |
| 1b | **Yahoo Finance Extended** | Revenue QoQ trend · EPS beat rate · Institutional ownership % · Asset growth · Buyback yield · EPS consistency | None |
| 2 | **Stooq CSV** | Price history fallback when yfinance returns < 63 days | None |
| 3 | **FRED API** | Recession probability · HY credit spread · Consumer sentiment · 10Y–2Y spread | Free key |
| 4 | **Finnhub** | Insider transactions (last 90d) · EPS surprise history (last 4Q) | Free key |
| 4b | **CFTC COT** | E-mini S&P 500 net speculator positioning (weekly ZIP) | None |
| 4c | **BLS JOLTS** | Job openings rate + monthly change | Free key |

### Tier 2 — Top 30 Only (enrichment after initial scoring)

| # | Source | Data Extracted | Key |
|---|--------|---------------|-----|
| 5 | **Alpha Vantage** | EPS beat rate + average surprise % (last 4 quarters) | Free key |
| 6 | **Financial Modeling Prep** | Analyst estimate revisions · Financial health rating · Revenue growth | Free key |
| 7 | **SEC EDGAR XBRL** | Revenue from 10-Q filings (when yfinance data is missing) | None |
| 8 | **SEC EDGAR Form 4** | Distinct insider buyer cluster (last 60 days) | None |
| 9 | **SEC EDGAR 8-K NLP** | Event sentiment: positive filings vs negative (write-offs, departures) | None |
| + | **Google Trends** | 90-day retail search interest change | None |
| + | **Reddit RSS** | r/stocks + r/investing mention sentiment | None |
| + | **Yahoo Finance options** | Put/call ratio · IV rank | None |

---

## The 7-Factor Scoring Model

Each stock is scored **0–100** on 7 independent factors using **rank-percentile normalisation** (outlier-robust), combined with a weight matrix tuned to your risk profile and time horizon.

```
composite = w1×momentum + w2×volatility + w3×value + w4×quality
          + w5×technical + w6×sentiment + w7×dividend

+ GARP bonus:      (momentum/100) × (quality/100) × 15    ← Asness (1997)
+ VQ bonus:        (value/100)    × (quality/100) × 10    ← Frazzini & Pedersen (2019)
− data quality:    stocks with <60% fundamental coverage lose up to −12 pts
− factor crowding: −4 pts if composite correlates >0.92 with a single factor
− price freshness: up to −7 pts for stale data (>7 days old)
− earnings risk:   −4/−2.5/−1/0 pts (by risk level) when earnings ≤7 days away
```

### Factor 1 — Momentum _(12-1 skip-month, academic grade)_
```
momentum = 0.10×r1m + 0.25×r3m + 0.35×r6m + 0.30×r12_1
```
**Boosts:** Short squeeze (>15% float) · sector outperformance vs ETF · EPS beat rate · revenue QoQ · Finnhub EPS surprise · **Jensen's alpha** (market-adjusted) · **52-week high proximity**

### Factor 2 — Volatility
Annualised daily σ — **inverted** (low vol = high score). Beta-aware scaling for regime adjustment.

### Factor 3 — Value _(3-signal composite)_
```
value = 0.40 × (P/E vs sector median)
      + 0.35 × (EV/EBITDA vs sector median)
      + 0.25 × (FCF yield)
      + shareholder yield bonus (dividends + buybacks)
```

### Factor 4 — Quality _(9 signals)_
```
quality = Piotroski(9pt) × 0.60
        + ROE/Profit Margin blend × 0.40
        + Accruals quality adjustment
        + Gross Profitability (Novy-Marx 2013)
        + Revenue QoQ trend
        + EPS beat rate
        + Institutional ownership
        − Asset growth penalty (Cooper 2008)
        − Beneish M-Score fraud penalty
```

**Beneish M-Score:**
```
M ≈ 4.679 × TATA + 0.892 × SGI − 3.0
# M > −1.78 → manipulation flag → quality penalty up to 12%
```

### Factor 5 — Technical _(5 sub-signals)_
| Sub-signal | Weight | Logic |
|------------|--------|-------|
| RSI (14d) | 25% | Sweet spot 40–65; oversold <30 = contrarian buy signal |
| MACD 12/26/9 | 30% | Line vs signal vs histogram direction |
| MA crossover | 20% | Price > SMA50 > SMA200; golden cross bonus +12pts |
| Bollinger %B | 15% | Buy zone 0.20–0.65; near upper band = overbought warning |
| OBV trend | 10% | 20d SMA > 50d SMA = smart-money accumulation |

### Factor 6 — Sentiment _(5-source composite in Tier 2)_
```
Tier 1:  0.45×news + 0.35×insider + 0.20×analyst
Tier 2:  0.30×news + 0.25×insider + 0.20×analyst + 0.15×options + 0.10×retail
```
News: 20 articles, negation-aware, recency-weighted (1.5× <3d · 1.2× <7d)

### Factor 7 — Dividend
Raw yield, capped at 15%. Heavily weighted for income-focused profiles.

---

## Valuation Engine — 4 Independent Methods

![Valuation Matrix](images/photo_2026-03-07_09-35-52.jpg)
*4-method valuation matrix — DCF, Graham Number, EV/EBITDA, FCF Yield — with fair value, entry zone, stop loss, R/R ratio, and signal*

| Method | Formula | Captures |
|--------|---------|----------|
| **DCF (2-stage)** | FCF/share × 5yr growth + terminal value, discounted at `rf + sector_erp + size_premium` | Future cash generation |
| **Graham Number** | `√(22.5 × EPS × Book/share)` | Graham's classic intrinsic value |
| **EV/EBITDA Target** | `EBITDA × sector_median_multiple → implied price` | How the market values sector peers |
| **FCF Yield Target** | `FCF/share ÷ (rf + 3%)` | Price at a dynamic FCF return target |

**Dynamic discount rate:**
```python
DR = rf_rate + SECTOR_ERP[sector] + size_premium
# Example: 4.5% rf + 6.0% tech ERP + 1.0% mid-cap = 11.5%
```

**Entry / Exit levels:**
```
Entry zone:  FV × 0.80  (20% margin of safety — Benjamin Graham rule)
Target:      FV × 1.20  (take profit)
Stop loss:   entry × 0.92  (8% downside protection)
```

**DCF Sensitivity — Bear / Base / Bull:**

![Valuation Charts](images/photo_2026-03-07_09-35-57.jpg)
*Entry price positioning (% vs fair value), valuation method spread scatter, and DCF sensitivity table across three growth scenarios*

| Scenario | Growth Assumption |
|----------|------------------|
| Bear | base growth × 0.50 |
| Base | blended revenue + earnings growth |
| Bull | base growth × 1.50 (capped at 30%) |

---

## Risk Engine — Full Institutional Suite

![Risk & Quality](images/photo_2026-03-07_09-36-09.jpg)
*Sharpe vs ROIC/WACC scatter (bubble size = spread magnitude), Piotroski F-Score bars, and Monte Carlo portfolio simulation (200 paths, 252 trading days)*

| Metric | Formula / Logic |
|--------|----------------|
| **Altman Z-Score** | 5-factor bankruptcy predictor → SAFE (>2.6) / GRAY / DISTRESS (<1.1) |
| **Sharpe Ratio** | `(annualised_return − rf) / annualised_vol` |
| **Sortino Ratio** | Sharpe using downside deviation only |
| **Max Drawdown** | Worst peak-to-trough % over the full period |
| **VaR 95% (1mo)** | 5th percentile of 21-day rolling return distribution |
| **ROIC / WACC** | `ROIC − WACC` spread → EXCEPTIONAL / STRONG / POSITIVE / NEUTRAL / DESTROYING VALUE |
| **Full WACC** | `(E/V)×cost_equity + (D/V)×cost_debt×(1−tax)` — Modigliani-Miller |
| **Accruals Ratio** | `(NI − OCF) / Assets` — negative = earnings backed by real cash |
| **Gross Profitability** | `(Revenue × Gross Margin) / Assets` — Novy-Marx (2013) |
| **Piotroski F-Score** | Full 9-point screen: profitability + leverage + efficiency |
| **IV Rank** | `(current_iv / hist_vol×1.15 − 0.5) × 1.25` — >0.70 = elevated fear |

---

## The 7-Gate Investment Protocol

![Protocol Gates](images/photo_2026-03-07_09-36-14.jpg)
*15×7 gate heatmap — green = PASS, amber = WARN, red = FAIL — with protocol summary showing conviction, valuation signal, and entry target per stock*

Every top-15 stock passes through a Warren Buffett–inspired 7-gate screen:

| Gate | Weight | What It Checks |
|------|--------|----------------|
| 1. Business Quality | 20% | ROA, ROE, FCF yield, profit margins, earnings growth |
| 2. Competitive Moat | 15% | Gross margins, operating margins, revenue scale |
| 3. Financial Health | 15% | Debt/equity, current ratio, interest coverage, Altman Z |
| 4. Valuation | 22% | ValuationEngine signal (65%) + P/E vs sector median (35%) |
| 5. Technical Entry | 10% | 52-week positioning, analyst consensus upside, forward P/E |
| 6. News & Sentiment | 8% | Multi-source sentiment score + analyst recommendation |
| 7. Trend Alignment | 10% | SMA200 trend, SMA50 crossover, 3-month momentum |

**Thresholds:** PASS ≥60 · WARN 35–59 · FAIL <35
**Conviction:** HIGH (≤1 fail, ≥70 overall, ≥6 pass) · MEDIUM (≤2 fails) · LOW (3+ fails)

---

## Portfolio Construction

![Portfolio Allocation](images/photo_2026-03-07_09-36-27.jpg)
*Sector allocation donut with Kelly-sized weights, position breakdown table showing dollar amounts and share counts at current prices*

```mermaid
flowchart TD
    A[Top 30 by\ncomposite score] --> B[Correlation Matrix\non daily returns]
    B --> C[Greedy Selection\n0.70 × quality + 0.30 × diversification]
    C --> D{Beta Check}
    D -->|portfolio β > target| E[Progressive 3-Pass Swap\nworst-β out · best alt in\n0.15 → 0.05 → any threshold]
    D -->|β within target| F
    E --> F{Sector\nConcentration}
    F -->|sector > 3 stocks| G[Trim overweight sectors\nfill with best alternatives]
    F -->|OK| H
    G --> H[VIX-Scaled Half-Kelly Sizing]
    H --> I[eff_vol = max hist_vol, VIX/100 + bear_penalty\nkelly_f = edge / eff_vol² / 2\nweight = min kelly_f / total, 0.15]
    I --> J[15-Stock Portfolio\n100% allocated]

    style J fill:#10B981,color:#fff
    style I fill:#3B82F6,color:#fff
```

### Beta Targets by Risk Level
| Risk Level | Label | Beta Target |
|-----------|-------|------------|
| 1 | Conservative | ≤ 0.90 |
| 2 | Balanced | ≤ 1.05 |
| 3 | Aggressive | ≤ 1.30 |
| 4 | Speculative | ≤ 1.60 |

### VIX-Scaled Kelly Position Sizing
```python
vix_implied = VIX / 100.0            # VIX=30 → 30% annual vol floor
bear_penalty = min(0.15, ...)         # extra floor when SPY >10% off high
eff_vol = max(hist_vol, vix_implied + bear_penalty)
kelly_f = (score/100) / (eff_vol² × 2)    # half-Kelly
weight  = min(kelly_f / Σkelly, 0.15)     # capped at 15%
```
> Automatically shifts weight toward lower-vol stocks in high-VIX environments — exactly the right behaviour in crashes.

---

## Macro Regime & Historical Performance

![Macro & Performance](images/photo_2026-03-07_09-36-32.jpg)
*Macro environment tiles (VIX, 10Y yield, regime), sector ETF 3-month returns, normalised 5-year performance vs S&P 500, and return correlation heatmap*

### 9 Crash Signals

| Signal | Trigger | Source |
|--------|---------|--------|
| VIX velocity | >7 pts in 5 days (panic spike) | yfinance ^VIX |
| HYG return | < −3% in 1 month (credit seizing) | yfinance HYG |
| Yield curve | 3M > 10Y (inverted) | yfinance ^IRX / ^TNX |
| SPY drawdown | >12% off 52-week high | yfinance SPY |
| VIX absolute | > 35 (extreme fear) | yfinance ^VIX |
| Recession probability | > 30% (RECPROUSM156N) | FRED API |
| HY credit spread | > 5% OAS (BAMLH0A0HYM2) | FRED API |
| CFTC COT | Net spec position < −0.30 | cftc.gov weekly ZIP |
| BLS JOLTS | Job openings rate MoM < −5% | BLS public API |

### Regime → Score Tilts

| Regime | Trigger | Sector Adjustments |
|--------|---------|-------------------|
| Risk-On | VIX < 16 | +4 Tech, +3 Consumer, −5 Utilities |
| Risk-Off | VIX > 27 | +7 Utilities, +5 Healthcare, −4 Tech |
| Rising Rate | 10Y up >0.35%/mo | +5 Financials, +3 Energy, −7 REITs, −5 Utilities |
| Falling Rate | 10Y down >0.30% | +5 REITs, +5 Utilities, +3 Tech, −3 Financials |
| Pre-Crisis | 2 crash signals | +8 Utilities, +6 Healthcare, −6 Tech, −6 REITs |
| Crisis | 3+ crash signals | +15 Utilities, +10 Healthcare, −15 Tech, −12 REITs |

---

## Smart Money Intelligence

The `advisor/smart_money.py` module adds two free institutional signal layers via SEC EDGAR (no API key):

| Signal | Source | Scoring Logic |
|--------|--------|---------------|
| **Form 4 Cluster** | EDGAR submissions JSON + XML | Counts *distinct* buyers in 60d window. 3+ buyers = 78 score; 5+ = 95 |
| **8-K Event Sentiment** | EDGAR 8-K item type classification | Item 1.01 (deals), 2.02 (earnings) = positive; 2.05 (write-offs), 4.02 (auditor change) = negative |

Composite score: equal-weight average → ±3pt boost to composite score.

---

## Macro Regime Detection (9 Crash Signals)

```mermaid
flowchart LR
    subgraph SIGNALS["9 Crash Signals"]
        S1[VIX velocity\n>7pts in 5d]
        S2[HYG 1mo return\n< −3%]
        S3[Yield curve\n3M > 10Y]
        S4[SPY drawdown\n>12% off high]
        S5[VIX absolute\n> 35]
        S6[FRED recession prob\n> 30%]
        S7[FRED HY spread\n> 5% OAS]
        S8[CFTC COT\nnet spec < −0.30]
        S9[BLS JOLTS\nMoM change < −5%]
    end

    SIGNALS --> COUNT{Signal Count}
    COUNT -->|0–1| RO[Risk-On]
    COUNT -->|2| PC[Pre-Crisis]
    COUNT -->|3+| CR[Crisis]

    style CR fill:#EF4444,color:#fff
    style PC fill:#F59E0B,color:#fff
    style RO fill:#10B981,color:#fff
```

---

## Adaptive Learning

Every session is saved to `memory/history.json`. After 5+ trading days, evaluation begins automatically across multiple horizons (5d → 21d → 63d → 126d).

Six stacked learning layers that compound across every session:

| Layer | What It Learns |
|-------|---------------|
| **1. Weight Adaptation** | Which of the 7 factors actually predicted your returns (adaptive LR: 4%→14%) |
| **2. Sector Intelligence** | Per-sector factor importance (momentum matters more in Tech than Utilities) |
| **3. Regime Intelligence** | Which factors work in each macro environment (quality beats momentum in risk-off) |
| **4. Pattern Matching** | Factor fingerprints of historical winners/losers → ±12pt bonus/penalty |
| **5. Valuation Calibration** | Tracks if STRONG_BUY/BUY signals were actually correct (win rate per signal) |
| **6. Dynamic Sector Tilts** | Learns which sectors actually beat the market in each macro regime |

**Fresh picks mode** (Q8): −22 pts to last 2 sessions' picks → forces entirely new recommendations.

---

## The 8-Question Investor Profile

| # | Question | Impact |
|---|----------|--------|
| 1 | Portfolio size ($1K – $1B) | Dollar allocations in the output table |
| 2 | Time horizon (1 / 3 / 5 yr) | Short → momentum-heavy; Long → value + quality |
| 3 | Risk tolerance (1–4) | Controls beta cap + factor weight distribution |
| 4 | Investment goal | Income goal lifts dividend weight; speculative lifts momentum |
| 5 | Drawdown tolerance | Adds volatility penalty when `drawdown_ok < 20%` |
| 6 | Sector focus / exclusions | Filters universe; preferred sectors get quality/momentum boost |
| 7 | Existing holdings | Removed from recommendations to avoid overlap |
| 8 | **Fresh picks mode** | −22 pts to last 2 sessions' picks → forces entirely new ideas |

---

## Weight Matrix

Factor weights auto-selected by `(risk_level, time_horizon)` and adapted over time by the learning engine:

| Profile | Momentum | Volatility | Value | Quality | Technical | Sentiment | Dividend |
|---------|----------|-----------|-------|---------|-----------|-----------|----------|
| Conservative / Short | 10% | 28% | 18% | 18% | 7% | 4% | 15% |
| Conservative / Long | 5% | 18% | 27% | 25% | 5% | 5% | 15% |
| Balanced / Medium | 18% | 14% | 22% | 25% | 11% | 5% | 5% |
| Aggressive / Short | 38% | 7% | 12% | 22% | 16% | 5% | 0% |
| Speculative / Medium | 35% | 4% | 12% | 28% | 16% | 5% | 0% |

---

## Backtest Strategy

Simulates the valuation entry strategy on historical daily closes:

| Rule | Trigger | Action |
|------|---------|--------|
| **Entry** | Price ≤ FV × 0.80 | Buy — 20% margin of safety |
| **Take Profit** | Price ≥ FV × 1.20 | Sell — 20% above fair value |
| **Stop Loss** | Price ≤ entry × 0.92 | Sell — 8% below entry price |

Output: Equal-weighted portfolio equity curve vs S&P 500 · Win rate · Portfolio return · Alpha · Per-stock breakdown · Per-trade log

---

## Streamlit Dashboard — 10 Tabs

| Tab | What You Get |
|-----|-------------|
| **1. Rankings** | Macro tiles · top-3 pick cards · full 15-stock table with signals · factor radar (7-factor fingerprint) · score distribution histogram · per-stock detail panel |
| **2. Valuation** | 4-method matrix · entry positioning chart · valuation method spread scatter · DCF sensitivity Bear/Base/Bull |
| **3. Risk & Quality** | Risk metrics table · Sharpe vs ROIC/WACC bubble scatter · Piotroski bars · Monte Carlo simulation (200 paths) |
| **4. Protocol Gates** | 15×7 gate heatmap · protocol summary with conviction + signal + entry target |
| **5. Portfolio** | Sector donut chart · position breakdown with Kelly weights · dollar amounts + share counts |
| **6. Macro & Performance** | VIX/yield/regime tiles · sector ETF bars · 5yr normalised price history vs S&P 500 · correlation heatmap · yield curve chart |
| **7. Stock Lookup** | Search any ticker — fresh full analysis: candlestick, valuation, risk, news, protocol |
| **8. History** | Past sessions · per-pick cards with factor score bars · entry/exit P&L · time-machine full analysis view |
| **9. Backtest** | Portfolio-wide equity curve vs S&P 500 · 5 aggregate tiles · per-stock breakdown |
| **10. Calendar** | Earnings timeline sorted by urgency: ≤7d RED · ≤14d AMBER · ≤30d BLUE · Wall Street-style quant analysis per stock |
| **⚙️ Settings** | Theme switcher (Terminal / Dark / Light / Warm) · CSV/JSON export · behaviour sliders |

---

## CLI Slash Commands

After the analysis pipeline completes, the terminal enters an interactive REPL:

```
/stock AAPL         → Full quant report: thesis · valuation · DCF sensitivity
                       risk metrics · analyst targets · technical · protocol
/news AAPL [15]     → Headlines with per-article sentiment colour coding
/chart AAPL [6mo]   → Dark-theme candlestick + SMA 20/50/200 + RSI panel
/compare AAPL MSFT  → Side-by-side: price · P/E · EV/EBITDA · Sharpe · Piotroski · signal
/add AAPL           → Add to persistent watchlist
/remove AAPL        → Remove from watchlist
/watchlist          → Show watchlist with live prices
/macro              → VIX · 10Y yield · regime · sector ETF returns · crash signal count
/history [n]        → Past sessions — win rate · alpha · per-pick returns
/exit               → Exit
```

---

## CLI Charts — 5 Dark-Theme PNGs

Auto-generated after each `python main.py` run:

| File | Contents |
|------|----------|
| `chart1_score_breakdown.png` | Stacked horizontal bars — 7 factor contributions per stock |
| `chart2_performance.png` | Normalised price history vs S&P 500 benchmark |
| `chart3_factor_heatmap.png` | 15 × 7 colour grid of all factor scores |
| `chart4_macro_dashboard.png` | VIX · 10Y yield history · sector ETF returns · correlation matrix |
| `chart5_quant_protocol.png` | Gate scorecard · entry price positioning · quant thesis per stock |

All charts use the dark Wall Street terminal palette:
```
Background: #0D1117    Panel: #161B22    Border: #30363D
Pass: #3FB950          Warn: #E3B341     Fail: #DA3633
```

---

## Excel Export — Book1.xlsx (6 Sheets)

| Sheet | Contents |
|-------|----------|
| Latest Picks | Top 15 with all 7 factor scores, composite, Kelly weight |
| Allocation | Weight %, dollar amounts, approx share counts |
| Macro Overview | VIX, 10Y yield, regime, 9-signal crash count, sector ETF rankings |
| History | All past sessions with tickers, entry prices, evaluations |
| Track Record | Evaluated sessions — avg return, S&P return, alpha |
| Deep Analysis | Gate scorecard · 4-method valuation · Risk metrics · Beneish M-Score |

---

## What's New

```mermaid
timeline
    title Version History
    v1 : yfinance · 7-factor scoring · 10-stock portfolio
    v2 : Valuation engine (DCF · Graham) · Risk engine · Protocol gates
    v3 : Adaptive learning · Fresh picks · CLI commands · Excel export
    v4 : Tier 2 enrichment · Monte Carlo · Factor radar · Macro FRED integration
    v5 : 800-stock universe · 9 crash signals · 15-stock portfolio
       : Smart Money (Form 4 · 8-K) · CFTC COT · BLS JOLTS
       : GARP + Value×Quality · Rank-percentile normalisation
       : Beneish M-Score fraud detection · VIX-scaled Kelly sizing
    v6 : S&P outperformance upgrades · Jensen's alpha · 52W high momentum
       : Shareholder yield · Asset growth penalty · EPS consistency
       : Sector concentration constraint · International ADRs (70+)
       : 3-pass progressive beta cap · Graveyard archetypes (loser patterns)
    v7 : Bug fixes · Earnings proximity scoring · Price freshness penalty
       : Forward alpha threshold fixed · Conviction logic corrected
       : EPS beat rate denominator fixed · SEC CIK deduplication
```

### v7 — Bug Fixes & Improvements

**Critical Bug Fixes:**

| File | Bug | Fix |
|------|-----|-----|
| `scorer.py` | Forward fundamental alpha **never fired** — threshold `_es > 2.0` required a 200% EPS surprise; Finnhub returns ratios (0.05 = 5%) | Threshold `0.02`, divisor `1.5` — signal now correctly activates for consistent EPS beaters |
| `protocol.py` | HIGH conviction checked `fail_c == 0` — one failing gate wrongly downgraded to MEDIUM | Fixed to `fail_c <= 1` per spec |
| `learner.py` | History tab showed null target price — `v.get("target")` key doesn't exist | Fixed to `v.get("target_price")` |
| `alternative_data.py` | EPS beat rate divided by `len(quarterly)` (total quarters) instead of quarters with valid EPS data | Added `valid_q` counter — beat rate no longer deflated by missing data |
| `learner.py` | Price evaluation failed on holidays — `period="2d"` returns empty on non-trading days | Changed to `period="5d"` |
| `collector.py` | All 7 questions labelled "X of 7" despite there being 8 questions | Fixed Q1–Q7 to "X of 8"; updated welcome banner to v7 |

**New Features:**

| Feature | Description |
|---------|-------------|
| **Earnings proximity adjustment** | Stocks ≤7 days from earnings get a composite penalty scaled by risk level: Conservative −4pts · Balanced −2.5pts · Aggressive −1pt · Speculative 0pts. Prevents recommending stocks on the eve of a gap-down risk event. |
| **Price freshness penalty** | `price_freshness` (computed per-ticker but never used in scoring) now applies up to −7pts for data older than 7 days. Prevents stale technicals from appearing clean. |
| **SEC CIK deduplication** | `enrich_top_n` now populates the module-level `_SEC_CIK_CACHE` from the already-fetched map, so `fetch_sec_revenue_trend` reuses it — eliminating a redundant HTTP request to SEC EDGAR per run. |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional — tool works without them)
```bash
cp .env.example .env
```

```ini
# .env — all optional
FINNHUB_KEY      = ""    # finnhub.io — free, 60 calls/min
FRED_KEY         = ""    # fred.stlouisfed.org — free, unlimited
ALPHAVANTAGE_KEY = ""    # alphavantage.co — free, 25 calls/day
FMP_KEY          = ""    # financialmodelingprep.com — free, 250 calls/day
BLS_KEY          = ""    # bls.gov — free JOLTS data
```

Without keys: yfinance + SEC EDGAR + Stooq + CFTC cover everything.

### 3. Run
```bash
# Terminal pipeline
python main.py

# Web dashboard
streamlit run app.py
```

---

## File Structure

```
portfolio/
├── main.py                   ← 16-step CLI pipeline + interactive REPL
├── app.py                    ← Streamlit dashboard (10 tabs + settings)
├── config.py                 ← Universe · weight matrix · sector multiples · API keys
├── requirements.txt
├── .env                      ← Your API keys (gitignored)
├── .env.example              ← Key template
├── .streamlit/config.toml    ← Wall Street terminal theme
├── images/                   ← Dashboard screenshots
├── advisor/
│   ├── collector.py          ← 8-question investor profile builder
│   ├── fetcher.py            ← yfinance + extended + Stooq + FRED + Finnhub + CFTC + BLS
│   ├── scorer.py             ← 7-factor MultiFactorScorer (GARP · Beneish · Jensen's alpha)
│   ├── alternative_data.py   ← Tier 2: options · Trends · Reddit · AV · FMP · SEC · Smart Money
│   ├── smart_money.py        ← SEC Form 4 insider cluster + 8-K event sentiment
│   ├── portfolio.py          ← Greedy selection + beta cap + sector limit + VIX-Kelly
│   ├── valuation.py          ← DCF · Graham · EV/EBITDA · FCF yield (dynamic DR)
│   ├── risk.py               ← Altman Z · Sharpe · Sortino · full WACC · Piotroski
│   ├── protocol.py           ← 7-gate Warren Buffett investment protocol
│   ├── learner.py            ← Session memory · adaptive weights · 6 learning layers
│   ├── news_fetcher.py       ← yfinance + RSS + Finnhub + NewsAPI (negation-aware)
│   ├── cli_commands.py       ← Interactive REPL (/stock /news /chart /compare …)
│   ├── charts.py             ← 5 dark-theme matplotlib charts + candlestick
│   ├── display.py            ← Rich terminal output + deep analysis
│   ├── exporter.py           ← Excel 6-sheet export
│   └── universe.py           ← Dynamic US market universe (NASDAQ API, cached 24h)
└── memory/
    ├── history.json          ← Auto-created session log (gitignored)
    ├── watchlist.json        ← CLI /add watchlist (gitignored)
    └── settings.json         ← Dashboard settings (gitignored)
```

---

## Dependencies

```
yfinance>=0.2.36       # primary data source
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
rich>=13.0.0           # terminal formatting
openpyxl>=3.1.0        # Excel export
streamlit>=1.32.0      # web dashboard
plotly>=5.18.0         # interactive charts
mplfinance>=0.12.9     # candlestick charts
feedparser>=6.0.0      # RSS news feeds
requests>=2.31.0       # HTTP (Stooq · FRED · FMP · AV · SEC · CFTC · BLS)
pytrends>=4.9.0        # Google Trends
python-dotenv>=1.0.0   # .env loader
```

No paid data subscriptions required.

---

## Disclaimer

This tool is for **educational and informational purposes only**.
Past performance does not guarantee future results.
Rankings, valuations, and backtest results are quantitative model outputs — **not financial advice**.
Always conduct your own due diligence before making investment decisions.
