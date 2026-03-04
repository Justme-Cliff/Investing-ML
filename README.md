# Stock Ranking Advisor v5

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Data](https://img.shields.io/badge/Data-100%25%20Free-brightgreen?style=flat-square)
![APIs](https://img.shields.io/badge/Paid%20APIs-None%20Required-brightgreen?style=flat-square)
![Theme](https://img.shields.io/badge/Theme-Wall%20Street%20Terminal-0A0D14?style=flat-square)
![Stocks](https://img.shields.io/badge/Universe-800%20Stocks-blueviolet?style=flat-square)
![Signals](https://img.shields.io/badge/Crash%20Signals-9-red?style=flat-square)

> Hedge-fund grade quantitative stock analysis — entirely on free data.
> No paid APIs. No AI subscriptions. Just math, discipline, and **9 independent data sources**.
> **Dark "Wall Street terminal" UI** by default — feels like Bloomberg without the $24k/year bill.

Scores up to **800 stocks** across your investor profile, runs a **7-gate Warren Buffett protocol**, computes intrinsic value via **4 independent valuation methods**, detects earnings manipulation via **Beneish M-Score**, monitors **9 live crash signals**, performs Monte Carlo simulations, backtests the strategy on historical prices, and delivers analysis that would cost thousands per month on a professional terminal — **for free**.

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
| Quant Charts | 10 (5 dark CLI + 5 interactive Plotly) |
| Monte Carlo Paths | 200 per run |
| Portfolio Size | **15 stocks** with sector concentration limit (max 3/sector) |
| Position Cap | 15% per position (half-Kelly, VIX-scaled) |
| Paid APIs Required | **0** |

---

## Two Ways to Run

### CLI Pipeline — terminal + 5 charts + Excel export
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

## Architecture Overview

```mermaid
flowchart TD
    A[👤 User Profile\n8 questions] --> B

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

    C[🧮 7-Factor Scorer\nrank-percentile normalisation · GARP · Value×Quality\nBeneish M-Score · data quality penalty]

    C --> D

    subgraph TIER2["Tier 2 — Top 30 Only"]
        D[Options P/C + IV rank]
        D2[Google Trends + Reddit RSS]
        D3[Alpha Vantage EPS beats]
        D4[FMP analyst revisions]
        D5[SEC EDGAR XBRL revenue]
        D6[🕵️ Smart Money\nForm 4 cluster · 8-K NLP]
    end

    D --> E
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    D6 --> E

    E[🏗️ Portfolio Construction\ngreedy correlation-aware · sector limit · beta cap\nVIX-scaled half-Kelly · 15 stocks]

    E --> F1[💰 ValuationEngine\nDCF · Graham · EV/EBITDA · FCF yield\nDCF sensitivity Bear/Base/Bull]
    E --> F2[⚠️ RiskEngine\nAltman Z · Sharpe · Sortino · ROIC/WACC\nMax DD · VaR · Piotroski]
    E --> F3[🔐 Protocol Analyzer\n7-Gate Buffett Screen]

    F1 --> G[📊 Output]
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

### Tier 1 — All Tickers (parallelized, runs on every stock)

| # | Source | Data Extracted | Key |
|---|--------|---------------|-----|
| 1 | **Yahoo Finance** | Price history, fundamentals, news, insider trades, options IV, earnings calendar | None |
| 1b | **Yahoo Finance Extended** | Revenue QoQ trend · EPS beat rate · Institutional ownership % | None |
| 2 | **Stooq CSV** | Price history fallback when yfinance returns < 63 days | None |
| 3 | **FRED API** | Recession probability · HY credit spread · Consumer sentiment · 10Y–2Y spread | Free key |
| 4 | **Finnhub** | Insider transactions (last 90d) · EPS surprise history (last 4Q) | Free key |
| 4b | **CFTC COT** | E-mini S&P 500 net speculator positioning (weekly ZIP) | None |
| 4c | **BLS JOLTS** | Job openings rate + monthly change (JTS000000000000000JOR) | Free key |

### Tier 2 — Top 30 Only (enrichment after initial scoring)

| # | Source | Data Extracted | Key |
|---|--------|---------------|-----|
| 5 | **Alpha Vantage** | EPS beat rate + average surprise % (last 4 quarters) | Free key |
| 6 | **Financial Modeling Prep (FMP)** | Analyst estimate revisions · Financial health rating · Revenue growth | Free key |
| 7 | **SEC EDGAR XBRL** | Revenue + net income from 10-Q filings (when yfinance data is missing) | None |
| 8 | **🕵️ SEC EDGAR Form 4** | Distinct insider buyer cluster (last 60 days) | None |
| 9 | **SEC EDGAR 8-K NLP** | Event sentiment: positive filings (deals, guidance) vs negative (losses, departures) | None |
| + | **Google Trends** (pytrends) | 90-day retail search interest change | None |
| + | **Reddit RSS** | r/stocks + r/investing mention sentiment | None |
| + | **Yahoo Finance options** | Put/call ratio · IV rank | None |

---

## The 7-Factor Scoring Model

Each stock is scored **0–100** on 7 independent factors using **rank-percentile normalisation** (outlier-robust — replaces min-max), combined using a weight matrix tuned to your risk profile and time horizon.

```
composite = w1×momentum + w2×volatility + w3×value + w4×quality
          + w5×technical + w6×sentiment + w7×dividend

+ GARP bonus: (momentum/100) × (quality/100) × 15    ← Asness (1997)
+ VQ bonus:   (value/100)    × (quality/100) × 10    ← Frazzini & Pedersen (2019)
− data quality penalty: stocks with <60% fundamental coverage lose up to 12 pts
− factor crowding penalty: −4 pts if composite correlates >0.92 with one factor only
```

```mermaid
flowchart LR
    M[Momentum\nr1m·r3m·r6m·r12] --> W{Weight\nMatrix}
    V[Volatility\nannualised σ] --> W
    VA[Value\nP/E · EV/EBITDA · FCF] --> W
    Q[Quality\nPiotroski · ROE · Accruals] --> W
    T[Technical\nRSI · MACD · MA · OBV] --> W
    S[Sentiment\nNews · Insider · Options] --> W
    D[Dividend\nyield capped 15%] --> W
    W --> CS[Composite\nScore 0–100]
    CS --> GARP[+GARP bonus\nmomentum × quality × 15]
    GARP --> VQ[+Value×Quality\nbonus × 10]
    VQ --> DQ[−Data Quality\npenalty]
    DQ --> BM[−Beneish\nfraud penalty]
    BM --> FINAL[Final Score]

    style FINAL fill:#10B981,color:#fff
    style BM fill:#EF4444,color:#fff
```

### Factor 1 — Momentum _(12-1 skip-month, academic grade)_
Avoids the 1-month reversal effect documented in academic literature:
```
momentum = 0.10×r1m + 0.25×r3m + 0.35×r6m + 0.30×r12_1
```
**Boosts:** Short squeeze (>15% float) · sector outperformance vs ETF · EPS beat rate · revenue QoQ acceleration · Finnhub earnings surprise magnitude · **Jensen's alpha** (market-adjusted outperformance) · **52-week high proximity** (52W_high/price - 1 signal)

### Factor 2 — Volatility
Annualised daily return standard deviation — **inverted** (low vol = high score). Includes **beta-aware scaling** for regime adjustment.

### Factor 3 — Value _(3-signal composite)_
```
value = 0.40 × (P/E vs sector median)
      + 0.35 × (EV/EBITDA vs sector median)
      + 0.25 × (FCF yield)
```

### Factor 4 — Quality _(8 signals, expanded)_
```
quality = Piotroski(9pt) × 0.60
        + ROE/Profit Margin blend × 0.40
        + Accruals quality adjustment   (−0.20 to +0.20)
        + Gross Profitability (Novy-Marx 2013)
        + Revenue trend QoQ             (−0.10 to +0.10)
        + EPS beat rate                 (−0.10 to +0.10)
        + Institutional ownership       (−0.05 to +0.06)
        + Shareholder yield bonus       (buybacks + dividends)
        − Asset growth penalty          (overexpansion destroys alpha)
        − EPS consistency penalty       (high variance earnings = risk)
        − Beneish M-Score penalty       (earnings manipulation flag)
```

**Beneish M-Score (fraud detection):**
```
M ≈ 4.679 × TATA + 0.892 × SGI − 3.0
# TATA = accruals ratio, SGI = 1 + revenue growth
# M > −1.78 → manipulation flag → quality penalty up to 12%
```

### Factor 5 — Technical _(5 sub-signals)_
| Sub-signal | Weight | Logic |
|------------|--------|-------|
| RSI (14d) | 25% | Sweet spot 40–65; oversold <30 = contrarian buy |
| MACD 12/26/9 | 30% | Line vs signal vs histogram direction |
| MA crossover | 20% | Price > SMA50 > SMA200 = golden alignment; golden cross bonus +12pts |
| Bollinger %B | 15% | Buy zone 0.20–0.65; near upper band = overbought warning |
| OBV trend | 10% | 20d SMA > 50d SMA = smart-money accumulation |

### Factor 6 — Sentiment _(5-source composite)_

**Tier 1 (all tickers):**
```
sentiment = 0.45 × news_score    (20 articles, negation-aware, recency-weighted 1.5×/1.2×/1.0×)
          + 0.35 × insider_score  (yfinance + Finnhub insider transactions, last 90d)
          + 0.20 × analyst_score  (rec key + target upside + coverage breadth)
```

**Tier 2 (top 30 — upgraded to full 5-source):**
```
sentiment = 0.30 × news_score
          + 0.25 × insider_score
          + 0.20 × analyst_score  (blended with FMP analyst revision)
          + 0.15 × options_score  (put/call ratio + IV premium)
          + 0.10 × retail_score   (Google Trends 60% + Reddit RSS 40%)
```

### Factor 7 — Dividend
Raw yield, capped at 15%. Heavily weighted only for income-focused profiles.

---

## Smart Money Intelligence (New in v5)

The `advisor/smart_money.py` module adds two free institutional signal layers via SEC EDGAR:

```mermaid
flowchart LR
    SEC[SEC EDGAR\nsubmissions API] --> F4[Form 4 filings\nlast 60 days]
    SEC --> EK[8-K filings\nlast 90 days]
    F4 --> BUYERS[Distinct\nInsider Buyers]
    BUYERS --> SCORE1[Cluster Score\n0→50  1→55  2→65\n3→78  4→88  5+→95]
    EK --> CLASS[Item Classification]
    CLASS --> POS["Positive Items\n1.01 deals · 2.02 earnings\n7.01 guidance · 8.01 announce"]
    CLASS --> NEG["Negative Items\n1.02 exit · 1.03 bankrupt\n2.05 write-off · 4.01 auditor"]
    POS --> SCORE2[Event Score 0–100]
    NEG --> SCORE2
    SCORE1 --> FINAL[Smart Money\nComposite Score]
    SCORE2 --> FINAL
    FINAL --> BOOST[±3pt boost\nto composite score]

    style FINAL fill:#8B5CF6,color:#fff
    style BOOST fill:#10B981,color:#fff
```

| Signal | Source | Scoring Logic |
|--------|--------|---------------|
| **Form 4 Cluster** | EDGAR submissions JSON + XML | Counts *distinct* buyers in 60d window (not volume). 3+ buyers = 78 score |
| **8-K Event Sentiment** | EDGAR 8-K item type classification | Item 1.01 (deals), 2.02 (earnings) = positive; 2.05 (write-offs), 4.02 (auditor change) = negative |

> No API key required — SEC EDGAR is public. Rate-limited gracefully with try/except.

---

## Valuation Engine — 4 Independent Methods

Every stock in the top 15 is valued four independent ways:

| Method | Formula | Captures |
|--------|---------|----------|
| **DCF (2-stage)** | FCF/share × 5yr growth + terminal value, discounted at `rf + sector_erp + size_premium` | Future cash generation |
| **Graham Number** | `√(22.5 × EPS × Book/share)` | Graham's classic intrinsic value |
| **EV/EBITDA Target** | `EBITDA × sector_median_multiple → implied price` | How market values sector peers |
| **FCF Yield Target** | `FCF/share ÷ (rf + 3%)` | Price at a dynamic FCF return target |

**Dynamic discount rate** — no more hard-coded 10%:
```python
DR = rf_rate + SECTOR_ERP[sector] + size_premium
# Example: 5% rf + 6.0% tech ERP + 1.0% mid-cap = 12.0%
```

**Signals:** `STRONG_BUY` (≤entry_low) · `BUY` (≤entry_high) · `HOLD_WATCH` (≤FV) · `WAIT` (≤FV×1.10) · `AVOID_PEAK` (>FV×1.10)

**DCF Sensitivity** — Bear / Base / Bull scenario table (50% / 100% / 150% of base growth):

| Scenario | Growth | Fair Value | Signal |
|----------|--------|-----------|--------|
| Bear | g × 0.50 | conservative | varies |
| Base | g × 1.00 | calculated | varies |
| Bull | g × 1.50 | optimistic | varies |

**Entry / Exit Rules (backtest + live signals):**
```
Entry zone:  FV × 0.80  (20% margin of safety — Benjamin Graham rule)
Target:      FV × 1.20  (take profit)
Stop loss:   entry × 0.92  (8% downside protection)
```

---

## Risk Engine — Full Institutional Suite

| Metric | Formula / Logic |
|--------|----------------|
| **Altman Z-Score** | 5-factor bankruptcy predictor → SAFE (>2.6) / GRAY / DISTRESS (<1.1) |
| **Sharpe Ratio** | `(annualised_return − rf) / annualised_vol` |
| **Sortino Ratio** | Sharpe using downside deviation only |
| **Max Drawdown** | Worst peak-to-trough % over the full period |
| **VaR 95% (1mo)** | 5th percentile of 21-day rolling return distribution |
| **ROIC / WACC** | `ROIC − WACC` spread → EXCEPTIONAL / STRONG / POSITIVE / NEUTRAL / DESTROYING VALUE |
| **Full WACC** | `(E/V)×cost_equity + (D/V)×cost_debt×(1−tax)` using Modigliani-Miller |
| **Accruals Ratio** | `(NI − OCF) / Assets` — negative = earnings backed by cash |
| **Gross Profitability** | `(Revenue × Gross Margin) / Assets` — Novy-Marx (2013) anomaly factor |
| **Piotroski F-Score** | Full 9-point screen: 3 profitability + 3 leverage + 3 efficiency |
| **IV Rank** | `(current_iv / hist_vol×1.15 − 0.5) × 1.25` — >0.70 = elevated fear |

---

## The 7-Gate Investment Protocol

Every top-15 stock is run through a Warren Buffett–inspired 7-gate screen:

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

## Macro Regime Detection

The MacroFetcher aggregates **9 crash signals** from live market data and FRED:

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

    SIGNALS --> COUNT{Signal\nCount}
    COUNT -->|0–1| RO[Risk-On\n+4 Tech +3 Consumer\n−5 Utilities]
    COUNT -->|2| PC[Pre-Crisis\n+8 Utilities +6 Health\n−6 Tech −6 REITs]
    COUNT -->|3+| CR[Crisis\n+15 Utilities +10 Health\n−15 Tech −12 REITs]

    style CR fill:#EF4444,color:#fff
    style PC fill:#F59E0B,color:#fff
    style RO fill:#10B981,color:#fff
```

| Signal | Trigger | Source |
|--------|---------|--------|
| VIX velocity | >7 pts in 5 days (panic spike) | yfinance ^VIX |
| HYG return | < −3% in 1 month (credit seizing) | yfinance HYG |
| Yield curve | 3M > 10Y (inverted) | yfinance ^IRX / ^TNX |
| SPY drawdown | >12% off 52-week high | yfinance SPY |
| VIX absolute | > 35 (extreme fear) | yfinance ^VIX |
| Recession probability | > 30% (RECPROUSM156N) | FRED API |
| HY credit spread | > 5% OAS (BAMLH0A0HYM2) | FRED API |
| **CFTC COT** | Net spec position < −0.30 | cftc.gov weekly ZIP |
| **BLS JOLTS** | Job openings rate MoM < −5% | BLS public API |

**Regime → Score Tilts:**

| Regime | Trigger | Sector Adjustments |
|--------|---------|-------------------|
| Risk-On | VIX < 16 | +4 Tech, +3 Consumer, −5 Utilities |
| Risk-Off | VIX > 27 | +7 Utilities, +5 Healthcare, −4 Tech |
| Rising Rate | 10Y up >0.35%/1mo | +5 Financials, +3 Energy, −7 REITs, −5 Utilities |
| Falling Rate | 10Y down >0.30% | +5 REITs, +5 Utilities, +3 Tech, −3 Financials |
| Pre-Crisis | 2 crash signals | +8 Utilities, +6 Healthcare, −6 Tech, −6 REITs |
| Crisis | 3+ crash signals | +15 Utilities, +10 Healthcare, −15 Tech, −12 REITs |

---

## Portfolio Construction

```mermaid
flowchart TD
    A[Top 30 by\ncomposite score] --> B[Correlation Matrix\non daily returns]
    B --> C[Greedy Selection\n0.70 × quality + 0.30 × diversification]
    C --> D{Beta\nCheck}
    D -->|portfolio β > target| E[Progressive 3-Pass Swap\nworst-β out · best alt in\nthresholds: 0.15 → 0.05 → any]
    D -->|β within target| F
    E --> F{Sector\nConcentration}
    F -->|sector > 3 stocks| G[Trim overweight sectors\nfill with best alternatives]
    F -->|OK| H
    G --> H[VIX-Scaled Half-Kelly Sizing]
    H --> I["eff_vol = max(hist_vol, VIX/100 + bear_penalty)\nkelly_f = edge / eff_vol² / 2\nweight = min(kelly_f / total, 0.15)"]
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
# Standard Kelly uses historical vol in denominator.
# In high-fear regimes, VIX signals future vol is higher than historical.
# We use max(hist_vol, vix_implied) so Kelly shrinks automatically in crashes.

vix_implied = VIX / 100.0          # VIX=30 → 30% annual vol floor
bear_penalty = min(0.15, ...)       # +0–15% extra floor if SPY >10% off high
eff_vol = max(hist_vol, vix_implied + bear_penalty)
kelly_f = (score/100) / (eff_vol² × 2)    # half-Kelly
weight  = min(kelly_f / Σkelly, 0.15)     # capped at 15% per position
```

> This shifts weight toward lower-volatility stocks in high-VIX environments — exactly the right behaviour in crashes.

### Portfolio Continuity Bonus
```
+3pt bonus to last session's picks (when fresh-picks mode is OFF)
→ simulates avoided round-trip transaction costs (~0.1–0.2%)
→ prevents churning positions that don't need to change
```

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

Factor weights auto-selected by `(risk_level, time_horizon)` and adapted over time:

| Profile | Momentum | Volatility | Value | Quality | Technical | Sentiment | Dividend |
|---------|----------|-----------|-------|---------|-----------|-----------|----------|
| Conservative / Short | 10% | 28% | 18% | 18% | 7% | 4% | 15% |
| Conservative / Long | 5% | 18% | 27% | 25% | 5% | 5% | 15% |
| Balanced / Medium | 18% | 14% | 22% | 25% | 11% | 5% | 5% |
| Aggressive / Short | 38% | 7% | 12% | 22% | 16% | 5% | 0% |
| Speculative / Medium | 35% | 4% | 12% | 28% | 16% | 5% | 0% |

Weights adapt over time via **adaptive learning** (see below). Sector-specific weight adjustments also applied independently.

---

## Backtest Strategy

Simulates our valuation entry strategy on historical daily closes:

| Rule | Trigger | Action |
|------|---------|--------|
| **Entry** | Price ≤ FV × 0.80 | Buy — 20% margin of safety |
| **Take Profit** | Price ≥ FV × 1.20 | Sell — 20% above fair value |
| **Stop Loss** | Price ≤ entry × 0.92 | Sell — 8% below entry price |

**Output:** Equal-weighted portfolio equity curve vs S&P 500 · Win rate · Portfolio return · Alpha · Per-stock breakdown · Per-trade log

> Uses current fair values as static levels — historical fundamentals vary; treat as illustrative.

---

## Adaptive Learning

Every session is saved to `memory/history.json`. After 30+ days:

```mermaid
flowchart LR
    S1[Save session\nentry prices + picks] --> WAIT[30+ days pass]
    WAIT --> EVAL[Fetch current prices\ncompute returns vs S&P]
    EVAL --> CORR[Pearson correlation:\nwhich factors predicted gains?]
    CORR --> UP["Factors with avg r > 0.30\n→ weight +4%"]
    CORR --> DOWN["Factors with avg r < 0.00\n→ weight −4%"]
    UP --> NORM[Renormalise + floor 3%]
    DOWN --> NORM
    NORM --> NEXT[Next run uses\nadapted weights]

    style UP fill:#10B981,color:#fff
    style DOWN fill:#EF4444,color:#fff
```

Additional learned layers:
- **Dynamic sector tilts** (observed sector performance by regime)
- **Sector-specific factor weight adjustments** (per-sector factor importance)
- **Pattern bonuses** (similarity to historical winners/losers)
- **Fresh picks penalty** (avoid repeating last 2 sessions — −22 pts)

---

## Streamlit Dashboard — 10 Tabs

| Tab | What You Get |
|-----|-------------|
| **1. Rankings** | Top-3 pick cards with signal + conviction + earnings badges; full rankings table; **factor radar chart** (7-factor fingerprint, top-5); **score distribution histogram** (portfolio vs universe); per-stock detail panel |
| **2. Valuation** | 4-method matrix; entry positioning chart; DCF sensitivity Bear/Base/Bull; **DCF scenario waterfall chart** (grouped bars with current price) |
| **3. Risk & Quality** | Risk metrics table; Sharpe vs ROIC/WACC bubble scatter; Piotroski bar chart; **Monte Carlo simulation** (200 paths, 252 days, P5/P25/P75/P95 bands) |
| **4. Protocol Gates** | 15×7 gate heatmap (Plotly); protocol summary with pass/warn/fail |
| **5. Portfolio** | Donut allocation chart; position breakdown with Kelly weights; sector concentration view |
| **6. Macro & Performance** | VIX + yield tiles; sector ETF bar chart; normalised price history vs S&P 500; correlation heatmap; **yield curve chart** (3M/2Y/10Y — red fill if inverted) |
| **7. Stock Lookup** | Search any ticker — full fresh analysis with candlestick, valuation, risk, news, protocol |
| **8. History** | Past sessions with per-pick cards · factor score bars · entry/exit P&L · "📊 Open Full Analysis" time-machine view |
| **9. Backtest** | Portfolio-wide equity curve vs S&P 500 · 5 aggregate tiles · per-stock breakdown |
| **10. Calendar** | Earnings timeline sorted by urgency (≤7d RED · ≤14d AMBER · ≤30d BLUE) · Wall Street-style quant recommendation per stock |
| **⚙️ Settings** | Theme switcher (**Terminal** / Dark / Light / Warm) · CSV/JSON export · behaviour sliders |

---

## CLI Charts — 5 Dark-Theme PNGs

Auto-generated after each `python main.py` run:

| File | Contents | Preview |
|------|----------|---------|
| `chart1_score_breakdown.png` | Stacked horizontal bars — 7 factor contributions per stock | Factor breakdown |
| `chart2_performance.png` | Normalised price history vs S&P 500 benchmark | Performance vs benchmark |
| `chart3_factor_heatmap.png` | 15 × 7 colour grid of all factor scores | Factor heatmap |
| `chart4_macro_dashboard.png` | VIX · 10Y yield history · sector ETF returns · correlation matrix | Macro view |
| `chart5_quant_protocol.png` | Gate scorecard · entry price positioning · quant thesis per stock | Protocol scorecard |

All charts use the **dark Wall Street terminal palette**:
```
Background: #0D1117    Panel: #161B22    Border: #30363D
Pass: #3FB950          Warn: #E3B341     Fail: #DA3633
```

---

## CLI Slash Commands

After the analysis pipeline completes, the terminal enters an interactive REPL:

```
/stock AAPL         → Full quant report: thesis · valuation · DCF sensitivity
                       risk metrics · analyst targets · technical · protocol · financials
/news AAPL [15]     → Headlines with per-article sentiment colour coding
/chart AAPL [6mo]   → Dark-theme candlestick + SMA 20/50/200 + RSI panel
/compare AAPL MSFT  → Side-by-side: price · P/E · EV/EBITDA · Sharpe · Piotroski · signal
/add AAPL           → Add to persistent watchlist (memory/watchlist.json)
/remove AAPL        → Remove from watchlist
/watchlist          → Show watchlist with live prices
/macro              → VIX · 10Y yield · regime · sector ETF returns · crash signal count
/history [n]        → Past sessions — win rate · alpha · per-pick returns
/exit               → Exit
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
| Deep Analysis | Gate scorecard · 4-method valuation detail · Risk metrics · Beneish M-Score |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys (optional — tool works without them)

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

```ini
# .env
FINNHUB_KEY      = ""    # finnhub.io — free, 60 calls/min (news, insider, earnings)
NEWSAPI_KEY      = ""    # newsapi.org — free, 100 req/day (broad news search)
FRED_KEY         = ""    # fred.stlouisfed.org — free, unlimited (macro series + recession prob)
ALPHAVANTAGE_KEY = ""    # alphavantage.co — free, 25 calls/day (EPS surprise history)
FMP_KEY          = ""    # financialmodelingprep.com — free, 250 calls/day (analyst revisions)
BLS_KEY          = ""    # bls.gov — free (JOLTS job openings rate)
```

All keys are optional. Without them the tool uses yfinance + EDGAR + Stooq + CFTC which are fully free and keyless.

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
├── main.py                   ← 16-step CLI pipeline + interactive command loop
├── app.py                    ← Streamlit dashboard (10 tabs + settings)
├── config.py                 ← Universe (800 tickers) · weight matrix · sector multiples
│                                · SECTOR_ERP · MACRO_TILTS · API keys · PORTFOLIO_N=15
├── requirements.txt
├── .env                      ← Your API keys (never committed — in .gitignore)
├── .env.example              ← Key template (committed, no values)
├── .streamlit/
│   └── config.toml           ← Wall Street terminal theme (dark bg · neon green/red)
├── advisor/
│   ├── collector.py          ← 8-question investor profile builder
│   ├── fetcher.py            ← yfinance + extended + Stooq fallback + FRED + Finnhub
│   │                            + CFTC COT (weekly ZIP) + BLS JOLTS
│   │                            data_quality_score + price_freshness per ticker
│   ├── scorer.py             ← 7-factor MultiFactorScorer
│   │                            rank-percentile normalisation · GARP · Value×Quality
│   │                            Beneish M-Score · factor crowding · data quality penalty
│   │                            Jensen's alpha · 52W high · asset growth · shareholder yield
│   ├── alternative_data.py   ← Tier 2 enrichment: options · Google Trends · Reddit
│   │                            · Alpha Vantage EPS · FMP revisions · SEC EDGAR XBRL
│   │                            · Smart Money (Form 4 cluster + 8-K NLP)
│   ├── smart_money.py        ← SEC EDGAR Form 4 buyer cluster + 8-K event sentiment
│   ├── portfolio.py          ← Greedy correlation-aware selection + 3-pass beta cap
│   │                            + sector concentration (max 3/sector) + VIX-scaled half-Kelly
│   │                            + portfolio continuity bonus
│   ├── valuation.py          ← DCF · Graham · EV/EBITDA · FCF yield (dynamic DR + accruals)
│   ├── risk.py               ← Altman Z · Sharpe · Sortino · full WACC · IV rank · Piotroski
│   ├── protocol.py           ← 7-gate Warren Buffett investment protocol
│   ├── learner.py            ← Session memory · adaptive weights · pattern bonuses
│   ├── news_fetcher.py       ← yfinance + RSS feeds + Finnhub + NewsAPI (negation-aware)
│   ├── cli_commands.py       ← Interactive REPL (/stock /news /chart /compare …)
│   ├── charts.py             ← 5 dark-theme matplotlib charts + candlestick
│   ├── display.py            ← Rich terminal output + deep analysis formatting
│   ├── exporter.py           ← Excel 6-sheet export
│   └── universe.py           ← Dynamic US market universe fetcher (NASDAQ API, cached 24h)
├── memory/
│   ├── history.json          ← Auto-created session log (gitignored)
│   ├── watchlist.json        ← CLI /add watchlist (gitignored)
│   └── settings.json         ← Web dashboard settings (gitignored)
└── Book1.xlsx                ← Auto-generated on each CLI run (gitignored)
```

---

## What's New in v5

```mermaid
timeline
    title Version History
    v1 : yfinance · 7-factor scoring · 10-stock portfolio
    v2 : Valuation engine (DCF · Graham) · Risk engine · Protocol gates
    v3 : Adaptive learning · Fresh picks · CLI commands · Excel export
    v4 : Tier 2 enrichment · Monte Carlo · Factor radar · Macro FRED integration
    v5 : 800-stock universe · 9 crash signals · 15-stock portfolio
       : Smart Money (Form 4 · 8-K) · CFTC COT · BLS JOLTS
       : GARP + Value×Quality interactions · Rank-percentile normalisation
       : Beneish M-Score fraud detection · VIX-scaled Kelly sizing
       : Sector concentration constraint · Jensen's alpha · Asset growth penalty
```

### v5 Highlights
- **800-stock universe** (up from 500) including 70+ international ADRs
- **9 crash signals** including CFTC COT net speculator positioning and BLS JOLTS job openings rate
- **15-stock portfolio** with max 3 stocks per sector (diversification guardrail)
- **Smart Money module** — SEC EDGAR Form 4 insider buyer cluster + 8-K event classification (no API key)
- **Beneish M-Score** — simplified 2-factor fraud/manipulation detector integrated into Quality factor
- **GARP & Value×Quality interactions** — non-linear factor combinations from Asness (1997) and Frazzini & Pedersen (2019)
- **Rank-percentile normalization** — replaces min-max, outlier-robust
- **VIX-scaled Kelly sizing** — position sizes shrink automatically in high-fear environments; VIX/100 used as minimum vol floor in Kelly denominator
- **Factor crowding detection** — penalizes stocks that score well on one factor only (crowded trades)
- **Data quality scoring** — penalizes stocks with sparse fundamental data coverage
- **Portfolio continuity bonus** — simulates avoided transaction costs for maintained positions

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
requests>=2.31.0       # HTTP (Stooq · FRED · FMP · AV · SEC EDGAR · CFTC · BLS)
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
