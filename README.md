# Stock Ranking Advisor v2

A fully local, free-data Python terminal tool that ranks stocks based on your personal investment profile using a 7-factor quantitative model with macro regime detection, adaptive learning, and Excel export.

**No paid APIs. No AI APIs. Just Yahoo Finance (free) + your laptop.**

---

## Quick Start

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Run the advisor
python main.py
```

---

## What It Does

1. **Asks 7 smart questions** to build your investor profile
2. **Fetches real-time data** for ~110 stocks via Yahoo Finance (free)
3. **Computes 7 factor scores** per stock using proven quant methods
4. **Detects the macro regime** (risk-on / risk-off / rising rates) from VIX + yields + sector ETFs
5. **Selects 10 stocks** using a greedy correlation-aware algorithm (diversified, not just top 10)
6. **Sizes positions** using half-Kelly criterion (risk-adjusted, capped at 20% per stock)
7. **Exports to `Book1.xlsx`** with 5 formatted sheets
8. **Generates 4 charts** and saves them as PNGs
9. **Remembers every session** and learns which factors actually predicted returns over time

---

## The 7 Questions

| # | Question | Why It Matters |
|---|----------|---------------|
| 1 | Portfolio size | Drives position sizing and share counts |
| 2 | Time horizon (1yr / 3yr / 5yr) | Short-term → momentum-heavy; long-term → value + quality heavy |
| 3 | Risk tolerance (1–4) | Controls beta/volatility filters and weight distribution |
| 4 | Investment goal | Income goal boosts dividend weight; speculative goal boosts momentum |
| 5 | Drawdown gut check | Extra volatility penalty if you can't stomach large drops |
| 6 | Sector focus / exclusions | Filters universe; adds score bonus to preferred sectors |
| 7 | Existing holdings | Removes overlap from recommendations |

---

## The 7-Factor Scoring Model

Each stock is scored 0–100 on seven independent factors, then combined using a weight matrix tuned to your risk profile and time horizon.

### Factor 1 — Momentum
Weighted combination of 1-month (20%), 3-month (35%), and 6-month (45%) price returns.
Higher recent returns score higher. Emphasised for aggressive / short-term profiles.

### Factor 2 — Volatility
Annualised standard deviation of daily returns, **inverted** — low volatility scores high.
Heavily weighted for conservative profiles; near-zero for speculative.

### Factor 3 — Value
Uses trailing P/E relative to the sector median. Cheaper than peers = higher score.
Enhanced with a free cash flow yield proxy (FCF / market cap) for quality validation.

### Factor 4 — Quality (Piotroski-inspired)
An 8-point fundamental checklist:
1. Return on Assets > 4%
2. Operating cash flow > 0
3. Free cash flow > 0
4. Debt / Equity < 1.0
5. Current ratio > 1.5
6. Profit margins > 10%
7. Revenue growth > 0
8. Earnings growth > 0

Score = (points / 8) × 100. Blended with ROE and profit margins for richness.

### Factor 5 — Technical
Computed entirely from price history — no extra data needed.
- **RSI (14-day, 30%)**: Ideal zone 40–65. Oversold gets contrarian bonus, overbought penalised.
- **MACD (12/26/9, 40%)**: Above signal + positive histogram = bullish (88/100).
- **MA crossover (30%)**: Price > SMA50 > SMA200 = strong uptrend (90/100). Golden cross = +12 bonus.

### Factor 6 — Sentiment
Fetches the last 7 news headlines for each ticker via Yahoo Finance (no extra API).
Scores headlines using 50+ curated positive/negative financial words.
Maps from −1/+1 range → 0–100.

### Factor 7 — Dividend
Raw dividend yield (capped at 15% to prevent outliers dominating).
Heavily weighted only for income-focused profiles (goal = "income").

---

## Weight Matrix

The 7 weights vary by (risk_level × time_horizon). Examples:

| Profile | Mom | Vol | Val | Qual | Tech | Sent | Div |
|---------|-----|-----|-----|------|------|------|-----|
| Low / Long | 5% | 18% | 27% | 25% | 5% | 5% | 15% |
| Moderate / Medium | 18% | 14% | 22% | 25% | 11% | 5% | 5% |
| High / Short | 38% | 7% | 12% | 22% | 16% | 5% | 0% |
| Speculative / Short | 45% | 4% | 8% | 22% | 16% | 5% | 0% |

---

## Macro Regime Detection

Fetched from Yahoo Finance (^VIX, ^TNX, sector ETFs XLK/XLV/XLF/...):

| Regime | Trigger | Effect |
|--------|---------|--------|
| Risk-On | VIX < 16 | +4 pts Tech, +3 Consumer, −5 Utilities |
| Risk-Off | VIX > 27 | +7 Utilities, +5 Healthcare, −4 Tech |
| Rising Rate | 10Y yield ↑ > 0.35% in 1 month | +5 Financials, −7 REITs, −5 Utilities |
| Falling Rate | 10Y yield ↓ > 0.30% | +5 REITs, +5 Utilities, +3 Tech |
| Neutral | Otherwise | No tilt |

Top-3 sector ETFs by 3-month return also get a +3 bonus.

---

## Portfolio Construction

After scoring all valid stocks:

1. **Top 30 candidates** selected by composite score
2. **Correlation matrix** computed (Pearson on daily returns)
3. **Greedy diversification**: each next pick chosen to maximise `score × (0.70 + 0.30 × (1 − avg_correlation_with_selected))`
4. This ensures the final 10 are both high-quality **and** not all moving together

**Position sizing** (half-Kelly):
- Kelly fraction = `(score/100) / vol² / 2`
- Capped at 20% per position
- Renormalised to sum = 100%

---

## Adaptive Learning

Every session is saved to `memory/history.json`.

After 30+ days, the tool automatically:
1. Fetches current prices for past picks
2. Calculates each pick's return vs the S&P 500
3. Measures which factors actually correlated with returns (Pearson r)
4. If a factor had avg r > 0.3 → its weight increases 4%
5. If a factor had avg r < 0.0 → its weight decreases 4%
6. Weights are renormalised and floored at 3%

The more sessions you run, the smarter the weights become for your specific profile.

---

## Excel Export (Book1.xlsx)

Five sheets, auto-formatted with conditional colour coding:

| Sheet | Contents |
|-------|----------|
| **Latest Picks** | Top 10 with all 7 factor scores (green/amber/red) |
| **Allocation** | Weight%, dollar amounts, approx shares |
| **Macro Overview** | VIX, 10Y yield, regime, sector ETF rankings |
| **History** | All past sessions with tickers |
| **Track Record** | Evaluated sessions with alpha vs S&P 500 |

---

## Charts (4 PNGs)

| File | Contents |
|------|----------|
| `chart1_score_breakdown.png` | Stacked horizontal bars — factor contribution per stock |
| `chart2_performance.png` | Normalised price history vs S&P 500 |
| `chart3_factor_heatmap.png` | 10 × 7 colour grid of all factor scores |
| `chart4_macro_dashboard.png` | VIX trend · 10Y yield · sector ETF returns · correlation matrix |

---

## File Structure

```
portfolio/
├── main.py                 ← Run this
├── config.py               ← Universe, weights, word lists
├── advisor/
│   ├── collector.py        ← 7-question profile builder
│   ├── fetcher.py          ← yfinance + technicals + sentiment
│   ├── scorer.py           ← 7-factor scoring + macro tilt
│   ├── portfolio.py        ← Correlation-aware selection + Kelly sizing
│   ├── learner.py          ← Session memory + weight adaptation
│   ├── charts.py           ← 4 matplotlib charts
│   ├── display.py          ← Terminal output
│   └── exporter.py         ← Excel export
├── memory/
│   └── history.json        ← Auto-created, grows over time
├── stock_advisor.py        ← Original simple version (v1)
├── requirements.txt
└── README.md
```

---

## Disclaimer

This tool is for **educational and informational purposes only**.
Past performance does not guarantee future results.
Rankings are quantitative metrics and are **not financial advice**.
Always conduct your own due diligence before investing.
