# config.py — Global constants: stock universe, weights, macro tickers, word lists

# ── Macro / benchmark tickers (all free via yfinance) ──────────────────────────
SP500_TICKER    = "^GSPC"
VIX_TICKER      = "^VIX"
YIELD_10Y_TICKER = "^TNX"

SECTOR_ETFS = {
    "Technology":  "XLK",
    "Healthcare":  "XLV",
    "Financials":  "XLF",
    "Consumer":    "XLY",
    "Energy":      "XLE",
    "Industrials": "XLI",
    "Utilities":   "XLU",
    "Real Estate": "XLRE",
    "Materials":   "XLB",
}

# ── Stock universe: ~110 liquid stocks across 9 sectors ───────────────────────
STOCK_UNIVERSE = {
    "Technology":  [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL",
        "CRM", "ADBE", "AMD", "QCOM", "TXN", "NOW", "INTU",
        "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "PANW",
    ],
    "Healthcare":  [
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        "DHR", "AMGN", "BMY", "GILD", "CI", "ISRG", "SYK",
        "BDX", "ZTS", "VRTX", "REGN", "HCA",
    ],
    "Financials":  [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP",
        "V", "MA", "C", "SCHW", "USB", "PNC", "COF",
        "ICE", "CME", "MCO", "SPGI", "TFC",
    ],
    "Consumer":    [
        "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX",
        "COST", "PG", "KO", "PEP", "PM", "CL", "KMB",
        "GIS", "K", "HSY", "MDLZ", "YUM",
    ],
    "Energy":      [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX",
        "OXY", "KMI", "WMB", "LNG", "DVN", "HES", "BKR",
    ],
    "Industrials": [
        "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE",
        "EMR", "ETN", "PH", "ROK", "ITW", "NSC", "UNP",
        "CSX", "FDX", "WM", "RSG", "FAST",
    ],
    "Utilities":   [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ED",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "AVB", "EQR",
    ],
    "Materials":   [
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "ALB",
    ],
}

# ── Sector median EV/EBITDA multiples (approximate) ──────────────────────────
SECTOR_EV_EBITDA = {
    "Technology":  22,
    "Healthcare":  16,
    "Financials":  12,
    "Consumer":    16,
    "Energy":       8,
    "Industrials": 14,
    "Utilities":   12,
    "Real Estate": 20,
    "Materials":   10,
    "Unknown":     14,
}

# ── Sector median P/E ratios (approximate, updated periodically) ──────────────
SECTOR_MEDIAN_PE = {
    "Technology":  28,
    "Healthcare":  22,
    "Financials":  14,
    "Consumer":    24,
    "Energy":      12,
    "Industrials": 20,
    "Utilities":   18,
    "Real Estate": 35,
    "Materials":   16,
    "Unknown":     20,
}

# ── 7-factor weight matrix ────────────────────────────────────────────────────
# Factors: [momentum, volatility, value, quality, technical, sentiment, dividend]
# Key: (risk_level 1–4, time_horizon "short"/"medium"/"long")
# All rows sum to 1.0
FACTOR_NAMES = ["momentum", "volatility", "value", "quality", "technical", "sentiment", "dividend"]

WEIGHT_MATRIX = {
    # Conservative / Low risk — emphasise stability, value, quality, dividends
    (1, "short"):  [0.10, 0.28, 0.18, 0.18, 0.07, 0.04, 0.15],
    (1, "medium"): [0.08, 0.23, 0.22, 0.20, 0.07, 0.05, 0.15],
    (1, "long"):   [0.05, 0.18, 0.27, 0.25, 0.05, 0.05, 0.15],

    # Moderate / Balanced
    (2, "short"):  [0.22, 0.17, 0.18, 0.20, 0.13, 0.05, 0.05],
    (2, "medium"): [0.18, 0.14, 0.22, 0.25, 0.11, 0.05, 0.05],
    (2, "long"):   [0.12, 0.12, 0.27, 0.30, 0.09, 0.05, 0.05],

    # Aggressive / High risk — momentum + technical lead
    (3, "short"):  [0.38, 0.07, 0.12, 0.22, 0.16, 0.05, 0.00],
    (3, "medium"): [0.28, 0.07, 0.18, 0.28, 0.14, 0.05, 0.00],
    (3, "long"):   [0.20, 0.07, 0.22, 0.35, 0.11, 0.05, 0.00],

    # Speculative / Very High risk
    (4, "short"):  [0.45, 0.04, 0.08, 0.22, 0.16, 0.05, 0.00],
    (4, "medium"): [0.35, 0.04, 0.12, 0.28, 0.16, 0.05, 0.00],
    (4, "long"):   [0.25, 0.04, 0.18, 0.38, 0.10, 0.05, 0.00],
}

# ── Labels ────────────────────────────────────────────────────────────────────
RISK_LABELS = {
    1: "Low / Conservative",
    2: "Moderate / Balanced",
    3: "High / Aggressive",
    4: "Very High / Speculative",
}

HORIZON_LABELS = {
    "short":  "Short (1 year)",
    "medium": "Medium (3 years)",
    "long":   "Long (5 years)",
}

GOAL_LABELS = {
    "retirement":  "Retirement / FIRE",
    "wealth":      "Long-term Wealth Building",
    "income":      "Income & Dividends",
    "speculative": "Speculative Growth",
}

# ── Macro regime sector tilts (points added to composite score post-normalise) ─
MACRO_TILTS = {
    "risk_on": {
        "Technology": +4, "Consumer": +3, "Financials": +2,
        "Utilities": -5, "Real Estate": -3,
    },
    "risk_off": {
        "Utilities": +7, "Healthcare": +5, "Consumer": +3,
        "Technology": -4, "Energy": -2,
    },
    "rising_rate": {
        "Financials": +5, "Energy": +3,
        "Real Estate": -7, "Utilities": -5, "Technology": -2,
    },
    "falling_rate": {
        "Real Estate": +5, "Utilities": +5, "Technology": +3,
        "Financials": -3,
    },
    "neutral": {},
}

# ── News sentiment word lists ─────────────────────────────────────────────────
POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "profit", "profits",
    "record", "growth", "upgrade", "upgrades", "buy", "outperform", "raise",
    "raised", "higher", "strong", "exceeds", "gain", "gains", "rise", "rises",
    "bullish", "acquisition", "deal", "breakthrough", "expand", "expansion",
    "exceed", "positive", "improve", "improvement", "dividend", "buyback",
    "innovation", "launch", "partner", "partnership", "win", "award", "boom",
    "robust", "accelerate", "momentum", "revenue", "milestone",
}

NEGATIVE_WORDS = {
    "miss", "misses", "drop", "drops", "fall", "falls", "loss", "losses",
    "weak", "weaker", "downgrade", "downgrades", "sell", "underperform", "cut",
    "cuts", "lower", "concern", "concerns", "risk", "risks", "decline",
    "bear", "bearish", "layoff", "layoffs", "investigation", "lawsuit", "fraud",
    "penalty", "fine", "warning", "recall", "debt", "bankruptcy", "default",
    "volatile", "uncertainty", "headwind", "pressure", "slowdown", "recession",
    "withdraw", "delay", "disappointing", "challenging", "probe", "scrutiny",
}
