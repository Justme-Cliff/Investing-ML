# config.py — Global constants: stock universe, weights, macro tickers, word lists

import os
from dotenv import load_dotenv

# Load .env from the project root (silently ignored if the file doesn't exist)
load_dotenv()

# ── Optional free API keys — loaded from .env (never hard-coded here) ─────────
# Keys are optional — the tool works without them; they enhance data quality.
# Copy .env.example → .env and fill in your own keys.
FINNHUB_KEY      = os.getenv("FINNHUB_KEY",      "")   # company news, insider trades, earnings calendar
NEWSAPI_KEY      = os.getenv("NEWSAPI_KEY",      "")   # broad financial news search
FRED_KEY         = os.getenv("FRED_KEY",         "")   # macro series: CPI, UNRATE, FEDFUNDS, T10Y2Y, etc.
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "")   # earnings, income statement supplements
FMP_KEY          = os.getenv("FMP_KEY",          "")   # Financial Modeling Prep: analyst revisions, ratings
BLS_KEY          = os.getenv("BLS_KEY",          "")   # Bureau of Labor Statistics (optional — 500/day with key vs 25 without)

# ── Dynamic universe settings ─────────────────────────────────────────────────
# When DYNAMIC_UNIVERSE = True the pipeline downloads ALL US-listed common
# stocks (~4,000–6,000) from NASDAQ's free public API at startup (cached 24h).
# UNIVERSE_MAX_TICKERS controls how many are actually scored per run.
# Selection: top-200 by market cap are always included; the remaining slots are
# filled with a RANDOM sample of the rest → different stocks surface every run.
# Raise UNIVERSE_MAX_TICKERS for deeper coverage (runtime scales linearly).
DYNAMIC_UNIVERSE        = True
UNIVERSE_MIN_MARKET_CAP = 100_000_000   # $100 M — filters out micro-cap noise
UNIVERSE_MAX_TICKERS    = 800           # stocks scored per run (expanded for international coverage)
PORTFOLIO_N             = 15            # final portfolio size (15 = better diversification vs 10)

# ── Sector equity risk premiums (% — added to rf_rate for sector-specific WACC/DCF) ─
SECTOR_ERP = {
    "Technology":    6.0,
    "Healthcare":    5.5,
    "Financials":    5.5,
    "Consumer":      5.0,
    "Energy":        5.5,
    "Industrials":   5.0,
    "Utilities":     3.5,
    "Real Estate":   4.0,
    "Materials":     5.5,
    "Communication": 5.5,
    "Unknown":       5.5,
}

# ── Macro / benchmark tickers (all free via yfinance) ──────────────────────────
SP500_TICKER    = "^GSPC"
VIX_TICKER      = "^VIX"
YIELD_10Y_TICKER = "^TNX"

SECTOR_ETFS = {
    "Technology":    "XLK",
    "Healthcare":    "XLV",
    "Financials":    "XLF",
    "Consumer":      "XLY",
    "Energy":        "XLE",
    "Industrials":   "XLI",
    "Utilities":     "XLU",
    "Real Estate":   "XLRE",
    "Materials":     "XLB",
    "Communication": "XLC",
}

# ── Stock universe: ~300 liquid stocks across 10 sectors ─────────────────────
STOCK_UNIVERSE = {
    "Technology": [
        # Mega-cap / established
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL",
        "CRM", "ADBE", "AMD", "QCOM", "TXN", "NOW", "INTU",
        "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "PANW",
        # Semiconductors
        "INTC", "CSCO", "MU", "NXPI", "MRVL", "ON", "MPWR",
        "HPQ", "DELL", "SMCI", "ANET", "PSTG",
        # Cloud / cybersecurity / software
        "CRWD", "ZS", "NET", "DDOG", "SNOW", "PLTR", "WDAY",
        "HUBS", "TEAM", "FTNT", "OKTA", "MDB", "GTLB",
        # International ADRs — Technology
        "ASML",   # ASML Holding (Netherlands) — world's only EUV lithography maker
        "TSM",    # Taiwan Semiconductor — largest foundry globally
        "SAP",    # SAP SE (Germany) — enterprise software leader
        "SHOP",   # Shopify (Canada) — e-commerce platform
        "SE",     # Sea Limited (Singapore) — Southeast Asia tech conglomerate
        "GRAB",   # Grab Holdings (Singapore) — SEA super-app
        "MELI",   # MercadoLibre (Argentina) — LatAm e-commerce + fintech
        "BIDU",   # Baidu (China) — search + AI
        "JD",     # JD.com (China) — e-commerce
    ],
    "Healthcare": [
        # Mega-cap / established
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
        "DHR", "AMGN", "BMY", "GILD", "CI", "ISRG", "SYK",
        "BDX", "ZTS", "VRTX", "REGN", "HCA",
        # Managed care / pharmacy
        "CVS", "MCK", "ELV", "HUM", "CAH",
        # Biotech / medtech
        "BIIB", "ILMN", "IDXX", "EW", "MTD", "BAX", "HOLX",
        "MRNA", "IQV", "DXCM", "PODD", "EXAS", "INCY", "ALNY", "GEHC",
        # International ADRs — Healthcare
        "NVO",    # Novo Nordisk (Denmark) — GLP-1 diabetes/obesity leader
        "AZN",    # AstraZeneca (UK) — oncology + rare disease
        "NVS",    # Novartis (Switzerland) — diversified pharma
        "SNY",    # Sanofi (France) — immunology + vaccines
        "GSK",    # GSK (UK) — vaccines + specialty medicines
        "RHHBY",  # Roche (Switzerland) — diagnostics + oncology
    ],
    "Financials": [
        # Mega-cap / diversified
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP",
        "V", "MA", "C", "SCHW", "USB", "PNC", "COF",
        "ICE", "CME", "MCO", "SPGI", "TFC",
        # Payments / fintech
        "FIS", "FISV", "GPN", "PYPL", "SQ", "SOFI", "AMP",
        # Insurance
        "AFL", "MET", "PRU", "ALL", "TRV",
        # Regional banks
        "MTB", "HBAN", "RF", "KEY", "CFG", "FITB",
        # Consumer finance
        "ALLY", "SYF",
        # International ADRs — Financials
        "ING",    # ING Groep (Netherlands) — European banking
        "SAN",    # Banco Santander (Spain) — global retail banking
        "MFC",    # Manulife Financial (Canada) — insurance + wealth mgmt
        "ITUB",   # Itaú Unibanco (Brazil) — largest LatAm bank
        "BBD",    # Bradesco (Brazil) — LatAm banking
    ],
    "Consumer": [
        # Staples / retail
        "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX",
        "COST", "PG", "KO", "PEP", "PM", "CL", "KMB",
        "GIS", "K", "HSY", "MDLZ", "YUM",
        # E-commerce / autos / marketplace
        "AMZN", "TSLA", "EBAY", "ETSY", "LULU", "ROST",
        "DG", "DLTR", "BBY", "F", "GM", "KR", "UBER",
        # Travel / leisure / restaurants
        "BKNG", "EXPE", "MAR", "HLT", "CMG", "DRI", "QSR",
        # International ADRs — Consumer
        "TM",     # Toyota Motor (Japan) — world's largest automaker
        "HMC",    # Honda Motor (Japan) — autos + motorcycles
        "SONY",   # Sony Group (Japan) — entertainment + electronics
        "UL",     # Unilever (UK/Netherlands) — consumer staples
        "DEO",    # Diageo (UK) — premium spirits leader
        "NSRGY",  # Nestlé (Switzerland) — largest food company
    ],
    "Energy": [
        # Integrated / E&P
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX",
        "OXY", "KMI", "WMB", "LNG", "DVN", "HES", "BKR",
        # E&P / oilfield services
        "APA", "FANG", "HAL", "CTRA", "RIG",
        # Midstream
        "TRGP", "OKE", "ET", "MPLX", "WES",
        # International ADRs — Energy
        "SHEL",   # Shell (UK/Netherlands) — major integrated oil & gas
        "BP",     # BP (UK) — integrated energy + renewables
        "SU",     # Suncor Energy (Canada) — oil sands + refining
        "TTE",    # TotalEnergies (France) — integrated energy
        "E",      # Eni (Italy) — integrated oil & gas
    ],
    "Industrials": [
        # Diversified / transport
        "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE",
        "EMR", "ETN", "PH", "ROK", "ITW", "NSC", "UNP",
        "CSX", "FDX", "WM", "RSG", "FAST",
        # Aerospace / defense
        "BA", "LHX", "GD", "NOC", "TDG", "LDOS",
        # Specialty / automation
        "CARR", "OTIS", "CPRT", "ROP", "VRSK", "CTAS",
        "GWW", "SWK", "XYL", "GNRC", "IR", "AME", "AXON",
        # International ADRs — Industrials
        "CNI",    # Canadian National Railway — top-tier NA railroad
        "CP",     # Canadian Pacific Kansas City — transcontinental railroad
        "SIEGY",  # Siemens (Germany) — industrial automation + electrification
        "ABB",    # ABB Ltd (Switzerland) — robotics + power grids
    ],
    "Utilities": [
        # Established
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ED",
        # Mid-cap utilities
        "AWK", "WEC", "ES", "EIX", "PPL", "CNP", "AEE", "NI", "DTE", "ETR",
    ],
    "Real Estate": [
        # REITs — established
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "AVB", "EQR",
        # REITs — diversified / specialty
        "DLR", "VTR", "ARE", "BXP", "KIM", "NNN", "STAG", "IRM", "VICI", "CBRE",
    ],
    "Materials": [
        # Chemicals / mining
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "ALB",
        # Specialty chemicals / packaging
        "DD", "PPG", "IFF", "RPM", "CE", "LYB", "MOS", "CF", "IP", "PKG",
    ],
    "Communication": [
        # Streaming / media
        "NFLX", "DIS", "CMCSA", "WBD",
        # Telecom
        "T", "VZ", "CHTR", "TMUS",
        # Entertainment / gaming / advertising
        "LYV", "OMC", "IPG", "TTWO", "EA", "FOXA",
    ],
}

# ── Sector median EV/EBITDA multiples (approximate) ──────────────────────────
SECTOR_EV_EBITDA = {
    "Technology":    22,
    "Healthcare":    16,
    "Financials":    12,
    "Consumer":      16,
    "Energy":         8,
    "Industrials":   14,
    "Utilities":     12,
    "Real Estate":   20,
    "Materials":     10,
    "Communication": 18,
    "Unknown":       14,
}

# ── Sector median P/E ratios (approximate, updated periodically) ──────────────
SECTOR_MEDIAN_PE = {
    "Technology":    28,
    "Healthcare":    22,
    "Financials":    14,
    "Consumer":      24,
    "Energy":        12,
    "Industrials":   20,
    "Utilities":     18,
    "Real Estate":   35,
    "Materials":     16,
    "Communication": 24,
    "Unknown":       20,
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
        "Technology": +4, "Consumer": +3, "Financials": +2, "Communication": +3,
        "Utilities": -5, "Real Estate": -3,
    },
    "risk_off": {
        "Utilities": +7, "Healthcare": +5, "Consumer": +3,
        "Technology": -4, "Energy": -2, "Communication": -2,
    },
    "rising_rate": {
        "Financials": +5, "Energy": +3,
        "Real Estate": -7, "Utilities": -5, "Technology": -2, "Communication": -2,
    },
    "falling_rate": {
        "Real Estate": +5, "Utilities": +5, "Technology": +3, "Communication": +2,
        "Financials": -3,
    },
    "crisis": {
        "Utilities": +15, "Healthcare": +10,
        "Technology": -15, "Real Estate": -12, "Financials": -10,
        "Consumer": -5, "Energy": -5, "Materials": -8, "Industrials": -8, "Communication": -5,
    },
    "pre_crisis": {
        "Utilities": +8, "Healthcare": +6,
        "Technology": -6, "Real Estate": -6, "Financials": -5, "Materials": -3,
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
    # Extended set
    "raised guidance", "margin expansion", "record revenue", "beat estimates",
    "record earnings", "accelerating", "exceeded", "rebound", "recovery",
    "upside", "boost", "strengthens", "raised forecast", "consensus beat",
    "market share gain",
}

NEGATIVE_WORDS = {
    "miss", "misses", "drop", "drops", "fall", "falls", "loss", "losses",
    "weak", "weaker", "downgrade", "downgrades", "sell", "underperform", "cut",
    "cuts", "lower", "concern", "concerns", "risk", "risks", "decline",
    "bear", "bearish", "layoff", "layoffs", "investigation", "lawsuit", "fraud",
    "penalty", "fine", "warning", "recall", "debt", "bankruptcy", "default",
    "volatile", "uncertainty", "headwind", "pressure", "slowdown", "recession",
    "withdraw", "delay", "disappointing", "challenging", "probe", "scrutiny",
    # Extended set
    "lowered guidance", "write-down", "restatement", "margin compression",
    "guidance cut", "missed estimates", "impairment", "suspended",
    "deteriorating", "revenue miss", "earnings miss", "negative outlook",
    "credit risk", "debt burden", "cost inflation",
}
