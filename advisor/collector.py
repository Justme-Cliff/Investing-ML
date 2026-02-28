# advisor/collector.py — UserProfile dataclass + enhanced 7-question InputCollector

from dataclasses import dataclass, field
from typing import List, Tuple
from config import RISK_LABELS, HORIZON_LABELS, GOAL_LABELS, STOCK_UNIVERSE


@dataclass
class UserProfile:
    # Core (questions 1–3)
    portfolio_size: float
    time_horizon: str           # "short" | "medium" | "long"
    time_horizon_years: int
    yf_period: str              # "1y" | "3y" | "5y"
    risk_label: str
    risk_level: int             # 1–4

    # Extended (questions 4–7)
    goal: str                   # "retirement" | "wealth" | "income" | "speculative"
    goal_label: str
    drawdown_ok: float          # Max acceptable drop: 0.10–0.40
    preferred_sectors: List[str] = field(default_factory=list)
    excluded_sectors: List[str] = field(default_factory=list)
    existing_tickers: List[str] = field(default_factory=list)

    @property
    def income_focused(self) -> bool:
        return self.goal == "income"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sep(title: str = ""):
    line = "─" * 62
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


class InputCollector:
    """Interactively collects 7 profile questions and returns a UserProfile."""

    ALL_SECTORS = list(STOCK_UNIVERSE.keys())

    def collect(self) -> UserProfile:
        self._welcome()
        while True:
            size                          = self._ask_size()
            horizon, hy, period           = self._ask_horizon()
            risk_label, risk_level        = self._ask_risk()
            goal, goal_label              = self._ask_goal()
            drawdown                      = self._ask_drawdown()
            pref_sectors, excl_sectors    = self._ask_sectors()
            existing                      = self._ask_existing()

            profile = UserProfile(
                portfolio_size      = size,
                time_horizon        = horizon,
                time_horizon_years  = hy,
                yf_period           = period,
                risk_label          = risk_label,
                risk_level          = risk_level,
                goal                = goal,
                goal_label          = goal_label,
                drawdown_ok         = drawdown,
                preferred_sectors   = pref_sectors,
                excluded_sectors    = excl_sectors,
                existing_tickers    = existing,
            )
            if self._confirm(profile):
                return profile
            print("\n  Starting over...\n")

    # ── Welcome ───────────────────────────────────────────────────────────────
    def _welcome(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║         PERSONALIZED  STOCK  RANKING  ADVISOR  v2           ║
║  7-factor model · macro regime · learning memory · Excel    ║
║  Real-time data from Yahoo Finance  ·  No API key needed    ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Q1: Portfolio size ────────────────────────────────────────────────────
    def _ask_size(self) -> float:
        _sep("QUESTION 1 of 7 — Portfolio Size")
        print("  How much are you planning to invest?")
        print("  Accepts: 5000  ·  10k  ·  1.5m  ·  $50,000\n")
        while True:
            raw = input("  Amount: ").strip()
            try:
                v = self._parse_amount(raw)
                if v <= 0:
                    print("  Must be positive.\n"); continue
                if v < 1_000:
                    print(f"  Note: Small portfolio (${v:,.0f}) — diversification may be limited.\n")
                elif v > 10_000_000:
                    print("  Note: Very large portfolio — consider a licensed advisor too.\n")
                print()
                return v
            except ValueError:
                print(f"  Cannot parse '{raw}'. Try: 10000  10k  1.5m\n")

    def _parse_amount(self, raw: str) -> float:
        raw = raw.strip().replace(",", "").replace("$", "").replace(" ", "")
        if not raw:
            raise ValueError
        mul = 1
        if raw.lower().endswith("k"):
            mul, raw = 1_000, raw[:-1]
        elif raw.lower().endswith("m"):
            mul, raw = 1_000_000, raw[:-1]
        return float(raw) * mul

    # ── Q2: Time horizon ──────────────────────────────────────────────────────
    def _ask_horizon(self) -> Tuple[str, int, str]:
        _sep("QUESTION 2 of 7 — Investment Time Horizon")
        print("  1.  Short  term  (up to 1 year)   — riding near-term momentum")
        print("  2.  Medium term  (2–5 years)       — balanced growth & stability")
        print("  3.  Long   term  (6+ years)        — compounding fundamentals\n")
        MAP = {
            "1": ("short", 1, "1y"), "short": ("short", 1, "1y"),
            "2": ("medium", 3, "3y"), "medium": ("medium", 3, "3y"),
            "3": ("long", 5, "5y"), "long": ("long", 5, "5y"),
        }
        while True:
            raw = input("  Choice (1 / 2 / 3): ").strip().lower()
            if raw in MAP:
                print(); return MAP[raw]
            print(f"  Invalid '{raw}'. Enter 1, 2, or 3.\n")

    # ── Q3: Risk tolerance ────────────────────────────────────────────────────
    def _ask_risk(self) -> Tuple[str, int]:
        _sep("QUESTION 3 of 7 — Risk Tolerance")
        print("  1.  Low / Conservative      — Capital preservation, steady income")
        print("  2.  Moderate / Balanced     — Growth + stability, some volatility OK")
        print("  3.  High / Aggressive       — Growth-focused, comfortable with big swings")
        print("  4.  Very High / Speculative — Max growth, very high volatility OK\n")
        MAP = {
            "1": (RISK_LABELS[1], 1), "low": (RISK_LABELS[1], 1),
            "conservative": (RISK_LABELS[1], 1),
            "2": (RISK_LABELS[2], 2), "moderate": (RISK_LABELS[2], 2),
            "balanced": (RISK_LABELS[2], 2), "medium": (RISK_LABELS[2], 2),
            "3": (RISK_LABELS[3], 3), "high": (RISK_LABELS[3], 3),
            "aggressive": (RISK_LABELS[3], 3),
            "4": (RISK_LABELS[4], 4), "very high": (RISK_LABELS[4], 4),
            "very_high": (RISK_LABELS[4], 4), "speculative": (RISK_LABELS[4], 4),
        }
        while True:
            raw = input("  Choice (1 / 2 / 3 / 4): ").strip().lower()
            if raw in MAP:
                print(); return MAP[raw]
            print(f"  Invalid '{raw}'. Enter 1, 2, 3, or 4.\n")

    # ── Q4: Investment goal ───────────────────────────────────────────────────
    def _ask_goal(self) -> Tuple[str, str]:
        _sep("QUESTION 4 of 7 — Primary Investment Goal")
        print("  1.  Retirement / FIRE        — Long horizon, protect capital near end")
        print("  2.  Long-term Wealth Build   — Compound returns over decades")
        print("  3.  Income & Dividends       — Cash flow, steady quarterly payments")
        print("  4.  Speculative Growth       — High-risk, potentially high-reward bets\n")
        MAP = {
            "1": ("retirement", GOAL_LABELS["retirement"]),
            "retirement": ("retirement", GOAL_LABELS["retirement"]),
            "fire": ("retirement", GOAL_LABELS["retirement"]),
            "2": ("wealth", GOAL_LABELS["wealth"]),
            "wealth": ("wealth", GOAL_LABELS["wealth"]),
            "3": ("income", GOAL_LABELS["income"]),
            "income": ("income", GOAL_LABELS["income"]),
            "dividends": ("income", GOAL_LABELS["income"]),
            "4": ("speculative", GOAL_LABELS["speculative"]),
            "speculative": ("speculative", GOAL_LABELS["speculative"]),
        }
        while True:
            raw = input("  Choice (1 / 2 / 3 / 4): ").strip().lower()
            if raw in MAP:
                print(); return MAP[raw]
            print(f"  Invalid '{raw}'. Enter 1, 2, 3, or 4.\n")

    # ── Q5: Drawdown gut check ────────────────────────────────────────────────
    def _ask_drawdown(self) -> float:
        _sep("QUESTION 5 of 7 — Emotional Risk Check")
        print("  Imagine your portfolio drops 30% in 3 months. What do you do?")
        print("  1.  Buy more aggressively   — I see it as a sale  (drawdown: 40%)")
        print("  2.  Hold steady             — I trust the thesis   (drawdown: 25%)")
        print("  3.  Feel very uncomfortable — Maybe sell some      (drawdown: 15%)")
        print("  4.  Sell everything         — I can't stomach it   (drawdown: 10%)\n")
        MAP = {"1": 0.40, "buy": 0.40, "2": 0.25, "hold": 0.25,
               "3": 0.15, "uncomfortable": 0.15, "4": 0.10, "sell": 0.10}
        while True:
            raw = input("  Choice (1 / 2 / 3 / 4): ").strip().lower()
            if raw in MAP:
                print(); return MAP[raw]
            print(f"  Invalid '{raw}'. Enter 1, 2, 3, or 4.\n")

    # ── Q6: Sector preferences ────────────────────────────────────────────────
    def _ask_sectors(self) -> Tuple[List[str], List[str]]:
        _sep("QUESTION 6 of 7 — Sector Preferences (optional)")
        print("  Available sectors:")
        for i, s in enumerate(self.ALL_SECTORS, 1):
            print(f"    {i:2}.  {s}")
        print()
        print("  FOCUS on specific sectors? (e.g. '1,3' or 'Technology,Financials')")
        print("  Press Enter to skip (no preference).")
        pref = self._parse_sector_input(input("  Focus: ").strip())
        print()
        print("  EXCLUDE any sectors? (e.g. 'Energy' or '5')")
        print("  Press Enter to skip.")
        excl = self._parse_sector_input(input("  Exclude: ").strip())
        print()
        return pref, excl

    def _parse_sector_input(self, raw: str) -> List[str]:
        if not raw:
            return []
        result = []
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            # Try as number
            try:
                idx = int(part) - 1
                if 0 <= idx < len(self.ALL_SECTORS):
                    result.append(self.ALL_SECTORS[idx])
                continue
            except ValueError:
                pass
            # Try as sector name (case-insensitive)
            for sector in self.ALL_SECTORS:
                if part.lower() in sector.lower():
                    result.append(sector)
                    break
        return list(dict.fromkeys(result))  # dedupe, preserve order

    # ── Q7: Existing holdings ─────────────────────────────────────────────────
    def _ask_existing(self) -> List[str]:
        _sep("QUESTION 7 of 7 — Existing Holdings (optional)")
        print("  List any stocks you already own so I can avoid recommending them.")
        print("  (e.g. 'AAPL, MSFT, NVDA')  Press Enter to skip.\n")
        raw = input("  Tickers: ").strip().upper()
        if not raw:
            return []
        tickers = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
        print(f"  Got it — {len(tickers)} existing ticker(s) noted.\n")
        return tickers

    # ── Confirmation ──────────────────────────────────────────────────────────
    def _confirm(self, p: UserProfile) -> bool:
        print("\n" + "═" * 62)
        print("  YOUR INVESTMENT PROFILE SUMMARY")
        print("═" * 62)
        print(f"  Portfolio      :  ${p.portfolio_size:>12,.2f}")
        print(f"  Time Horizon   :  {HORIZON_LABELS[p.time_horizon]}")
        print(f"  Risk Tolerance :  {p.risk_label}")
        print(f"  Goal           :  {p.goal_label}")
        print(f"  Max Drawdown   :  {p.drawdown_ok*100:.0f}%  (your gut threshold)")
        if p.preferred_sectors:
            print(f"  Focus Sectors  :  {', '.join(p.preferred_sectors)}")
        if p.excluded_sectors:
            print(f"  Exclude Sectors:  {', '.join(p.excluded_sectors)}")
        if p.existing_tickers:
            print(f"  Already Own    :  {', '.join(p.existing_tickers)}")
        print("═" * 62)
        while True:
            ans = input("  Proceed with this profile? (y / n): ").strip().lower()
            if ans in ("y", "yes"): return True
            if ans in ("n", "no"):  return False
            print("  Please enter y or n.")
