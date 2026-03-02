# advisor/learner.py — Enhanced learning engine v2
"""
Six stacked learning layers that compound across every session you ever run:

  1. WEIGHT ADAPTATION     — which of the 7 factors actually predicted your returns
                             (adaptive learning rate: grows with data, up to 14%)
  2. SECTOR INTELLIGENCE   — learns different factor importance per sector
                             (e.g. momentum matters more in Tech than Utilities)
  3. REGIME INTELLIGENCE   — learns which factors work in each macro environment
                             (e.g. quality beats momentum in risk-off regimes)
  4. PATTERN MATCHING      — stores the factor fingerprint of every winner and loser
                             and gives a ±12pt bonus/penalty based on similarity
  5. VALUATION CALIBRATION — tracks if STRONG_BUY/BUY signals were actually right
  6. DYNAMIC SECTOR TILTS  — learns which sectors actually beat the market in which
                             macro regimes and updates the score tilts automatically

The model NEVER resets. Every run teaches it something new.
Data is stored in memory/history.json and survives forever.
"""

import os
import json
import uuid
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import io
import contextlib
import warnings
import numpy as np
import yfinance as yf

from config import WEIGHT_MATRIX, FACTOR_NAMES

warnings.filterwarnings("ignore")

MEMORY_DIR  = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "history.json")
SCHEMA_VER  = 2

# ── Learning hyperparameters ───────────────────────────────────────────────────
MIN_LR            = 0.04    # learning rate floor — starts here
MAX_LR            = 0.14    # learning rate ceiling — approached as data grows
MIN_WEIGHT        = 0.02    # no factor weight can drop below 2%
MAX_PATTERN_BONUS = 12.0    # max points added from pattern-match similarity
MAX_PATTERN_PENALTY = 6.0   # max points removed for loser-pattern similarity
WINNER_THRESHOLD  = 0.75    # top 25% of returns = "winner" pattern stored
LOSER_THRESHOLD   = 0.25    # bottom 25% of returns = "loser" pattern stored
MAX_PATTERNS      = 300     # cap on stored patterns (oldest pruned first)

# ── Graveyard archetypes — survivorship bias mitigation ─────────────────────
# Pre-seeded loser fingerprints based on documented financial distress patterns.
# Factor vector: [momentum, volatility, value, quality, technical, sentiment, dividend]
# All values on 0–100 scale (same as live session factor scores).
# Injected on first load when loser_patterns has < 5 real entries.
_GRAVEYARD_ARCHETYPES = [
    {   # Classic zombie: burning cash, no profit, high debt — "looks cheap" value trap
        "f": [20, 15, 45, 10, 20, 20, 0],
        "sector": "Unknown", "regime": "neutral",
        "r": -0.65, "alpha": -0.60, "source": "graveyard",
    },
    {   # Meme/speculative pump: high momentum, no fundamental backing
        "f": [85, 10, 20, 12, 70, 80, 0],
        "sector": "Unknown", "regime": "neutral",
        "r": -0.72, "alpha": -0.68, "source": "graveyard",
    },
    {   # Value trap: cheap for a reason + high yield that gets cut
        "f": [15, 25, 70, 15, 18, 30, 30],
        "sector": "Unknown", "regime": "neutral",
        "r": -0.50, "alpha": -0.45, "source": "graveyard",
    },
    {   # Over-leveraged rising-rate: debt-heavy, rate-sensitive (REIT/utility in wrong macro)
        "f": [25, 20, 40, 20, 22, 25, 40],
        "sector": "Unknown", "regime": "rising_rate",
        "r": -0.58, "alpha": -0.53, "source": "graveyard",
    },
    {   # Deteriorating quality: past winner in decline, fundamentals eroding
        "f": [30, 30, 55, 18, 30, 35, 15],
        "sector": "Unknown", "regime": "neutral",
        "r": -0.55, "alpha": -0.50, "source": "graveyard",
    },
]

# Evaluation horizons (trading days) per time horizon type
# 5-day checkpoint fires quickly so learning starts within one week
# even if you run the tool many times per day.
# Primary learning signal uses the first horizon >= 21 days for stability.
EVAL_HORIZONS = {
    "short":  [5, 21],            # 1 week + 1 month
    "medium": [5, 21, 63],        # 1 week + 1 month + 3 months
    "long":   [5, 21, 63, 126],   # 1 week + 1 month + 3 months + 6 months
}
# Minimum days before any evaluation fires
MIN_EVAL_DAYS = 5


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [−1, 1]."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))


def _adaptive_lr(n_sessions: int) -> float:
    """Learning rate grows with accumulated data (sqrt schedule), bounded."""
    return min(MAX_LR, MIN_LR * math.sqrt(max(1, n_sessions)))


class SessionMemory:
    """
    Persists all recommendation sessions and learns from their real-world outcomes.
    The more sessions accumulate, the more powerful the model becomes.
    """

    def __init__(self):
        self._data: dict = {
            "schema_version":        SCHEMA_VER,
            "sessions":              [],
            "learned_weights":       {},    # "risk_horizon" → [7 weights]
            "sector_factor_corr":    {},    # sector → {factor → [correlation list]}
            "regime_factor_corr":    {},    # regime → {factor → [correlation list]}
            "winner_patterns":       [],    # top-quartile picks' factor vectors + context
            "loser_patterns":        [],    # bottom-quartile picks' factor vectors + context
            "valuation_accuracy":    {},    # signal → {wins, total, avg_alpha list}
            "sector_regime_returns": {},    # "regime|sector" → [alpha list]
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def load(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        if not os.path.exists(MEMORY_FILE):
            self._seed_graveyard_archetypes()
            return
        try:
            with open(MEMORY_FILE, "r") as f:
                self._data = self._migrate(json.load(f))
        except Exception:
            pass  # corrupt file — start fresh
        self._seed_graveyard_archetypes()

    def _seed_graveyard_archetypes(self):
        """
        Inject pre-built loser archetypes when the database is fresh.
        Only seeds when fewer than 5 real (non-graveyard) loser entries exist.
        Marked with source='graveyard' so they survive pruning and are never re-seeded.
        """
        losers = self._data.setdefault("loser_patterns", [])
        real_losers = [p for p in losers if p.get("source") != "graveyard"]
        if len(real_losers) >= 5:
            return   # enough real data — no seeding needed
        if any(p.get("source") == "graveyard" for p in losers):
            return   # already seeded in a prior session
        losers.extend(_GRAVEYARD_ARCHETYPES)
        self._data["loser_patterns"] = losers

    def _migrate(self, data: dict) -> dict:
        """Upgrade schema v1 → v2: add any missing top-level keys."""
        defaults = {
            "schema_version":        SCHEMA_VER,
            "sessions":              [],
            "learned_weights":       {},
            "sector_factor_corr":    {},
            "regime_factor_corr":    {},
            "winner_patterns":       [],
            "loser_patterns":        [],
            "valuation_accuracy":    {},
            "sector_regime_returns": {},
        }
        for key, val in defaults.items():
            if key not in data:
                data[key] = val
        # Migrate old single-evaluation sessions to new evaluations dict
        for s in data.get("sessions", []):
            if "evaluations" not in s:
                s["evaluations"] = {}
                if s.get("evaluation"):          # old schema had singular "evaluation"
                    s["evaluations"]["21d"] = s["evaluation"]
            if "macro_regime" not in s:
                s["macro_regime"] = "neutral"
            # Ensure picks have sector field
            for p in s.get("picks", []):
                if "sector" not in p:
                    p["sector"] = "Unknown"
        data["schema_version"] = SCHEMA_VER
        return data

    def save(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"  Warning: could not save memory — {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save a new session
    # ─────────────────────────────────────────────────────────────────────────

    def save_session(
        self,
        profile,
        top10_df,
        sp500_price: float,
        macro_data:        dict = None,
        valuation_results: dict = None,
        universe_data:     dict = None,
        risk_results:      dict = None,
        protocol_results:  list = None,   # list from ProtocolAnalyzer.analyze_all()
    ):
        """
        Save a completed analysis session.
        Stores full valuation, risk, and protocol data per pick for the history time-machine.
        """
        regime = (macro_data or {}).get("regime", "neutral")

        # Macro snapshot for history time-machine view
        macro_snapshot = {}
        if macro_data:
            macro_snapshot = {
                "regime":        regime,
                "vix":           macro_data.get("vix"),
                "yield_10y":     macro_data.get("yield_10y"),
                "crash_signals": macro_data.get("crash_signals", 0),
            }

        # Convert protocol list → dict keyed by ticker for easy lookup
        proto_map: dict = {}
        if protocol_results:
            for p in protocol_results:
                t = p.get("ticker")
                if t:
                    proto_map[t] = p

        session = {
            "session_id":     str(uuid.uuid4())[:8],
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "macro_regime":   regime,
            "macro_snapshot": macro_snapshot,
            "profile": {
                "risk_level":   profile.risk_level,
                "time_horizon": profile.time_horizon,
                "goal":         profile.goal,
            },
            "sp500_entry": sp500_price,
            "evaluations": {},
            "evaluation":  None,     # backward-compat field for display code
            "evaluated":   False,
            "picks":       [],
        }

        factor_score_cols = [f"{n}_score" for n in FACTOR_NAMES]
        for _, row in top10_df.iterrows():
            ticker = row["ticker"]
            pick = {
                "ticker":           ticker,
                "price_entry":      float(row["current_price"]),
                "composite_score":  float(row["composite_score"]),
                "sector":           str(row.get("sector", "Unknown")),
                "factors":          {},
                "valuation_signal": None,
            }
            for col in factor_score_cols:
                name = col.replace("_score", "")
                pick["factors"][name] = float(row.get(col, 50.0))

            # Valuation signal (backward-compat) + full snapshot for time-machine
            if valuation_results and ticker in valuation_results:
                v = valuation_results[ticker]
                pick["valuation_signal"] = v.get("signal")
                pick["valuation"] = {
                    "fair_value":  v.get("fair_value"),
                    "entry_low":   v.get("entry_low"),
                    "entry_high":  v.get("entry_high"),
                    "target":      v.get("target"),
                    "stop_loss":   v.get("stop_loss"),
                    "signal":      v.get("signal"),
                    "upside_pct":  v.get("upside_pct"),
                    "rr_ratio":    v.get("rr_ratio"),
                    "premium_pct": v.get("premium_pct"),
                    "estimates":   v.get("estimates", {}),
                    "sensitivity": v.get("sensitivity", {}),
                }

            # Full risk snapshot for time-machine
            if risk_results and ticker in risk_results:
                r = risk_results[ticker]
                pick["risk"] = {
                    "altman_z":         r.get("altman_z", {}),
                    "sharpe":           r.get("sharpe"),
                    "sortino":          r.get("sortino"),
                    "max_drawdown_pct": r.get("max_drawdown_pct"),
                    "var_95_pct":       r.get("var_95_pct"),
                    "roic_wacc":        r.get("roic_wacc", {}),
                    "piotroski":        r.get("piotroski", {}),
                }

            # Full protocol snapshot for time-machine
            if ticker in proto_map:
                pr = proto_map[ticker]
                pick["protocol"] = {
                    "gates":         pr.get("gates", []),
                    "gate_statuses": pr.get("gate_statuses", []),
                    "pass_count":    pr.get("pass_count", 0),
                    "warn_count":    pr.get("warn_count", 0),
                    "fail_count":    pr.get("fail_count", 0),
                    "overall_score": pr.get("overall_score"),
                    "conviction":    pr.get("conviction"),
                }

            session["picks"].append(pick)

        self._data["sessions"].append(session)

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-horizon evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_pending(self) -> List[dict]:
        """
        Check all sessions for newly matured evaluation horizons.
        Triggers all 6 learning algorithms for each newly evaluated session.
        Returns list of sessions that gained new evaluation data this run.
        """
        updated = []
        now     = datetime.now(timezone.utc)

        for session in self._data["sessions"]:
            ts         = datetime.fromisoformat(session["timestamp"])
            horizon    = session["profile"].get("time_horizon", "medium")
            days_since = (now - ts).days
            horizons   = EVAL_HORIZONS.get(horizon, [21])
            any_new    = False

            for eval_days in horizons:
                key = f"{eval_days}d"
                if key in session.get("evaluations", {}):
                    continue        # already done
                if days_since < eval_days:
                    continue        # not ready yet

                result = self._evaluate_at(session, now)
                if result:
                    session.setdefault("evaluations", {})[key] = result
                    # Also set backward-compat singular field for the first horizon
                    if not session.get("evaluation"):
                        session["evaluation"] = result
                    any_new = True

            if any_new:
                all_done = all(
                    f"{d}d" in session.get("evaluations", {})
                    for d in horizons
                )
                session["evaluated"] = all_done
                self._learn_from_session(session)
                updated.append(session)

        return updated

    def _evaluate_at(self, session: dict, now: datetime) -> Optional[dict]:
        sp500_then = session.get("sp500_entry", 0)
        sp500_now  = self._fetch_price("^GSPC")
        sp500_ret  = (sp500_now / sp500_then - 1) if sp500_then and sp500_now else None

        pick_results = []
        for pick in session["picks"]:
            cur = self._fetch_price(pick["ticker"])
            if cur and pick["price_entry"]:
                ret   = cur / pick["price_entry"] - 1
                alpha = (ret - sp500_ret) if sp500_ret is not None else None
                pick_results.append({
                    "ticker":           pick["ticker"],
                    "price_exit":       cur,
                    "return":           round(ret, 4),
                    "alpha":            round(alpha, 4) if alpha is not None else None,
                    "sector":           pick.get("sector", "Unknown"),
                    "factors_at_rec":   pick["factors"],
                    "valuation_signal": pick.get("valuation_signal"),
                    "composite_at_rec": pick["composite_score"],
                })

        if not pick_results:
            return None

        avg_ret   = float(np.mean([r["return"] for r in pick_results]))
        alphas    = [r["alpha"] for r in pick_results if r["alpha"] is not None]
        avg_alpha = float(np.mean(alphas)) if alphas else None

        return {
            "evaluation_date": now.isoformat(),
            "sp500_return":    round(sp500_ret, 4) if sp500_ret is not None else None,
            "avg_pick_return": round(avg_ret, 4),
            "alpha":           round(avg_alpha, 4) if avg_alpha is not None else None,
            "picks":           pick_results,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Learning algorithms (all 6 layers)
    # ─────────────────────────────────────────────────────────────────────────

    def _learn_from_session(self, session: dict):
        """Trigger all 6 learning layers after a session gains evaluation data."""
        for eval_data in session.get("evaluations", {}).values():
            picks = eval_data.get("picks", [])
            if len(picks) < 3:
                continue
            regime = session.get("macro_regime", "neutral")
            # Layer 2: Sector factor correlations
            self._update_sector_factor_corr(picks)
            # Layer 3: Regime factor correlations
            self._update_regime_factor_corr(picks, regime)
            # Layer 4: Winner / loser patterns
            self._update_winner_loser_patterns(picks, regime)
            # Layer 5: Valuation signal accuracy
            self._update_valuation_accuracy(picks)
            # Layer 6: Sector returns by regime
            self._update_sector_regime_returns(picks, regime)
        # Layer 1: Recompute all global weights
        self._recompute_all_weights()

    def _update_sector_factor_corr(self, picks: List[dict]):
        """Layer 2 — learn which factors are predictive per sector."""
        by_sector = defaultdict(list)
        for p in picks:
            sec = p.get("sector", "Unknown")
            if sec and sec != "Unknown":
                by_sector[sec].append(p)

        sfc = self._data.setdefault("sector_factor_corr", {})
        for sector, spicks in by_sector.items():
            if len(spicks) < 2:
                continue
            returns      = [p["return"] for p in spicks]
            sector_corrs = sfc.setdefault(sector, {})
            for factor in FACTOR_NAMES:
                scores = [p["factors_at_rec"].get(factor, 50) for p in spicks]
                if len(set(scores)) < 2:
                    continue
                try:
                    corr = float(np.corrcoef(scores, returns)[0, 1])
                    if not math.isnan(corr):
                        lst = sector_corrs.setdefault(factor, [])
                        lst.append(corr)
                        sector_corrs[factor] = lst[-30:]  # keep last 30 observations
                except Exception:
                    pass

    def _update_regime_factor_corr(self, picks: List[dict], regime: str):
        """Layer 3 — learn which factors work in each macro environment."""
        if not regime or regime == "neutral":
            return
        returns = [p["return"] for p in picks]
        rfc = self._data.setdefault("regime_factor_corr", {})
        regime_corrs = rfc.setdefault(regime, {})
        for factor in FACTOR_NAMES:
            scores = [p["factors_at_rec"].get(factor, 50) for p in picks]
            if len(set(scores)) < 2:
                continue
            try:
                corr = float(np.corrcoef(scores, returns)[0, 1])
                if not math.isnan(corr):
                    lst = regime_corrs.setdefault(factor, [])
                    lst.append(corr)
                    regime_corrs[factor] = lst[-40:]
            except Exception:
                pass

    def _update_winner_loser_patterns(self, picks: List[dict], regime: str):
        """Layer 4 — store factor fingerprints of best and worst performers."""
        if len(picks) < 4:
            return
        returns  = [p["return"] for p in picks]
        high_cut = float(np.percentile(returns, 100 * WINNER_THRESHOLD))
        low_cut  = float(np.percentile(returns, 100 * LOSER_THRESHOLD))

        winners = self._data.setdefault("winner_patterns", [])
        losers  = self._data.setdefault("loser_patterns",  [])

        for p in picks:
            fvec = [p["factors_at_rec"].get(f, 50.0) for f in FACTOR_NAMES]
            pat  = {
                "f":      fvec,
                "sector": p.get("sector", "Unknown"),
                "regime": regime,
                "r":      p["return"],
                "alpha":  p.get("alpha") or 0.0,
            }
            if p["return"] >= high_cut:
                winners.append(pat)
            elif p["return"] <= low_cut:
                losers.append(pat)
                # Triple-weight catastrophic losses (<-40%): rare but highly informative.
                # Simulates the extra historical data we'd have if we'd seen more
                # bankruptcies — gives the pattern matcher stronger avoidance signal.
                if p["return"] < -0.40:
                    losers.append(pat)
                    losers.append(pat)

        # Keep best winners (highest alpha) and worst losers (lowest alpha)
        self._data["winner_patterns"] = sorted(
            winners, key=lambda x: x.get("alpha", 0), reverse=True
        )[:MAX_PATTERNS]
        self._data["loser_patterns"] = sorted(
            losers, key=lambda x: x.get("alpha", 0)
        )[:MAX_PATTERNS]

    def _update_valuation_accuracy(self, picks: List[dict]):
        """Layer 5 — track whether valuation signals were accurate."""
        va = self._data.setdefault("valuation_accuracy", {})
        for p in picks:
            sig = p.get("valuation_signal")
            if not sig:
                continue
            alpha = p.get("alpha")
            if alpha is None:
                continue
            entry = va.setdefault(sig, {"wins": 0, "total": 0, "avg_alpha": []})
            entry["total"] += 1
            if alpha > 0:
                entry["wins"] += 1
            entry.setdefault("avg_alpha", []).append(alpha)
            entry["avg_alpha"] = entry["avg_alpha"][-50:]

    def _update_sector_regime_returns(self, picks: List[dict], regime: str):
        """Layer 6 — track which sectors actually outperform in each macro regime."""
        if not regime or regime == "neutral":
            return
        srr = self._data.setdefault("sector_regime_returns", {})
        by_sector = defaultdict(list)
        for p in picks:
            sec = p.get("sector", "Unknown")
            if sec != "Unknown":
                by_sector[sec].append(p.get("alpha") or 0.0)
        for sector, alphas in by_sector.items():
            key = f"{regime}|{sector}"
            lst = srr.setdefault(key, [])
            lst.append(float(np.mean(alphas)))
            srr[key] = lst[-30:]

    def _recompute_all_weights(self):
        """Layer 1 — recompute adapted weights for every profile key seen so far."""
        seen = set()
        for s in self._data["sessions"]:
            p = s["profile"]
            seen.add((p["risk_level"], p["time_horizon"]))
        for (rl, th) in seen:
            w = self._compute_weights(rl, th)
            if w:
                self._data["learned_weights"][f"{rl}_{th}"] = w

    def _compute_weights(self, risk_level: int, time_horizon: str) -> Optional[List[float]]:
        """
        Compute adapted weights for a profile using ALL evaluated sessions.
        Weights correlations by return magnitude (big wins teach more than small wins).
        Adaptive learning rate grows with session count.
        """
        relevant = [
            s for s in self._data["sessions"]
            if s.get("evaluations")
            and s["profile"]["risk_level"] == risk_level
            and s["profile"]["time_horizon"] == time_horizon
        ]
        if len(relevant) < 2:
            return self._data["learned_weights"].get(f"{risk_level}_{time_horizon}")

        # Use 21-day horizon as primary learning signal (5-day is too noisy for weights)
        all_horizons = EVAL_HORIZONS.get(time_horizon, [21])
        primary_days = next((h for h in all_horizons if h >= 21), all_horizons[-1])
        horizon_key  = f"{primary_days}d"

        # Collect (correlation, avg_abs_return) per factor
        factor_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for session in relevant:
            eval_data = session["evaluations"].get(horizon_key)
            if not eval_data:
                continue
            picks = eval_data.get("picks", [])
            if len(picks) < 3:
                continue
            returns     = [p["return"] for p in picks]
            avg_mag     = float(np.mean([abs(r) for r in returns]))
            for factor in FACTOR_NAMES:
                scores = [p["factors_at_rec"].get(factor, 50) for p in picks]
                if len(set(scores)) < 2:
                    continue
                try:
                    corr = float(np.corrcoef(scores, returns)[0, 1])
                    if not math.isnan(corr):
                        factor_data[factor].append((corr, avg_mag))
                except Exception:
                    pass

        lr     = _adaptive_lr(len(relevant))
        base_w = list(WEIGHT_MATRIX.get((risk_level, time_horizon), [1 / 7] * 7))

        for i, factor in enumerate(FACTOR_NAMES):
            pairs = factor_data.get(factor, [])
            if not pairs:
                continue
            # Magnitude-weighted average correlation
            total_mag = sum(m for _, m in pairs) or 1.0
            avg_corr  = sum(c * m for c, m in pairs) / total_mag

            if avg_corr > 0.20:
                # Scale nudge proportionally to how predictive the factor is
                boost = lr * min(avg_corr / 0.20, 2.5)
                base_w[i] *= (1.0 + boost)
            elif avg_corr < -0.05:
                drop = lr * min(abs(avg_corr) / 0.15, 1.8)
                base_w[i] *= (1.0 - drop)

        base_w = [max(w, MIN_WEIGHT) for w in base_w]
        total  = sum(base_w)
        return [w / total for w in base_w]

    # ─────────────────────────────────────────────────────────────────────────
    # Query methods — called by the scoring pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def get_adapted_weights(self, risk_level: int, time_horizon: str) -> Optional[List[float]]:
        """Return learned factor weights for this risk/horizon. None if not enough data."""
        return self._data["learned_weights"].get(f"{risk_level}_{time_horizon}")

    def get_pattern_bonus(
        self, factor_scores: dict, sector: str, regime: str
    ) -> float:
        """
        Layer 4 query — cosine similarity to historical winners/losers.
        Returns a score in [−MAX_PENALTY, +MAX_BONUS] to add to composite_score.
        0.0 if no patterns exist yet.
        """
        winners = self._data.get("winner_patterns", [])
        losers  = self._data.get("loser_patterns",  [])
        if not winners:
            return 0.0

        candidate = [factor_scores.get(f, 50.0) for f in FACTOR_NAMES]

        def _weighted_sim(patterns: list, sec_boost: float, reg_boost: float) -> float:
            total_w, total_s = 0.0, 0.0
            for pat in patterns:
                sim    = _cosine_sim(candidate, pat["f"])
                weight = abs(pat.get("alpha", pat.get("r", 0.05))) + 0.01
                if pat.get("sector") == sector:
                    weight *= sec_boost
                if pat.get("regime") == regime:
                    weight *= reg_boost
                total_s += sim * weight
                total_w += weight
            return total_s / total_w if total_w > 0 else 0.0

        win_sim  = _weighted_sim(winners, 1.6, 1.3)
        lose_sim = _weighted_sim(losers,  1.6, 1.3)

        # Net signal: winners reward more than losers penalise
        net = win_sim - lose_sim * 0.55
        # Scale to ±bonus range
        bonus = net * (MAX_PATTERN_BONUS + MAX_PATTERN_PENALTY) * 1.1 - MAX_PATTERN_PENALTY * 0.1
        return float(max(-MAX_PATTERN_PENALTY, min(MAX_PATTERN_BONUS, bonus)))

    def get_all_pattern_bonuses(
        self,
        universe_data: dict,
        ranked_df=None,
        regime: str = "neutral",
    ) -> Dict[str, float]:
        """
        Compute pattern bonuses for every scored stock.
        Uses factor scores from ranked_df if provided (more accurate).
        Returns {ticker: bonus_pts}.
        """
        if not self._data.get("winner_patterns"):
            return {}

        bonuses: Dict[str, float] = {}
        for ticker, data in universe_data.items():
            sector = data.get("sector", "Unknown")

            # Pull factor scores from the ranked dataframe
            factor_scores = {f: 50.0 for f in FACTOR_NAMES}
            if ranked_df is not None:
                row = ranked_df[ranked_df["ticker"] == ticker]
                if not row.empty:
                    for f in FACTOR_NAMES:
                        col = f"{f}_score"
                        if col in row.columns:
                            factor_scores[f] = float(row[col].iloc[0])

            bonus = self.get_pattern_bonus(factor_scores, sector, regime)
            if abs(bonus) > 0.05:
                bonuses[ticker] = bonus

        return bonuses

    def get_dynamic_sector_tilts(self, regime: str) -> Dict[str, float]:
        """
        Layer 6 query — learned sector tilt adjustments for the current regime.
        Blends observed sector alpha data into ±8pt adjustments.
        """
        srr   = self._data.get("sector_regime_returns", {})
        tilts: Dict[str, float] = {}
        for key, returns in srr.items():
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue
            stored_regime, sector = parts
            if stored_regime != regime or not returns:
                continue
            avg_alpha = float(np.mean(returns))
            # +1pt per +2% average alpha, capped ±8
            tilt = float(np.clip(avg_alpha * 50, -8, 8))
            tilts[sector] = round(tilt, 1)
        return tilts

    def get_sector_weight_adjustments(self, sector: str) -> Dict[str, float]:
        """
        Layer 2 query — factor importance multipliers learned for a sector.
        e.g. {"momentum": 1.35, "value": 0.75} for Technology.
        Used by the scorer to compute sector-adjusted composite scores.
        """
        sfc  = self._data.get("sector_factor_corr", {})
        data = sfc.get(sector, {})
        if not data:
            return {}
        adj: Dict[str, float] = {}
        for factor, corrs in data.items():
            if not corrs or len(corrs) < 2:
                continue
            avg = float(np.mean(corrs))
            if avg > 0.15:
                adj[factor] = min(1.0 + avg * 0.9, 1.5)   # up to 1.5×
            elif avg < -0.10:
                adj[factor] = max(1.0 + avg * 0.7, 0.55)  # down to 0.55×
        return adj

    def get_regime_weight_adjustments(self, regime: str) -> Dict[str, float]:
        """
        Layer 3 query — factor importance multipliers for the current macro regime.
        """
        rfc  = self._data.get("regime_factor_corr", {})
        data = rfc.get(regime, {})
        if not data:
            return {}
        adj: Dict[str, float] = {}
        for factor, corrs in data.items():
            if not corrs or len(corrs) < 2:
                continue
            avg = float(np.mean(corrs))
            if avg > 0.20:
                adj[factor] = min(1.0 + avg * 0.8, 1.4)
            elif avg < -0.10:
                adj[factor] = max(1.0 + avg * 0.6, 0.60)
        return adj

    def get_valuation_signal_stats(self) -> Dict[str, dict]:
        """Layer 5 query — accuracy stats per valuation signal."""
        out = {}
        for sig, data in self._data.get("valuation_accuracy", {}).items():
            total = data.get("total", 0)
            if total < 2:
                continue
            wins     = data.get("wins", 0)
            alphas   = data.get("avg_alpha", [])
            out[sig] = {
                "win_rate":  round(wins / total, 3),
                "total":     total,
                "avg_alpha": round(float(np.mean(alphas)), 4) if alphas else None,
            }
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Existing public interface (unchanged for backward compatibility)
    # ─────────────────────────────────────────────────────────────────────────

    def get_recent_tickers(self, n_sessions: int = 2) -> List[str]:
        sessions = self._data.get("sessions", [])
        if not sessions:
            return []
        recent  = sorted(sessions, key=lambda s: s.get("timestamp", ""), reverse=True)
        tickers = []
        for s in recent[:n_sessions]:
            for p in s.get("picks", []):
                t = p.get("ticker")
                if t and t not in tickers:
                    tickers.append(t)
        return tickers

    def get_track_record(self) -> List[dict]:
        evald = [
            s for s in self._data["sessions"]
            if s.get("evaluations") or s.get("evaluation")
        ]
        evald.sort(key=lambda s: s["timestamp"], reverse=True)
        return evald

    def total_sessions(self) -> int:
        return len(self._data["sessions"])

    def _fetch_price(self, ticker: str) -> Optional[float]:
        try:
            sink = io.StringIO()
            with contextlib.redirect_stderr(sink):
                hist = yf.Ticker(ticker).history(period="2d")
            if hist is not None and len(hist) > 0:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None
