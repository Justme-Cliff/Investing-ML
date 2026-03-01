# advisor/learner.py — Session memory, performance tracking, adaptive weight learning

import os
import json
import uuid
import math
from datetime import datetime, timezone
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
SCHEMA_VER  = 1
MIN_DAYS_TO_EVAL = 30
LEARNING_RATE    = 0.04   # nudge per evaluation cycle
MIN_WEIGHT       = 0.03


class SessionMemory:
    """
    Persists recommendation sessions and tracks their performance over time.
    Adapts factor weights based on what actually predicted returns well.
    """

    def __init__(self):
        self._data: dict = {"schema_version": SCHEMA_VER, "learned_weights": {}, "sessions": []}

    # ── Load / Save ───────────────────────────────────────────────────────────
    def load(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        if not os.path.exists(MEMORY_FILE):
            return
        try:
            with open(MEMORY_FILE, "r") as f:
                self._data = json.load(f)
        except Exception:
            pass   # corrupt file — start fresh

    def save(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"  Warning: could not save memory — {e}")

    # ── Save a new recommendation session ────────────────────────────────────
    def save_session(self, profile, top10_df, sp500_price: float):
        session = {
            "session_id":  str(uuid.uuid4())[:8],
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "profile": {
                "risk_level":    profile.risk_level,
                "time_horizon":  profile.time_horizon,
                "goal":          profile.goal,
            },
            "sp500_entry": sp500_price,
            "evaluated":   False,
            "evaluation":  None,
            "picks": [],
        }

        factor_score_cols = [f"{n}_score" for n in FACTOR_NAMES]
        for _, row in top10_df.iterrows():
            pick = {
                "ticker":          row["ticker"],
                "price_entry":     float(row["current_price"]),
                "composite_score": float(row["composite_score"]),
                "factors":         {},
            }
            for col in factor_score_cols:
                name = col.replace("_score", "")
                pick["factors"][name] = float(row.get(col, 50.0))
            session["picks"].append(pick)

        self._data["sessions"].append(session)

    # ── Evaluate old sessions ─────────────────────────────────────────────────
    def evaluate_pending(self) -> List[dict]:
        """
        Fetch current prices for sessions older than MIN_DAYS_TO_EVAL.
        Returns list of newly-evaluated session dicts.
        """
        evaluated = []
        now = datetime.now(timezone.utc)

        for session in self._data["sessions"]:
            if session.get("evaluated"):
                continue
            ts = datetime.fromisoformat(session["timestamp"])
            if (now - ts).days < MIN_DAYS_TO_EVAL:
                continue

            # Fetch current prices
            tickers    = [p["ticker"] for p in session["picks"]]
            sp500_then = session.get("sp500_entry", 0)
            sp500_now  = self._fetch_price("^GSPC")
            sp500_ret  = (sp500_now / sp500_then - 1) if sp500_then and sp500_now else None

            pick_results = []
            for pick in session["picks"]:
                cur = self._fetch_price(pick["ticker"])
                if cur and pick["price_entry"]:
                    ret = cur / pick["price_entry"] - 1
                    pick_results.append({
                        "ticker":       pick["ticker"],
                        "price_exit":   cur,
                        "return":       round(ret, 4),
                        "composite_at_rec": pick["composite_score"],
                        "factors_at_rec":   pick["factors"],
                    })

            if pick_results:
                avg_ret = float(np.mean([r["return"] for r in pick_results]))
                alpha   = (avg_ret - sp500_ret) if sp500_ret is not None else None
                session["evaluation"] = {
                    "evaluation_date": now.isoformat(),
                    "sp500_return":    round(sp500_ret, 4) if sp500_ret is not None else None,
                    "avg_pick_return": round(avg_ret, 4),
                    "alpha":           round(alpha, 4) if alpha is not None else None,
                    "picks":           pick_results,
                }
                session["evaluated"] = True
                evaluated.append(session)

        return evaluated

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

    # ── Recent picks (for avoid-repeat logic) ────────────────────────────────
    def get_recent_tickers(self, n_sessions: int = 2) -> List[str]:
        """
        Return tickers recommended in the last n_sessions sessions.
        Used to penalise repeat picks so each run feels fresh.
        """
        sessions = self._data.get("sessions", [])
        if not sessions:
            return []
        recent = sorted(sessions, key=lambda s: s.get("timestamp", ""), reverse=True)
        tickers = []
        for s in recent[:n_sessions]:
            for p in s.get("picks", []):
                t = p.get("ticker")
                if t and t not in tickers:
                    tickers.append(t)
        return tickers

    # ── Track record summary ──────────────────────────────────────────────────
    def get_track_record(self) -> List[dict]:
        """Return evaluated sessions sorted by date desc."""
        evald = [s for s in self._data["sessions"] if s.get("evaluated")]
        evald.sort(key=lambda s: s["timestamp"], reverse=True)
        return evald

    def total_sessions(self) -> int:
        return len(self._data["sessions"])

    # ── Adapted weight learning ───────────────────────────────────────────────
    def get_adapted_weights(self, risk_level: int, time_horizon: str) -> Optional[List[float]]:
        """
        Returns factor weights nudged by historical predictive power.
        Returns None if not enough history to adapt.
        """
        key = f"{risk_level}_{time_horizon}"

        # Gather all evaluated sessions for this profile key
        relevant = [
            s for s in self._data["sessions"]
            if s.get("evaluated")
            and s["profile"]["risk_level"] == risk_level
            and s["profile"]["time_horizon"] == time_horizon
        ]

        if len(relevant) < 2:
            # Check if we have stored learned weights
            stored = self._data["learned_weights"].get(key)
            return stored   # may be None — caller uses default

        # Compute per-factor predictive power (Pearson correlation with actual return)
        factor_pred: Dict[str, List[float]] = {f: [] for f in FACTOR_NAMES}

        for session in relevant:
            picks = session["evaluation"]["picks"]
            if len(picks) < 3:
                continue
            returns = [p["return"] for p in picks]
            for factor in FACTOR_NAMES:
                scores = [p["factors_at_rec"].get(factor, 50) for p in picks]
                if len(set(scores)) < 2:
                    continue
                try:
                    corr = float(np.corrcoef(scores, returns)[0, 1])
                    if not math.isnan(corr):
                        factor_pred[factor].append(corr)
                except Exception:
                    pass

        # Nudge base weights
        base_w = list(WEIGHT_MATRIX[(risk_level, time_horizon)])
        for i, factor in enumerate(FACTOR_NAMES):
            preds = factor_pred[factor]
            if not preds:
                continue
            avg_pred = float(np.mean(preds))
            if avg_pred > 0.3:
                base_w[i] *= (1 + LEARNING_RATE)
            elif avg_pred < 0.0:
                base_w[i] *= (1 - LEARNING_RATE)

        # Clamp minimums and renormalise
        base_w = [max(w, MIN_WEIGHT) for w in base_w]
        total  = sum(base_w)
        base_w = [w / total for w in base_w]

        # Store for next time
        self._data["learned_weights"][key] = base_w
        return base_w
