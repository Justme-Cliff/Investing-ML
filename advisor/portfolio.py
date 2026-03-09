# advisor/portfolio.py — Correlation-aware greedy stock selection + Kelly position sizing

import math
import warnings as _warnings
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from advisor.fetcher import DataFetcher
from config import PIPELINE_CONFIG


class PortfolioConstructor:
    """
    Selects the best N stocks from a scored DataFrame, maximising both
    composite score and diversification (low pairwise correlation).

    Constraints applied after greedy selection:
      - Portfolio beta cap: progressive multi-swap (3 passes)
      - Sector concentration: max 3 stocks per sector
      - Position sizing: half-Kelly, capped at 15% per position

    Position sizing uses a half-Kelly criterion:
        f_i = (score_i/100) / vol_i^2  →  halved  →  capped at 15%  →  renormalised
    Falls back to score-weighted if Kelly produces degenerate sizes.
    """

    def __init__(self, n: int = 10, candidate_pool: int = None):
        self.n              = n
        self.candidate_pool = candidate_pool or PIPELINE_CONFIG["candidate_pool"]
        self.kelly_fallback = False

    def select(self, ranked_df: pd.DataFrame, universe_data: Dict,
               risk_level: int = 2) -> pd.DataFrame:
        """Return the top-n stocks chosen with correlation-aware greedy algorithm."""
        if len(ranked_df) <= self.n:
            return ranked_df.copy()

        candidates = ranked_df.head(self.candidate_pool)
        corr       = self._correlation_matrix(candidates["ticker"].tolist(), universe_data)

        selected: List[str] = [candidates.iloc[0]["ticker"]]   # start with #1 ranked

        for _ in range(self.n - 1):
            best_adjusted = -np.inf
            best_ticker   = None

            for _, row in candidates.iterrows():
                t = row["ticker"]
                if t in selected:
                    continue

                # Average absolute correlation with already-selected tickers
                if corr is not None and t in corr.index and len(selected) > 0:
                    corr_vals  = [abs(corr.loc[t, s]) for s in selected if s in corr.columns]
                    avg_corr   = float(np.mean(corr_vals)) if corr_vals else 0.5
                else:
                    avg_corr = 0.5   # neutral if no correlation data

                # Adjusted score: 70% quality, 30% diversification
                adjusted = float(row["composite_score"]) * (0.70 + 0.30 * (1 - avg_corr))

                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_ticker   = t

            if best_ticker:
                selected.append(best_ticker)

        # ── Portfolio beta cap — progressive multi-swap ────────────────────────
        # After greedy selection, check the portfolio's average beta. Loop up to
        # 3 passes: each pass swaps out the highest-beta holding for the best
        # lower-beta alternative from the extended pool (top-50).
        #
        # Threshold relaxes each pass (0.15 → 0.05 → any lower beta) so the
        # algorithm doesn't give up prematurely when good candidates exist but
        # fall just outside the strict cutoff. Stops early when portfolio beta
        # is within target or no better alternative is available.
        _beta_targets = PIPELINE_CONFIG["beta_targets"]
        beta_target   = _beta_targets.get(risk_level, 1.10)
        extended      = ranked_df.head(min(50, len(ranked_df)))

        if "beta" in ranked_df.columns and len(selected) > 0:
            for _pass, _threshold in enumerate([0.15, 0.05, 0.0]):
                sel_df    = ranked_df[ranked_df["ticker"].isin(selected)][
                    ["ticker", "beta", "composite_score"]
                ].copy()
                # Weighted portfolio beta (score-weighted, not simple mean)
                _total_sc = sel_df["composite_score"].sum()
                if _total_sc > 0:
                    sel_df["_wt"] = sel_df["composite_score"] / _total_sc
                    port_beta = float((sel_df["beta"] * sel_df["_wt"]).sum())
                else:
                    port_beta = float(sel_df["beta"].mean())

                if port_beta <= beta_target:
                    break   # within target — no more swaps needed

                # Highest-beta holding is the prime swap candidate
                worst_row    = sel_df.sort_values("beta", ascending=False).iloc[0]
                worst_ticker = worst_row["ticker"]
                worst_beta   = float(worst_row["beta"])

                not_sel  = extended[~extended["ticker"].isin(selected)]
                low_beta = not_sel[not_sel["beta"] < worst_beta - _threshold]

                if low_beta.empty:
                    continue   # relax threshold on next pass

                replacement = (
                    low_beta.sort_values("composite_score", ascending=False)
                    .iloc[0]["ticker"]
                )
                selected = [replacement if t == worst_ticker else t for t in selected]

        # ── Sector concentration constraint ───────────────────────────────────
        # Cap at 3 stocks per sector to prevent the portfolio from becoming a
        # single-sector bet. If a sector is overweight, the excess picks are
        # replaced with the best remaining stocks from other sectors.
        # This reduces idiosyncratic sector risk without sacrificing overall quality.
        if "sector" in ranked_df.columns:
            sector_counts: Counter = Counter()
            final_selected: List[str] = []
            for t in selected:
                row = ranked_df[ranked_df["ticker"] == t]
                if row.empty:
                    final_selected.append(t)
                    continue
                sec = row.iloc[0]["sector"]
                if sector_counts[sec] < 3:
                    final_selected.append(t)
                    sector_counts[sec] += 1

            # If any were dropped, fill with best unconstrained alternatives
            n_needed = self.n - len(final_selected)
            if n_needed > 0:
                selected_set = set(final_selected)
                for _, row in ranked_df.iterrows():
                    if n_needed <= 0:
                        break
                    t   = row["ticker"]
                    sec = row.get("sector", "Unknown")
                    if t not in selected_set and sector_counts[sec] < 3:
                        final_selected.append(t)
                        selected_set.add(t)
                        sector_counts[sec] += 1
                        n_needed -= 1

            selected = final_selected

        final = ranked_df[ranked_df["ticker"].isin(selected)].copy()
        final = final.sort_values("composite_score", ascending=False).reset_index(drop=True)
        final["rank"] = range(1, len(final) + 1)
        return final

    def size_positions(self, top10: pd.DataFrame, portfolio_size: float,
                       macro_data: dict = None, universe_data: dict = None) -> pd.DataFrame:
        """Add weight%, dollar amount, approx shares columns using half-Kelly sizing.

        VIX-scaled Kelly: when VIX > 20, raw Kelly fractions are scaled down by
        (20/VIX). This corrects for the fact that Kelly assumes stable variance —
        in high-VIX regimes realised vol can double, making standard Kelly sizes 2×
        too aggressive versus the actual risk being taken on.

        Market drawdown scalar: if SPY is >10% off its 52W high, additional
        exposure reduction of up to 30% is applied (at -40% drawdown).

        Both scalars affect the *pre-normalisation* Kelly fractions, so final
        weights still sum to 100% (capital stays fully invested). The effect is
        that lower-vol stocks receive relatively more weight in high-fear periods.
        """
        df = top10.copy()

        # ── VIX-adjusted variance for Kelly denominator ───────────────────────
        # Standard Kelly uses historical realised vol. In high-fear periods VIX
        # spikes, signalling that FUTURE vol is likely to be higher than historical.
        # We use max(hist_vol, vix_implied_vol) as the denominator so that Kelly
        # sizes shrink when forward vol is elevated — naturally shifting weight
        # toward lower-beta stocks. VIX of 20 → 20% annual implied vol.
        vix_implied: float = 0.0
        dd_penalty:  float = 0.0
        if macro_data is not None:
            try:
                vix = float(macro_data.get("vix") or 0.0)
                if vix > 0:
                    vix_implied = vix / 100.0    # e.g. VIX=30 → 0.30 annual vol
            except Exception:
                pass
            try:
                dd_pct = float(macro_data.get("market_drawdown_pct") or 0.0)
                if dd_pct < -10.0:
                    # Add up to 0.15 to the VIX-implied vol floor in bear markets
                    dd_penalty = min(0.15, abs(dd_pct + 10.0) / 100.0)
            except Exception:
                pass

        vol_floor = max(0.0, vix_implied + dd_penalty)   # 0 when no macro_data

        # ── CVaR-adjusted Kelly sizing ────────────────────────────────────────
        # Standard Kelly uses vol^2. CVaR-adjusted Kelly uses max(hist_vol, cvar_vol)
        # as the effective volatility — more conservative when tail risk is elevated.
        # CVaR captures crash severity; vol does not. In fat-tail conditions
        # (CVaR >> VaR), the effective denominator grows, naturally sizing down
        # the riskiest positions even if their average returns look attractive.
        #
        # Convert monthly CVaR → annualised vol equivalent:
        #   cvar_annual_vol = |cvar_monthly| × sqrt(12)
        # Then use max(hist_vol, vix_floor, cvar_annual_vol) as floor.
        cvar_map: dict = {}
        if universe_data:
            for _t in df["ticker"]:
                _ud = universe_data.get(_t, {})
                _hist = _ud.get("history")
                if _hist is not None and not _hist.empty:
                    _close = _hist["Close"].dropna()
                    if len(_close) >= 63:
                        _monthly = _close.pct_change(21).dropna()
                        if len(_monthly) >= 10:
                            _cut  = float(_monthly.quantile(0.05))
                            _tail = _monthly[_monthly <= _cut]
                            if len(_tail) > 0:
                                cvar_map[_t] = abs(float(_tail.mean()))

        kelly_raw = []
        for _, row in df.iterrows():
            _t       = row["ticker"]
            edge     = float(row["composite_score"]) / 100
            hist_vol = float(row["vol"])
            eff_vol  = max(hist_vol, vol_floor)
            if _t in cvar_map:
                # Annualise monthly CVaR and use as volatility floor
                cvar_vol = cvar_map[_t] * (12 ** 0.5)
                eff_vol  = max(eff_vol, cvar_vol)
            var = max(eff_vol ** 2, 0.01)
            kelly_raw.append(edge / var / 2)

        total_k = sum(kelly_raw)
        if total_k <= 0 or any(math.isnan(k) for k in kelly_raw):
            # Fallback: score-weighted
            _warnings.warn(
                "Kelly sizing failed (degenerate fractions) — using score-weighted allocation.",
                RuntimeWarning, stacklevel=2,
            )
            self.kelly_fallback = True
            total_s = df["composite_score"].sum()
            df["weight"] = df["composite_score"] / total_s
        else:
            self.kelly_fallback = False
            weights = [k / total_k for k in kelly_raw]
            _pos_cap = PIPELINE_CONFIG["position_cap"]
            weights = [min(w, _pos_cap) for w in weights]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            df["weight"] = weights

        df["dollar_amount"]   = df["weight"] * portfolio_size
        df["approx_shares"]   = (df["dollar_amount"] / df["current_price"]).apply(
            lambda x: int(x) if x >= 1 else round(x, 2)
        )
        return df

    # ── Internals ─────────────────────────────────────────────────────────────
    def _correlation_matrix(self, tickers: List[str], universe_data: Dict) -> Optional[pd.DataFrame]:
        returns: Dict[str, pd.Series] = {}
        for t in tickers:
            if t not in universe_data:
                continue
            close = universe_data[t]["history"]["Close"].dropna()
            close = DataFetcher.strip_tz(close)
            ret   = close.pct_change().dropna()
            if len(ret) > 20:
                returns[t] = ret

        if len(returns) < 2:
            return None

        ret_df = pd.DataFrame(returns)
        ret_df = ret_df.dropna()
        return ret_df.corr(method="spearman")
