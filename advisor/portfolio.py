# advisor/portfolio.py — Correlation-aware greedy stock selection + Kelly position sizing

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from advisor.fetcher import DataFetcher


class PortfolioConstructor:
    """
    Selects the best 10 stocks from a scored DataFrame, maximising both
    composite score and diversification (low pairwise correlation).

    Position sizing uses a half-Kelly criterion:
        f_i = (score_i/100) / vol_i^2  →  halved  →  capped at 20%  →  renormalised
    Falls back to score-weighted if Kelly produces degenerate sizes.
    """

    def __init__(self, n: int = 10, candidate_pool: int = 30):
        self.n              = n
        self.candidate_pool = candidate_pool

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

        # ── Portfolio beta cap ─────────────────────────────────────────────────
        # After greedy selection, check the portfolio's average beta. If it
        # exceeds the risk-level target, swap out the highest-beta holding for
        # the best-scoring lower-beta alternative from the extended pool (top-50).
        # This prevents the portfolio from being a leveraged S&P 500 proxy that
        # amplifies market moves in both directions.
        _beta_targets = {1: 0.90, 2: 1.05, 3: 1.30, 4: 1.60}
        beta_target   = _beta_targets.get(risk_level, 1.10)

        if "beta" in ranked_df.columns and len(selected) > 0:
            sel_df    = ranked_df[ranked_df["ticker"].isin(selected)][
                ["ticker", "beta", "composite_score"]
            ]
            port_beta = float(sel_df["beta"].mean())

            if port_beta > beta_target:
                # Highest-beta stock in the portfolio is the prime swap candidate
                worst_row    = sel_df.sort_values("beta", ascending=False).iloc[0]
                worst_ticker = worst_row["ticker"]
                worst_beta   = float(worst_row["beta"])

                # Search extended pool (top-50) for unselected lower-beta alternatives
                extended = ranked_df.head(min(50, len(ranked_df)))
                not_sel  = extended[~extended["ticker"].isin(selected)]
                low_beta = not_sel[not_sel["beta"] < worst_beta - 0.15]

                if not low_beta.empty:
                    replacement = (
                        low_beta.sort_values("composite_score", ascending=False)
                        .iloc[0]["ticker"]
                    )
                    selected = [replacement if t == worst_ticker else t for t in selected]

        final = ranked_df[ranked_df["ticker"].isin(selected)].copy()
        final = final.sort_values("composite_score", ascending=False).reset_index(drop=True)
        final["rank"] = range(1, len(final) + 1)
        return final

    def size_positions(self, top10: pd.DataFrame, portfolio_size: float) -> pd.DataFrame:
        """Add weight%, dollar amount, approx shares columns using half-Kelly sizing."""
        df = top10.copy()

        # Half-Kelly: edge / variance / 2
        kelly_raw = []
        for _, row in df.iterrows():
            edge = float(row["composite_score"]) / 100
            var  = float(row["vol"]) ** 2
            var  = max(var, 0.01)          # floor to avoid division issues
            kelly_raw.append(edge / var / 2)

        total_k = sum(kelly_raw)
        if total_k <= 0 or any(math.isnan(k) for k in kelly_raw):
            # Fallback: score-weighted
            total_s = df["composite_score"].sum()
            df["weight"] = df["composite_score"] / total_s
        else:
            weights = [k / total_k for k in kelly_raw]
            # Cap each at 20%
            weights = [min(w, 0.20) for w in weights]
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
        return ret_df.corr()
