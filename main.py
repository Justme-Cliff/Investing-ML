"""
main.py — Entry point for the Stock Ranking Advisor v3

Pure quantitative analysis — no paid AI APIs required.
Uses hedge-fund grade math: DCF, Graham, EV/EBITDA, FCF yield,
Altman Z, Sharpe/Sortino, ROIC/WACC, Piotroski 9pt, and more.

Run:  python main.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from config import (
    STOCK_UNIVERSE, DYNAMIC_UNIVERSE,
    UNIVERSE_MIN_MARKET_CAP, UNIVERSE_MAX_TICKERS,
    WEIGHT_MATRIX, PORTFOLIO_N,
    VERSION, FRESH_PICKS_PENALTY, check_api_keys,
)

from advisor.collector    import InputCollector
from advisor.fetcher      import DataFetcher, MacroFetcher
from advisor.scorer       import MultiFactorScorer
from advisor.portfolio    import PortfolioConstructor
from advisor.learner      import SessionMemory
from advisor.protocol     import ProtocolAnalyzer
from advisor.valuation    import ValuationEngine
from advisor.risk         import RiskEngine
from advisor.charts       import ChartEngine
from advisor.display      import TerminalDisplay
from advisor.exporter     import ExcelExporter
from advisor.cli_commands import CommandHandler


def main():
    display = TerminalDisplay()
    memory  = SessionMemory()

    # ── Version banner ────────────────────────────────────────────────────────
    print(f"\n  Stock Ranking Advisor  v{VERSION}\n")

    # ── API key health check ──────────────────────────────────────────────────
    print("  Checking API key status...")
    _key_status = check_api_keys()
    _icons = {True: "✓", False: "✗", None: "—"}
    for _svc, _ok in _key_status.items():
        _label = "active" if _ok else ("invalid" if _ok is False else "not configured")
        print(f"    {_icons[_ok]}  {_svc:<14} {_label}")
    print()

    # ── 1. Load memory + show track record ───────────────────────────────────
    memory.load()

    print("\nChecking past session performance...")
    newly_evaluated = memory.evaluate_pending()
    if newly_evaluated:
        print(f"  Evaluated {len(newly_evaluated)} new session(s).")
    memory.save()

    track_record = memory.get_track_record()
    display.show_welcome(track_record)

    # ── 2. Collect user profile ───────────────────────────────────────────────
    profile = InputCollector().collect()

    # Adapted weights from past performance (None if not enough history)
    adapted_weights = memory.get_adapted_weights(profile.risk_level, profile.time_horizon)
    display.show_weight_adaptation(adapted_weights)

    # ── 3. Fetch stock data ───────────────────────────────────────────────────
    if DYNAMIC_UNIVERSE:
        from advisor.universe import fetch_us_universe
        all_tickers = fetch_us_universe(
            min_market_cap=UNIVERSE_MIN_MARKET_CAP,
            max_tickers=UNIVERSE_MAX_TICKERS,
        )
        # Remove tickers the user already holds
        all_tickers = [t for t in all_tickers if t not in profile.existing_tickers]
    else:
        all_tickers = [
            t for sector, tlist in STOCK_UNIVERSE.items()
            for t in tlist
            if sector not in profile.excluded_sectors
            and t not in profile.existing_tickers
        ]
        all_tickers = list(dict.fromkeys(all_tickers))

    fetcher       = DataFetcher(profile.yf_period)
    universe_data = fetcher.fetch_universe(all_tickers)

    # Post-fetch sector exclusion (applies in both modes)
    if profile.excluded_sectors and universe_data:
        before = len(universe_data)
        universe_data = {
            t: d for t, d in universe_data.items()
            if d.get("sector", "Unknown") not in profile.excluded_sectors
        }
        removed = before - len(universe_data)
        if removed:
            print(f"  Excluded {removed} stocks in sectors: {profile.excluded_sectors}")
    sp500_hist    = fetcher.fetch_sp500()

    if not universe_data:
        print("\nError: No stock data loaded. Check your internet connection.")
        sys.exit(1)

    # ── 4. Fetch macro data ───────────────────────────────────────────────────
    macro_data = MacroFetcher().fetch()
    display.show_macro(macro_data)

    # Risk-free rate from live 10Y yield (fallback: 4.5%)
    rf_rate = (macro_data.get("yield_10y") or 4.5) / 100

    # ── 5. Score stocks ───────────────────────────────────────────────────────
    print("Scoring and ranking stocks...")
    scorer    = MultiFactorScorer(profile, macro_data, adapted_weights)
    ranked_df = scorer.score_all(universe_data, sp500_hist=sp500_hist)

    if ranked_df.empty:
        print("No stocks passed the profile filters. Try a higher risk level.")
        sys.exit(0)

    print(f"  Done — {len(ranked_df)} stocks scored.\n")

    # ── 5b. Apply learned intelligence (pattern + sector + regime) ────────────
    regime = macro_data.get("regime", "neutral")

    # Dynamic sector tilts (Layer 6: learned from observed sector performance)
    dynamic_tilts = memory.get_dynamic_sector_tilts(regime)
    if dynamic_tilts:
        for sector, tilt in dynamic_tilts.items():
            mask = ranked_df["sector"] == sector
            ranked_df.loc[mask, "composite_score"] = (
                ranked_df.loc[mask, "composite_score"] + tilt
            ).clip(0, 100)

    # Sector-specific factor weight adjustments (Layer 2)
    sector_adj_applied = 0
    for sector in ranked_df["sector"].unique():
        adj = memory.get_sector_weight_adjustments(sector)
        if not adj:
            continue
        base_w = list(adapted_weights or WEIGHT_MATRIX.get(
            (profile.risk_level, profile.time_horizon), [1/7]*7
        ))
        factor_names_local = ["momentum", "volatility", "value", "quality",
                               "technical", "sentiment", "dividend"]
        adjusted_w = [base_w[i] * adj.get(factor_names_local[i], 1.0)
                      for i in range(len(base_w))]
        total = sum(adjusted_w) or 1.0
        adjusted_w = [w / total for w in adjusted_w]
        mask = ranked_df["sector"] == sector
        score_cols = [f"{f}_score" for f in factor_names_local]
        available  = [c for c in score_cols if c in ranked_df.columns]
        if len(available) == len(factor_names_local):
            adj_composite = sum(
                adjusted_w[i] * ranked_df.loc[mask, score_cols[i]]
                for i in range(len(factor_names_local))
            )
            ranked_df.loc[mask, "composite_score"] = (
                ranked_df.loc[mask, "composite_score"] * 0.6 + adj_composite * 0.4
            ).clip(0, 100)
            sector_adj_applied += int(mask.sum())

    # Pattern bonus (Layer 4: similarity to historical winners/losers)
    pattern_bonuses = memory.get_all_pattern_bonuses(universe_data, ranked_df, regime)
    if pattern_bonuses:
        for ticker, bonus in pattern_bonuses.items():
            mask = ranked_df["ticker"] == ticker
            if mask.any():
                ranked_df.loc[mask, "composite_score"] = (
                    ranked_df.loc[mask, "composite_score"] + bonus
                ).clip(0, 100)
        n_adj = sum(1 for b in pattern_bonuses.values() if abs(b) > 0.5)
        print(f"  Intelligence applied: "
              f"{n_adj} pattern bonuses · "
              f"{len(dynamic_tilts)} sector tilts · "
              f"{sector_adj_applied} sector-weight adjustments\n")

    ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = range(1, len(ranked_df) + 1)

    # ── 5d. Tier 2 enrichment — options flow + Google Trends + Reddit ─────────
    try:
        from advisor.alternative_data import enrich_top_n
        ranked_df = enrich_top_n(ranked_df, universe_data, macro_data, n=30)
        print("  Enrichment complete — options flow + retail sentiment applied.\n")
    except Exception as _e:
        print(f"  Enrichment skipped ({_e}).\n")

    # ── 5b-cont. Portfolio continuity bonus (after enrichment) ────────────────
    # Applied AFTER Tier 2 enrichment so enrichment scores don't overwrite the bonus.
    # Disabled in fresh-picks mode so the user's rotation intent is respected.
    if not profile.avoid_recent:
        _cont_tickers = memory.get_recent_tickers(n_sessions=1)
        if _cont_tickers:
            _cont_mask = ranked_df["ticker"].isin(_cont_tickers)
            ranked_df.loc[_cont_mask, "composite_score"] = (
                ranked_df.loc[_cont_mask, "composite_score"] + 3.0
            ).clip(upper=100)
            ranked_df = ranked_df.sort_values(
                "composite_score", ascending=False
            ).reset_index(drop=True)
            ranked_df["rank"] = range(1, len(ranked_df) + 1)

    # ── 5c. Apply fresh-picks penalty (optional) ──────────────────────────────
    if profile.avoid_recent:
        recent_tickers = memory.get_recent_tickers(n_sessions=2)
        if recent_tickers:
            PENALTY = FRESH_PICKS_PENALTY
            mask = ranked_df["ticker"].isin(recent_tickers)
            ranked_df.loc[mask, "composite_score"] = (
                ranked_df.loc[mask, "composite_score"] - PENALTY
            ).clip(lower=0)
            ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            print(f"  Fresh picks mode: -{PENALTY:.0f}pt penalty applied to {mask.sum()} recent picks ({', '.join(recent_tickers)}).\n")

    # ── 6. Correlation-aware selection + Kelly position sizing ────────────────
    constructor = PortfolioConstructor(n=PORTFOLIO_N)
    top10       = constructor.select(ranked_df, universe_data, profile.risk_level)
    top10_sized = constructor.size_positions(top10, profile.portfolio_size,
                                             macro_data=macro_data,
                                             universe_data=universe_data)

    # ── 7. Terminal display: quantitative results ─────────────────────────────
    display.show_results(top10_sized)
    display.show_allocation(top10_sized, profile.portfolio_size)

    # ── 8+9. Valuation + Risk in parallel (independent analyses) ─────────────
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TOut
    from config import PIPELINE_CONFIG as _PC
    _timeout = _PC.get("engine_timeout", 120)

    print("Running multi-method valuation + risk analysis (parallel)...")
    _engine_v = ValuationEngine(rf_rate)
    _engine_r = RiskEngine()

    with ThreadPoolExecutor(max_workers=2) as _exe:
        _fv = _exe.submit(_engine_v.analyze_all, top10_sized, universe_data)
        _fr = _exe.submit(_engine_r.analyze_all, top10_sized, universe_data, rf_rate)
        try:
            valuation_results = _fv.result(timeout=_timeout)
            print(f"  Valuation complete — {len(valuation_results)} stocks valued.")
        except _TOut:
            print("  Warning: Valuation timed out — using partial results.")
            valuation_results = {}
        try:
            risk_results = _fr.result(timeout=_timeout)
            print(f"  Risk analysis complete — {len(risk_results)} stocks analyzed.\n")
        except _TOut:
            print("  Warning: Risk analysis timed out — using partial results.\n")
            risk_results = {}

    # ── 10. Investment protocol analysis (7 gates) ────────────────────────────
    print("Running 7-gate investment protocol analysis...")
    protocol_analyzer = ProtocolAnalyzer()
    protocol_results  = protocol_analyzer.analyze_all(top10_sized, universe_data, valuation_results)
    print(f"  Protocol complete — {len(protocol_results)} stocks evaluated.\n")

    display.show_protocol(protocol_results)

    # ── 11. Deep quantitative analysis (hedge-fund grade output) ─────────────
    display.show_deep_analysis(top10_sized, valuation_results, risk_results, universe_data)

    # ── 12. Charts ────────────────────────────────────────────────────────────
    print("Generating charts...")
    charts = ChartEngine(profile, sp500_hist, macro_data)
    fig1   = charts.score_breakdown(top10_sized)
    fig2   = charts.performance(top10_sized, universe_data)
    fig3   = charts.factor_heatmap(top10_sized)
    fig4   = charts.macro_dashboard(top10_sized, universe_data)
    fig5   = charts.thought_process(top10_sized, protocol_results, valuation_results, risk_results)
    charts.save_all([fig1, fig2, fig3, fig4, fig5])

    # ── 13. Excel export ──────────────────────────────────────────────────────
    print("Exporting to Excel...")
    ExcelExporter().export(
        top10_sized, macro_data, profile, memory, top10_sized,
        protocol_results=protocol_results,
        valuation_results=valuation_results,
        risk_results=risk_results,
    )

    # ── 14. Save session ──────────────────────────────────────────────────────
    sp500_price = 0.0
    if sp500_hist is not None and len(sp500_hist) > 0:
        sp500_price = float(sp500_hist["Close"].iloc[-1])
    memory.save_session(
        profile, top10_sized, sp500_price,
        macro_data=macro_data,
        valuation_results=valuation_results,
        universe_data=universe_data,
        risk_results=risk_results,
        protocol_results=protocol_results,
    )
    memory.save()

    # ── 15. Disclaimer + show charts ─────────────────────────────────────────
    display.show_disclaimer()

    import matplotlib.pyplot as plt
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        print("  (Open the chart PNG files to view visualisations.)\n")

    # ── 16. Interactive command loop ──────────────────────────────────────────
    CommandHandler(
        universe_data     = universe_data,
        valuation_results = valuation_results,
        risk_results      = risk_results,
        protocol_results  = protocol_results,
        rf_rate           = rf_rate,
        macro_data        = macro_data,
    ).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Check your internet connection and try again.")
        raise
