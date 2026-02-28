"""
main.py — Entry point for the Stock Ranking Advisor v2

Run:  python main.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from config import STOCK_UNIVERSE

from advisor.collector  import InputCollector
from advisor.fetcher    import DataFetcher, MacroFetcher
from advisor.scorer     import MultiFactorScorer
from advisor.portfolio  import PortfolioConstructor
from advisor.learner    import SessionMemory
from advisor.charts     import ChartEngine
from advisor.display    import TerminalDisplay
from advisor.exporter   import ExcelExporter


def main():
    display = TerminalDisplay()
    memory  = SessionMemory()

    # ── 1. Load memory + show track record ───────────────────────────────────
    memory.load()

    print("\nChecking past session performance...")
    newly_evaluated = memory.evaluate_pending()
    if newly_evaluated:
        print(f"  Evaluated {len(newly_evaluated)} new session(s).")
    memory.save()   # persist evaluations

    track_record = memory.get_track_record()
    display.show_welcome(track_record)

    # ── 2. Collect user profile ───────────────────────────────────────────────
    profile = InputCollector().collect()

    # Adapted weights from past performance (None if not enough history)
    adapted_weights = memory.get_adapted_weights(profile.risk_level, profile.time_horizon)
    display.show_weight_adaptation(adapted_weights)

    # ── 3. Fetch stock data ───────────────────────────────────────────────────
    all_tickers = [
        t for sector, tlist in STOCK_UNIVERSE.items()
        for t in tlist
        if sector not in profile.excluded_sectors
        and t not in profile.existing_tickers
    ]
    all_tickers = list(dict.fromkeys(all_tickers))   # dedupe

    fetcher       = DataFetcher(profile.yf_period)
    universe_data = fetcher.fetch_universe(all_tickers)
    sp500_hist    = fetcher.fetch_sp500()

    if not universe_data:
        print("\nError: No stock data loaded. Check your internet connection.")
        sys.exit(1)

    # ── 4. Fetch macro data ───────────────────────────────────────────────────
    macro_data = MacroFetcher().fetch()
    display.show_macro(macro_data)

    # ── 5. Score stocks ───────────────────────────────────────────────────────
    print("Scoring and ranking stocks...")
    scorer    = MultiFactorScorer(profile, macro_data, adapted_weights)
    ranked_df = scorer.score_all(universe_data)

    if ranked_df.empty:
        print("No stocks passed the profile filters. Try a higher risk level.")
        sys.exit(0)

    print(f"  Done — {len(ranked_df)} stocks scored.\n")

    # ── 6. Correlation-aware selection + Kelly position sizing ────────────────
    constructor    = PortfolioConstructor()
    top10          = constructor.select(ranked_df, universe_data)
    top10_sized    = constructor.size_positions(top10, profile.portfolio_size)

    # ── 7. Terminal display ───────────────────────────────────────────────────
    display.show_results(top10_sized)
    display.show_allocation(top10_sized, profile.portfolio_size)

    # ── 8. Charts ─────────────────────────────────────────────────────────────
    print("Generating charts...")
    charts = ChartEngine(profile, sp500_hist, macro_data)
    fig1   = charts.score_breakdown(top10_sized)
    fig2   = charts.performance(top10_sized, universe_data)
    fig3   = charts.factor_heatmap(top10_sized)
    fig4   = charts.macro_dashboard(top10_sized, universe_data)
    charts.save_all([fig1, fig2, fig3, fig4])

    # ── 9. Excel export ───────────────────────────────────────────────────────
    print("Exporting to Excel...")
    ExcelExporter().export(top10_sized, macro_data, profile, memory, top10_sized)

    # ── 10. Save session ──────────────────────────────────────────────────────
    sp500_price = 0.0
    if sp500_hist is not None and len(sp500_hist) > 0:
        sp500_price = float(sp500_hist["Close"].iloc[-1])
    memory.save_session(profile, top10_sized, sp500_price)
    memory.save()

    # ── 11. Disclaimer + show charts ─────────────────────────────────────────
    display.show_disclaimer()

    import matplotlib.pyplot as plt
    try:
        plt.show()
    except Exception:
        print("  (Open the chart PNG files to view visualisations.)\n")


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
