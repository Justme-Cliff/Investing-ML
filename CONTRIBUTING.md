# Contributing to Stock Ranking Advisor

Thank you for your interest in contributing. This project is a fully local, quantitative stock ranking and deep analysis tool — no paid AI APIs required. Real money decisions are made based on this output, so quality, accuracy, and precision are non-negotiable.

Read this document fully before opening a pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [What We Are Looking For](#what-we-are-looking-for)
3. [What We Are NOT Looking For](#what-we-are-not-looking-for)
4. [Project Architecture Overview](#project-architecture-overview)
5. [Getting Started — Development Setup](#getting-started--development-setup)
6. [Branching Strategy](#branching-strategy)
7. [Making Your First Contribution](#making-your-first-contribution)
8. [Pull Request Process](#pull-request-process)
9. [Coding Standards](#coding-standards)
10. [Adding a New Data Source](#adding-a-new-data-source)
11. [Adding a New Scoring Factor](#adding-a-new-scoring-factor)
12. [Adding a New Valuation Method](#adding-a-new-valuation-method)
13. [Adding a New Streamlit Tab or UI Component](#adding-a-new-streamlit-tab-or-ui-component)
14. [Testing Your Changes](#testing-your-changes)
15. [Reporting Bugs](#reporting-bugs)
16. [Suggesting Features](#suggesting-features)
17. [Commit Message Format](#commit-message-format)
18. [File Ownership Map](#file-ownership-map)
19. [Design Principles (Never Break These)](#design-principles-never-break-these)
20. [FAQ](#faq)

---

## Code of Conduct

- Be respectful. Disagreements about implementation are fine; personal attacks are not.
- This is a real-money tool. Safety, accuracy, and explainability come before cleverness.
- If you are unsure whether a change is appropriate, open an issue first and discuss.

---

## What We Are Looking For

We welcome contributions in the following areas:

- **Bug fixes** — incorrect calculations, broken data fetches, UI rendering errors
- **New free data sources** — Tier 1 (all tickers, fast) or Tier 2 (top-30, enrichment)
- **New scoring factors** — must be quantitatively justified with citations
- **New valuation methods** — DCF variants, relative valuation, NAV, etc.
- **Streamlit UI improvements** — better charts, new tabs, improved UX
- **CLI command additions** — new slash commands in `advisor/cli_commands.py`
- **Backtesting improvements** — more realistic simulation logic
- **Performance optimizations** — faster data fetching, better parallelism
- **Documentation** — correcting inaccuracies, adding examples
- **Landing page** (`landing_page/`) — HTML/CSS/JS improvements within the existing black/white design system

---

## What We Are NOT Looking For

- **Paid API integrations as hard requirements** — all core features must work without any API key
- **External AI/LLM API calls** — this tool is intentionally pure math (DCF, Piotroski, etc.)
- **Breaking changes to the `UserProfile` dataclass** without a migration plan
- **New Python dependencies** without clear justification — we keep the dependency footprint small
- **Opinionated refactors** — do not restructure working code just for style preference
- **Changes to the dark CLI theme** or light Streamlit theme without prior discussion
- **Hard-coded API keys** — all keys go in `.env` only, loaded via `python-dotenv`

---

## Project Architecture Overview

```
portfolio/
├── main.py                  # 16-step CLI pipeline
├── app.py                   # Streamlit web dashboard (10 tabs)
├── config.py                # Universe, weights, sector maps, API keys
├── requirements.txt
├── advisor/
│   ├── collector.py         # UserProfile + 8-question CLI input
│   ├── fetcher.py           # DataFetcher (yfinance + Stooq + Finnhub + FRED)
│   ├── scorer.py            # 7-factor MultiFactorScorer
│   ├── alternative_data.py  # Tier 2 enrichment for top-30
│   ├── portfolio.py         # Greedy correlation-aware selection + half-Kelly
│   ├── protocol.py          # 7-gate investment protocol
│   ├── valuation.py         # DCF, Graham, EV/EBITDA, FCF yield
│   ├── risk.py              # Altman Z, Sharpe, Sortino, WACC, Piotroski
│   ├── news_fetcher.py      # News + RSS + sentiment scoring
│   ├── cli_commands.py      # Interactive slash command REPL
│   ├── charts.py            # 5 dark-theme matplotlib charts (CLI only)
│   ├── display.py           # Rich terminal output
│   ├── exporter.py          # Excel export (6 sheets)
│   ├── universe.py          # Dynamic ticker universe (NASDAQ API, 24h cache)
│   ├── learner.py           # Session memory + adaptive weights
│   └── smart_money.py       # SEC Form 4 cluster + 8-K event scoring
└── landing_page/            # Static HTML/CSS/JS marketing site
```

### Data Flow

```
UserProfile
    |
    v
DataFetcher.fetch_all()          <- Tier 1: yfinance + Stooq + Finnhub + FRED (all tickers)
    |
    v
MultiFactorScorer.score_all()   <- 7 factors, normalized 0-100
    |
    v
enrich_top_n()                   <- Tier 2: Options + Trends + AV + FMP + SEC (top-30 only)
    |
    v
Portfolio construction           <- Greedy correlation-aware + half-Kelly sizing
    |
    v
ValuationEngine + RiskEngine + ProtocolAnalyzer
    |
    v
Display / Charts / Export / Streamlit tabs
```

---

## Getting Started — Development Setup

### Prerequisites

- Python 3.10+
- Git
- Windows 11 / macOS / Linux (developed on Windows 11)
- A virtual environment manager (`venv` or `conda`)

### Step 1 — Fork and Clone

1. Fork the repository on GitHub.
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/portfolio.git
cd portfolio
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/ORIGINAL-OWNER/portfolio.git
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure API Keys (Optional)

API keys are optional — the tool degrades gracefully without them. For full Tier 2 enrichment during development, create a `.env` file in the project root:

```
FINNHUB_KEY=your_key_here
NEWSAPI_KEY=your_key_here
FRED_KEY=your_key_here
ALPHAVANTAGE_KEY=your_key_here
FMP_KEY=your_key_here
BLS_KEY=your_key_here
```

**Never commit `.env` to the repository.** It is already listed in `.gitignore`.

### Step 5 — Run the App

```bash
# CLI
python main.py

# Streamlit web dashboard
streamlit run app.py
```

---

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, production-ready code |
| `dev` | Integration branch for new features |
| `fix/short-description` | Bug fixes |
| `feat/short-description` | New features |
| `docs/short-description` | Documentation changes only |
| `refactor/short-description` | Refactors that do not change behavior |

**Always branch from `main` for bug fixes. Branch from `dev` for new features.**

```bash
# For a bug fix
git checkout main
git pull upstream main
git checkout -b fix/valuation-dcf-nan-crash

# For a new feature
git checkout dev
git pull upstream dev
git checkout -b feat/new-rss-source
```

---

## Making Your First Contribution

### Beginner-Friendly Areas

If you are new to the codebase, start here:

1. **`advisor/news_fetcher.py`** — Add a new RSS feed. Each feed is one line in the `RSS_FEEDS` list.
2. **`config.py`** — Add tickers to `STOCK_UNIVERSE`. Find the right sector list and append the symbol.
3. **`landing_page/`** — Fix typos or improve the documentation at `docs.html` or `changelog.html`.
4. **`advisor/cli_commands.py`** — Add a new slash command. Follow the pattern of existing commands.

### Workflow

1. Pick an open issue or create one describing what you plan to do.
2. Assign yourself or comment that you are working on it to avoid duplicate effort.
3. Create your branch (see above).
4. Make your changes.
5. Test manually (see [Testing Your Changes](#testing-your-changes)).
6. Commit with a clear message (see [Commit Message Format](#commit-message-format)).
7. Push to your fork and open a pull request against `main` (fixes) or `dev` (features).

---

## Pull Request Process

1. **One concern per PR.** Do not combine a bug fix with an unrelated feature.
2. **Fill out the PR template** completely. Describe what changed and why.
3. **Link the relevant issue** using `Closes #123` or `Fixes #123` in the PR description.
4. **Self-review your diff** before requesting review. Check for debug prints, commented-out code, and hardcoded values.
5. **All CI checks must pass** (if configured). Do not merge with failing checks.
6. **At least one maintainer approval** is required before merging.
7. PRs against `main` are squash-merged. PRs against `dev` may use merge commits.

### PR Description Template

```
## What does this PR do?
Brief description of the change.

## Why is it needed?
What problem does it solve or what improvement does it make?

## How was it tested?
Describe the manual testing steps you performed.

## Related issue
Closes #

## Checklist
- [ ] I read CONTRIBUTING.md
- [ ] No hardcoded API keys or credentials
- [ ] No new required paid APIs
- [ ] Graceful try/except if a data source can fail
- [ ] Estimate dict keys remain short: "dcf", "graham", "ev_ebitda", "fcf_yield"
- [ ] Dark theme for CLI charts, light theme for Streamlit (unless changing themes)
- [ ] No plt.show() blocking calls — use plt.show(block=False) + plt.pause(0.1)
```

---

## Coding Standards

### Python Style

- Follow **PEP 8** with a line length of 100 characters.
- Use **type hints** for all new function signatures.
- Use **f-strings** — not `.format()` or `%` formatting.
- Variables and functions: `snake_case`. Classes: `PascalCase`. Constants: `UPPER_SNAKE_CASE`.
- Do not add docstrings to code you did not write unless the maintainer requests it.
- Keep functions focused. If a function exceeds 80 lines, consider splitting it.

### Error Handling

Every data fetch must be wrapped in `try/except`. The tool must always produce output even when sources fail:

```python
# Correct
try:
    result = fetch_something(ticker)
except Exception:
    result = None

# Wrong — naked fetches that can crash the pipeline
result = fetch_something(ticker)
```

Log failures silently or with a brief warning. Never `raise` from a data fetcher unless the failure is unrecoverable.

### No Hard-Coded Secrets

```python
# Correct
import os
key = os.getenv("FINNHUB_KEY")
if not key:
    return None  # degrade gracefully

# Wrong
key = "abc123myrealapikey"
```

### Streamlit Conventions

- All color constants are defined at the top of `app.py`. Use them — do not introduce new hex literals.
- All helper functions (`badge`, `mtile`, `shdr`, `sbar`) are defined at the top of `app.py`. Use them instead of duplicating HTML.
- `st.session_state` keys are documented in `CLAUDE.md`. Do not add undocumented keys without updating the docs.
- All Plotly figures must use `_plotly_base()` for consistent layout.

### CLI / Rich Conventions

- Use `rich` for all terminal output in the CLI pipeline and REPL.
- Chart files use the dark palette from `charts.py`. Do not introduce new color constants.
- `plt.show(block=False)` + `plt.pause(0.1)` — never block the command loop.

---

## Adding a New Data Source

### Tier 1 Source (all tickers, fast)

Tier 1 sources run inside `DataFetcher._fetch_one()` in `advisor/fetcher.py` inside a `ThreadPoolExecutor`. They must be:

- **Fast** (< 2s per ticker on average)
- **Free** (no paid key required, or key is optional with graceful fallback)
- **Wrapped in try/except**

Steps:
1. Add your fetch logic inside `_fetch_one()`, appending the new field to the returned dict.
2. Document the new field in the return dict docstring and in `CLAUDE.md` under "fetcher.py".
3. If the field is used in scoring, update `advisor/scorer.py` (see next section).
4. If it requires a new env key, add it to `config.py` with `os.getenv("YOUR_KEY")` and list it in `CLAUDE.md`.

### Tier 2 Source (top-30 only, enrichment)

Tier 2 sources run in `advisor/alternative_data.py` inside `enrich_top_n()`. They are called once per run, only for the top-30 scored stocks.

Steps:
1. Write a standalone fetch function `fetch_my_source(ticker, ...)` in `alternative_data.py`.
2. Call it inside the `enrich_top_n()` loop after the existing sources.
3. Apply the result as a direct adjustment to `ranked_df` `composite_score` (clipped 0–100).
4. Add the source to the Tier 2 table in `CLAUDE.md` and `landing_page/docs.html`.

---

## Adding a New Scoring Factor

The scorer uses 7 named factors. Adding an 8th is a significant change — open an issue first.

To modify an existing factor or add a sub-component:

1. **`advisor/scorer.py` — `_score_one()`**: Compute your sub-score (0–100 scale). Apply it as a weighted addition to the relevant factor (value, momentum, quality, etc.).
2. **`config.py` — `WEIGHT_MATRIX`**: If you are changing factor weights, update all 4 risk-level rows. Weights per row must sum to 1.0.
3. **`advisor/charts.py`**: If you added a new top-level factor, update the factor heatmap labels.
4. **`app.py` — `_factor_bars_html()`** and the factor radar chart: update factor name lists.
5. **`CLAUDE.md`**: Update the scorer section.

Every factor must have a quantitative justification. Cite academic papers or established financial formulas in your PR description.

---

## Adding a New Valuation Method

Valuation methods live in `advisor/valuation.py` inside `ValuationEngine.analyze()`.

1. Compute your estimate and add it to the `estimates` dict with a **short lowercase key**:

```python
estimates = {
    "dcf":       dcf_value,
    "graham":    graham_value,
    "ev_ebitda": ev_ebitda_value,
    "fcf_yield": fcf_yield_value,
    "my_method": my_value,       # <-- new
}
```

2. Include it in the `valid_estimates` filter (must be > 0 and finite).
3. Update the fair value averaging logic if appropriate.
4. Expose it in `app.py` Valuation tab and in `advisor/cli_commands.py` `/stock` command.
5. Document the formula and key assumptions in a comment in the code.

**Important:** The estimate key must remain short lowercase. Never use the long human-readable name as a dict key.

---

## Adding a New Streamlit Tab or UI Component

1. Write the tab as a standalone function: `def tab_my_feature(arg1, arg2): ...`
2. Add it to the `tabs` list in the main `app.py` body alongside the existing 10 tabs.
3. Use existing color constants and helper functions (`badge`, `mtile`, `shdr`, `sbar`, `_plotly_base`).
4. All Plotly charts must use `st.plotly_chart(fig, use_container_width=True)`.
5. Add the new tab name to `CLAUDE.md` and `landing_page/docs.html` (keep the tab list accurate).
6. If you introduce new `st.session_state` keys, document them in `CLAUDE.md`.

---

## Testing Your Changes

There is no automated test suite currently. All testing is manual. Here is the expected verification checklist for each change type:

### Data Source Change
- [ ] Run `python main.py` with a small universe (edit `config.py` temporarily to 10 tickers)
- [ ] Confirm the new field appears in `ranked_df` and is not all-None
- [ ] Confirm the pipeline completes without exception when the source returns None (simulate by temporarily setting the key to `None`)
- [ ] Run `streamlit run app.py` and confirm no crash in the UI

### Scoring Change
- [ ] Confirm composite scores are still in range 0–100
- [ ] Confirm portfolio construction produces 10 picks (or PORTFOLIO_N picks)
- [ ] Verify the ranking order makes intuitive sense for well-known tickers (AAPL, MSFT, etc.)

### Valuation Change
- [ ] Verify fair value is positive and non-NaN for at least 5 major tickers
- [ ] Verify `entry_low`, `entry_high`, `target_price`, `stop_loss` are all derived correctly
- [ ] Check the Valuation tab in Streamlit renders without error

### UI Change
- [ ] Test in Streamlit with a fresh run (click "Run Analysis")
- [ ] Test edge cases: empty results, None fields, tickers with missing data
- [ ] Check both Terminal/Dark and Light themes if your change involves colors

### CLI Change
- [ ] Run `python main.py` through the full pipeline
- [ ] Manually invoke your new command in the REPL and verify output

---

## Reporting Bugs

Open a GitHub issue with the following information:

```
**Bug description**
What happened vs. what you expected.

**Steps to reproduce**
1. Run `python main.py`
2. Enter portfolio size: 10000
3. ...

**Error output**
Paste the full traceback here.

**Environment**
- OS: Windows 11 / macOS 14 / Ubuntu 22.04
- Python version: 3.10.x
- Relevant package versions (yfinance, streamlit, pandas)

**API keys configured**
List which optional keys are set (do NOT paste the actual keys).
```

For data accuracy bugs (wrong valuation, wrong score), include:
- The ticker symbol
- The value you observed
- The value you expected and why (with source/formula)

---

## Suggesting Features

Open a GitHub issue with label `enhancement`. Include:

1. **Problem statement** — What limitation or gap does this address?
2. **Proposed solution** — How would it work? Which files would change?
3. **Data source** — If it requires new data, what is the source and is it free?
4. **Financial justification** — Why is this metric or feature useful for real investment decisions?
5. **Scope** — Is this Tier 1 (all tickers) or Tier 2 (top-30 only)?

Features that require paid APIs will only be accepted as optional enhancements with complete graceful fallback.

---

## Commit Message Format

Use the **Conventional Commits** format:

```
type(scope): short imperative description

Optional longer body explaining the why, not the what.

Closes #123
```

### Types

| Type | When to use |
|------|-------------|
| `fix` | Bug fix — corrects incorrect behavior |
| `feat` | New feature or capability |
| `refactor` | Code restructure with no behavior change |
| `perf` | Performance improvement |
| `docs` | Documentation only (CLAUDE.md, README, comments) |
| `style` | Formatting only (whitespace, line length) |
| `chore` | Dependency updates, config changes, build tasks |

### Scopes

`fetcher`, `scorer`, `valuation`, `risk`, `protocol`, `portfolio`, `learner`, `alternative_data`, `smart_money`, `news`, `cli`, `app`, `charts`, `display`, `exporter`, `config`, `landing_page`

### Examples

```
fix(valuation): prevent NaN crash when EBITDA is None

Graham Number and EV/EBITDA methods now return None (not NaN)
when required fields are missing, allowing the averaging logic
to skip them gracefully.

Closes #47
```

```
feat(fetcher): add Quandl SHARADAR fallback for revenue data

Optional — requires QUANDL_KEY in .env. Falls back to yfinance
when key is absent. Revenue trend populated from SHARADAR/SF1
annual table when yfinance quarterly_financials is empty.

Closes #83
```

```
docs(landing_page): correct tab count from 9 to 10 in docs.html
```

---

## File Ownership Map

Use this to find the right file for your change:

| What you want to change | File |
|------------------------|------|
| Add a stock ticker | `config.py` → `STOCK_UNIVERSE` |
| Change factor weights | `config.py` → `WEIGHT_MATRIX` |
| Change sector PE / EV ratios | `config.py` → `SECTOR_MEDIAN_PE`, `SECTOR_EV_EBITDA` |
| Add / fix a data field | `advisor/fetcher.py` → `_fetch_one()` |
| Add / fix macro data | `advisor/fetcher.py` → `MacroFetcher.fetch()` |
| Change scoring logic | `advisor/scorer.py` → `_score_one()` |
| Add Tier 2 enrichment | `advisor/alternative_data.py` → `enrich_top_n()` |
| Change valuation math | `advisor/valuation.py` → `ValuationEngine.analyze()` |
| Change risk metrics | `advisor/risk.py` → `RiskEngine.analyze()` |
| Change protocol gates | `advisor/protocol.py` → `ProtocolAnalyzer` |
| Change portfolio construction | `advisor/portfolio.py` |
| Change CLI questions | `advisor/collector.py` |
| Add a slash command | `advisor/cli_commands.py` → `CommandHandler` |
| Change session memory / weights | `advisor/learner.py` |
| Add a CLI chart | `advisor/charts.py` |
| Change terminal display | `advisor/display.py` |
| Change Excel export | `advisor/exporter.py` |
| Change news / sentiment | `advisor/news_fetcher.py` |
| Change smart money signals | `advisor/smart_money.py` |
| Change Streamlit UI | `app.py` |
| Change Streamlit theme | `.streamlit/config.toml` + `app.py` → `_apply_theme_css()` |
| Change landing page | `landing_page/` |

---

## Design Principles (Never Break These)

These are enforced at review time. PRs that violate them will not be merged.

1. **No paid data APIs as hard requirements** — optional keys must degrade gracefully.
2. **No external AI/LLM APIs** — all analysis is pure math.
3. **Real money context** — never be vague. Output must be precise and defensible.
4. **Graceful degradation** — every data source has `try/except`. The tool always produces output.
5. **Non-blocking CLI charts** — `plt.show(block=False)` + `plt.pause(0.1)` only.
6. **Short estimate keys** — `"dcf"`, `"graham"`, `"ev_ebitda"`, `"fcf_yield"` — never the long form.
7. **Tier separation** — expensive/slow sources run in Tier 2 (top-30 only).
8. **API keys in `.env` only** — never hard-coded, never committed.
9. **Warren Buffett philosophy** — valuation and margin of safety always required; no momentum-only picks.
10. **CLI dark theme / Streamlit light theme** — do not swap these without explicit discussion.

---

## FAQ

**Q: Can I add a new required Python package?**
Yes, but justify it in your PR. We prefer using libraries already in `requirements.txt`. If you add one, add it to `requirements.txt` with a minimum version pin.

**Q: Can I restructure the folder layout?**
No. The module structure is intentional. Open an issue to discuss before making structural changes.

**Q: How do I test with a small universe to go faster?**
Temporarily edit `config.py` and reduce `STOCK_UNIVERSE` to a few tickers, or set `UNIVERSE_MAX_TICKERS` to a small number. Revert before committing.

**Q: Can I contribute without writing code?**
Yes. Documentation fixes, issue triage, bug reports with detailed reproduction steps, and landing page copy improvements are all valuable contributions.

**Q: My PR touches `app.py`. It is a very large file — is that normal?**
Yes. `app.py` is intentionally a single file for the Streamlit dashboard. Do not split it without prior maintainer approval.

**Q: Where should I ask questions?**
Open a GitHub Discussion or comment on the relevant issue. Do not open a new issue just to ask a question.

**Q: Can I add support for non-US stocks?**
The universe already includes 70+ international ADRs. Full international exchange support is on the roadmap. For now, contributions should target US-listed tickers and ADRs.

**Q: What Python version should I target?**
Python 3.10. Do not use syntax or features introduced in 3.11 or later.

---

*Thank you for contributing. Every improvement to this tool directly affects the quality of real investment decisions.*
