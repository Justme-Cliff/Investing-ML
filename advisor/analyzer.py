# advisor/analyzer.py — Claude AI deep analysis with extended thinking
"""
Uses Claude claude-opus-4-5 with extended thinking to do rigorous qualitative analysis
on the top pre-screened stocks.

Requires: ANTHROPIC_API_KEY environment variable
Install:  pip install anthropic

Gracefully degrades to None if API key not set or anthropic not installed.
The rest of the system works fine without it — Claude's analysis is additive,
displayed in the thought-process chart and terminal output.
"""

import os
import json
from typing import List, Optional

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


_SYSTEM = """You are a rigorous investment analyst following Warren Buffett and Charlie Munger's value investing philosophy.

YOUR CORE RULES:
1. A great company at a bad price is still a bad investment — always assess valuation honestly
2. Always require a margin of safety (15-20% below intrinsic value) before recommending a buy
3. Flag any stock trading within 5% of its 52-week high as a risky entry point
4. If P/E exceeds 1.5× sector median without exceptional growth justification, it's expensive
5. Prefer businesses with durable moats: high margins, network effects, switching costs, brand value
6. Think in 3-5 year time horizons, not quarters
7. Be direct and intellectually honest — if something is overpriced, say so clearly
8. Never recommend buying the entire list — prioritize the 3-4 best opportunities

You have just completed a deep thinking session where you carefully analyzed each company's business model, competitive position, financial health, and valuation. Your analysis is thorough, contrarian where warranted, and grounded in fundamental analysis."""

_PROMPT = """CURRENT MARKET CONDITIONS:
{macro}

TOP CANDIDATE STOCKS (pre-screened by quantitative model, ranked by composite score):
{stocks}

You have already thought deeply about each company. Now provide your final investment verdicts.

For each stock, deliver:
- A honest business quality assessment (not marketing speak)
- A clear valuation verdict (is this cheap, fair, or expensive right now?)
- A specific entry recommendation with price rationale
- The 2 most important risks to be aware of
- The 2 most compelling catalysts

Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks):
{{
  "market_assessment": "<1-2 sentences on current market conditions and what they mean for investors>",
  "stocks": [
    {{
      "ticker": "<TICKER>",
      "conviction": "HIGH|MEDIUM|LOW|AVOID",
      "business_quality": "<2-3 sentences on business fundamentals, moat, and durability>",
      "valuation_view": "<honest 1-2 sentence valuation assessment — is it cheap, fair, or expensive?>",
      "entry_signal": "STRONG_BUY|BUY|HOLD_WATCH|WAIT|AVOID_PEAK",
      "entry_price_comment": "<specific: at what price or under what condition would you buy?>",
      "key_risks": ["<specific risk 1>", "<specific risk 2>"],
      "key_catalysts": ["<specific catalyst 1>", "<specific catalyst 2>"],
      "one_line": "<single sentence that captures your investment view — be direct>"
    }}
  ],
  "portfolio_view": "<2-3 sentences on the portfolio as a whole — diversification, timing, concentration risk>",
  "top_picks": ["<ticker1>", "<ticker2>", "<ticker3>"],
  "stocks_to_watch": ["<ticker: condition that would make you buy>"]
}}"""


class AIAnalyzer:
    """
    Uses Claude claude-opus-4-5 with extended thinking for deep stock analysis.

    Extended thinking gives Claude time to reason carefully through valuations,
    business quality, and entry timing before returning its final verdict.
    """

    def __init__(self):
        self.client    = None
        self.available = False

        if not _ANTHROPIC_AVAILABLE:
            print("  [AI] anthropic package not installed — run: pip install anthropic")
            return

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("  [AI] ANTHROPIC_API_KEY not set — running quantitative-only mode")
            print("       Set it to enable Claude deep analysis and entry price AI reasoning")
            return

        try:
            self.client    = anthropic.Anthropic(api_key=key)
            self.available = True
            print("  [AI] Claude deep analysis: ENABLED  (claude-opus-4-5 + extended thinking)")
        except Exception as e:
            print(f"  [AI] Could not initialise Anthropic client: {e}")

    def analyze(
        self,
        top10,
        universe_data: dict,
        macro_data:    dict,
        protocol_results: List[dict],
    ) -> Optional[dict]:
        """
        Run Claude extended-thinking analysis on the top stocks.
        Returns structured dict or None if unavailable.
        """
        if not self.available:
            return None

        prompt = _PROMPT.format(
            macro=self._fmt_macro(macro_data),
            stocks=self._fmt_stocks(top10, universe_data, protocol_results),
        )

        try:
            print("  Running Claude extended-thinking analysis (allow ~60s)...")
            msg = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 10000},
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text block (the thinking block is separate and handled internally)
            text = next((b.text for b in msg.content if b.type == "text"), "")
            if not text:
                print("  Claude returned no text response.")
                return None

            result = self._parse(text)
            n = len(result.get("stocks", []))
            print(f"  Claude analysis complete — {n} stocks reviewed.")
            return result

        except Exception as e:
            print(f"  Claude analysis error: {e}")
            return None

    # ── Prompt formatting ─────────────────────────────────────────────────────

    def _fmt_stocks(self, top10, universe_data: dict, proto_list: List[dict]) -> str:
        pm    = {p["ticker"]: p for p in proto_list}
        parts = []

        for _, row in top10.iterrows():
            t     = row["ticker"]
            data  = universe_data.get(t, {})
            info  = data.get("info", {})
            proto = pm.get(t, {})
            entry = proto.get("entry_analysis", {})
            news  = "; ".join(data.get("news_titles", [])[:4]) or "No recent headlines"

            # Format key metrics cleanly
            def fmt(v, fmt_str=".2f", prefix="", suffix=""):
                try:
                    return f"{prefix}{float(v):{fmt_str}}{suffix}"
                except (TypeError, ValueError):
                    return "N/A"

            parts.append(
                f"─── {t} ({row.get('sector','?')})  Rank #{int(row.get('rank',0))}  Score {row.get('composite_score',0):.1f}/100 ───\n"
                f"  Price:       ${row.get('current_price',0):.2f}\n"
                f"  52W Range:   ${info.get('fiftyTwoWeekLow','?')} – ${info.get('fiftyTwoWeekHigh','?')}\n"
                f"  P/E:         {fmt(info.get('trailingPE'),',.1f')}  |  Forward P/E: {fmt(info.get('forwardPE'),',.1f')}  |  PEG: {fmt(info.get('pegRatio'),'.2f')}\n"
                f"  ROE:         {fmt(info.get('returnOnEquity'),'.1%')}  |  Profit Margin: {fmt(info.get('profitMargins'),'.1%')}\n"
                f"  Rev Growth:  {fmt(info.get('revenueGrowth'),'.1%')}  |  EPS Growth: {fmt(info.get('earningsGrowth'),'.1%')}\n"
                f"  Analyst Tgt: ${fmt(info.get('targetMeanPrice'),',.2f')}  |  Recommendation: {info.get('recommendationKey','N/A')}\n"
                f"  Protocol:    {proto.get('pass_count','?')}/7 gates passed  |  Conviction: {proto.get('conviction','?')}  |  Score: {proto.get('overall_score','?')}/100\n"
                f"  Entry Data:  Current=${entry.get('current_price','?')}  FairValue=${entry.get('fair_value','?')}  Target=${entry.get('entry_target','?')}  Signal={entry.get('signal','?')}  Premium={fmt(entry.get('premium_pct'),'.1f')}%\n"
                f"  Headlines:   {news}"
            )

        return "\n\n".join(parts)

    def _fmt_macro(self, macro_data: dict) -> str:
        vix    = macro_data.get("vix")
        y10    = macro_data.get("yield_10y")
        regime = macro_data.get("regime", "neutral").upper()
        etf    = macro_data.get("sector_etf", {})
        top4   = sorted(etf.items(), key=lambda x: x[1], reverse=True)[:4]
        sects  = " | ".join(f"{s} {r:+.1f}%" for s, r in top4)
        return (
            f"Regime: {regime} | VIX: {f'{vix:.1f}' if vix else 'N/A'} | 10Y Yield: {f'{y10:.2f}%' if y10 else 'N/A'}\n"
            f"Leading Sectors (3-month): {sects}"
        )

    def _parse(self, text: str) -> dict:
        """Extract and parse JSON from Claude's response text."""
        # Strip any markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise ValueError("No valid JSON found in Claude response")
