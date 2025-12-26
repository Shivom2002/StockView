from __future__ import annotations

from typing import Any, Dict

from .utils import format_currency, format_currency_abbrev, format_number, format_percent


def render_markdown(results: Dict[str, Any]) -> str:
    tech = results.get("technicals", {})
    fundamentals = results.get("fundamentals", {})
    dcf = results.get("dcf", {})
    rec = results.get("recommendation", {})

    lines = []
    lines.append(f"# Research Report: {results.get('ticker', 'N/A')}")
    lines.append("")
    lines.append(f"*Last updated:* {results.get('timestamp', 'N/A')}")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Current price: {format_currency(results.get('price'))}")
    lines.append(f"- 1D change: {format_percent(results.get('price_change_pct'))}")
    lines.append(f"- Market cap: {format_currency_abbrev(fundamentals.get('market_cap'))}")
    lines.append(f"- Trailing P/E: {format_number(fundamentals.get('trailing_pe'))}")
    lines.append(f"- Forward P/E: {format_number(fundamentals.get('forward_pe'))}")
    lines.append("")

    lines.append("## Technicals")
    lines.append(f"- RSI(14): {format_number(tech.get('rsi'))}")
    lines.append(f"- Annualized volatility: {format_percent(tech.get('volatility'))}")
    lines.append(f"- 1Y max drawdown: {format_percent(tech.get('max_drawdown'))}")
    lines.append("")

    lines.append("## Fundamentals")
    lines.append(f"- Market cap: {format_currency_abbrev(fundamentals.get('market_cap'))}")
    lines.append(f"- P/S: {format_number(fundamentals.get('price_to_sales'))}")
    lines.append(f"- Profit margin: {format_percent(fundamentals.get('profit_margin'))}")
    lines.append(f"- Revenue growth: {format_percent(fundamentals.get('revenue_growth'))}")
    lines.append(f"- FCF: {format_currency_abbrev(fundamentals.get('fcf'))}")
    lines.append(f"- FCF margin: {format_percent(fundamentals.get('fcf_margin'))}")
    lines.append(f"- Total debt: {format_currency_abbrev(fundamentals.get('total_debt'))}")
    lines.append("")

    if dcf.get("enabled"):
        lines.append("## DCF Lite")
        lines.append(f"- Base intrinsic value per share: {format_currency(dcf.get('base_value'))}")
        lines.append(f"- Bear intrinsic value per share: {format_currency(dcf.get('bear_value'))}")
        lines.append(f"- Bull intrinsic value per share: {format_currency(dcf.get('bull_value'))}")
        lines.append(
            f"- Base assumptions: growth {format_percent(dcf.get('base_growth'))}, discount {format_percent(dcf.get('base_discount'))}, terminal {format_percent(dcf.get('base_terminal'))}"
        )
        lines.append("")
    else:
        lines.append("## DCF Lite")
        lines.append("- FCF unavailable; DCF disabled in Phase 1.")
        lines.append("")

    lines.append("## Recommendation Summary")
    lines.append(rec.get("summary", "N/A"))
    lines.append("")
    lines.append("**Reasons**")
    for item in rec.get("reasons", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**Risks / Watch items**")
    for item in rec.get("risks", []):
        lines.append(f"- {item}")

    return "\n".join(lines)
