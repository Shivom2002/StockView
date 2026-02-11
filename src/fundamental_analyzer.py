from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import streamlit as st

# Try to import Google Generative AI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


try:
    _cache_data = st.cache_data
except AttributeError:  # Streamlit < 1.18
    _cache_data = st.cache


def _get_gemini_client():
    """
    Get Gemini client with API key from environment.

    Returns (client, error_message).
    """
    if not GENAI_AVAILABLE:
        return None, "google-genai package not installed. Run: pip install google-genai"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "GEMINI_API_KEY environment variable not set"

    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Failed to initialize Gemini client: {str(e)}"


def _get_gemini_model() -> str:
    """
    Get Gemini model name from environment variable with fallback to default.
    """
    default_model = "gemini-3-flash-preview"
    model = os.environ.get("GEMINI_MODEL", default_model)
    return model.strip() if model else default_model


def _build_fundamental_narrative_prompt(fundamentals: Dict[str, Any], ticker: str, info: Dict[str, Any]) -> str:
    """
    Build comprehensive prompt with all fundamental data for narrative generation.
    """
    # Extract context
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    business_summary = info.get("longBusinessSummary", "")[:500]  # Truncate

    # Get fundamental metrics
    market_cap = fundamentals.get('market_cap', 0)
    trailing_pe = fundamentals.get('trailing_pe')
    forward_pe = fundamentals.get('forward_pe')
    price_to_sales = fundamentals.get('price_to_sales')
    profit_margin = fundamentals.get('profit_margin', 0)
    revenue_growth = fundamentals.get('revenue_growth', 0)
    total_revenue = fundamentals.get('total_revenue', 0)
    fcf = fundamentals.get('fcf', 0)
    fcf_margin = fundamentals.get('fcf_margin', 0)
    total_debt = fundamentals.get('total_debt', 0)

    # Get additional context from info
    debt_to_equity = info.get('debtToEquity')
    current_ratio = info.get('currentRatio')
    roe = info.get('returnOnEquity')
    roa = info.get('returnOnAssets')
    operating_margins = info.get('operatingMargins')
    ebitda_margins = info.get('ebitdaMargins')
    quick_ratio = info.get('quickRatio')

    # Format metrics for prompt
    def fmt_num(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        return f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"

    def fmt_pct(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        return f"{val*100:.2f}%"

    import math

    prompt = f"""You are a financial analyst providing a comprehensive assessment of {ticker}'s financial health for investors.

Company Context:
- Ticker: {ticker}
- Sector: {sector}
- Industry: {industry}
- Business: {business_summary or "N/A"}

Current Fundamental Metrics:
Valuation:
- Market Cap: ${fmt_num(market_cap)}
- Trailing P/E: {fmt_num(trailing_pe)}
- Forward P/E: {fmt_num(forward_pe)}
- Price-to-Sales: {fmt_num(price_to_sales)}

Profitability:
- Profit Margin: {fmt_pct(profit_margin)}
- Operating Margins: {fmt_pct(operating_margins)}
- EBITDA Margins: {fmt_pct(ebitda_margins)}
- ROE: {fmt_pct(roe)}
- ROA: {fmt_pct(roa)}

Growth:
- Revenue Growth (YoY): {fmt_pct(revenue_growth)}
- Total Revenue: ${fmt_num(total_revenue)}

Cash Generation:
- Free Cash Flow: ${fmt_num(fcf)}
- FCF Margin: {fmt_pct(fcf_margin)}

Balance Sheet:
- Total Debt: ${fmt_num(total_debt)}
- Debt-to-Equity: {fmt_num(debt_to_equity)}
- Current Ratio: {fmt_num(current_ratio)}
- Quick Ratio: {fmt_num(quick_ratio)}

Provide a comprehensive financial health assessment in JSON format:
{{
  "overall_health": "<Strong|Moderate|Weak|Critical>",
  "health_score": <0-100>,
  "narrative": "A 3-4 paragraph narrative covering: (1) Overall financial position and business model strength, (2) Profitability and cash generation analysis with specific metrics, (3) Growth trajectory and sustainability, (4) Balance sheet strength and risk factors",
  "strengths": [
    "List 3-5 key financial strengths with specific metrics cited",
    "Example: 'Strong FCF generation with {fmt_pct(fcf_margin)} margins indicates efficient operations'"
  ],
  "weaknesses": [
    "List 3-5 key financial weaknesses or concerns with specifics",
    "Example: 'Declining revenue growth of {fmt_pct(revenue_growth)} suggests market saturation'"
  ],
  "red_flags": [
    "List critical concerns that could threaten the business (if any)",
    "Example: 'Debt-to-equity ratio of {fmt_num(debt_to_equity)} raises solvency concerns'",
    "Leave empty array if no major red flags"
  ],
  "trends": {{
    "profitability": "<Improving|Declining|Stable>",
    "growth": "<Improving|Declining|Stable>",
    "leverage": "<Improving|Declining|Stable>",
    "efficiency": "<Improving|Declining|Stable>"
  }},
  "peer_context": "1-2 sentences on how these metrics compare to {sector} sector averages or typical ranges. Note if metrics are above/below industry standards."
}}

Guidelines for assessment:
- Be specific: cite actual metrics from the data provided
- Be balanced: include both positives and negatives
- Be actionable: focus on what matters for investment decisions
- Consider sector context: tech margins differ from retail, growth stocks have different profiles than value stocks
- Identify trends: Note if profitability appears to be improving or declining (though historical data is limited)
- Flag risks: high debt, negative FCF, declining margins, very high or very low P/E relative to growth
- Assess sustainability: can current growth and margins continue?
- Health score guidance:
  - 80-100: Strong (healthy margins, good growth, solid balance sheet, reasonable valuation)
  - 60-79: Moderate (decent fundamentals but some concerns)
  - 40-59: Weak (significant red flags or poor fundamentals)
  - 0-39: Critical (severe financial distress indicators)

IMPORTANT:
- Respond with ONLY valid, complete JSON
- No markdown code blocks (no ```json or ```)
- No additional text before or after the JSON
- Ensure all JSON strings are properly closed with quotes
- Ensure all arrays and objects are properly closed with brackets
- If a metric is N/A, acknowledge it in the narrative but don't let it block analysis

Output the JSON now:"""

    return prompt


def _parse_narrative_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse Gemini's response for fundamental narrative.

    Returns (parsed_data, error_message).
    """
    try:
        # Similar parsing logic to gemini_summarizer
        text = response_text.strip()

        # Remove markdown code blocks
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('```'):
                start_idx = i + 1
                break
            elif stripped.startswith('{'):
                start_idx = i
                break

        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped == '```':
                end_idx = i
                break
            elif stripped.endswith('}'):
                end_idx = i + 1
                break

        json_lines = lines[start_idx:end_idx]
        text = '\n'.join(json_lines).strip()

        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        # Try to fix truncation issues
        if not text.endswith('}'):
            open_braces = text.count('{')
            close_braces = text.count('}')
            open_brackets = text.count('[')
            close_brackets = text.count(']')

            if text.count('"') % 2 != 0:
                text += '"'

            for _ in range(open_brackets - close_brackets):
                text += ']'

            for _ in range(open_braces - close_braces):
                text += '}'

        # Parse JSON
        data = json.loads(text)

        # Validate structure
        if not isinstance(data, dict):
            return None, "Response is not a JSON object"

        overall_health = data.get("overall_health", "Moderate")
        health_score = data.get("health_score", 50)
        narrative = data.get("narrative", "")
        strengths = data.get("strengths", [])
        weaknesses = data.get("weaknesses", [])
        red_flags = data.get("red_flags", [])
        trends = data.get("trends", {})
        peer_context = data.get("peer_context", "")

        # Validate types
        if not isinstance(strengths, list):
            strengths = []
        if not isinstance(weaknesses, list):
            weaknesses = []
        if not isinstance(red_flags, list):
            red_flags = []
        if not isinstance(trends, dict):
            trends = {}

        # Clamp health score
        health_score = max(0, min(100, int(health_score)))

        # Limit list lengths
        strengths = strengths[:5]
        weaknesses = weaknesses[:5]
        red_flags = red_flags[:5]

        return {
            "overall_health": overall_health,
            "health_score": health_score,
            "narrative": narrative,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "red_flags": red_flags,
            "trends": trends,
            "peer_context": peer_context,
        }, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error parsing response: {str(e)}"


@_cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def generate_fundamental_narrative(
    fundamentals: Dict[str, Any],
    ticker: str,
    info: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate comprehensive financial health narrative using LLM.

    Args:
        fundamentals: Parsed fundamental metrics from parse_fundamentals()
        ticker: Stock ticker symbol
        info: Raw yfinance info dict for additional context

    Returns:
        ({
            "overall_health": str,
            "health_score": int (0-100),
            "narrative": str,
            "strengths": List[str],
            "weaknesses": List[str],
            "red_flags": List[str],
            "trends": dict,
            "peer_context": str
        }, error_message)
    """
    if not fundamentals:
        return None, "No fundamentals provided"

    # Get Gemini client
    client, error = _get_gemini_client()
    if client is None:
        return None, error

    try:
        # Build prompt
        prompt = _build_fundamental_narrative_prompt(fundamentals, ticker, info)

        # Get model name
        model_name = _get_gemini_model()

        # Call Gemini API
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,  # Slightly higher for narrative
                    max_output_tokens=6000,  # Longer narrative
                    response_modalities=["TEXT"],
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            # Fallback if response_mime_type not supported
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=6000,
                    response_modalities=["TEXT"],
                ),
            )

        # Extract text from response
        if not response or not response.text:
            return None, "Empty response from Gemini API"

        response_text = response.text

        # Parse response
        parsed, parse_error = _parse_narrative_response(response_text)
        if parsed is None:
            return None, parse_error

        return parsed, None

    except Exception as e:
        return None, f"Gemini API error: {str(e)}"
