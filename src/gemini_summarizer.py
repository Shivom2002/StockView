from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Try to import Google Generative AI SDK
# This is optional - the app works without it, just won't have AI summaries
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
    If client is None, error_message will contain the reason.
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

    Returns the model name to use.
    """
    default_model = "gemini-3-flash"
    model = os.environ.get("GEMINI_MODEL", default_model)
    return model.strip() if model else default_model


def _build_prompt(items: List[Dict[str, Any]], ticker: str, max_items: int = 10) -> str:
    """
    Build a prompt for Gemini to summarize news articles.
    """
    # Limit the number of items to avoid token limits
    items_to_use = items[:max_items]

    # Build article list
    articles_text = []
    for i, item in enumerate(items_to_use, 1):
        title = item.get("title", "")
        publisher = item.get("publisher", "Unknown")
        published_at = item.get("published_at")
        summary = item.get("summary", "")

        time_str = published_at.strftime("%Y-%m-%d %H:%M UTC") if published_at else "Unknown"

        article_entry = f"{i}. {title}\n   Publisher: {publisher}\n   Published: {time_str}"
        if summary:
            article_entry += f"\n   Snippet: {summary[:200]}"  # Limit snippet length
        articles_text.append(article_entry)

    articles_section = "\n\n".join(articles_text)

    prompt = f"""You are summarizing recent news headlines about {ticker} for an investor dashboard. Provide a neutral, factual summary highlighting key recent developments.

Recent articles:

{articles_section}

Based on these articles, provide a summary in JSON format with the following structure:
{{
  "bullets": ["bullet point 1", "bullet point 2", ...],
  "takeaway": "overall takeaway in 2-3 sentences",
  "watch_items": ["item 1", "item 2", ...]
}}

Requirements:
- bullets: 4-6 concise bullet points (each under 150 characters) covering key themes, announcements, or market movements
- takeaway: A brief overall assessment (2-3 sentences, under 300 characters) for investors
- watch_items: 2-3 things investors should monitor (optional, can be empty list)

IMPORTANT:
- Respond with ONLY valid, complete JSON
- No markdown code blocks (no ```json or ```)
- No additional text before or after the JSON
- Ensure all JSON strings are properly closed with quotes
- Ensure all arrays and objects are properly closed with brackets

Output the JSON now:"""

    return prompt


def _parse_gemini_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse Gemini's response text into structured data.

    Returns (parsed_data, error_message).
    If parsing fails, returns (None, error_message).
    """
    try:
        # Try to find JSON in the response
        # Sometimes the model may wrap JSON in markdown code blocks
        text = response_text.strip()

        # More robust markdown code block removal
        # Handle various markdown formats
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)

        # Find start of JSON (skip markdown fence)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('```'):
                start_idx = i + 1
                break
            elif stripped.startswith('{'):
                start_idx = i
                break

        # Find end of JSON (skip closing markdown fence)
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped == '```':
                end_idx = i
                break
            elif stripped.endswith('}'):
                end_idx = i + 1
                break

        # Reconstruct JSON
        json_lines = lines[start_idx:end_idx]
        text = '\n'.join(json_lines).strip()

        # If still has markdown artifacts, try simple strip
        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        # Try to fix common JSON truncation issues
        if not text.endswith('}'):
            # JSON might be truncated - try to close it
            # Count opening and closing braces/brackets
            open_braces = text.count('{')
            close_braces = text.count('}')
            open_brackets = text.count('[')
            close_brackets = text.count(']')

            # Add missing closing quotes if needed
            if text.count('"') % 2 != 0:
                text += '"'

            # Close arrays
            for _ in range(open_brackets - close_brackets):
                text += ']'

            # Close objects
            for _ in range(open_braces - close_braces):
                text += '}'

        # Parse JSON
        data = json.loads(text)

        # Validate structure
        if not isinstance(data, dict):
            return None, "Response is not a JSON object"

        bullets = data.get("bullets", [])
        takeaway = data.get("takeaway", "")
        watch_items = data.get("watch_items", [])

        if not isinstance(bullets, list):
            return None, "bullets field must be a list"

        if not isinstance(takeaway, str):
            return None, "takeaway field must be a string"

        if not isinstance(watch_items, list):
            watch_items = []

        # Limit lengths
        bullets = bullets[:6]
        watch_items = watch_items[:3]

        return {
            "bullets": bullets,
            "takeaway": takeaway,
            "watch_items": watch_items,
        }, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error parsing response: {str(e)}"


@_cache_data(show_spinner=False, ttl=600)  # Cache for 10 minutes
def summarize_news(
    items: List[Dict[str, Any]], ticker: str, max_items_for_context: int = 10
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Summarize news items using Gemini API.

    Returns (summary_dict, error_message).
    - If successful: ({"bullets": [...], "takeaway": "...", "watch_items": [...]}, None)
    - If failed: (None, error_message)
    """
    if not items:
        return None, "No news items to summarize"

    # Get Gemini client
    client, error = _get_gemini_client()
    if client is None:
        return None, error

    try:
        # Build prompt
        prompt = _build_prompt(items, ticker, max_items_for_context)

        # Get model name (configurable via GEMINI_MODEL env var)
        model_name = _get_gemini_model()

        # Call Gemini API
        # Note: Some models support response_mime_type for structured output
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more factual output
                    max_output_tokens=4096,  # Increased to avoid truncation
                    response_modalities=["TEXT"],
                    response_mime_type="application/json",  # Request JSON output
                ),
            )
        except Exception:
            # Fallback if response_mime_type not supported
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                    response_modalities=["TEXT"],
                ),
            )

        # Extract text from response
        if not response or not response.text:
            return None, "Empty response from Gemini API"

        response_text = response.text

        # Parse response
        parsed, parse_error = _parse_gemini_response(response_text)
        if parsed is None:
            # If parsing failed, try to extract at least some useful info from raw text
            # Clean up the text and split into sentences
            clean_text = response_text.replace('\n', ' ').strip()

            # Try to extract any complete sentences (ending with period)
            sentences = []
            for s in clean_text.split('.'):
                s = s.strip()
                if s and len(s) > 10:  # Only include substantial sentences
                    # Remove any incomplete quote marks or brackets
                    s = s.replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').strip()
                    if s:
                        sentences.append(s + '.')

            # If no complete sentences, try splitting by common delimiters
            if not sentences:
                for delimiter in ['\n', ',', ';']:
                    parts = [p.strip() for p in clean_text.split(delimiter) if p.strip() and len(p.strip()) > 10]
                    if parts:
                        sentences = parts[:4]
                        break

            bullets = sentences[:4] if sentences else ["AI summary temporarily unavailable due to parsing error."]

            return {
                "bullets": bullets,
                "takeaway": "Summary parsing incomplete. Please refresh for updated results.",
                "watch_items": [],
            }, f"JSON parsing failed: {parse_error}"

        return parsed, None

    except Exception as e:
        return None, f"Gemini API error: {str(e)}"
