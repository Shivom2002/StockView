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
    default_model = "gemini-3-flash-preview"
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


def _build_article_sentiment_prompt(article: Dict[str, Any], ticker: str) -> str:
    """
    Build a prompt for Gemini to analyze a single article with sentiment and entity extraction.
    """
    title = article.get("title", "")
    publisher = article.get("publisher", "Unknown")
    published_at = article.get("published_at")
    summary = article.get("summary", "")
    full_content = article.get("full_content")

    time_str = published_at.strftime("%Y-%m-%d %H:%M UTC") if published_at else "Unknown"

    # Use full content if available, otherwise use snippet
    content_to_analyze = full_content if full_content else summary
    content_label = "Article Content" if full_content else "Article Snippet"

    prompt = f"""You are analyzing a news article about {ticker} for investment decision-making.

Article Title: {title}
Publisher: {publisher}
Published: {time_str}
{content_label}: {content_to_analyze}

Provide analysis in JSON format:
{{
  "sentiment_score": <float from -1.0 to 1.0>,
  "sentiment_label": <one of: "Very Negative", "Negative", "Neutral", "Positive", "Very Positive">,
  "entities": {{
    "partnerships": [list any partnerships, collaborations, or deals mentioned],
    "products": [list product launches, updates, or announcements],
    "people": [list key executives, board members, or influential people],
    "competitors": [list competitors or competitive dynamics mentioned],
    "regulatory": [list regulatory approvals, investigations, or policy changes]
  }},
  "key_points": [2-3 most important takeaways for investors],
  "relevance_score": <0-10 score for how much this could impact stock price>
}}

Scoring guidelines:
- sentiment_score: -1.0 = very bearish (major negative news), 0 = neutral, +1.0 = very bullish (major positive news)
- sentiment_label:
  - "Very Negative": sentiment_score < -0.6
  - "Negative": -0.6 <= sentiment_score < -0.2
  - "Neutral": -0.2 <= sentiment_score <= 0.2
  - "Positive": 0.2 < sentiment_score <= 0.6
  - "Very Positive": sentiment_score > 0.6
- relevance_score: 0 = no impact, 5 = moderate impact, 10 = major catalyst

Focus on stock price impact, not general news tone.

IMPORTANT:
- Respond with ONLY valid, complete JSON
- No markdown code blocks (no ```json or ```)
- No additional text before or after the JSON
- Ensure all JSON strings are properly closed with quotes
- Ensure all arrays and objects are properly closed with brackets

Output the JSON now:"""

    return prompt


def _parse_article_sentiment_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse Gemini's response for article sentiment analysis.

    Returns (parsed_data, error_message).
    """
    try:
        # Similar parsing logic to _parse_gemini_response
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

        sentiment_score = data.get("sentiment_score", 0)
        sentiment_label = data.get("sentiment_label", "Neutral")
        entities = data.get("entities", {})
        key_points = data.get("key_points", [])
        relevance_score = data.get("relevance_score", 0)

        # Validate types
        if not isinstance(sentiment_score, (int, float)):
            return None, "sentiment_score must be a number"

        if not isinstance(entities, dict):
            entities = {}

        if not isinstance(key_points, list):
            key_points = []

        # Clamp values
        sentiment_score = max(-1.0, min(1.0, float(sentiment_score)))
        relevance_score = max(0, min(10, int(relevance_score)))

        # Limit list lengths
        key_points = key_points[:3]

        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "entities": entities,
            "key_points": key_points,
            "relevance_score": relevance_score,
        }, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error parsing response: {str(e)}"


@_cache_data(show_spinner=False, ttl=21600)  # Cache for 6 hours
def analyze_article_with_sentiment(
    article: Dict[str, Any],
    ticker: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Analyze single article with sentiment and entity extraction.

    Args:
        article: Article dict with title, summary, link, full_content (optional)
        ticker: Stock ticker symbol

    Returns:
        ({
            "sentiment_score": float (-1.0 to 1.0),
            "sentiment_label": str,
            "entities": dict,
            "key_points": list,
            "relevance_score": int (0-10)
        }, error_message)
    """
    if not article:
        return None, "No article provided"

    # Get Gemini client
    client, error = _get_gemini_client()
    if client is None:
        return None, error

    try:
        # Build prompt
        prompt = _build_article_sentiment_prompt(article, ticker)

        # Get model name
        model_name = _get_gemini_model()

        # Call Gemini API
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
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
                    temperature=0.3,
                    max_output_tokens=2048,
                    response_modalities=["TEXT"],
                ),
            )

        # Extract text from response
        if not response or not response.text:
            return None, "Empty response from Gemini API"

        response_text = response.text

        # Parse response
        parsed, parse_error = _parse_article_sentiment_response(response_text)
        if parsed is None:
            return None, parse_error

        return parsed, None

    except Exception as e:
        return None, f"Gemini API error: {str(e)}"


def _build_aggregate_sentiment_prompt(analyzed_articles: List[Dict[str, Any]], ticker: str) -> str:
    """
    Build a prompt for Gemini to create aggregate summary with sentiment analysis.
    """
    # Calculate sentiment statistics
    sentiment_scores = [
        article.get('analysis', {}).get('sentiment_score', 0)
        for article in analyzed_articles
        if article.get('analysis')
    ]

    if not sentiment_scores:
        avg_sentiment = 0
        positive_count = 0
        neutral_count = 0
        negative_count = 0
    else:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        positive_count = sum(1 for s in sentiment_scores if s > 0.2)
        neutral_count = sum(1 for s in sentiment_scores if -0.2 <= s <= 0.2)
        negative_count = sum(1 for s in sentiment_scores if s < -0.2)

    # Build article summaries section
    articles_text = []
    for i, article in enumerate(analyzed_articles, 1):
        title = article.get("title", "")
        published_at = article.get("published_at")
        time_str = published_at.strftime("%Y-%m-%d") if published_at else "Unknown"

        analysis = article.get("analysis", {})
        sentiment_score = analysis.get("sentiment_score", 0)
        key_points = analysis.get("key_points", [])
        entities = analysis.get("entities", {})

        article_entry = f"{i}. {title} ({time_str})\n   Sentiment: {sentiment_score:.2f}"

        if key_points:
            points_str = "; ".join(key_points[:2])
            article_entry += f"\n   Key Points: {points_str}"

        # Include entities if present
        entity_mentions = []
        for category, items in entities.items():
            if items and category in ['partnerships', 'products', 'regulatory']:
                entity_mentions.extend(items[:2])
        if entity_mentions:
            article_entry += f"\n   Mentions: {', '.join(entity_mentions[:3])}"

        articles_text.append(article_entry)

    articles_section = "\n\n".join(articles_text[:15])  # Limit to 15 articles

    prompt = f"""You are summarizing recent news about {ticker} based on {len(analyzed_articles)} analyzed articles with sentiment scores.

Per-article summaries:

{articles_section}

Sentiment statistics:
- Average sentiment: {avg_sentiment:.2f}
- Positive articles (>0.2): {positive_count}
- Neutral articles (-0.2 to 0.2): {neutral_count}
- Negative articles (<-0.2): {negative_count}

Provide aggregate analysis in JSON format:
{{
  "sentiment_trend": <"Improving", "Declining", or "Stable">,
  "bullets": [4-6 key themes across all articles, each under 150 characters],
  "takeaway": "2-3 sentence overall assessment for investors (under 300 characters)",
  "watch_items": [2-3 critical developments to monitor],
  "top_entities": {{
    "partnerships": [most significant partnerships across all articles],
    "products": [most significant product developments],
    "people": [most frequently mentioned key people],
    "competitors": [most relevant competitors],
    "regulatory": [most important regulatory developments]
  }}
}}

Instructions:
- Identify overarching themes, not individual article summaries
- Consider sentiment trend: compare older vs newer articles to determine if improving/declining/stable
- Highlight contrasting perspectives if sentiment varies significantly
- Focus on actionable insights for investors
- For top_entities, deduplicate and prioritize the most significant mentions across all articles
- Each entity category can have 0-3 items (empty list if none found)

IMPORTANT:
- Respond with ONLY valid, complete JSON
- No markdown code blocks (no ```json or ```)
- No additional text before or after the JSON
- Ensure all JSON strings are properly closed with quotes
- Ensure all arrays and objects are properly closed with brackets

Output the JSON now:"""

    return prompt


def _parse_aggregate_sentiment_response(
    response_text: str,
    analyzed_articles: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse Gemini's response for aggregate sentiment summary.

    Returns (parsed_data, error_message).
    """
    try:
        # Similar parsing logic to _parse_gemini_response
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

        # Calculate aggregate sentiment from analyzed articles
        sentiment_scores = [
            article.get('analysis', {}).get('sentiment_score', 0)
            for article in analyzed_articles
            if article.get('analysis')
        ]
        aggregate_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        # Calculate sentiment distribution
        positive_count = sum(1 for s in sentiment_scores if s > 0.2)
        neutral_count = sum(1 for s in sentiment_scores if -0.2 <= s <= 0.2)
        negative_count = sum(1 for s in sentiment_scores if s < -0.2)

        # Extract fields
        sentiment_trend = data.get("sentiment_trend", "Stable")
        bullets = data.get("bullets", [])
        takeaway = data.get("takeaway", "")
        watch_items = data.get("watch_items", [])
        top_entities = data.get("top_entities", {})

        # Validate types
        if not isinstance(bullets, list):
            bullets = []
        if not isinstance(watch_items, list):
            watch_items = []
        if not isinstance(top_entities, dict):
            top_entities = {}

        # Limit lengths
        bullets = bullets[:6]
        watch_items = watch_items[:3]

        return {
            "aggregate_sentiment": aggregate_sentiment,
            "sentiment_trend": sentiment_trend,
            "bullets": bullets,
            "takeaway": takeaway,
            "watch_items": watch_items,
            "sentiment_distribution": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count,
            },
            "top_entities": top_entities,
        }, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error parsing response: {str(e)}"


@_cache_data(show_spinner=False, ttl=600)  # Cache for 10 minutes
def summarize_news_with_sentiment(
    analyzed_articles: List[Dict[str, Any]],
    ticker: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate aggregate summary with sentiment analysis.

    Args:
        analyzed_articles: List of articles with 'analysis' field from analyze_article_with_sentiment
        ticker: Stock ticker symbol

    Returns:
        ({
            "aggregate_sentiment": float,
            "sentiment_trend": str,
            "bullets": list,
            "takeaway": str,
            "watch_items": list,
            "sentiment_distribution": dict,
            "top_entities": dict
        }, error_message)
    """
    if not analyzed_articles:
        return None, "No articles to summarize"

    # Filter to articles with analysis
    articles_with_analysis = [a for a in analyzed_articles if a.get('analysis')]
    if not articles_with_analysis:
        return None, "No analyzed articles available"

    # Get Gemini client
    client, error = _get_gemini_client()
    if client is None:
        return None, error

    try:
        # Build prompt
        prompt = _build_aggregate_sentiment_prompt(articles_with_analysis, ticker)

        # Get model name
        model_name = _get_gemini_model()

        # Call Gemini API
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,  # Slightly higher for narrative synthesis
                    max_output_tokens=4096,
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
                    max_output_tokens=4096,
                    response_modalities=["TEXT"],
                ),
            )

        # Extract text from response
        if not response or not response.text:
            return None, "Empty response from Gemini API"

        response_text = response.text

        # Parse response
        parsed, parse_error = _parse_aggregate_sentiment_response(response_text, articles_with_analysis)
        if parsed is None:
            return None, parse_error

        return parsed, None

    except Exception as e:
        return None, f"Gemini API error: {str(e)}"


def _build_direct_sentiment_prompt(articles: List[Dict[str, Any]], ticker: str) -> str:
    """
    Build prompt for direct analysis of raw articles (single API call).
    """
    # Build article summaries
    articles_text = []
    for i, article in enumerate(articles, 1):
        title = article.get("title", "")
        publisher = article.get("publisher", "Unknown")
        published_at = article.get("published_at")
        time_str = published_at.strftime("%Y-%m-%d %H:%M UTC") if published_at else "Unknown"

        # Use full content if available, otherwise snippet
        full_content = article.get("full_content")
        summary = article.get("summary", "")
        content = full_content if full_content else summary
        content_type = "Full Article" if full_content else "Snippet"

        article_entry = f"{i}. {title}\n   Publisher: {publisher}\n   Published: {time_str}\n   {content_type}: {content[:1000]}..."  # Limit to 1000 chars per article
        articles_text.append(article_entry)

    articles_section = "\n\n".join(articles_text[:15])  # Limit to 15 articles

    prompt = f"""You are analyzing recent news about {ticker} for an investor dashboard. Analyze these {len(articles)} articles and provide comprehensive sentiment analysis with entity extraction.

Articles:

{articles_section}

Provide analysis in JSON format:
{{
  "aggregate_sentiment": <float from -1.0 to 1.0>,
  "sentiment_trend": <"Improving", "Declining", or "Stable">,
  "sentiment_distribution": {{
    "positive": <count of positive articles>,
    "neutral": <count of neutral articles>,
    "negative": <count of negative articles>
  }},
  "bullets": [4-6 key themes across all articles, each under 150 characters],
  "takeaway": "2-3 sentence overall assessment for investors (under 300 characters)",
  "watch_items": [2-3 critical developments to monitor],
  "top_entities": {{
    "partnerships": [most significant partnerships, collaborations, or deals mentioned],
    "products": [most significant product launches, updates, or announcements],
    "people": [most frequently mentioned key executives, board members, or influential people],
    "competitors": [most relevant competitors or competitive dynamics mentioned],
    "regulatory": [most important regulatory approvals, investigations, or policy changes]
  }}
}}

Guidelines:
- aggregate_sentiment: -1.0 = very bearish overall, 0 = neutral, +1.0 = very bullish overall
- sentiment_distribution: Count how many articles are positive (>0.2), neutral (-0.2 to 0.2), or negative (<-0.2)
- sentiment_trend: Compare older vs newer articles - is sentiment improving, declining, or stable over time?
- bullets: Identify overarching themes, not individual article summaries
- takeaway: Overall assessment focusing on actionable insights for investors
- watch_items: Critical developments to monitor going forward
- top_entities: Extract and deduplicate the most significant mentions across all articles (0-3 items per category)

IMPORTANT:
- Respond with ONLY valid, complete JSON
- No markdown code blocks (no ```json or ```)
- No additional text before or after the JSON
- Ensure all JSON strings are properly closed with quotes
- Ensure all arrays and objects are properly closed with brackets

Output the JSON now:"""

    return prompt


def _parse_direct_sentiment_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse Gemini's response for direct sentiment analysis.
    """
    try:
        # Similar parsing logic
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

        # Try to fix truncation
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

        if not isinstance(data, dict):
            return None, "Response is not a JSON object"

        # Extract and validate fields
        aggregate_sentiment = data.get("aggregate_sentiment", 0)
        sentiment_trend = data.get("sentiment_trend", "Stable")
        sentiment_distribution = data.get("sentiment_distribution", {})
        bullets = data.get("bullets", [])
        takeaway = data.get("takeaway", "")
        watch_items = data.get("watch_items", [])
        top_entities = data.get("top_entities", {})

        # Validate types
        if not isinstance(aggregate_sentiment, (int, float)):
            aggregate_sentiment = 0
        aggregate_sentiment = max(-1.0, min(1.0, float(aggregate_sentiment)))

        if not isinstance(bullets, list):
            bullets = []
        if not isinstance(watch_items, list):
            watch_items = []
        if not isinstance(sentiment_distribution, dict):
            sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        if not isinstance(top_entities, dict):
            top_entities = {}

        # Limit lengths
        bullets = bullets[:6]
        watch_items = watch_items[:3]

        return {
            "aggregate_sentiment": aggregate_sentiment,
            "sentiment_trend": sentiment_trend,
            "sentiment_distribution": sentiment_distribution,
            "bullets": bullets,
            "takeaway": takeaway,
            "watch_items": watch_items,
            "top_entities": top_entities,
        }, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error parsing response: {str(e)}"


@_cache_data(show_spinner=False, ttl=600)  # Cache for 10 minutes
def summarize_news_with_sentiment_direct(
    articles: List[Dict[str, Any]],
    ticker: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate aggregate summary with sentiment analysis directly from raw articles.
    Single API call - much more efficient than per-article analysis.

    Args:
        articles: List of raw articles (with optional full_content field)
        ticker: Stock ticker symbol

    Returns:
        ({
            "aggregate_sentiment": float,
            "sentiment_trend": str,
            "sentiment_distribution": dict,
            "bullets": list,
            "takeaway": str,
            "watch_items": list,
            "top_entities": dict
        }, error_message)
    """
    if not articles:
        return None, "No articles to summarize"

    # Get Gemini client
    client, error = _get_gemini_client()
    if client is None:
        return None, error

    try:
        # Build prompt
        prompt = _build_direct_sentiment_prompt(articles, ticker)

        # Get model name
        model_name = _get_gemini_model()

        # Call Gemini API
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=4096,
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
                    max_output_tokens=4096,
                    response_modalities=["TEXT"],
                ),
            )

        # Extract text from response
        if not response or not response.text:
            return None, "Empty response from Gemini API"

        response_text = response.text

        # Parse response
        parsed, parse_error = _parse_direct_sentiment_response(response_text)
        if parsed is None:
            return None, parse_error

        return parsed, None

    except Exception as e:
        return None, f"Gemini API error: {str(e)}"
