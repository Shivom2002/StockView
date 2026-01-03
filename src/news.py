from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
import yfinance as yf


try:
    _cache_data = st.cache_data
except AttributeError:  # Streamlit < 1.18
    _cache_data = st.cache


def normalize_news_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a news item from yfinance to a consistent format.

    Returns None if the item is invalid or missing critical fields.
    """
    if not isinstance(item, dict):
        return None

    # yfinance news structure has changed - data is nested under 'content'
    content = item.get("content", {})
    if not content:
        # Fallback: try old structure
        content = item

    # Extract title
    title = content.get("title", "").strip()
    if not title:
        return None

    # Extract link - try multiple locations
    link = None
    canonical_url = content.get("canonicalUrl")
    click_through_url = content.get("clickThroughUrl")

    if canonical_url and isinstance(canonical_url, dict):
        link = canonical_url.get("url")
    elif click_through_url and isinstance(click_through_url, dict):
        link = click_through_url.get("url")
    else:
        # Fallback for old structure
        link = content.get("link") or content.get("url", "")

    if not link:
        return None

    # Extract publisher
    provider = content.get("provider", {})
    if isinstance(provider, dict):
        publisher = provider.get("displayName", "Unknown")
    else:
        publisher = content.get("publisher", "Unknown")

    # Extract timestamp
    published_at = None

    # Try ISO format first (new structure)
    pub_date = content.get("pubDate")
    if pub_date:
        try:
            # Parse ISO format: "2026-01-02T19:11:13Z"
            published_at = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    # Try UNIX timestamp (old structure)
    if published_at is None:
        timestamp_raw = content.get("providerPublishTime")
        if timestamp_raw:
            try:
                published_at = datetime.fromtimestamp(timestamp_raw, tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                pass

    # If we couldn't parse timestamp, skip this item
    if published_at is None:
        return None

    # Extract summary/snippet if available
    summary = content.get("summary") or content.get("description", "")

    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "published_at": published_at,
        "summary": summary.strip() if summary else "",
        "source": "yfinance",
    }


def filter_recent_news(items: List[Dict[str, Any]], max_age_hours: int = 72) -> List[Dict[str, Any]]:
    """
    Filter news items to only include those within max_age_hours.

    Returns items sorted by published_at descending (most recent first).
    """
    if not items:
        return []

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    recent = []
    for item in items:
        published_at = item.get("published_at")
        if published_at and published_at >= cutoff_time:
            recent.append(item)

    # Sort by published_at descending
    recent.sort(key=lambda x: x.get("published_at", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

    return recent


def pick_top_links(items: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Pick the top k most recent news items.

    Assumes items are already sorted by recency.
    """
    return items[:k]


@_cache_data(show_spinner=False, ttl=600)  # Cache for 10 minutes
def fetch_news_yfinance(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch news for a ticker using yfinance.

    Returns a list of normalized news items, or empty list on error.
    """
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news

        if not raw_news:
            return []

        normalized = []
        for item in raw_news:
            normalized_item = normalize_news_item(item)
            if normalized_item:
                normalized.append(normalized_item)

        return normalized

    except Exception as e:
        # Return empty list on any error
        return []
