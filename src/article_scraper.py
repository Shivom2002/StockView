from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st

# Try to import scraping libraries
# These are optional - the app works without them, just won't have full article content
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    Article = None

try:
    from bs4 import BeautifulSoup
    import requests
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None
    requests = None

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


try:
    _cache_data = st.cache_data
except AttributeError:  # Streamlit < 1.18
    _cache_data = st.cache


def is_paywall_or_blocked(url: str, content: str) -> bool:
    """
    Detect common paywall or anti-scraping patterns.

    Args:
        url: The article URL
        content: The scraped content

    Returns:
        True if content appears to be blocked/paywalled, False otherwise
    """
    if not content:
        return True

    # Check for minimal content (likely blocked)
    if len(content.strip()) < 100:
        return True

    # Common paywall indicators
    paywall_keywords = [
        "subscribe",
        "subscription required",
        "paywall",
        "premium content",
        "sign in to continue",
        "register to read",
        "403 forbidden",
        "access denied",
        "please disable your ad blocker"
    ]

    content_lower = content.lower()
    for keyword in paywall_keywords:
        if keyword in content_lower:
            return True

    return False


@_cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def scrape_article_content(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Scrape full article content from URL with fallback handling.

    Uses newspaper3k first, falls back to BeautifulSoup if that fails.

    Args:
        url: The article URL to scrape
        timeout: Timeout in seconds for HTTP requests

    Returns:
        {
            "success": bool,
            "content": str,  # Full article text
            "title": str,
            "authors": List[str],
            "publish_date": datetime or None,
            "error": Optional[str]
        }
    """
    result = {
        "success": False,
        "content": "",
        "title": "",
        "authors": [],
        "publish_date": None,
        "error": None
    }

    if not url:
        result["error"] = "No URL provided"
        return result

    # Try newspaper3k first (best for news articles)
    if NEWSPAPER_AVAILABLE:
        try:
            article = Article(url)
            article.download()
            article.parse()

            content = article.text

            # Check for paywall/blocking
            if is_paywall_or_blocked(url, content):
                result["error"] = "Content appears to be paywalled or blocked"
            else:
                result["success"] = True
                result["content"] = content
                result["title"] = article.title
                result["authors"] = article.authors
                result["publish_date"] = article.publish_date
                return result

        except Exception as e:
            logger.warning(f"newspaper3k failed for {url}: {str(e)}")
            result["error"] = f"newspaper3k error: {str(e)}"

    # Fallback to BeautifulSoup
    if BS4_AVAILABLE and requests:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')

            # Try common article selectors
            article_content = None
            selectors = [
                'article',
                'div[class*="article"]',
                'div[class*="content"]',
                'div[class*="story"]',
                'div[id*="article"]',
                'main',
            ]

            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    # Get all paragraph tags within the element
                    paragraphs = elements[0].find_all('p')
                    if paragraphs:
                        article_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        break

            # If no article content found, try all paragraphs
            if not article_content:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    article_content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

            if article_content and not is_paywall_or_blocked(url, article_content):
                result["success"] = True
                result["content"] = article_content

                # Try to extract title
                title_tag = soup.find('title') or soup.find('h1')
                if title_tag:
                    result["title"] = title_tag.get_text().strip()

                return result
            else:
                result["error"] = "No content found or content is blocked"

        except requests.Timeout:
            result["error"] = f"Request timeout after {timeout} seconds"
        except requests.HTTPError as e:
            result["error"] = f"HTTP error: {e.response.status_code}"
        except Exception as e:
            logger.warning(f"BeautifulSoup failed for {url}: {str(e)}")
            result["error"] = f"BeautifulSoup error: {str(e)}"

    # If we get here, both methods failed
    if not BS4_AVAILABLE and not NEWSPAPER_AVAILABLE:
        result["error"] = "No scraping libraries available (install newspaper3k or beautifulsoup4)"
    elif not result["error"]:
        result["error"] = "All scraping methods failed"

    return result


def scrape_multiple_articles(
    news_items: List[Dict[str, Any]],
    max_workers: int = 5,
    max_articles: int = 15
) -> List[Dict[str, Any]]:
    """
    Scrape multiple articles in parallel using ThreadPoolExecutor.

    Adds 'full_content' field to each news item.
    If scraping fails, full_content will be None.

    Args:
        news_items: List of news items with 'link' field
        max_workers: Number of parallel workers
        max_articles: Maximum number of articles to scrape

    Returns:
        List of news items with added 'full_content' field
    """
    if not news_items:
        return []

    # Limit to top N articles
    items_to_scrape = news_items[:max_articles]

    # Prepare results list
    enriched_items = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping jobs
        future_to_item = {
            executor.submit(scrape_article_content, item.get('link', '')): item
            for item in items_to_scrape
        }

        # Collect results as they complete
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                scrape_result = future.result(timeout=15)  # Overall timeout per article

                if scrape_result.get('success'):
                    item['full_content'] = scrape_result.get('content')
                    item['scraping_success'] = True
                else:
                    item['full_content'] = None
                    item['scraping_success'] = False
                    item['scraping_error'] = scrape_result.get('error')

            except Exception as e:
                logger.warning(f"Exception scraping {item.get('link', 'unknown')}: {str(e)}")
                item['full_content'] = None
                item['scraping_success'] = False
                item['scraping_error'] = f"Scraping exception: {str(e)}"

            enriched_items.append(item)

    # Sort back to original order (futures may complete out of order)
    # Match by link
    sorted_items = []
    for original_item in items_to_scrape:
        for enriched_item in enriched_items:
            if enriched_item.get('link') == original_item.get('link'):
                sorted_items.append(enriched_item)
                break
        else:
            # If not found in enriched (shouldn't happen), add original
            original_item['full_content'] = None
            original_item['scraping_success'] = False
            sorted_items.append(original_item)

    return sorted_items
