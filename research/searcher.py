"""
research/searcher.py — Live Internet Access Module
====================================================

This module gives the AI real internet access by:
  1. Querying DuckDuckGo (free, no API key required)
  2. Fetching full page content from any URL
  3. Pulling structured summaries from Wikipedia

The results are returned as clean text, which the vectorizer
then converts into numbers the neural network can process.
"""

import re
import urllib.request
import urllib.parse
import json
from typing import Optional

# Try importing the fast DDG library; fall back to urllib if not installed
try:
    from ddgs import DDGS
    _HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _HAS_DDGS = True
    except ImportError:
        _HAS_DDGS = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False


# ---------------------------------------------------------------------------
# DuckDuckGo Search
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo and return a list of results.

    Each result is a dict with keys:
      - 'title'  : Page title
      - 'url'    : Full URL
      - 'body'   : Short snippet / description

    Parameters
    ----------
    query : str
        The search query.
    max_results : int
        How many results to return (default 5).

    Returns
    -------
    list[dict]
        List of result dicts. Empty list on failure.
    """
    if _HAS_DDGS:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            # Normalize key names (DDG uses 'href' not 'url')
            normalized = []
            for r in results:
                normalized.append({
                    "title": r.get("title", ""),
                    "url":   r.get("href", r.get("url", "")),
                    "body":  r.get("body", ""),
                })
            return normalized
        except Exception as e:
            print(f"[SEARCH] DDG error: {e}")
            return []
    else:
        # Fallback: DuckDuckGo Instant Answer API (JSON, no key needed)
        return _ddg_instant(query, max_results)


def _ddg_instant(query: str, max_results: int) -> list[dict]:
    """Fallback: use DuckDuckGo's public Instant Answer JSON API."""
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        results = []
        # Related topics from the instant answer
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:60],
                    "url":   topic.get("FirstURL", ""),
                    "body":  topic.get("Text", ""),
                })
        return results
    except Exception as e:
        print(f"[SEARCH] Instant API error: {e}")
        return []


# ---------------------------------------------------------------------------
# Wikipedia Summary Fetch
# ---------------------------------------------------------------------------

def wikipedia_summary(topic: str, sentences: int = 5) -> Optional[str]:
    """
    Fetch a plain-text summary of a Wikipedia article.

    Uses Wikipedia's public REST API — no key needed.

    Parameters
    ----------
    topic : str
        The topic or article title to look up.
    sentences : int
        (Unused by Wikipedia REST, but kept for interface consistency.)

    Returns
    -------
    str or None
        The article summary text, or None if not found.
    """
    try:
        encoded = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "DeepMindAI-Research/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        return data.get("extract", None)
    except Exception as e:
        print(f"[WIKI] Error fetching '{topic}': {e}")
        return None


# ---------------------------------------------------------------------------
# Full Page Fetch + Clean
# ---------------------------------------------------------------------------

def fetch_page(url: str, max_chars: int = 3000) -> Optional[str]:
    """
    Download a web page and extract its readable text content.

    Parameters
    ----------
    url : str
        The URL to fetch.
    max_chars : int
        Maximum number of characters to return (default 3000).

    Returns
    -------
    str or None
        Cleaned plain text, or None on failure.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode(errors="replace")

        if _HAS_BS4:
            soup = BeautifulSoup(raw, "html.parser")
            # Remove scripts, styles, nav
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        else:
            # Simple regex fallback: strip all HTML tags
            text = re.sub(r"<[^>]+>", " ", raw)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    except Exception as e:
        print(f"[FETCH] Error fetching '{url}': {e}")
        return None


# ---------------------------------------------------------------------------
# Convenience: Search + Fetch full content
# ---------------------------------------------------------------------------

def research(query: str, max_results: int = 3, fetch_full: bool = False) -> list[dict]:
    """
    High-level research call: search DuckDuckGo, optionally fetch full pages.

    Parameters
    ----------
    query : str
        Research query.
    max_results : int
        Number of results to retrieve.
    fetch_full : bool
        If True, also download and clean the full page text for each result.

    Returns
    -------
    list[dict]
        Each dict has 'title', 'url', 'body', and optionally 'full_text'.
    """
    results = search(query, max_results=max_results)

    if fetch_full:
        for r in results:
            if r.get("url"):
                r["full_text"] = fetch_page(r["url"])
            else:
                r["full_text"] = None

    return results
