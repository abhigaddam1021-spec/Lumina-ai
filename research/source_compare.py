"""
research/source_compare.py — Multi-Source Fact Comparison Engine
=================================================================

For any topic, this module fetches summaries from four independent,
authoritative sources and lets the AI (and user) compare them:

  1. Wikipedia          — Crowd-sourced, vast coverage
  2. Britannica         — Professional editorial standard since 1768
  3. .gov site          — Official U.S. government information
  4. "Most reputable"   — Nature / Stanford Encyclopedia / Scientific American
                          (the highest-trust open-access academic sources)

The AI can then be trained on the DIFFERENCES between sources — teaching
it to recognize bias, depth, and authority in information.
"""

import re
import urllib.request
import urllib.parse
import json
import time
from typing import Optional

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

try:
    from ddgs import DDGS
    _HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _HAS_DDGS = True
    except ImportError:
        _HAS_DDGS = False


# ---------------------------------------------------------------------------
# Shared HTML → Text helper
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_and_clean(url: str, max_chars: int = 2500, timeout: int = 12) -> Optional[str]:
    """Download a URL and return its clean readable text."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode(errors="replace")

        if _HAS_BS4:
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form", "button"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        else:
            text = re.sub(r"<[^>]+>", " ", raw)

        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars] if text else None
    except Exception as e:
        return None


def _site_search(query: str, site: str, max_results: int = 3) -> list[dict]:
    """Use DuckDuckGo to find pages on a specific site."""
    if not _HAS_DDGS:
        return []
    full_query = f"site:{site} {query}"
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(full_query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url":   r.get("href", r.get("url", "")),
                "body":  r.get("body", ""),
            }
            for r in raw
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 1: Wikipedia
# ---------------------------------------------------------------------------

def fetch_wikipedia(topic: str) -> dict:
    """
    Fetch a Wikipedia article summary using the REST API.

    Returns
    -------
    dict with keys: source, url, text, status
    """
    try:
        encoded = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        extract = data.get("extract", "")
        wiki_url = data.get("content_urls", {}).get("desktop", {}).get("page", url)

        return {
            "source":      "Wikipedia",
            "icon":        "📖",
            "url":         wiki_url,
            "text":        extract,
            "status":      "ok" if extract else "empty",
            "credibility": "Community-edited, 6.7M+ articles, generally reliable for overviews",
        }
    except Exception as e:
        return {
            "source": "Wikipedia", "icon": "📖", "url": "",
            "text": None, "status": f"error: {e}",
            "credibility": "Community-edited",
        }


# ---------------------------------------------------------------------------
# Source 2: Britannica
# ---------------------------------------------------------------------------

def fetch_britannica(topic: str) -> dict:
    """
    Fetch the Britannica article summary for a topic.

    Britannica offers free article previews; we grab the visible text.
    """
    # Search Britannica directly
    slug = topic.lower().replace(" ", "-")
    direct_url = f"https://www.britannica.com/search?query={urllib.parse.quote_plus(topic)}"

    # Try to find the article via DDG site-search first
    results = _site_search(topic, "britannica.com", max_results=2)
    article_url = results[0]["url"] if results else direct_url

    text = _fetch_and_clean(article_url, max_chars=2500)

    # Britannica pages have a lot of nav noise — try to isolate the article body
    if text:
        # Heuristic: article text usually starts after the topic title appears
        idx = text.lower().find(topic.lower().split()[0])
        if idx > 200:                    # skip header junk
            text = text[idx:idx + 2000]
        else:
            text = text[:2000]

    return {
        "source":      "Britannica",
        "icon":        "📚",
        "url":         article_url,
        "text":        text,
        "status":      "ok" if text else "blocked/empty",
        "credibility": "Professional editorial team, expert-written since 1768",
    }


# ---------------------------------------------------------------------------
# Source 3: .gov site
# ---------------------------------------------------------------------------

# Map common topics to the best .gov agency
_GOV_AGENCY_HINTS = {
    "health":    "cdc.gov",
    "disease":   "cdc.gov",
    "medicine":  "nih.gov",
    "drug":      "fda.gov",
    "space":     "nasa.gov",
    "climate":   "noaa.gov",
    "weather":   "noaa.gov",
    "energy":    "energy.gov",
    "history":   "loc.gov",
    "law":       "usa.gov",
    "education": "ed.gov",
    "science":   "science.gov",
    "nutrition": "nutrition.gov",
    "nature":    "nps.gov",
    "ocean":     "noaa.gov",
    "geology":   "usgs.gov",
    "finance":   "treasury.gov",
}


def _pick_gov_domain(topic: str) -> str:
    """Pick the most relevant .gov domain for the topic."""
    tl = topic.lower()
    for keyword, domain in _GOV_AGENCY_HINTS.items():
        if keyword in tl:
            return domain
    return ".gov"  # broad site:*.gov fallback


def fetch_gov(topic: str) -> dict:
    """
    Search .gov sites for the topic and return the best match.
    """
    domain = _pick_gov_domain(topic)

    # Try the specific agency first, then fall back to broad .gov
    results = _site_search(topic, domain, max_results=3)
    if not results:
        results = _site_search(topic, ".gov", max_results=3)

    if not results:
        return {
            "source": ".gov", "icon": "🏛️", "url": "",
            "text": None, "status": "no results",
            "credibility": "Official U.S. government agencies",
        }

    best = results[0]
    text = _fetch_and_clean(best["url"], max_chars=2500) or best["body"]

    return {
        "source":      f".gov ({domain})",
        "icon":        "🏛️",
        "url":         best["url"],
        "text":        text,
        "status":      "ok" if text else "empty",
        "credibility": "Official U.S. government source — authoritative for policy & science",
    }


# ---------------------------------------------------------------------------
# Source 4: Most Reputable Academic / Science Source
# ---------------------------------------------------------------------------

# Ranked by trust / peer-review status
_REPUTABLE_SOURCES = [
    ("nature.com",                "Nature (peer-reviewed journal, est. 1869)"),
    ("scientificamerican.com",    "Scientific American (expert science journalism)"),
    ("plato.stanford.edu",        "Stanford Encyclopedia of Philosophy (peer-reviewed)"),
    ("ncbi.nlm.nih.gov/pmc",      "PubMed Central (open-access biomedical research)"),
    ("scholar.google.com",        "Google Scholar (academic literature index)"),
    ("newscientist.com",          "New Scientist (science journalism)"),
    ("theconversation.com",       "The Conversation (academics writing for public)"),
]


def fetch_reputable(topic: str) -> dict:
    """
    Try each top-tier reputable source in order and return the first hit.
    """
    for domain, description in _REPUTABLE_SOURCES:
        results = _site_search(topic, domain, max_results=2)
        if results:
            best = results[0]
            text = _fetch_and_clean(best["url"], max_chars=2500) or best["body"]
            if text and len(text) > 100:
                return {
                    "source":      description.split("(")[0].strip(),
                    "icon":        "🔬",
                    "url":         best["url"],
                    "text":        text,
                    "status":      "ok",
                    "credibility": description,
                }

    return {
        "source": "Reputable Academic", "icon": "🔬", "url": "",
        "text": None, "status": "no results",
        "credibility": "Nature / Scientific American / Stanford / PubMed",
    }


# ---------------------------------------------------------------------------
# Master: compare_sources()
# ---------------------------------------------------------------------------

SourceResult = dict   # typed alias for clarity


def compare_sources(topic: str, delay: float = 0.5) -> list[SourceResult]:
    """
    Fetch information about a topic from all four authoritative sources.

    Parameters
    ----------
    topic : str
        The topic to research (e.g. "black holes", "vaccines", "democracy").
    delay : float
        Seconds to wait between requests to be polite to servers.

    Returns
    -------
    list[dict]
        One dict per source, each with keys:
          source, icon, url, text, status, credibility
    """
    fetchers = [
        ("Wikipedia",        fetch_wikipedia),
        ("Britannica",       fetch_britannica),
        (".gov",             fetch_gov),
        ("Reputable Source", fetch_reputable),
    ]

    results = []
    for name, fn in fetchers:
        print(f"  [COMPARE] Fetching {name}...")
        result = fn(topic)
        results.append(result)
        time.sleep(delay)   # polite pause between requests

    return results


# ---------------------------------------------------------------------------
# Text similarity: how much do two sources agree?
# ---------------------------------------------------------------------------

def word_overlap_score(text_a: str, text_b: str) -> float:
    """
    Compute a simple word-overlap similarity score (Jaccard index).
    Returns 0.0 (no overlap) to 1.0 (identical).
    """
    def tokenize(t: str) -> set:
        return set(re.findall(r"[a-z]{4,}", t.lower()))

    if not text_a or not text_b:
        return 0.0

    words_a = tokenize(text_a)
    words_b = tokenize(text_b)
    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    union        = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def build_agreement_matrix(results: list[SourceResult]) -> list[list]:
    """
    Build a pairwise similarity matrix between all fetched sources.

    Returns
    -------
    list[list[float]]  (n_sources × n_sources)
    """
    n = len(results)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = word_overlap_score(
                    results[i].get("text") or "",
                    results[j].get("text") or "",
                )
    return matrix
