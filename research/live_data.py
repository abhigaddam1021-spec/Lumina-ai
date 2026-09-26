"""
research/live_data.py — Live Internet Training Data Generator
==============================================================

This module automatically fetches real-world text from the internet
and converts it into training data the neural network can learn from.

It searches for articles across two (or more) topics and turns the
web results into a labeled dataset — ready to be fed into the network!
"""

from typing import Optional
import numpy as np

from .searcher import search, wikipedia_summary
from .vectorizer import build_dataset, TFIDFVectorizer


# ---------------------------------------------------------------------------
# Live Dataset Builder
# ---------------------------------------------------------------------------

def fetch_live_dataset(
    topics: list[str],
    results_per_topic: int = 8,
    use_wikipedia: bool = True,
    vocab_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str], TFIDFVectorizer, dict]:
    """
    Automatically build a labeled training dataset from live internet data.

    Searches for each topic, gathers the text snippets and optionally 
    Wikipedia summaries, then converts everything to TF-IDF vectors.

    Parameters
    ----------
    topics : list[str]
        List of topic labels to search for (e.g. ["quantum physics", "basketball"]).
        Each topic becomes one class the network learns to recognize.
    results_per_topic : int
        How many web results to gather per topic.
    use_wikipedia : bool
        Whether to also pull a Wikipedia summary for each topic.
    vocab_size : int
        Size of the TF-IDF vocabulary / number of input features.

    Returns
    -------
    X            : np.ndarray [n_samples, vocab_size]
    y            : np.ndarray [n_samples, n_classes]  (one-hot)
    class_names  : list[str]
    vectorizer   : fitted TFIDFVectorizer
    source_log   : dict {topic: [list of fetched texts]}  (for the UI)
    """
    text_groups: dict[str, list[str]] = {}
    source_log:  dict[str, list[str]] = {}

    for topic in topics:
        texts = []
        log   = []

        # 1. Web search results
        results = search(topic, max_results=results_per_topic)
        for r in results:
            combined = f"{r.get('title', '')} {r.get('body', '')}".strip()
            if combined:
                texts.append(combined)
                log.append(f"[WEB] {r.get('title','')[:60]} — {r.get('url','')[:50]}")

        # 2. Wikipedia summary
        if use_wikipedia:
            wiki = wikipedia_summary(topic)
            if wiki:
                texts.append(wiki)
                log.append(f"[WIKI] Wikipedia: {topic}")

        text_groups[topic] = texts
        source_log[topic]  = log
        print(f"  [LIVE] '{topic}': gathered {len(texts)} text samples")

    if not any(text_groups.values()):
        raise RuntimeError("No data fetched! Check your internet connection.")

    X, y, class_names, vectorizer = build_dataset(text_groups, vocab_size=vocab_size)
    return X, y, class_names, vectorizer, source_log


# ---------------------------------------------------------------------------
# Preset Research Themes
# ---------------------------------------------------------------------------

RESEARCH_PRESETS = {
    "Science vs Sports": ["quantum physics", "football"],
    "Tech vs Nature":    ["artificial intelligence", "rainforest ecology"],
    "Space vs Ocean":    ["space exploration", "deep sea creatures"],
    "Custom":            [],  # User fills in their own topics
}
