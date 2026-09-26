"""
research/vectorizer.py — Text → Numbers for the Neural Network
================================================================

The neural network only understands numbers. This module converts
raw text (from web searches, Wikipedia, etc.) into numerical vectors
using TF-IDF (Term Frequency - Inverse Document Frequency).

HOW TF-IDF WORKS:
  - TF  = how often a word appears in THIS document
  - IDF = how rare that word is ACROSS all documents
  - TF-IDF = TF × IDF → high score = a word is important & unique to this doc

The output is a fixed-size vector (e.g., 64 numbers) that the
neural network can be trained to classify topics from.
"""

import re
import math
import numpy as np
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> list[str]:
    """
    Tokenize and clean a text string into a list of lowercase words.

    Removes punctuation, numbers, and common stop words.
    """
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "that",
        "this", "it", "its", "they", "their", "there", "as", "if", "not",
        "also", "which", "who", "what", "when", "where", "how", "all",
        "more", "one", "two", "three", "about", "than", "then", "so",
        "such", "into", "onto", "up", "out", "over", "after", "before",
    }
    # Lowercase + keep only letters
    text = text.lower()
    words = re.findall(r"[a-z]{3,}", text)  # only words 3+ chars
    return [w for w in words if w not in STOP_WORDS]


# ---------------------------------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------------------------------

class TFIDFVectorizer:
    """
    Converts a collection of text documents into TF-IDF feature vectors.

    Usage:
        vec = TFIDFVectorizer(vocab_size=64)
        vec.fit(list_of_strings)
        X = vec.transform(list_of_strings)   # → numpy array [n_docs, 64]
    """

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.vocabulary_: Optional[list[str]] = None
        self.idf_: Optional[np.ndarray] = None
        self._word_to_idx: dict[str, int] = {}

    def fit(self, documents: list[str]) -> "TFIDFVectorizer":
        """
        Learn the vocabulary and IDF weights from a list of documents.

        Parameters
        ----------
        documents : list[str]
            Raw text strings.
        """
        n_docs = len(documents)
        tokenized = [clean_text(d) for d in documents]

        # --- Build vocabulary: top vocab_size words by total frequency ---
        all_words: Counter = Counter()
        for tokens in tokenized:
            all_words.update(tokens)

        self.vocabulary_ = [w for w, _ in all_words.most_common(self.vocab_size)]
        self._word_to_idx = {w: i for i, w in enumerate(self.vocabulary_)}

        # --- Compute IDF for each vocabulary word ---
        idf = np.zeros(len(self.vocabulary_))
        for i, word in enumerate(self.vocabulary_):
            # Count how many docs contain this word
            df = sum(1 for tokens in tokenized if word in tokens)
            # IDF formula: log((1 + n) / (1 + df)) + 1 (smoothed)
            idf[i] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf_ = idf
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        """
        Convert documents to TF-IDF vectors.

        Parameters
        ----------
        documents : list[str]

        Returns
        -------
        np.ndarray, shape [n_docs, vocab_size]
        """
        if self.vocabulary_ is None:
            raise RuntimeError("Call fit() before transform().")

        X = np.zeros((len(documents), len(self.vocabulary_)))
        for d_idx, doc in enumerate(documents):
            tokens = clean_text(doc)
            if not tokens:
                continue
            counts = Counter(tokens)
            for word, count in counts.items():
                if word in self._word_to_idx:
                    w_idx = self._word_to_idx[word]
                    tf = count / len(tokens)          # normalized TF
                    X[d_idx, w_idx] = tf * self.idf_[w_idx]

        # L2-normalize each row so long docs don't dominate
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit on documents, then immediately transform them."""
        return self.fit(documents).transform(documents)

    @property
    def n_features(self) -> int:
        """Number of output features (columns in the vector)."""
        return len(self.vocabulary_) if self.vocabulary_ else 0


# ---------------------------------------------------------------------------
# Quick utility: text list → labeled dataset
# ---------------------------------------------------------------------------

def build_dataset(
    text_groups: dict[str, list[str]],
    vocab_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str], TFIDFVectorizer]:
    """
    Convert labeled text groups into a numerical dataset.

    Parameters
    ----------
    text_groups : dict[str, list[str]]
        Keys are class labels (e.g. "science", "sports").
        Values are lists of raw text strings.
    vocab_size : int
        Size of the TF-IDF vocabulary.

    Returns
    -------
    X : np.ndarray, shape [n_samples, vocab_size]
    y : np.ndarray, shape [n_samples, n_classes]  (one-hot)
    class_names : list[str]
    vectorizer : TFIDFVectorizer (fitted)
    """
    class_names = sorted(text_groups.keys())
    all_texts, all_labels = [], []

    for label_idx, cls in enumerate(class_names):
        for text in text_groups[cls]:
            all_texts.append(text)
            all_labels.append(label_idx)

    vectorizer = TFIDFVectorizer(vocab_size=vocab_size)
    X = vectorizer.fit_transform(all_texts)

    # One-hot encode labels
    n_classes = len(class_names)
    y = np.zeros((len(all_labels), n_classes))
    for i, lbl in enumerate(all_labels):
        y[i, lbl] = 1.0

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm], class_names, vectorizer
