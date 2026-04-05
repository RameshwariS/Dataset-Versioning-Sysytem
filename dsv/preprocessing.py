"""
preprocessing.py
----------------
Preprocessing pipeline for text datasets.
All operations are applied in a fixed, deterministic order to guarantee
reproducibility across runs and machines.

Preprocessing order:
    1. strip_whitespace
    2. lowercase
    3. remove_punctuation
    4. remove_duplicates
    5. tokenize
    6. remove_stopwords
"""

import re
import string
from typing import List

# Predefined English stopwords list
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "his", "she", "her", "they", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "as", "than", "then", "just", "because", "while", "although",
    "about", "above", "after", "before", "between", "into", "through",
    "during", "such", "up", "out", "over", "also", "more", "most",
    "other", "some", "any", "each", "there", "here", "all"
}


class PreprocessingPipeline:
    """
    Applies a sequence of configurable text preprocessing steps to a list
    of text rows. The execution order is always fixed regardless of the order
    keys appear in the config dict, ensuring deterministic outputs.
    """

    # Canonical execution order — never changes
    ORDERED_STEPS = [
        "strip_whitespace",
        "lowercase",
        "remove_punctuation",
        "remove_duplicates",
        "tokenize",
        "remove_stopwords",
    ]

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Mapping of step name -> bool (True = enabled).
            Unknown keys are silently ignored.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Individual preprocessing operations
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_whitespace(rows: List[str]) -> List[str]:
        """Strip leading/trailing whitespace and collapse internal spaces."""
        return [" ".join(row.split()) for row in rows]

    @staticmethod
    def _lowercase(rows: List[str]) -> List[str]:
        """Convert every character to lowercase."""
        return [row.lower() for row in rows]

    @staticmethod
    def _remove_punctuation(rows: List[str]) -> List[str]:
        """Remove all punctuation characters from each row."""
        table = str.maketrans("", "", string.punctuation)
        return [row.translate(table) for row in rows]

    @staticmethod
    def _remove_duplicates(rows: List[str]) -> List[str]:
        """Remove duplicate rows while preserving original order."""
        seen = set()
        unique = []
        for row in rows:
            if row not in seen:
                seen.add(row)
                unique.append(row)
        return unique

    @staticmethod
    def _tokenize(rows: List[str]) -> List[str]:
        """
        Tokenize each row by whitespace and rejoin with single spaces.
        This normalises token boundaries so downstream steps work on
        clean token-separated strings.
        """
        return [" ".join(row.split()) for row in rows]

    @staticmethod
    def _remove_stopwords(rows: List[str]) -> List[str]:
        """Remove predefined stopwords from each row."""
        result = []
        for row in rows:
            tokens = row.split()
            filtered = [t for t in tokens if t.lower() not in STOPWORDS]
            result.append(" ".join(filtered))
        return result

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    def run(self, rows: List[str]) -> List[str]:
        """
        Apply enabled preprocessing steps in the canonical order.

        Parameters
        ----------
        rows : List[str]
            Raw text rows (one sentence / document per element).

        Returns
        -------
        List[str]
            Preprocessed rows.
        """
        # Map step names to their handler methods
        handlers = {
            "strip_whitespace": self._strip_whitespace,
            "lowercase": self._lowercase,
            "remove_punctuation": self._remove_punctuation,
            "remove_duplicates": self._remove_duplicates,
            "tokenize": self._tokenize,
            "remove_stopwords": self._remove_stopwords,
        }

        processed = list(rows)  # work on a copy

        for step in self.ORDERED_STEPS:
            if self.config.get(step, False):
                processed = handlers[step](processed)

        # Always drop completely empty rows produced by the pipeline
        processed = [row for row in processed if row.strip()]

        return processed
