"""
metrics.py
----------
Computes and stores dataset metrics for a processed list of text rows.

Metrics computed:
    - num_rows          : total number of rows (after preprocessing)
    - vocabulary_size   : number of unique tokens across all rows
    - avg_sentence_len  : mean number of tokens per row
    - total_tokens      : sum of tokens across all rows
"""

from typing import List, Dict, Any


class MetricsCalculator:
    """Stateless helper that computes descriptive metrics for text data."""

    @staticmethod
    def compute(rows: List[str]) -> Dict[str, Any]:
        """
        Compute dataset metrics from a list of preprocessed text rows.

        Parameters
        ----------
        rows : List[str]
            Preprocessed text rows.

        Returns
        -------
        dict
            Dictionary containing all computed metrics.
        """
        if not rows:
            return {
                "num_rows": 0,
                "vocabulary_size": 0,
                "avg_sentence_len": 0.0,
                "total_tokens": 0,
            }

        # Tokenise each row by whitespace for metric calculations
        tokenised = [row.split() for row in rows]

        token_counts = [len(tokens) for tokens in tokenised]
        total_tokens = sum(token_counts)

        vocabulary: set = set()
        for tokens in tokenised:
            vocabulary.update(t.lower() for t in tokens)

        avg_sentence_len = total_tokens / len(rows) if rows else 0.0

        return {
            "num_rows": len(rows),
            "vocabulary_size": len(vocabulary),
            "avg_sentence_len": round(avg_sentence_len, 4),
            "total_tokens": total_tokens,
        }

    @staticmethod
    def diff(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a dict of numeric differences (b - a) between two metrics dicts.

        Parameters
        ----------
        metrics_a, metrics_b : dict
            Metric dicts produced by :meth:`compute`.

        Returns
        -------
        dict
            Keys map to (value_a, value_b, difference) tuples for numeric
            fields.
        """
        numeric_keys = ["num_rows", "vocabulary_size", "avg_sentence_len", "total_tokens"]
        result = {}
        for key in numeric_keys:
            a = metrics_a.get(key, 0)
            b = metrics_b.get(key, 0)
            result[key] = {
                "version_1": a,
                "version_2": b,
                "difference": round(b - a, 4) if isinstance(b, float) or isinstance(a, float)
                              else b - a,
            }
        return result
