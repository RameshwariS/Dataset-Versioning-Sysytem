"""
model.py
--------
Bag-of-Words (TF-IDF) + Multinomial Naive Bayes text classifier.

Built entirely from scratch using only Python standard library + numpy.
No scikit-learn or any ML framework is used.

Design rationale
----------------
Naive Bayes is an ideal "toy" model for demonstrating dataset version effects:
  - It is fully interpretable: we can inspect learned word log-probabilities.
  - It trains instantly, so version comparisons are fast.
  - It is sensitive to vocabulary changes, making preprocessing effects visible.
  - The math is transparent and easy to reason about.

TF-IDF vectorisation
--------------------
  TF(t, d)  = count(t in d) / total_tokens(d)
  IDF(t)    = log((1 + N) / (1 + df(t))) + 1   [sklearn-style smoothed]
  TF-IDF(t, d) = TF * IDF

  The vocabulary is built from the *training* set only (no leakage).
  At inference time, unknown tokens are simply ignored.

Multinomial Naive Bayes
-----------------------
  For class c and document d:
      log P(c|d)  ∝  log P(c)  +  Σ_t  count(t,d) * log P(t|c)

  Word probabilities use Laplace (add-1) smoothing to handle unseen tokens.

The classifier is intentionally simple. The goal is to show HOW dataset
preprocessing changes the feature space, vocabulary coverage, and ultimately
the model's accuracy — not to maximise accuracy itself.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── TF-IDF Vectoriser ───────────────────────────────────────────────────────────

class TFIDFVectorizer:
    """
    Fits a TF-IDF vocabulary on training documents and transforms documents
    into TF-IDF weighted token-count vectors (represented as sparse dicts).
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}      # token → index
        self.idf: Dict[str, float] = {}           # token → IDF weight
        self._n_docs: int = 0

    # ------------------------------------------------------------------

    def fit(self, docs: List[str]) -> "TFIDFVectorizer":
        """
        Build vocabulary and IDF weights from a list of training documents.

        Parameters
        ----------
        docs : List[str]
            Each element is a whitespace-tokenised text string.
        """
        self._n_docs = len(docs)

        # Document frequency: how many docs contain each token
        df: Counter = Counter()
        for doc in docs:
            tokens = set(doc.lower().split())
            df.update(tokens)

        # Build vocabulary (sorted for determinism)
        all_tokens = sorted(df.keys())
        self.vocabulary = {tok: idx for idx, tok in enumerate(all_tokens)}

        # Smoothed IDF (sklearn convention)
        n = self._n_docs
        self.idf = {
            tok: math.log((1 + n) / (1 + count)) + 1.0
            for tok, count in df.items()
        }

        return self

    def transform(self, docs: List[str]) -> List[Dict[int, float]]:
        """
        Transform documents into TF-IDF sparse vectors.

        Returns a list of dicts {vocab_index: tfidf_weight}.
        Unknown tokens (not in vocabulary) are ignored.
        """
        vectors = []
        for doc in docs:
            tokens = doc.lower().split()
            if not tokens:
                vectors.append({})
                continue

            tf_raw = Counter(tokens)
            n = len(tokens)
            vec: Dict[int, float] = {}
            for tok, cnt in tf_raw.items():
                if tok in self.vocabulary:
                    tf = cnt / n
                    tfidf = tf * self.idf[tok]
                    vec[self.vocabulary[tok]] = tfidf
            vectors.append(vec)
        return vectors

    def fit_transform(self, docs: List[str]) -> List[Dict[int, float]]:
        self.fit(docs)
        return self.transform(docs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)


# ── Multinomial Naive Bayes ─────────────────────────────────────────────────────

class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes trained on TF-IDF weighted sparse vectors.

    Uses Laplace smoothing (alpha=1) so unseen-vocabulary tokens do not
    produce -inf log-probabilities.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: List[str] = []
        self.log_class_priors_: Dict[str, float] = {}
        self.log_word_probs_: Dict[str, Dict[int, float]] = {}
        self._vocab_size: int = 0

    # ------------------------------------------------------------------

    def fit(
        self,
        vectors: List[Dict[int, float]],
        labels: List[str],
        vocab_size: int,
    ) -> "NaiveBayesClassifier":
        """
        Train the classifier.

        Parameters
        ----------
        vectors   : TF-IDF sparse vectors from TFIDFVectorizer.transform()
        labels    : Parallel list of class labels (strings).
        vocab_size: Total vocabulary size (needed for Laplace smoothing).
        """
        self._vocab_size = vocab_size
        self.classes_ = sorted(set(labels))

        # Class counts
        class_counts: Counter = Counter(labels)
        n_total = len(labels)

        # Log class priors  P(c)
        self.log_class_priors_ = {
            c: math.log(count / n_total)
            for c, count in class_counts.items()
        }

        # Aggregate TF-IDF weights per class per feature
        # word_weights[c][feat_idx] = sum of tfidf weights for that feature in class c
        word_weights: Dict[str, Counter] = {c: Counter() for c in self.classes_}
        for vec, label in zip(vectors, labels):
            word_weights[label].update(vec)

        # Log word probabilities with Laplace smoothing
        self.log_word_probs_ = {}
        for c in self.classes_:
            total_weight = sum(word_weights[c].values()) + self.alpha * vocab_size
            self.log_word_probs_[c] = {}
            for feat_idx in range(vocab_size):
                raw = word_weights[c].get(feat_idx, 0.0)
                prob = (raw + self.alpha) / total_weight
                self.log_word_probs_[c][feat_idx] = math.log(prob)

        return self

    def predict_log_proba(self, vector: Dict[int, float]) -> Dict[str, float]:
        """
        Return un-normalised log P(c|d) for each class.

        Parameters
        ----------
        vector : sparse TF-IDF dict for a single document.
        """
        scores: Dict[str, float] = {}
        for c in self.classes_:
            score = self.log_class_priors_[c]
            for feat_idx, weight in vector.items():
                if feat_idx in self.log_word_probs_[c]:
                    score += weight * self.log_word_probs_[c][feat_idx]
            scores[c] = score
        return scores

    def predict_one(self, vector: Dict[int, float]) -> str:
        """Predict the class for a single sparse vector."""
        scores = self.predict_log_proba(vector)
        return max(scores, key=scores.get)

    def predict(self, vectors: List[Dict[int, float]]) -> List[str]:
        """Predict classes for a list of sparse vectors."""
        return [self.predict_one(v) for v in vectors]


# ── Evaluation helpers ──────────────────────────────────────────────────────────

def evaluate(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> Dict[str, Any]:
    """
    Compute accuracy, per-class precision / recall / F1, and macro-F1.

    Parameters
    ----------
    y_true, y_pred : parallel lists of true and predicted labels.
    classes        : ordered list of class names.

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, per_class (dict of {class: {precision, recall, f1}})
    """
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / n if n else 0.0

    # Confusion matrix entries per class
    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

    return {
        "accuracy":  round(accuracy, 4),
        "macro_f1":  round(macro_f1, 4),
        "per_class": per_class,
        "n_test":    n,
        "n_correct": correct,
    }


def top_features(
    clf: NaiveBayesClassifier,
    vec: TFIDFVectorizer,
    class_name: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Return the top-N most discriminative tokens for a given class.
    Discriminativeness = log P(token|class) - avg log P(token|other classes).

    Parameters
    ----------
    clf        : trained NaiveBayesClassifier
    vec        : fitted TFIDFVectorizer (needed for index→token mapping)
    class_name : the class to explain
    top_n      : how many features to return

    Returns
    -------
    List of (token, score) tuples, sorted descending by score.
    """
    idx_to_token = {idx: tok for tok, idx in vec.vocabulary.items()}
    other_classes = [c for c in clf.classes_ if c != class_name]

    scores: List[Tuple[str, float]] = []
    for idx in range(vec.vocab_size):
        log_prob_c = clf.log_word_probs_[class_name].get(idx, 0.0)
        avg_others = (
            sum(clf.log_word_probs_[c].get(idx, 0.0) for c in other_classes)
            / len(other_classes)
            if other_classes else 0.0
        )
        disc = log_prob_c - avg_others
        scores.append((idx_to_token.get(idx, f"<{idx}>"), disc))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]
