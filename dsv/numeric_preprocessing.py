"""
numeric_preprocessing.py
------------------------
Preprocessing pipeline for numeric/tabular datasets (CSV format).

All operations are applied in a fixed, deterministic order to guarantee
reproducibility across runs and machines.

Preprocessing order:
    1. drop_missing           - Remove rows with missing/NaN values
    2. drop_duplicates        - Remove duplicate rows
    3. normalize              - Min-max normalize numeric columns to [0, 1]
    4. standardize            - Z-score standardize numeric columns (mean=0, std=1)
    5. remove_outliers        - Remove rows with values beyond 3 std deviations
    6. round_decimals         - Round numeric values to a fixed number of decimal places

NOTE: normalize and standardize are mutually exclusive; normalize takes priority.
"""

import csv
import io
import statistics
from typing import List, Dict, Any, Tuple


class NumericPreprocessingPipeline:
    """
    Applies a sequence of configurable numeric preprocessing steps to tabular data.
    Input/output format: list of strings (CSV rows, first row = header).
    The execution order is always fixed regardless of config key order.
    """

    # Canonical execution order — never changes
    ORDERED_STEPS = [
        "drop_missing",
        "drop_duplicates",
        "normalize",
        "standardize",
        "remove_outliers",
        "round_decimals",
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
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(rows: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
        """Parse CSV rows into (headers, list-of-dicts)."""
        if not rows:
            return [], []
        reader = csv.DictReader(io.StringIO("\n".join(rows)))
        headers = reader.fieldnames or []
        data = [dict(row) for row in reader]
        return list(headers), data

    @staticmethod
    def _serialize_csv(headers: List[str], data: List[Dict[str, str]]) -> List[str]:
        """Serialize (headers, list-of-dicts) back to CSV string rows."""
        if not data:
            return [",".join(headers)] if headers else []
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
        lines = buf.getvalue().strip().split("\n")
        return lines

    @staticmethod
    def _numeric_columns(headers: List[str], data: List[Dict[str, str]]) -> List[str]:
        """Return columns where all non-empty values are numeric."""
        numeric_cols = []
        for col in headers:
            values = [row[col] for row in data if row.get(col, "").strip() != ""]
            if not values:
                continue
            try:
                [float(v) for v in values]
                numeric_cols.append(col)
            except ValueError:
                pass
        return numeric_cols

    # ------------------------------------------------------------------
    # Individual preprocessing operations
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_missing(headers: List[str], data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove rows that have any empty or missing cell."""
        cleaned = []
        for row in data:
            if all(row.get(col, "").strip() != "" for col in headers):
                cleaned.append(row)
        return cleaned

    @staticmethod
    def _drop_duplicates(headers: List[str], data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate rows while preserving original order."""
        seen = set()
        unique = []
        for row in data:
            key = tuple(row.get(col, "") for col in headers)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        return unique

    def _normalize(self, headers: List[str], data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Min-max normalize all numeric columns to [0, 1]."""
        if not data:
            return data
        num_cols = self._numeric_columns(headers, data)
        result = [dict(row) for row in data]
        for col in num_cols:
            vals = [float(row[col]) for row in result]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1.0
            for row in result:
                row[col] = str(round((float(row[col]) - mn) / rng, 6))
        return result

    def _standardize(self, headers: List[str], data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Z-score standardize all numeric columns (mean=0, std=1)."""
        if not data:
            return data
        num_cols = self._numeric_columns(headers, data)
        result = [dict(row) for row in data]
        for col in num_cols:
            vals = [float(row[col]) for row in result]
            if len(vals) < 2:
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            if std == 0:
                std = 1.0
            for row in result:
                row[col] = str(round((float(row[col]) - mean) / std, 6))
        return result

    def _remove_outliers(self, headers: List[str], data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove rows where any numeric column value is >3 std devs from mean."""
        if not data:
            return data
        num_cols = self._numeric_columns(headers, data)
        # Compute per-column mean and std
        col_stats: Dict[str, Tuple[float, float]] = {}
        for col in num_cols:
            vals = [float(row[col]) for row in data]
            if len(vals) < 2:
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            col_stats[col] = (mean, std if std > 0 else 1.0)

        result = []
        for row in data:
            keep = True
            for col, (mean, std) in col_stats.items():
                val = float(row[col])
                if abs(val - mean) > 3 * std:
                    keep = False
                    break
            if keep:
                result.append(row)
        return result

    @staticmethod
    def _round_decimals(headers: List[str], data: List[Dict[str, str]],
                        decimals: int = 4) -> List[Dict[str, str]]:
        """Round all numeric column values to a fixed number of decimal places."""
        if not data:
            return data
        result = []
        for row in data:
            new_row = dict(row)
            for col in headers:
                val = row.get(col, "").strip()
                try:
                    new_row[col] = str(round(float(val), decimals))
                except (ValueError, TypeError):
                    pass  # non-numeric columns left unchanged
            result.append(new_row)
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
            Raw CSV rows (first row must be the header).

        Returns
        -------
        List[str]
            Preprocessed CSV rows (header + data rows).
        """
        if not rows:
            return rows

        headers, data = self._parse_csv(rows)

        if not data:
            return rows

        # If both normalize and standardize are enabled, normalize wins
        normalize_on = self.config.get("normalize", False)
        standardize_on = self.config.get("standardize", False)
        if normalize_on and standardize_on:
            standardize_on = False  # normalize takes priority

        handlers = {
            "drop_missing":    lambda h, d: (h, self._drop_missing(h, d)),
            "drop_duplicates": lambda h, d: (h, self._drop_duplicates(h, d)),
            "normalize":       lambda h, d: (h, self._normalize(h, d)) if normalize_on else (h, d),
            "standardize":     lambda h, d: (h, self._standardize(h, d)) if standardize_on else (h, d),
            "remove_outliers": lambda h, d: (h, self._remove_outliers(h, d)),
            "round_decimals":  lambda h, d: (h, self._round_decimals(h, d)),
        }

        for step in self.ORDERED_STEPS:
            if self.config.get(step, False):
                headers, data = handlers[step](headers, data)

        return self._serialize_csv(headers, data)


# ---------------------------------------------------------------------------
# Numeric metrics helper
# ---------------------------------------------------------------------------

def compute_numeric_metrics(rows: List[str]) -> Dict[str, Any]:
    """
    Compute basic descriptive metrics for a numeric/tabular dataset.

    Parameters
    ----------
    rows : List[str]
        CSV rows (first row = header).

    Returns
    -------
    dict
        Metrics dictionary compatible with the existing MetricsCalculator shape,
        with additional numeric-specific fields.
    """
    if not rows or len(rows) < 2:
        return {
            "num_rows": 0,
            "num_columns": 0,
            "numeric_columns": 0,
            "missing_cells": 0,
            "dataset_type": "numeric",
            # Kept for storage/versioning compatibility
            "vocabulary_size": 0,
            "avg_sentence_len": 0.0,
            "total_tokens": 0,
        }

    reader = csv.DictReader(io.StringIO("\n".join(rows)))
    headers = list(reader.fieldnames or [])
    data = [dict(row) for row in reader]

    num_rows = len(data)
    num_cols = len(headers)
    missing = sum(1 for row in data for col in headers if row.get(col, "").strip() == "")

    # Detect numeric columns
    numeric_cols = []
    for col in headers:
        vals = [row.get(col, "").strip() for row in data]
        try:
            [float(v) for v in vals if v]
            numeric_cols.append(col)
        except ValueError:
            pass

    # Compute per-column stats for numeric cols
    col_stats = {}
    for col in numeric_cols:
        vals = [float(row[col]) for row in data if row.get(col, "").strip()]
        if vals:
            col_stats[col] = {
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }

    return {
        "num_rows": num_rows,
        "num_columns": num_cols,
        "numeric_columns": len(numeric_cols),
        "missing_cells": missing,
        "column_stats": col_stats,
        "dataset_type": "numeric",
        # Kept for storage/versioning compatibility
        "vocabulary_size": num_cols,
        "avg_sentence_len": round(num_rows / num_cols, 4) if num_cols else 0.0,
        "total_tokens": num_rows * num_cols,
    }
