"""
versioning.py
-------------
Core versioning logic: hash computation, version creation, DAG lineage.

Hashing strategy
----------------
SHA-256 over UTF-8 bytes of:
    raw_file_content  +  "||"  +  canonical_json(config)

Canonical JSON = sorted keys, no extra whitespace → same config always
produces the same bytes regardless of insertion order.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from .numeric_preprocessing import NumericPreprocessingPipeline, compute_numeric_metrics
from .preprocessing import PreprocessingPipeline
from .metrics import MetricsCalculator
from .storage import StorageManager


DATASET_TYPE_TXT = "txt"
DATASET_TYPE_NUMERIC = "numeric"
DATASET_TYPE_KEY = "__dataset_type__"


class DatasetVersionManager:
    """
    Orchestrates creation, retrieval, listing, comparison, and DAG lineage
    of dataset versions.
    """

    def __init__(self, storage: StorageManager):
        self.storage = storage

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_version_id(raw_content: str, config: Dict[str, Any]) -> str:
        """Deterministic SHA-256 from raw content + canonical config JSON."""
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        payload   = raw_content + "||" + canonical
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _choose_storage_mode(
        self,
        parent_id: Optional[str],
        version_id: str,
    ) -> Dict[str, Any]:
        """Choose whether the new version should be a snapshot or a delta."""
        if not parent_id:
            return {
                "storage_type": "snapshot",
                "delta_depth": 0,
                "base_checkpoint_id": version_id,
                "parent_rows": None,
            }

        parent_data = self.storage.load_version(parent_id)
        parent_meta = parent_data["meta"]
        checkpoint_interval = self.storage.get_checkpoint_interval()
        next_depth = int(parent_meta.get("delta_depth", 0)) + 1

        if checkpoint_interval > 1 and next_depth < checkpoint_interval:
            return {
                "storage_type": "delta",
                "delta_depth": next_depth,
                "base_checkpoint_id": (
                    parent_meta.get("base_checkpoint_id")
                    or parent_meta.get("version_id")
                    or parent_id
                ),
                "parent_rows": parent_data["rows"],
            }

        return {
            "storage_type": "snapshot",
            "delta_depth": 0,
            "base_checkpoint_id": version_id,
            "parent_rows": parent_data["rows"],
        }

    # ------------------------------------------------------------------
    # Version creation
    # ------------------------------------------------------------------

    def create_version(
        self,
        data_path: str,
        config: Dict[str, Any],
        name: str,
        message: str,
        parent_ref: Optional[str] = None,
        transformation_step: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new immutable dataset version.

        Steps
        -----
        1. Read raw file content.
        2. Resolve parent (explicit ref, or auto-latest, or None for root).
        3. Compute SHA-256 version ID.
        4. Return existing version if hash already stored (idempotent).
        5. Run preprocessing pipeline (label-aware).
        6. Compute metrics.
        7. Persist all artefacts including lineage fields.

        Parameters
        ----------
        data_path           : Path to the raw dataset file.
        config              : Preprocessing configuration dict.
        name                : Human-readable version name.
        message             : Commit-style description.
        parent_ref          : Name or hash of parent version (None = auto-detect latest).
        transformation_step : Label describing the change vs parent (e.g. "added stopword removal").
        """
        raw_content = self.storage.read_raw(data_path)
        version_id  = self.compute_version_id(raw_content, config)

        # ── Resolve parent ───────────────────────────────────────────────
        parent_name: Optional[str] = None
        parent_id:   Optional[str] = None

        if parent_ref is not None:
            # Explicit parent supplied by user
            try:
                resolved_pid = self.resolve_version(parent_ref)
                parent_data  = self.storage.load_version(resolved_pid)
                parent_name  = parent_data["meta"].get("name", parent_ref)
                parent_id    = resolved_pid
            except (FileNotFoundError, KeyError):
                print(f"  [warn] Parent '{parent_ref}' not found — creating as root node.")
        else:
            # Auto-detect: use the most recently created version as parent
            latest = self.storage.get_latest_version()
            if latest:
                parent_name = latest["name"]
                parent_id   = latest["version_id"]

        # ── Idempotency check ────────────────────────────────────────────
        if self.storage.version_exists(version_id):
            stored = self.storage.load_version(version_id)
            return {
                "version_id": version_id,
                "name":       stored["meta"].get("name", ""),
                "metrics":    stored["metrics"],
                "created":    False,
            }

        # ── Preprocessing ────────────────────────────────────────────────
        raw_lines: List[str] = [l for l in raw_content.splitlines() if l.strip()]

        # ── Numeric dataset branch ────────────────────────────────────
        dataset_type = config.get(DATASET_TYPE_KEY, "txt")
        if dataset_type == DATASET_TYPE_NUMERIC:
            pipeline       = NumericPreprocessingPipeline(config)
            processed_rows = pipeline.run(raw_lines)
            metrics        = compute_numeric_metrics(processed_rows)
            storage_plan   = self._choose_storage_mode(parent_id, version_id)
            self.storage.save_version(
                version_id, processed_rows, config, metrics,
                name, message,
                parent_name=parent_name,
                parent_id=parent_id,
                transformation_step=transformation_step,
                storage_type=storage_plan["storage_type"],
                delta_depth=storage_plan["delta_depth"],
                base_checkpoint_id=storage_plan["base_checkpoint_id"],
                parent_rows=storage_plan["parent_rows"],
            )
            return {
                "version_id":  version_id,
                "name":        name,
                "metrics":     metrics,
                "created":     True,
                "parent_name": parent_name,
                "storage_type": storage_plan["storage_type"],
                "delta_depth": storage_plan["delta_depth"],
                "base_checkpoint_id": storage_plan["base_checkpoint_id"],
            }

        # Detect labeled rows using MAJORITY VOTE (not all()), so that adding
        # a single unlabeled line does not break the whole dataset.
        # A row is "labeled" if it contains "|" and has non-empty text+label
        # on both sides of the last "|".
        def _is_labeled(line: str) -> bool:
            if "|" not in line:
                return False
            sep = line.rfind("|")
            return bool(line[:sep].strip()) and bool(line[sep + 1:].strip())

        labeled_count   = sum(1 for l in raw_lines if _is_labeled(l))
        unlabeled_count = len(raw_lines) - labeled_count
        # Treat as a labeled dataset if more than half the rows have labels
        is_labeled_dataset = labeled_count > unlabeled_count

        if is_labeled_dataset:
            if unlabeled_count > 0:
                print(f"  [warn] {unlabeled_count} row(s) have no label — "
                      f"they will be stored without a label and skipped during training.")

            pipeline = PreprocessingPipeline(config)
            processed_rows = []
            for line in raw_lines:
                if _is_labeled(line):
                    sep   = line.rfind("|")
                    text  = line[:sep].strip()
                    label = line[sep + 1:].strip()
                    result = pipeline.run([text])
                    if result:                          # non-empty after preprocessing
                        processed_rows.append(result[0] + "|" + label)
                else:
                    # Unlabeled row: preprocess text and store without label
                    result = pipeline.run([line])
                    if result:
                        processed_rows.append(result[0])
        else:
            pipeline       = PreprocessingPipeline(config)
            processed_rows = pipeline.run(raw_lines)

        # Metrics on text portion only
        text_only = [
            r[:r.rfind("|")].strip() if "|" in r else r
            for r in processed_rows
        ]
        metrics = MetricsCalculator.compute(text_only)

        # ── Persist ──────────────────────────────────────────────────────
        storage_plan = self._choose_storage_mode(parent_id, version_id)

        self.storage.save_version(
            version_id, processed_rows, config, metrics,
            name, message,
            parent_name=parent_name,
            parent_id=parent_id,
            transformation_step=transformation_step,
            storage_type=storage_plan["storage_type"],
            delta_depth=storage_plan["delta_depth"],
            base_checkpoint_id=storage_plan["base_checkpoint_id"],
            parent_rows=storage_plan["parent_rows"],
        )

        return {
            "version_id": version_id,
            "name":       name,
            "metrics":    metrics,
            "created":    True,
            "parent_name": parent_name,
            "storage_type": storage_plan["storage_type"],
            "delta_depth": storage_plan["delta_depth"],
            "base_checkpoint_id": storage_plan["base_checkpoint_id"],
        }

    # ------------------------------------------------------------------
    # Version inspection
    # ------------------------------------------------------------------

    def resolve_version(self, version_ref: str) -> str:
        """Accept a version name or 64-char SHA-256 hash, return hash."""
        if len(version_ref) == 64 and all(c in "0123456789abcdef" for c in version_ref):
            return version_ref
        return self.storage.resolve_name(version_ref)

    def get_version(self, version_ref: str) -> Dict[str, Any]:
        """Retrieve all artefacts. Accepts name or hash."""
        return self.storage.load_version(self.resolve_version(version_ref))

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return versions ordered by creation time (oldest first)."""
        return self.storage.list_versions()

    # ------------------------------------------------------------------
    # DAG / lineage
    # ------------------------------------------------------------------

    def build_dag(self) -> Dict[str, Any]:
        """
        Build the complete lineage DAG from all stored versions.

        Returns
        -------
        dict with:
            nodes : list of node dicts (one per version)
            edges : list of {source, target, label} dicts
        """
        versions = self.storage.list_versions()

        nodes = []
        edges = []

        for v in versions:
            version_id = v["version_id"]
            # Load full data for metrics and training report
            try:
                data = self.storage.load_version(version_id)
            except Exception:
                continue

            metrics = data["metrics"]
            meta    = data["meta"]
            config  = data["config"]

            # Load training report — try name-based folder first, then hash-based
            version_name = v["name"]
            name_report  = self.storage.versions_path / version_name / "training_report.json"
            hash_report  = self.storage.versions_path / version_id   / "training_report.json"
            report_path  = name_report if name_report.exists() else hash_report
            training = {}
            if report_path.exists():
                try:
                    training = json.loads(report_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # Load AI insights if they exist
            name_insights = self.storage.versions_path / version_name / "ai_insights.json"
            hash_insights = self.storage.versions_path / version_id   / "ai_insights.json"
            insights_path = name_insights if name_insights.exists() else hash_insights
            ai_insights = None
            if insights_path.exists():
                try:
                    ai_insights = json.loads(insights_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            node = {
                "id":           version_id,
                "name":         v["name"],
                "message":      v["message"],
                "created_at":   v["created_at"],
                "parent_id":    v.get("parent_id"),
                "parent_name":  v.get("parent_name"),
                "transformation_step": v.get("transformation_step"),
                "storage_type": meta.get("storage_type", "snapshot"),
                "delta_depth": meta.get("delta_depth", 0),
                "base_checkpoint_id": meta.get("base_checkpoint_id"),
                "metrics": {
                    "num_rows":         metrics.get("num_rows", 0),
                    "vocabulary_size":  metrics.get("vocabulary_size", 0),
                    "avg_sentence_len": metrics.get("avg_sentence_len", 0),
                    "total_tokens":     metrics.get("total_tokens", 0),
                },
                "config": config,
                "training": {
                    "test_accuracy": (
                        training.get("test_metrics", {}).get("accuracy")
                        if training else None
                    ),
                    "macro_f1": (
                        training.get("test_metrics", {}).get("macro_f1")
                        if training else None
                    ),
                    "top_features": training.get("top_features", {}),
                },
                "has_training_report": bool(training),
                "ai_insights": ai_insights,
            }
            nodes.append(node)

            # Build edge from parent → this node
            if v.get("parent_id"):
                edges.append({
                    "source": v["parent_id"],
                    "target": version_id,
                    "label":  v.get("transformation_step") or "preprocessed",
                })

        return {"nodes": nodes, "edges": edges}

    def get_lineage_path(self, version_ref: str) -> List[str]:
        """
        Walk up the parent chain from a version to the root.
        Returns ordered list of version names: [root, ..., version].
        """
        versions  = {v["version_id"]: v for v in self.storage.list_versions()}
        target_id = self.resolve_version(version_ref)

        path = []
        current = versions.get(target_id)
        while current:
            path.append(current["name"] or current["version_id"][:12])
            pid = current.get("parent_id")
            current = versions.get(pid) if pid else None

        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Version comparison
    # ------------------------------------------------------------------

    def compare_versions(self, version_ref_1: str, version_ref_2: str) -> Dict[str, Any]:
        vid1 = self.resolve_version(version_ref_1)
        vid2 = self.resolve_version(version_ref_2)
        v1   = self.storage.load_version(vid1)
        v2   = self.storage.load_version(vid2)

        metrics_diff = MetricsCalculator.diff(v1["metrics"], v2["metrics"])
        config_diff  = self._diff_configs(v1["config"], v2["config"])

        identical = (
            vid1 == vid2
            or (not config_diff and all(
                entry["difference"] == 0 for entry in metrics_diff.values()
            ))
        )

        return {
            "version_id_1": vid1,
            "version_id_2": vid2,
            "name_1":       v1["meta"].get("name", version_ref_1),
            "name_2":       v2["meta"].get("name", version_ref_2),
            "metrics_diff": metrics_diff,
            "config_diff":  config_diff,
            "identical":    identical,
        }

    @staticmethod
    def _diff_configs(config_a: Dict, config_b: Dict) -> Dict:
        all_keys = set(config_a) | set(config_b)
        diff = {}
        for key in sorted(all_keys):
            va, vb = config_a.get(key), config_b.get(key)
            if va != vb:
                diff[key] = {"version_1": va, "version_2": vb}
        return diff
