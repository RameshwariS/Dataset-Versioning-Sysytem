"""
storage.py
----------
Filesystem-based storage layer for dataset versions.

Repository layout
-----------------
dataset_repo/
    raw/                        <- user places source datasets here
    versions/
        <version_name>/         <- folder named by human-readable name
            dataset.txt         <- full snapshot for checkpoint versions
            delta.json          <- row-level delta for non-checkpoint versions
            config.json         <- preprocessing configuration used
            metrics.json        <- computed dataset metrics
            meta.json           <- name, hash, message, timestamp, parent, step
    registry.json               <- index: name -> version_id

Hybrid storage strategy
-----------------------
Versions are stored as a snapshot at the root and then as row-level deltas
against the immediate parent. After a configurable number of chained deltas,
the next version is stored as a fresh snapshot checkpoint again.
"""

import json
import time as _time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional


class StorageManager:
    """
    Manages reading and writing of version artefacts on the local filesystem.
    Folders are named by version name (e.g. v1-raw), not by hash.
    The SHA-256 hash is stored inside meta.json for integrity checking.
    """

    VERSIONS_DIR  = "versions"
    RAW_DIR       = "raw"
    DATASET_FILE  = "dataset.txt"
    DELTA_FILE    = "delta.json"
    CONFIG_FILE   = "config.json"
    METRICS_FILE  = "metrics.json"
    META_FILE     = "meta.json"
    REGISTRY_FILE  = "registry.json"
    REPO_META_FILE = ".dsv_repo_meta.json"
    DEFAULT_CHECKPOINT_INTERVAL = 5

    def __init__(self, repo_path: str = "dataset_repo"):
        self.repo_path     = Path(repo_path)
        self.versions_path = self.repo_path / self.VERSIONS_DIR
        self.raw_path      = self.repo_path / self.RAW_DIR
        self.registry_path = self.repo_path / self.REGISTRY_FILE

    # ------------------------------------------------------------------
    # Repository initialisation
    # ------------------------------------------------------------------

    def init(self, dataset_type: str = "txt", checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL) -> None:
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({})
        # Save dataset type in repo metadata
        meta_path = self.repo_path / self.REPO_META_FILE
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta["dataset_type"] = dataset_type
        meta["checkpoint_interval"] = max(1, int(meta.get("checkpoint_interval", checkpoint_interval)))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def get_dataset_type(self) -> str:
        """Return the dataset type ('txt' or 'numeric')."""
        meta_path = self.repo_path / self.REPO_META_FILE
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return meta.get("dataset_type", "txt")
            except Exception:
                pass
        return "txt"

    def get_checkpoint_interval(self) -> int:
        """Return the configured snapshot checkpoint interval."""
        meta_path = self.repo_path / self.REPO_META_FILE
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return max(1, int(meta.get("checkpoint_interval", self.DEFAULT_CHECKPOINT_INTERVAL)))
            except Exception:
                pass
        return self.DEFAULT_CHECKPOINT_INTERVAL

    def is_initialised(self) -> bool:
        return self.repo_path.exists() and self.versions_path.exists()

    # ------------------------------------------------------------------
    # Registry  (name -> name, kept for API compatibility)
    # ------------------------------------------------------------------

    def _read_registry(self) -> Dict[str, str]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, registry: Dict[str, str]) -> None:
        self.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def name_exists(self, name: str) -> bool:
        return (self.versions_path / name).exists()

    def resolve_name(self, name: str) -> str:
        """
        Accept a version name and return the version_id (hash) from meta.json.
        Also accepts a raw 64-char hash for backwards compatibility.
        """
        # Direct name lookup — folder named by version name
        folder = self.versions_path / name
        if folder.exists():
            meta_path = folder / self.META_FILE
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return meta.get("version_id", name)
            return name

        # Fallback: scan all folders for matching hash in meta.json
        if len(name) == 64:
            for d in self.versions_path.iterdir():
                if not d.is_dir(): continue
                meta_path = d / self.META_FILE
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("version_id") == name:
                        return name

        raise KeyError(
            f"Version '{name}' not found. "
            f"Use 'python dsv.py list' to see available versions."
        )

    def register_name(self, name: str, version_id: str) -> None:
        registry = self._read_registry()
        registry[name] = version_id
        self._write_registry(registry)

    def _read_meta(self, version_dir: Path) -> Dict[str, Any]:
        meta_path = version_dir / self.META_FILE
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _resolve_version_dir(self, version_ref: str) -> Path:
        """Resolve a version name or hash to its directory."""
        direct = self.versions_path / version_ref
        if direct.exists():
            return direct

        for version_dir in self.versions_path.iterdir():
            if not version_dir.is_dir():
                continue
            meta = self._read_meta(version_dir)
            if meta.get("version_id") == version_ref or version_dir.name == version_ref:
                return version_dir

        raise FileNotFoundError(f"Version '{version_ref}' not found in repository.")

    @staticmethod
    def _compute_row_delta(base_rows: List[str], target_rows: List[str]) -> Dict[str, Any]:
        """Create a compact row-level patch from base_rows -> target_rows."""
        operations: List[Dict[str, Any]] = []
        matcher = SequenceMatcher(a=base_rows, b=target_rows)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            operations.append({
                "op": tag,
                "start": i1,
                "end": i2,
                "rows": target_rows[j1:j2],
            })
        return {
            "format": "row_ops_v1",
            "operations": operations,
        }

    @staticmethod
    def _apply_row_delta(base_rows: List[str], delta_payload: Dict[str, Any]) -> List[str]:
        """Apply a stored row-level patch and return the reconstructed rows."""
        operations = list(delta_payload.get("operations", []))
        rows = list(base_rows)
        offset = 0

        for op in operations:
            start = int(op.get("start", 0)) + offset
            end = int(op.get("end", start)) + offset
            replacement = list(op.get("rows", []))
            rows[start:end] = replacement
            offset += len(replacement) - (end - start)

        return rows

    # ------------------------------------------------------------------
    # Version persistence
    # ------------------------------------------------------------------

    def version_exists(self, version_id: str) -> bool:
        """Check if a version with this hash already exists (scan meta.json files)."""
        for d in self.versions_path.iterdir():
            if not d.is_dir(): continue
            meta_path = d / self.META_FILE
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("version_id") == version_id:
                    return True
        return False

    def save_version(
        self,
        version_id: str,
        rows: List[str],
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        name: str,
        message: str,
        parent_name: Optional[str] = None,
        parent_id:   Optional[str] = None,
        transformation_step: Optional[str] = None,
        storage_type: str = "snapshot",
        delta_depth: int = 0,
        base_checkpoint_id: Optional[str] = None,
        parent_rows: Optional[List[str]] = None,
    ) -> Path:
        """
        Persist a dataset version to disk.
        Folder is named by version name (e.g. v1-raw), not by hash.
        Hash is stored inside meta.json.
        """
        # Folder named by version name — human readable
        version_dir = self.versions_path / name
        version_dir.mkdir(parents=True, exist_ok=True)

        if storage_type == "delta":
            if parent_rows is None:
                raise ValueError("parent_rows are required to save a delta version.")
            delta_payload = self._compute_row_delta(parent_rows, rows)
            (version_dir / self.DELTA_FILE).write_text(
                json.dumps(delta_payload, indent=2),
                encoding="utf-8",
            )
        else:
            (version_dir / self.DATASET_FILE).write_text("\n".join(rows), encoding="utf-8")
        (version_dir / self.CONFIG_FILE).write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        (version_dir / self.METRICS_FILE).write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

        registry      = self._read_registry()
        creation_index = len(registry)

        meta: Dict[str, Any] = {
            "name":                name,
            "message":             message,
            "version_id":          version_id,   # SHA-256 hash stored here
            "created_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_ts":          _time.time(),
            "creation_index":      creation_index,
            "parent_name":         parent_name,
            "parent_id":           parent_id,
            "transformation_step": transformation_step,
            "storage_type":        storage_type,
            "delta_depth":         int(delta_depth),
            "base_checkpoint_id":  base_checkpoint_id or version_id,
        }
        (version_dir / self.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self.register_name(name, version_id)
        return version_dir

    # ------------------------------------------------------------------
    # Version retrieval
    # ------------------------------------------------------------------

    def load_version(self, version_id: str) -> Dict[str, Any]:
        """
        Load all artefacts for a version.
        Accepts either a version name or a SHA-256 hash.
        """
        target_dir = self._resolve_version_dir(version_id)
        config = json.loads((target_dir / self.CONFIG_FILE).read_text(encoding="utf-8"))
        metrics = json.loads((target_dir / self.METRICS_FILE).read_text(encoding="utf-8"))
        meta = self._read_meta(target_dir)

        dataset_file = target_dir / self.DATASET_FILE
        delta_file = target_dir / self.DELTA_FILE

        if dataset_file.exists():
            rows = dataset_file.read_text(encoding="utf-8").splitlines()
            meta.setdefault("storage_type", "snapshot")
            meta.setdefault("delta_depth", 0)
            meta.setdefault("base_checkpoint_id", meta.get("version_id"))
        elif delta_file.exists():
            parent_id = meta.get("parent_id")
            if not parent_id:
                raise ValueError(
                    f"Delta version '{meta.get('name', version_id)}' is missing parent metadata."
                )
            parent_version = self.load_version(parent_id)
            delta_payload = json.loads(delta_file.read_text(encoding="utf-8"))
            rows = self._apply_row_delta(parent_version["rows"], delta_payload)
            meta.setdefault("storage_type", "delta")
            meta.setdefault("delta_depth", parent_version["meta"].get("delta_depth", 0) + 1)
            meta.setdefault(
                "base_checkpoint_id",
                parent_version["meta"].get("base_checkpoint_id") or parent_id,
            )
        else:
            rows = []
            meta.setdefault("storage_type", "snapshot")
            meta.setdefault("delta_depth", 0)
            meta.setdefault("base_checkpoint_id", meta.get("version_id"))

        return {"rows": rows, "config": config, "metrics": metrics, "meta": meta}

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all versions ordered by creation time (oldest first)."""
        if not self.versions_path.exists():
            return []

        entries = []
        for d in self.versions_path.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / self.META_FILE
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            entries.append({
                # version_id is now the hash stored inside meta.json
                "version_id":          meta.get("version_id", d.name),
                "name":                meta.get("name", d.name),
                "message":             meta.get("message", ""),
                "created_at":          meta.get("created_at", ""),
                "created_ts":          meta.get("created_ts", 0.0),
                "creation_index":      meta.get("creation_index", 0),
                "parent_name":         meta.get("parent_name"),
                "parent_id":           meta.get("parent_id"),
                "transformation_step": meta.get("transformation_step"),
                "storage_type":        meta.get(
                    "storage_type",
                    "snapshot" if (d / self.DATASET_FILE).exists() else "delta",
                ),
                "delta_depth":         meta.get("delta_depth", 0),
                "base_checkpoint_id":  meta.get("base_checkpoint_id", meta.get("version_id", d.name)),
            })

        entries.sort(key=lambda e: (
            e.get("created_ts", 0.0),
            e.get("creation_index", 0),
        ))
        return entries

    def get_latest_version(self) -> Optional[Dict[str, Any]]:
        versions = self.list_versions()
        return versions[-1] if versions else None

    # ------------------------------------------------------------------
    # Raw data helpers
    # ------------------------------------------------------------------

    def read_raw(self, data_path: str) -> str:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        return path.read_text(encoding="utf-8")
