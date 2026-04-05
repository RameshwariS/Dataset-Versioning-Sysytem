"""
watch.py
--------
Watches dataset_repo/raw/ for file changes and automatically creates a
new dataset version whenever a raw file is modified or a new file is added.

Works on Windows, macOS, and Linux using only the Python standard library
(no watchdog or inotify dependency).

Usage
-----
    python watch.py
    python watch.py --data dataset_repo/raw/dataset.txt
    python watch.py --interval 3        # check every 3 seconds (default: 2)
    python watch.py --no-train          # skip auto-training after version creation
    python watch.py --no-dashboard      # skip auto-refreshing dashboard data

How it works
------------
Every <interval> seconds the watcher computes an MD5 hash of each watched
file.  If the hash differs from the last recorded value, a change is detected
and a new version is created automatically.

The version name is generated as:
    auto-<filename-stem>-<YYYYMMDD-HHMMSS>

The transformation step is set to:
    "auto: file changed at <timestamp>"

The preprocessing config used is the LAST manually created version's config
(so the automatic version inherits whatever settings were last used).
If no versions exist yet, a sensible default config is used.

Ctrl-C to stop watching.
"""

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from .storage import StorageManager
from .versioning import DatasetVersionManager
from .train import train_and_evaluate


# ── Default config used when no previous versions exist ─────────
DEFAULT_CONFIG = {
    "lowercase":           True,
    "strip_whitespace":    True,
    "remove_punctuation":  False,
    "remove_duplicates":   True,
    "tokenize":            False,
    "remove_stopwords":    False,
}


def file_hash(path: Path) -> str:
    """Return MD5 hex digest of file contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def get_last_config(storage: StorageManager) -> dict:
    """
    Return the preprocessing config of the most recently created version,
    or DEFAULT_CONFIG if no versions exist yet.
    """
    latest = storage.get_latest_version()
    if not latest:
        return DEFAULT_CONFIG
    try:
        data = storage.load_version(latest["version_id"])
        return data["config"]
    except Exception:
        return DEFAULT_CONFIG


def make_version_name(stem: str, storage: StorageManager) -> str:
    """
    Generate a unique version name like auto-dataset-20240318-142301.
    If that name is already taken (rare collision), append a counter.
    """
    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"auto-{stem}-{ts}"
    name = base
    i    = 1
    while storage.name_exists(name):
        name = f"{base}-{i}"
        i += 1
    return name


def handle_change(
    path: Path,
    storage: StorageManager,
    manager: DatasetVersionManager,
    auto_train: bool,
    auto_dashboard: bool,
) -> None:
    """
    Called when a watched file changes.
    Creates a new version, optionally trains, optionally refreshes dashboard.
    """
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name    = make_version_name(path.stem, storage)
    config  = get_last_config(storage)
    message = f"Auto-version: {path.name} changed at {ts}"
    step    = f"auto: file changed at {ts}"

    print(f"\n[watch] Change detected in '{path.name}' at {ts}")
    print(f"        Creating version '{name}' ...")

    try:
        result = manager.create_version(
            str(path),
            config,
            name=name,
            message=message,
            parent_ref=None,       # auto-parent = latest
            transformation_step=step,
        )
    except Exception as exc:
        print(f"[watch] [error] Version creation failed: {exc}")
        return

    if result["created"]:
        m = result["metrics"]
        print(f"        [ok] Version '{name}' created.")
        print(f"             hash  = {result['version_id'][:20]}…")
        print(f"             rows  = {m['num_rows']}  "
              f"vocab = {m['vocabulary_size']}  "
              f"avg_len = {m['avg_sentence_len']}")

        if result.get("parent_name"):
            print(f"             parent = {result['parent_name']}")
    else:
        print(f"        [skip] Content unchanged — same hash as existing version.")
        return

    # ── Auto-train ───────────────────────────────────────────────
    if auto_train:
        print(f"        Training model on '{name}' ...")
        try:
            report = train_and_evaluate(name, storage, manager, verbose=False)
            if report.get("skipped"):
                print(f"        [skip] Training skipped: {report['reason']}")
            else:
                ta = report["test_metrics"]["accuracy"]
                f1 = report["test_metrics"]["macro_f1"]
                print(f"        [ok] accuracy={ta:.4f}  macro_f1={f1:.4f}")
        except (Exception, SystemExit) as exc:
            msg = str(exc)
            if not msg or msg == "1":
                print("        [skip] Training skipped — no labeled rows found.")
                print("               Rows must follow: text|label format to enable training.")
            else:
                print("        [warn] Training failed: " + msg)

    # ── Refresh dashboard data ───────────────────────────────────
    if auto_dashboard:
        try:
            dag       = manager.build_dag()
            data_file = ROOT / "dashboard_data.json"
            data_file.write_text(json.dumps(dag, indent=2), encoding="utf-8")
            print(f"        [ok] dashboard_data.json refreshed "
                  f"({len(dag['nodes'])} nodes).")
            print(f"        Reload your browser to see the new version in the DAG.")
        except Exception as exc:
            print(f"        [warn] Dashboard refresh failed: {exc}")


def watch(
    data_files: list,
    interval: float,
    storage: StorageManager,
    manager: DatasetVersionManager,
    auto_train: bool,
    auto_dashboard: bool,
) -> None:
    """
    Poll watched files every <interval> seconds.
    Detects: file modified, file created, file replaced.
    """
    # Seed initial hashes
    hashes: dict = {}
    for p in data_files:
        try:
            hashes[str(p)] = file_hash(p)
        except FileNotFoundError:
            hashes[str(p)] = None

    print(f"\n[watch] Watching {len(data_files)} file(s) — polling every {interval}s")
    for p in data_files:
        print(f"        · {p}")
    print(f"[watch] Ctrl-C to stop.\n")

    while True:
        time.sleep(interval)
        for p in data_files:
            key = str(p)
            try:
                current = file_hash(p)
            except FileNotFoundError:
                current = None

            previous = hashes.get(key)

            if current != previous:
                if current is None:
                    print(f"[watch] '{p.name}' was deleted — ignoring.")
                else:
                    handle_change(p, storage, manager, auto_train, auto_dashboard)
                hashes[key] = current


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="watch",
        description="Auto-create dataset versions when raw files change.",
    )
    parser.add_argument(
        "--repo", default="dataset_repo", metavar="PATH",
        help="Path to the dataset repository (default: dataset_repo).",
    )
    parser.add_argument(
        "--data", metavar="FILE", action="append", dest="files",
        help="File to watch (can be repeated). Default: all files in raw/.",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, metavar="SECONDS",
        help="Polling interval in seconds (default: 2).",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Skip automatic model training after version creation.",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Skip automatic dashboard_data.json refresh.",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo)
    storage   = StorageManager(repo_path=str(repo_path))

    if not storage.is_initialised():
        print("[error] Repository not initialised. Run 'python dsv.py init' first.")
        sys.exit(1)

    manager = DatasetVersionManager(storage=storage)

    # Resolve files to watch
    if args.files:
        data_files = [Path(f) for f in args.files]
        missing    = [f for f in data_files if not f.exists()]
        if missing:
            print(f"[error] File(s) not found: {', '.join(str(f) for f in missing)}")
            sys.exit(1)
    else:
        raw_dir    = repo_path / "raw"
        data_files = [f for f in raw_dir.iterdir()
                      if f.is_file() and not f.name.startswith(".")]
        if not data_files:
            print(f"[error] No files found in {raw_dir}")
            sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   DSV File Watcher — Auto Dataset Versioning            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        watch(
            data_files  = data_files,
            interval    = args.interval,
            storage     = storage,
            manager     = manager,
            auto_train  = not args.no_train,
            auto_dashboard = not args.no_dashboard,
        )
    except KeyboardInterrupt:
        print("\n[watch] Stopped.")


if __name__ == "__main__":
    main()
