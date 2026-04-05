"""
demo.py — DSV one-command demo
================================
Reachable as:  ``dsv demo``

What it does
------------
1. Resets the repository (removes old versions) — skipped with ``--no-reset``
2. Creates 5 dataset versions with a proper DAG branching tree
3. Trains a Naive Bayes classifier on every version
4. Opens the interactive dashboard in your browser

This is a self-contained showcase of the full DSV workflow.

Usage
-----
    dsv demo
    dsv demo --no-reset        # keep existing versions, just rebuild dashboard
    dsv demo --repo my_repo    # use a non-default repo path
"""

import json
import shutil
import sys
import webbrowser
from pathlib import Path


# ---------------------------------------------------------------------------
# Version definitions
# ---------------------------------------------------------------------------

VERSIONS = [
    (
        "v1-raw",
        "Root: lowercase + strip only",
        None,
        "initial ingestion",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": False, "remove_duplicates": False,
         "tokenize": False, "remove_stopwords": False},
    ),
    (
        "v2-dedup-punct",
        "Branch A: add dedup + punctuation removal",
        "v1-raw",
        "remove_duplicates + remove_punctuation",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": False, "remove_stopwords": False},
    ),
    (
        "v3-tokenized",
        "Branch A cont.: tokenize on top of v2",
        "v2-dedup-punct",
        "tokenize",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": True, "remove_stopwords": False},
    ),
    (
        "v3-stopwords",
        "Branch B: stopwords removed (no tokenize)",
        "v2-dedup-punct",
        "remove_stopwords",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": False, "remove_stopwords": True},
    ),
    (
        "v4-full",
        "Full pipeline: all steps enabled",
        "v3-tokenized",
        "add remove_stopwords",
        {"lowercase": True, "strip_whitespace": True,
         "remove_punctuation": True, "remove_duplicates": True,
         "tokenize": True, "remove_stopwords": True},
    ),
]


# ---------------------------------------------------------------------------
# ANSI helpers (kept self-contained — no import from dsv.py)
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _c(code, text):
    import os
    if os.environ.get("NO_COLOR") or not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return text
    return code + text + RESET

def _ok(msg):   print(_c(GREEN, "  [ok] ") + msg)
def _info(msg): print(_c(CYAN,  "  [--] ") + msg)
def _step(msg): print(_c(BOLD,  "\n  >>> ") + msg)


# ---------------------------------------------------------------------------
# Demo logic
# ---------------------------------------------------------------------------

def run_demo(repo_path: str = "dataset_repo", reset: bool = True) -> None:
    """Run the full DSV demo end-to-end."""
    from .storage import StorageManager
    from .versioning import DatasetVersionManager
    from .train import train_and_evaluate

    repo     = Path(repo_path)
    raw_file = repo / "raw" / "dataset.txt"

    print()
    print(_c(BOLD, "  ╔══════════════════════════════════════════════════╗"))
    print(_c(BOLD, "  ║           DSV — One-command demo                 ║"))
    print(_c(BOLD, "  ╚══════════════════════════════════════════════════╝"))
    print()

    # ── 1. Reset ──────────────────────────────────────────────────────────
    if reset:
        _step("Resetting repository...")
        versions_dir = repo / "versions"
        registry     = repo / "registry.json"
        if versions_dir.exists():
            shutil.rmtree(versions_dir)
            _ok("Removed old versions/")
        if registry.exists():
            registry.unlink()
            _ok("Removed registry.json")
        # Remove head/stage files from cwd
        for f in [".dsv_head", ".dsv_stage.json"]:
            p = Path.cwd() / f
            if p.exists():
                p.unlink()
    else:
        _info("--no-reset: keeping existing versions")

    # ── 2. Init ───────────────────────────────────────────────────────────
    _step("Initialising repository...")
    storage = StorageManager(repo_path=repo_path)
    storage.init()
    _ok(f"Repository ready at '{repo_path}/'")

    # ── 3. Check raw dataset ──────────────────────────────────────────────
    if not raw_file.exists():
        print()
        print(_c(YELLOW, f"  [warn] No dataset found at {raw_file}"))
        print(_c(YELLOW,  "         Place a labeled dataset there first:"))
        print(_c(CYAN,   f"           {raw_file}"))
        print(_c(DIM,     "         Format: one row per line, text|label"))
        sys.exit(1)

    # ── 4. Create versions ────────────────────────────────────────────────
    _step(f"Creating {len(VERSIONS)} versions with a branching DAG...")
    manager = DatasetVersionManager(storage=storage)

    for name, message, parent_ref, step_label, config in VERSIONS:
        result = manager.create_version(
            str(raw_file), config,
            name=name, message=message,
            parent_ref=parent_ref, transformation_step=step_label,
        )
        status = "created" if result["created"] else "already exists"
        m = result["metrics"]
        _ok(
            f"{_c(CYAN, name):30s}  "
            f"rows={m['num_rows']}  "
            f"vocab={m['vocabulary_size']}  "
            f"({status})"
        )

    # ── 5. Train on all versions ──────────────────────────────────────────
    _step("Training a Naive Bayes classifier on all versions...")
    versions_list = storage.list_versions()
    for v_entry in versions_list:
        try:
            train_and_evaluate(v_entry["name"], storage, manager, verbose=False)
            _ok(f"Trained  {_c(CYAN, v_entry['name'])}")
        except Exception as exc:
            print(_c(YELLOW, f"  [warn] Skipped '{v_entry['name']}': {exc}"))

    # ── 6. Write dashboard data ───────────────────────────────────────────
    _step("Generating dashboard data...")
    dag          = manager.build_dag()
    versions_all = storage.list_versions()
    head_file    = Path.cwd() / ".dsv_head"
    dag["latest_id"] = versions_all[-1]["version_id"] if versions_all else None
    dag["head_name"] = head_file.read_text(encoding="utf-8").strip() if head_file.exists() else None
    data_file    = Path.cwd() / "dashboard_data.json"
    data_file.write_text(json.dumps(dag, indent=2), encoding="utf-8")
    _ok(f"dashboard_data.json written → {data_file}")

    # ── 7. Check dashboard.html ───────────────────────────────────────────
    dash_html = Path.cwd() / "dashboard.html"
    if not dash_html.exists():
        pkg_dash = Path(__file__).parent / "dashboard.html"
        if pkg_dash.exists():
            import shutil as _sh
            _sh.copy2(pkg_dash, dash_html)
            _ok(f"dashboard.html copied   → {dash_html}")
        else:
            print(_c(YELLOW, f"  [warn] dashboard.html not found. Run `dsv init` first."))

    # ── 8. Summary ────────────────────────────────────────────────────────
    print()
    print(_c(BOLD, "  ╔══════════════════════════════════════════════════╗"))
    print(_c(BOLD, "  ║  Demo complete!                                  ║"))
    print(_c(BOLD, "  ╚══════════════════════════════════════════════════╝"))
    print()
    print(f"  {len(VERSIONS)} versions created and trained.")
    print(f"  Try these commands:")
    print()
    print(_c(CYAN, "    dsv list"))
    print(_c(CYAN, "    dsv lineage"))
    print(_c(CYAN, "    dsv show v1-raw"))
    print(_c(CYAN, "    dsv compare v1-raw v4-full"))
    print(_c(CYAN, "    dsv train --all"))
    print(_c(CYAN, "    dsv dashboard"))
    print()

    # ── 9. Open dashboard ─────────────────────────────────────────────────
    if dash_html.exists():
        _step("Opening dashboard in browser...")
        _launch_dashboard(repo_path)


def _launch_dashboard(repo_path: str) -> None:
    """Serve and open the dashboard (same logic as cmd_dashboard)."""
    import http.server
    import socket
    import threading
    import time
    import signal

    serve_dir = str(Path.cwd().resolve())
    port = 8765

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)
        def log_message(self, fmt, *a):
            pass

    for candidate in range(port, port + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", candidate))
            port = candidate
            break
        except OSError:
            continue

    server = http.server.HTTPServer(("localhost", port), _Handler)
    url    = f"http://localhost:{port}/dashboard.html"

    print()
    _ok(f"Dashboard running at: {_c(CYAN, url)}")
    print(f"     Press {_c(BOLD, 'Ctrl-C')} to stop.")

    def _open_browser():
        time.sleep(0.6)
        webbrowser.open(url)
    threading.Thread(target=_open_browser, daemon=True).start()

    def _shutdown(signum=None, frame=None):
        print()
        _ok("Server stopped.")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        _shutdown()


# ---------------------------------------------------------------------------
# Entry point when called standalone (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demo()
