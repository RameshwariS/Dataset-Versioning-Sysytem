"""
dsv.py  — Dataset Versioning System CLI

Git-style workflow (recommended):
    dsv init
    dsv add --data FILE [--config FILE]
    dsv commit --name NAME --message MSG [--parent V] [--step S]
    dsv status

Other commands:
    python dsv.py list
    python dsv.py show VERSION
    python dsv.py compare VERSION_A VERSION_B
    python dsv.py lineage [VERSION]
    python dsv.py dashboard [--port PORT]
    python dsv.py watch [--data FILE] [--interval N] [--no-train] [--no-dashboard]
    python dsv.py create --data FILE --name NAME --message MSG [--parent V] [--step S]
"""

import argparse
import hashlib
import http.server
import json
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .storage import StorageManager
from .versioning import DatasetVersionManager

PREPROCESSING_STEPS = [
    "lowercase", "remove_punctuation", "remove_duplicates",
    "strip_whitespace", "tokenize", "remove_stopwords",
]

NUMERIC_PREPROCESSING_STEPS = [
    "drop_missing", "drop_duplicates", "normalize",
    "standardize", "remove_outliers", "round_decimals",
]

DATASET_TYPE_TXT     = "txt"
DATASET_TYPE_NUMERIC = "numeric"
DATASET_TYPE_KEY     = "__dataset_type__"

DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"
DASHBOARD_DATA = Path.cwd() / "dashboard_data.json"
STAGE_FILE     = Path.cwd() / ".dsv_stage.json"
HEAD_FILE      = Path.cwd() / ".dsv_head"

# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR_ON = _supports_color()

def _c(color, text):
    if _COLOR_ON:
        return color + text + RESET
    return text

def ok(msg):   print(_c(GREEN,  "[ok] ")    + msg)
def err(msg):  print(_c(RED,    "[error] ") + msg)
def info(msg): print(_c(CYAN,   "[info] ")  + msg)
def warn(msg): print(_c(YELLOW, "[warn] ")  + msg)


# ---------------------------------------------------------------------------
# Staging area helpers
# ---------------------------------------------------------------------------

def _read_stage():
    if not STAGE_FILE.exists():
        return {}
    try:
        return json.loads(STAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_stage(stage):
    STAGE_FILE.write_text(json.dumps(stage, indent=2), encoding="utf-8")

def _clear_stage():
    if STAGE_FILE.exists():
        STAGE_FILE.unlink()

def _md5(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# HEAD helpers
# ---------------------------------------------------------------------------

def _read_head():
    """Return the current HEAD version name, or None."""
    if not HEAD_FILE.exists():
        return None
    return HEAD_FILE.read_text(encoding="utf-8").strip() or None

def _write_head(version_name):
    HEAD_FILE.write_text(version_name, encoding="utf-8")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _require_repo(storage):
    if not storage.is_initialised():
        err("Repository not initialised. Run " + _c(BOLD, "dsv init") + " first.")
        sys.exit(1)

def _prompt_config():
    print("\n  Configure preprocessing steps (Enter = default y):")
    print("  " + "-" * 45)
    config = {}
    for step in PREPROCESSING_STEPS:
        while True:
            raw = input("  Enable '" + step + "'? [y/n] (default: y): ").strip().lower()
            if raw in ("", "y", "yes"):
                config[step] = True; break
            elif raw in ("n", "no"):
                config[step] = False; break
            else:
                print("    Please enter y or n.")
    config[DATASET_TYPE_KEY] = DATASET_TYPE_TXT
    print()
    return config

def _select_dataset_type() -> str:
    """
    Prompt the user to choose between a text dataset and a numeric/tabular dataset.
    Returns DATASET_TYPE_TXT or DATASET_TYPE_NUMERIC.
    """
    print()
    print("  " + _c(BOLD, "═" * 50))
    print("  " + _c(BOLD, "  DSV — Dataset Versioning System"))
    print("  " + _c(BOLD, "═" * 50))
    print()
    print("  " + _c(CYAN, "Select your dataset type:"))
    print()
    print("    " + _c(GREEN,  "[1]") + "  " + _c(BOLD, "Text Dataset") +
          "       (.txt — sentences, documents, text corpus)")
    print("    " + _c(YELLOW, "[2]") + "  " + _c(BOLD, "Numeric Dataset") +
          "    (.csv — tabular data with numeric columns)")
    print()
    while True:
        choice = input("  Enter choice [1/2] (default: 1): ").strip()
        if choice in ("", "1"):
            print()
            ok("Text dataset selected.")
            return DATASET_TYPE_TXT
        elif choice == "2":
            print()
            ok("Numeric dataset selected.")
            return DATASET_TYPE_NUMERIC
        else:
            print("    Please enter 1 or 2.")


def _prompt_numeric_config():
    """Prompt for numeric preprocessing config."""
    print("\n  Configure numeric preprocessing steps (Enter = default y):")
    print("  " + "-" * 55)
    print("  " + _c(DIM, "Note: 'normalize' and 'standardize' are mutually exclusive;"))
    print("  " + _c(DIM, "      if both are enabled, normalize takes priority."))
    print()
    config = {}
    for step in NUMERIC_PREPROCESSING_STEPS:
        while True:
            raw = input("  Enable '" + step + "'? [y/n] (default: y): ").strip().lower()
            if raw in ("", "y", "yes"):
                config[step] = True; break
            elif raw in ("n", "no"):
                config[step] = False; break
            else:
                print("    Please enter y or n.")
    config[DATASET_TYPE_KEY] = DATASET_TYPE_NUMERIC
    print()
    return config


def _validate_name(name, storage):
    if not name:
        err("Version name cannot be empty."); sys.exit(1)
    if not re.match(r'^[\w\.\-]+$', name):
        err("Invalid name. Use letters, digits, hyphens, underscores, dots.")
        sys.exit(1)
    if storage.name_exists(name):
        err("Version name '" + name + "' is already in use."); sys.exit(1)

def _display_config(config):
    dataset_type = config.get(DATASET_TYPE_KEY, DATASET_TYPE_TXT)
    steps = NUMERIC_PREPROCESSING_STEPS if dataset_type == DATASET_TYPE_NUMERIC else PREPROCESSING_STEPS
    dtype_label = _c(YELLOW, "[numeric]") if dataset_type == DATASET_TYPE_NUMERIC else _c(CYAN, "[text]")
    print("  " + _c(DIM, "Dataset type: ") + dtype_label)
    for key in steps:
        enabled = config.get(key, False)
        mark    = _c(GREEN, "[+]") if enabled else _c(RED, "[-]")
        status  = _c(GREEN, "enabled") if enabled else _c(DIM, "disabled")
        print("  " + mark + " " + key.ljust(25) + " " + status)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    import shutil as _shutil
    storage = StorageManager(repo_path=args.repo)
    dataset_type = _select_dataset_type()
    storage.init(dataset_type=dataset_type)

    # --- copy dashboard.html from package into cwd so `dsv dashboard` works ---
    _pkg_dir = Path(__file__).parent          # dsv/ inside site-packages or editable src
    _dash_src = _pkg_dir / "dashboard.html"
    _dash_dst = Path.cwd() / "dashboard.html"
    if _dash_src.exists() and not _dash_dst.exists():
        _shutil.copy2(_dash_src, _dash_dst)

    # --- write a starter config.json if none exists ---
    _cfg_dst = Path.cwd() / "config.json"
    if not _cfg_dst.exists():
        _cfg_dst.write_text(
            json.dumps({
                "_comment": (
                    "Edit this file to change preprocessing. "
                    "Then run: dsv add --config config.json && "
                    "dsv commit --name <name> --message <msg>"
                ),
                "lowercase":           True,
                "strip_whitespace":    True,
                "remove_punctuation":  True,
                "remove_duplicates":   True,
                "tokenize":            False,
                "remove_stopwords":    False,
            }, indent=2),
            encoding="utf-8",
        )

    ok("Initialised repository at " + _c(BOLD, "'" + args.repo + "/'"))
    print("     Place raw datasets in " + _c(CYAN, "'" + args.repo + "/raw/'"))
    if (_dash_src.exists()):
        print("     " + _c(DIM, "dashboard.html") + " copied  → " + str(_dash_dst))
    print("     " + _c(DIM, "config.json")    + "    written → " + str(_cfg_dst) if not _cfg_dst.exists() else
          "     " + _c(DIM, "config.json")    + "    ready   → edit to change preprocessing steps")
    print()
    print(_c(DIM, "  Next steps:"))
    print(_c(DIM, "    dsv add --data " + args.repo + "/raw/<your_dataset.txt>"))
    print(_c(DIM, "    dsv add --config config.json        # optional: tweak preprocessing first"))
    print(_c(DIM, "    dsv commit --name v1 --message 'initial version'"))


def cmd_add(args):
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)

    stage   = _read_stage()
    changed = False

    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            err("Dataset file not found: " + args.data); sys.exit(1)
        stage["data"]     = str(data_path.resolve())
        stage["data_md5"] = _md5(data_path)
        print(_c(GREEN, "  staged: ") + _c(BOLD, "data") + "   → " + str(data_path))
        changed = True

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            err("Config file not found: " + args.config); sys.exit(1)
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            err("Config file is not valid JSON: " + args.config); sys.exit(1)
        # Strip _comment keys — they are for humans, not hashing
        config_data = {k: v for k, v in config_data.items() if not k.startswith("_")}
        stage["config"]      = config_data
        stage["config_path"] = str(config_path.resolve())
        stage["config_md5"]  = _md5(config_path)
        print(_c(GREEN, "  staged: ") + _c(BOLD, "config") + " → " + str(config_path))
        changed = True

    if not changed:
        warn("Nothing to stage. Use --data FILE and/or --config FILE.")
        sys.exit(1)

    _write_stage(stage)
    print()
    info("Changes staged. Run " + _c(BOLD, "dsv commit --name NAME --message MSG") + " to create a version.")
    info("Run " + _c(BOLD, "dsv status") + " to see what is staged.")


def cmd_commit(args):
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)

    stage = _read_stage()

    if not stage:
        err("Nothing staged. Run " + _c(BOLD, "dsv add --data FILE [--config FILE]") + " first.")
        sys.exit(1)

    # If only config was staged (no data), reuse the latest version's raw data file
    if "data" not in stage:
        if "config" not in stage:
            err("Nothing staged. Run " + _c(BOLD, "dsv add --data FILE [--config FILE]") + " first.")
            sys.exit(1)
        # Find latest version and reuse its raw data path
        latest = storage.get_latest_version()
        if not latest:
            err("No previous version found. Cannot infer data file.")
            err("Run " + _c(BOLD, "dsv add --data FILE --config FILE") + " to stage both.")
            sys.exit(1)
        latest_data = storage.load_version(latest["version_id"])
        # The raw data is stored in raw/ — use the default dataset path
        raw_files = list(storage.raw_path.glob("*")) if storage.raw_path.exists() else []
        raw_files = [f for f in raw_files if f.is_file()]
        if not raw_files:
            err("No raw data file found in " + str(storage.raw_path))
            err("Run " + _c(BOLD, "dsv add --data FILE --config FILE") + " to stage both.")
            sys.exit(1)
        # Use the first (or only) raw file
        data_path = raw_files[0]
        info("No data staged — reusing raw data file: " + _c(BOLD, str(data_path)))
    else:
        data_path = Path(stage["data"])
        if not data_path.exists():
            err("Staged data file no longer exists: " + str(data_path))
            err("Run " + _c(BOLD, "dsv add --data FILE") + " again.")
            sys.exit(1)
        current_md5 = _md5(data_path)
        if current_md5 != stage.get("data_md5"):
            warn("Data file has changed since it was staged!")
            warn("Run " + _c(BOLD, "dsv add --data " + str(data_path)) + " again to re-stage.")
            sys.exit(1)

    _validate_name(args.name, storage)

    if "config" in stage:
        config = stage["config"]
        print("\n  " + _c(CYAN, "Using staged config:"))
        _display_config(config)
        print()
    else:
        # Route to correct config prompt based on stored dataset type
        dataset_type = storage.get_dataset_type()
        if dataset_type == DATASET_TYPE_NUMERIC:
            config = _prompt_numeric_config()
        else:
            config = _prompt_config()
        print("  Final config:")
        _display_config(config)

    parent_ref = getattr(args, "parent", None)
    step_label = getattr(args, "step", None)

    if parent_ref:
        print("\n  " + _c(DIM, "Parent  :") + " " + parent_ref)
    else:
        latest = storage.get_latest_version()
        if latest:
            print("\n  " + _c(DIM, "Parent  :") + " " + _c(YELLOW, latest["name"]) + "  (auto-detected latest)")
        else:
            print("\n  " + _c(DIM, "Parent  :") + " none  (this will be the root version)")

    print("  " + _c(DIM, "Name    :") + " " + _c(BOLD, args.name))
    print("  " + _c(DIM, "Message :") + " " + args.message)
    if step_label:
        print("  " + _c(DIM, "Step    :") + " " + step_label)
    print("  " + _c(DIM, "Data    :") + " " + str(data_path))

    confirm = input("\n  Proceed? [y/n] (default: y): ").strip().lower()
    if confirm not in ("", "y", "yes"):
        warn("Commit cancelled."); sys.exit(0)

    manager = DatasetVersionManager(storage=storage)
    result  = manager.create_version(
        str(data_path), config,
        name=args.name, message=args.message,
        parent_ref=parent_ref, transformation_step=step_label,
    )

    vid = result["version_id"]
    print()
    if result["created"]:
        ok("Version " + _c(BOLD, "'" + result["name"] + "'") + " committed successfully.")
        if result.get("parent_name"):
            print("     " + _c(DIM, "Parent :") + " " + result["parent_name"])
        print("     " + _c(DIM, "Hash   :") + " " + _c(CYAN, vid[:16]) + _c(DIM, "... (64-char SHA-256)"))
    else:
        info("Already exists as " + _c(BOLD, "'" + result["name"] + "'") + " (same hash — no new version created).")
        print("     " + _c(DIM, "Hash   :") + " " + _c(CYAN, vid[:16]) + "...")

    m = result["metrics"]
    dataset_type = config.get(DATASET_TYPE_KEY, DATASET_TYPE_TXT)
    print()
    if dataset_type == DATASET_TYPE_NUMERIC:
        print("     " + _c(DIM, "Rows          :") + " " + str(m.get("num_rows", 0)))
        print("     " + _c(DIM, "Columns       :") + " " + str(m.get("num_columns", 0)))
        print("     " + _c(DIM, "Numeric cols  :") + " " + str(m.get("numeric_columns", 0)))
        print("     " + _c(DIM, "Missing cells :") + " " + str(m.get("missing_cells", 0)))
    else:
        print("     " + _c(DIM, "Rows          :") + " " + str(m["num_rows"]))
        print("     " + _c(DIM, "Vocabulary    :") + " " + str(m["vocabulary_size"]) + " unique tokens")
        print("     " + _c(DIM, "Total tokens  :") + " " + str(m["total_tokens"]))
        print("     " + _c(DIM, "Avg sent. len :") + " " + str(m["avg_sentence_len"]) + " tokens")

    _clear_stage()
    _write_head(result["name"])
    print()
    ok("Staging area cleared. Working tree is clean.")
    info("HEAD is now at: " + _c(CYAN, result["name"]))


def cmd_status(args):
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)

    stage  = _read_stage()
    latest = storage.get_latest_version()

    head_name = _read_head()
    print()
    if head_name:
        print("  " + _c(DIM, "HEAD           :") + " " + _c(CYAN, head_name))
    if latest:
        print("  " + _c(DIM, "Latest version :") + " " + _c(BOLD, latest["name"]) +
              "  " + _c(DIM, "(" + latest["created_at"] + ")"))
    else:
        print("  " + _c(DIM, "No versions committed yet."))
    print()

    if stage:
        print(_c(GREEN, "  Changes staged for commit:"))
        print(_c(DIM,   "  (use \"dsv commit --name NAME --message MSG\" to create a version)"))
        print()

        if "data" in stage:
            data_path = Path(stage["data"])
            if not data_path.exists():
                print(_c(RED,    "  missing:  ") + _c(BOLD, str(data_path)) + _c(RED, "  ← file deleted after staging!"))
            elif _md5(data_path) != stage.get("data_md5"):
                print(_c(YELLOW, "  modified: ") + _c(BOLD, str(data_path)) + _c(YELLOW, "  ← changed since staged, re-run add"))
            else:
                print(_c(GREEN,  "  staged:   ") + _c(BOLD, str(data_path)))

        if "config" in stage:
            config_path_str = stage.get("config_path", "")
            if config_path_str and Path(config_path_str).exists():
                cp = Path(config_path_str)
                if _md5(cp) != stage.get("config_md5"):
                    print(_c(YELLOW, "  modified: ") + _c(BOLD, str(cp)) + _c(YELLOW, "  ← changed since staged, re-run add"))
                else:
                    print(_c(GREEN,  "  staged:   ") + _c(BOLD, str(cp)))
            else:
                print(_c(GREEN,  "  staged:   ") + _c(BOLD, "config (inline)"))

            print()
            print(_c(DIM, "  Staged preprocessing config:"))
            _display_config(stage["config"])
    else:
        print(_c(DIM, "  Nothing staged."))
        print(_c(DIM, "  (use \"dsv add --data FILE [--config FILE]\" to stage changes)"))

        raw_path = storage.raw_path
        if raw_path.exists():
            raw_files = [f for f in raw_path.glob("*") if f.is_file()]
            if raw_files:
                print()
                print(_c(YELLOW, "  Untracked / potentially modified files in raw/:"))
                print(_c(DIM,    "  (use \"dsv add --data FILE\" to stage)"))
                print()
                for f in raw_files:
                    print(_c(YELLOW, "  untracked: ") + str(f))

    print()
    default_config = Path.cwd() / "config.json"
    if default_config.exists():
        print(_c(BOLD, "  To change preprocessing config:"))
        print(_c(DIM,  "    1. Edit ") + _c(CYAN, "config.json") + _c(DIM, " in this folder (enable/disable steps)"))
        print(_c(DIM,  "    2. Run  ") + _c(BOLD, "dsv add --config config.json"))
        print(_c(DIM,  "    3. Run  ") + _c(BOLD, "dsv commit --name NAME --message MSG"))
        print(_c(DIM,  "  (config_full.json no longer exists — config.json is the only config file)"))
    print()


def cmd_create(args):
    """Legacy: create a version directly without staging."""
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)

    data_path = Path(args.data)
    if not data_path.exists():
        err("Dataset file not found: " + args.data); sys.exit(1)

    _validate_name(args.name, storage)

    parent_ref = getattr(args, "parent", None)
    step_label = getattr(args, "step", None)

    if parent_ref:
        print("\n  " + _c(DIM, "Parent  :") + " " + parent_ref)
    else:
        latest = storage.get_latest_version()
        if latest:
            print("\n  " + _c(DIM, "Parent  :") + " " + _c(YELLOW, latest["name"]) + "  (auto-detected latest)")
        else:
            print("\n  " + _c(DIM, "Parent  :") + " none  (root version)")

    print("  " + _c(DIM, "Name    :") + " " + _c(BOLD, args.name))
    print("  " + _c(DIM, "Message :") + " " + args.message)
    if step_label:
        print("  " + _c(DIM, "Step    :") + " " + step_label)
    print("  " + _c(DIM, "Dataset :") + " " + args.data)

    # Route to correct config prompt based on stored dataset type
    dataset_type = storage.get_dataset_type()
    if dataset_type == DATASET_TYPE_NUMERIC:
        config = _prompt_numeric_config()
    else:
        config = _prompt_config()
    print("  Final config:")
    _display_config(config)

    confirm = input("\n  Proceed? [y/n] (default: y): ").strip().lower()
    if confirm not in ("", "y", "yes"):
        warn("Version creation cancelled."); sys.exit(0)

    manager = DatasetVersionManager(storage=storage)
    result  = manager.create_version(
        str(data_path), config,
        name=args.name, message=args.message,
        parent_ref=parent_ref, transformation_step=step_label,
    )

    vid = result["version_id"]
    print()
    if result["created"]:
        ok("Version " + _c(BOLD, "'" + result["name"] + "'") + " created.")
        if result.get("parent_name"):
            print("     " + _c(DIM, "Parent :") + " " + result["parent_name"])
        print("     " + _c(DIM, "Hash   :") + " " + _c(CYAN, vid[:16]) + _c(DIM, "... (64-char SHA-256)"))
    else:
        info("Already exists as " + _c(BOLD, "'" + result["name"] + "'") + " (same hash).")
        print("     " + _c(DIM, "Hash   :") + " " + _c(CYAN, vid[:16]) + "...")

    m = result["metrics"]
    dataset_type = config.get(DATASET_TYPE_KEY, DATASET_TYPE_TXT)
    print()
    if dataset_type == DATASET_TYPE_NUMERIC:
        print("     " + _c(DIM, "Rows          :") + " " + str(m.get("num_rows", 0)))
        print("     " + _c(DIM, "Columns       :") + " " + str(m.get("num_columns", 0)))
        print("     " + _c(DIM, "Numeric cols  :") + " " + str(m.get("numeric_columns", 0)))
        print("     " + _c(DIM, "Missing cells :") + " " + str(m.get("missing_cells", 0)))
    else:
        print("     " + _c(DIM, "Rows          :") + " " + str(m["num_rows"]))
        print("     " + _c(DIM, "Vocabulary    :") + " " + str(m["vocabulary_size"]) + " unique tokens")
        print("     " + _c(DIM, "Total tokens  :") + " " + str(m["total_tokens"]))
        print("     " + _c(DIM, "Avg sent. len :") + " " + str(m["avg_sentence_len"]) + " tokens")


def cmd_list(args):
    storage  = StorageManager(repo_path=args.repo)
    _require_repo(storage)
    manager  = DatasetVersionManager(storage=storage)
    versions = manager.list_versions()

    if not versions:
        info("No versions found."); return

    latest_id  = versions[-1]["version_id"] if versions else None
    head_name  = _read_head()

    print()
    print("  " + _c(BOLD, "NAME".ljust(22)) + "  " + _c(BOLD, "TYPE".ljust(10)) + "  " + _c(BOLD, "PARENT".ljust(18)) + "  " +
          _c(BOLD, "CREATED".ljust(20)) + "  " + _c(BOLD, "ROWS".rjust(5)) + "  " + _c(BOLD, "VOC".rjust(5)))
    print("  " + _c(DIM, "-" * 92))

    for entry in versions:
        data      = storage.load_version(entry["version_id"])
        m         = data["metrics"]
        parent    = entry.get("parent_name") or "-"
        is_latest = entry["version_id"] == latest_id
        is_head   = entry["name"] == head_name
        stype     = entry.get("storage_type", "snapshot")

        markers = ""
        if is_head:   markers += _c(CYAN,  " ← HEAD")
        if is_latest and not is_head: markers += _c(GREEN, " ← latest")
        if is_head and is_latest:     markers = _c(CYAN, " ← HEAD") + _c(GREEN, ", latest")

        type_display = _c(YELLOW, stype.ljust(10)) if stype == "delta" else _c(CYAN, stype.ljust(10))

        row = ("  " + _c(CYAN if is_head else (GREEN if is_latest else RESET), entry["name"].ljust(22)) +
               "  " + type_display +
               "  " + parent.ljust(18) +
               "  " + entry["created_at"].ljust(20) +
               "  " + str(m["num_rows"]).rjust(5) +
               "  " + str(m["vocabulary_size"]).rjust(5) +
               "  " + markers)
        print(row)
        if entry["message"]:
            msg = entry["message"][:68] + ("..." if len(entry["message"]) > 68 else "")
            print("  " + " " * 22 + "  " + _c(DIM, "-> " + msg))
    print()


def cmd_show(args):
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)
    manager = DatasetVersionManager(storage=storage)
    try:
        data = manager.get_version(args.version_ref)
    except (FileNotFoundError, KeyError) as exc:
        err(str(exc)); sys.exit(1)

    m, c, meta, rows = data["metrics"], data["config"], data["meta"], data["rows"]
    print("\n" + _c(BOLD, "=" * 62))
    print("  " + _c(BOLD, "Name    :") + " " + _c(CYAN, meta.get("name", args.version_ref)))
    print("  " + _c(DIM,  "Hash    :") + " " + meta.get("version_id", ""))
    print("  " + _c(DIM,  "Created :") + " " + meta.get("created_at", ""))
    print("  " + _c(DIM,  "Message :") + " " + meta.get("message", ""))
    if meta.get("parent_name"):
        print("  " + _c(DIM, "Parent  :") + " " + meta["parent_name"])
    if meta.get("transformation_step"):
        print("  " + _c(DIM, "Step    :") + " " + meta["transformation_step"])
    
    stype = meta.get("storage_type", "snapshot")
    type_label = _c(YELLOW, stype) if stype == "delta" else _c(CYAN, stype)
    print("  " + _c(DIM,  "Storage :") + " " + type_label)
    if stype == "delta":
        print("  " + _c(DIM,  "Depth   :") + " " + str(meta.get("delta_depth", 0)))
    print(_c(BOLD, "=" * 62))

    print("\n  " + _c(BOLD, "METRICS"))
    print("  " + _c(DIM, "Rows".ljust(25))               + ": " + _c(CYAN, str(m["num_rows"])))
    print("  " + _c(DIM, "Vocabulary size".ljust(25))    + ": " + _c(CYAN, str(m["vocabulary_size"]) + " unique tokens"))
    print("  " + _c(DIM, "Total tokens".ljust(25))       + ": " + _c(CYAN, str(m["total_tokens"])))
    print("  " + _c(DIM, "Avg sentence length".ljust(25))+ ": " + _c(CYAN, str(m["avg_sentence_len"]) + " tokens"))

    print("\n  " + _c(BOLD, "PREPROCESSING CONFIG"))
    _display_config(c)

    print("\n  " + _c(BOLD, "SAMPLE ROWS") + _c(DIM, " (first 5 of " + str(len(rows)) + ")"))
    for i, row in enumerate(rows[:5], 1):
        preview = row[:80] + ("..." if len(row) > 80 else "")
        print("  " + _c(DIM, str(i).rjust(3) + ". ") + preview)
    print()


def cmd_compare(args):
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)
    manager = DatasetVersionManager(storage=storage)
    try:
        result = manager.compare_versions(args.version_ref_1, args.version_ref_2)
    except (FileNotFoundError, KeyError) as exc:
        err(str(exc)); sys.exit(1)

    n1, n2 = result["name_1"], result["name_2"]
    print("\n" + _c(BOLD, "=" * 66))
    print("  Comparing  " + _c(CYAN, "A: " + n1) + "   " + _c(YELLOW, "B: " + n2))
    print(_c(BOLD, "=" * 66))

    if result["identical"]:
        ok("Versions are IDENTICAL\n"); return

    print("\n  " + _c(BOLD, "METRIC DIFFERENCES") + _c(DIM, "  (B − A)"))
    print("  " + _c(DIM, "Metric".ljust(25) + "  " + n1.rjust(16) + "  " + n2.rjust(16) + "  " + "Delta".rjust(10)))
    print("  " + _c(DIM, "-" * 72))
    for key, entry in result["metrics_diff"].items():
        delta = entry["difference"]
        sign  = "+" if isinstance(delta, (int, float)) and delta > 0 else ""
        delta_str = sign + str(delta)
        if isinstance(delta, (int, float)):
            delta_colored = _c(GREEN, delta_str) if delta > 0 else (_c(RED, delta_str) if delta < 0 else _c(DIM, delta_str))
        else:
            delta_colored = delta_str
        print("  " + key.ljust(25) + "  " + str(entry["version_1"]).rjust(16) +
              "  " + str(entry["version_2"]).rjust(16) + "  " + delta_colored.rjust(10 + (len(delta_colored) - len(delta_str))))

    print("\n  " + _c(BOLD, "CONFIG DIFFERENCES"))
    if not result["config_diff"]:
        print(_c(DIM, "  (configs are identical)"))
    else:
        print("  " + _c(DIM, "Key".ljust(25) + "  " + n1.rjust(16) + "  " + n2.rjust(16)))
        print("  " + _c(DIM, "-" * 60))
        for key, entry in result["config_diff"].items():
            v1_s = str(entry["version_1"])
            v2_s = str(entry["version_2"])
            print("  " + key.ljust(25) + "  " + _c(RED, v1_s).rjust(16) + "  " + _c(GREEN, v2_s).rjust(16))
    print()


def cmd_lineage(args):
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)
    manager = DatasetVersionManager(storage=storage)

    if args.version_ref:
        try:
            path = manager.get_lineage_path(args.version_ref)
        except (FileNotFoundError, KeyError) as exc:
            err(str(exc)); sys.exit(1)
        print("\n  Lineage path for " + _c(BOLD, "'" + args.version_ref + "'") + ":")
        print("  " + " → ".join(_c(CYAN, p) for p in path))
    else:
        versions = manager.list_versions()
        if not versions:
            info("No versions found."); return

        children = {}
        id_to_v  = {v["version_id"]: v for v in versions}
        roots    = []
        for v in versions:
            pid = v.get("parent_id")
            if pid and pid in id_to_v:
                children.setdefault(pid, []).append(v["version_id"])
            else:
                roots.append(v["version_id"])

        print("\n  " + _c(BOLD, "VERSION LINEAGE DAG"))
        print("  " + _c(DIM, "-" * 50))

        def _print_tree(vid, prefix="", is_last=True):
            v    = id_to_v[vid]
            conn = "L- " if is_last else "|- "
            step = "  [" + v["transformation_step"] + "]" if v.get("transformation_step") else ""
            print("  " + _c(DIM, prefix + conn) + _c(CYAN, v["name"]) +
                  _c(YELLOW, step) + _c(DIM, "  (" + v["created_at"] + ")"))
            kids = children.get(vid, [])
            for i, kid in enumerate(kids):
                ext = "   " if is_last else "|  "
                _print_tree(kid, prefix + ext, i == len(kids) - 1)

        for i, root in enumerate(roots):
            _print_tree(root, "", i == len(roots) - 1)
    print()


def cmd_dashboard(args):
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)
    manager = DatasetVersionManager(storage=storage)

    if not DASHBOARD_HTML.exists():
        err("dashboard.html not found at " + str(DASHBOARD_HTML))
        sys.exit(1)

    dag = manager.build_dag()
    if not dag["nodes"]:
        info("No versions found — create some first.")
        sys.exit(0)

    # Inject latest and HEAD info into dashboard data
    versions = storage.list_versions()
    latest_id = versions[-1]["version_id"] if versions else None
    head_name = _read_head()
    dag["latest_id"]  = latest_id
    dag["head_name"]  = head_name

    DASHBOARD_DATA.write_text(json.dumps(dag, indent=2), encoding="utf-8")
    ok("Dashboard data written → " + str(DASHBOARD_DATA))

    serve_dir = str(Path.cwd().resolve())
    port = getattr(args, "port", 8765)

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
    url    = "http://localhost:" + str(port) + "/dashboard.html"

    ok("Dashboard running at: " + _c(CYAN, url))
    print("     Press " + _c(BOLD, "Ctrl-C") + " to stop.")

    def _open_browser():
        time.sleep(0.5)
        webbrowser.open(url)
    threading.Thread(target=_open_browser, daemon=True).start()

    # Windows-safe shutdown — handle both KeyboardInterrupt and SystemExit
    # and explicitly call server.shutdown() to unblock serve_forever()
    import signal

    def _shutdown(signum=None, frame=None):
        print()
        ok("Server stopped.")
        threading.Thread(target=server.shutdown, daemon=True).start()

    # Register Ctrl+C handler explicitly (more reliable on Windows)
    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        _shutdown()


def cmd_watch(args):
    warn("Watch mode creates versions automatically on every file save.")
    warn("For intentional versioning, prefer: " + _c(BOLD, "dsv add → dsv commit"))
    print()
    from .watch import main as watch_main
    sys.argv = ["watch.py"]
    if getattr(args, "files", None):
        for f in args.files:
            sys.argv += ["--data", f]
    if getattr(args, "interval", None):
        sys.argv += ["--interval", str(args.interval)]
    if getattr(args, "no_train", False):
        sys.argv.append("--no-train")
    if getattr(args, "no_dashboard", False):
        sys.argv.append("--no-dashboard")
    sys.argv += ["--repo", args.repo]
    watch_main()


def cmd_rollback(args):
    """Restore a previous version\'s config into config.json ready to re-commit."""
    storage = StorageManager(repo_path=args.repo)
    _require_repo(storage)
    manager = DatasetVersionManager(storage=storage)

    try:
        data = manager.get_version(args.version_ref)
    except (FileNotFoundError, KeyError) as exc:
        err(str(exc)); sys.exit(1)

    meta   = data["meta"]
    config = data["config"]
    name   = meta.get("name", args.version_ref)

    print()
    print("  " + _c(BOLD, "Rolling back config to version: ") + _c(CYAN, name))
    print("  " + _c(DIM, "Hash    : ") + meta.get("version_id", "")[:16] + "...")
    print("  " + _c(DIM, "Created : ") + meta.get("created_at", ""))
    print("  " + _c(DIM, "Message : ") + meta.get("message", ""))
    print()
    print("  " + _c(BOLD, "Config that will be restored:"))
    _display_config(config)
    print()
    warn("This will OVERWRITE your current config.json.")
    warn("No versions will be deleted — rollback only restores the working config.")
    print()

    confirm = input("  Proceed? [y/n] (default: y): ").strip().lower()
    if confirm not in ("", "y", "yes"):
        warn("Rollback cancelled."); sys.exit(0)

    # Write the restored config to config.json, preserving the _comment
    config_path = Path.cwd() / "config.json"
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    comment = existing.get("_comment", "Edit this file to change preprocessing. Then run: dsv add --config config.json && dsv commit --name <n> --message <msg>")

    restored = {"_comment": comment}
    restored.update(config)
    config_path.write_text(json.dumps(restored, indent=2), encoding="utf-8")

    # Move HEAD to the rolled-back version
    _write_head(name)

    # Clear any staged changes to avoid confusion
    _clear_stage()

    print()
    ok("Rolled back to version " + _c(CYAN, name) + " successfully.")
    print()
    print(_c(DIM, "  HEAD is now at: ") + _c(CYAN, name))
    print(_c(DIM, "  config.json has been restored to match this version."))
    print(_c(DIM, "  No versions were deleted — all history is intact."))
    print()
    print(_c(DIM, "  To branch off from here with new changes:"))
    print("    " + _c(BOLD, "  edit config.json or dataset_repo/raw/dataset.txt"))
    print("    " + _c(BOLD, "  dsv add --config config.json"))
    print("    " + _c(BOLD, "  dsv commit --name NEW_NAME --message MSG"))
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="dsv",
        description="Dataset Versioning System — Git-style CLI",
    )
    parser.add_argument("--repo", default="dataset_repo", metavar="PATH")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    sub.add_parser("init", help="Initialise a new repository.")

    p_add = sub.add_parser("add", help="Stage data and/or config changes.")
    p_add.add_argument("--data",   default=None, metavar="FILE", help="Raw dataset file to stage.")
    p_add.add_argument("--config", default=None, metavar="FILE", help="Preprocessing config JSON to stage.")

    p_commit = sub.add_parser("commit", help="Create a new version from staged changes.")
    p_commit.add_argument("--name",    required=True, metavar="NAME")
    p_commit.add_argument("--message", required=True, metavar="MSG")
    p_commit.add_argument("--parent",  default=None,  metavar="VERSION")
    p_commit.add_argument("--step",    default=None,  metavar="DESCRIPTION")

    sub.add_parser("status", help="Show staged changes and working tree state.")

    sub.add_parser("list", help="List all committed versions.")

    p_show = sub.add_parser("show", help="Show version details.")
    p_show.add_argument("version_ref")

    p_compare = sub.add_parser("compare", help="Compare two versions.")
    p_compare.add_argument("version_ref_1")
    p_compare.add_argument("version_ref_2")

    p_lineage = sub.add_parser("lineage", help="Show lineage DAG in terminal.")
    p_lineage.add_argument("version_ref", nargs="?", default=None)

    p_dash = sub.add_parser("dashboard", help="Serve dashboard via local HTTP server.")
    p_dash.add_argument("--port", type=int, default=8765, metavar="PORT")

    p_create = sub.add_parser("create", help="[Legacy] Create a version directly (bypasses staging).")
    p_create.add_argument("--data",    required=True, metavar="FILE")
    p_create.add_argument("--name",    required=True, metavar="NAME")
    p_create.add_argument("--message", required=True, metavar="MSG")
    p_create.add_argument("--parent",  default=None,  metavar="VERSION")
    p_create.add_argument("--step",    default=None,  metavar="DESCRIPTION")

    p_rollback = sub.add_parser("rollback", help="Restore a previous version\'s config into config.json.")
    p_rollback.add_argument("version_ref", metavar="VERSION", help="Version name to roll back to.")

    p_watch = sub.add_parser("watch", help="[Auto] Watch raw files and auto-version on change.")
    p_watch.add_argument("--data",         metavar="FILE", action="append", dest="files")
    p_watch.add_argument("--interval",     type=float, default=2.0, metavar="SECONDS")
    p_watch.add_argument("--no-train",     action="store_true", dest="no_train")
    p_watch.add_argument("--no-dashboard", action="store_true", dest="no_dashboard")

    return parser


COMMANDS = {
    "init":      cmd_init,
    "add":       cmd_add,
    "commit":    cmd_commit,
    "status":    cmd_status,
    "create":    cmd_create,
    "list":      cmd_list,
    "show":      cmd_show,
    "compare":   cmd_compare,
    "lineage":   cmd_lineage,
    "dashboard": cmd_dashboard,
    "rollback":  cmd_rollback,
    "watch":     cmd_watch,
}


def main():
    parser = build_parser()
    args   = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
