"""
cli.py — DSV Command-Line Interface Entry Point
================================================
Exposes a single ``main()`` function registered as the ``dsv`` console
script by setuptools.  All subcommand logic lives in ``dsv.py``; this
module only wires up the argument parser and dispatches to it.

Usage after ``pip install -e .``::

    dsv init
    dsv add --data dataset_repo/raw/dataset.txt
    dsv commit --name v1 --message "initial version"
    dsv list
    dsv show v1
    dsv compare v1 v2
    dsv lineage
    dsv dashboard
    dsv train --version v1
    dsv train --compare v1 v2
    dsv train --all
    dsv watch [--data FILE] [--interval N] [--no-train] [--no-dashboard]
    dsv rollback v1
  dsv insights version v1
  dsv insights compare v1 v2
  dsv insights suggest
  dsv insights report
  dsv demo
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# Re-export helpers from dsv.py so external callers can import from cli too
# ---------------------------------------------------------------------------
from dsv.dsv import (
    cmd_init,
    cmd_add,
    cmd_commit,
    cmd_status,
    cmd_create,
    cmd_list,
    cmd_show,
    cmd_compare,
    cmd_lineage,
    cmd_dashboard,
    cmd_rollback,
    cmd_watch,
    _c, ok, err, info, warn,
    BOLD, CYAN, DIM, RESET,
)
from dsv import __version__
from dsv.insights import cmd_insights
from dsv.demo import run_demo


# ---------------------------------------------------------------------------
# Train command — delegates to train.py's main logic
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Train a Naive-Bayes classifier on one or more dataset versions.

    Delegates entirely to the existing ``train.py`` logic by reconstructing
    the argument namespace that ``train.main()`` expects.
    """
    import importlib
    # Import the packaged copy of train.py
    train_mod = importlib.import_module("dsv.train")

    # Build a fresh parser from train.py so we respect its exact arg shape,
    # then manually populate it from *our* already-parsed args.
    train_parser = train_mod.build_parser()

    # Reconstruct sys.argv so train.main() → train.build_parser().parse_args()
    # picks up the right flags.
    argv = ["dsv train", "--repo", args.repo]

    if getattr(args, "version", None):
        argv += ["--version", args.version]
    elif getattr(args, "compare_versions", None):
        argv += ["--compare"] + list(args.compare_versions)
    elif getattr(args, "all_versions", False):
        argv += ["--all"]
    else:
        err("Specify --version NAME, --compare A B, or --all")
        sys.exit(1)

    # Temporarily override sys.argv and run train.main()
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        train_mod.main()
    finally:
        sys.argv = old_argv



def cmd_demo(args: argparse.Namespace) -> None:
    """Run the full DSV demo: 5 versions, train all, open dashboard."""
    from dsv.demo import run_demo
    run_demo(
        repo_path=args.repo,
        reset=not getattr(args, "no_reset", False),
    )

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser for the ``dsv`` CLI."""
    parser = argparse.ArgumentParser(
        prog="dsv",
        description=(
            "DSV — Dataset Versioning System\n"
            "A git-style CLI for versioning, comparing, and training on datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dsv init\n"
            "  dsv add --data dataset_repo/raw/dataset.txt\n"
            "  dsv commit --name v1 --message 'initial version'\n"
            "  dsv list\n"
            "  dsv show v1\n"
            "  dsv compare v1 v2\n"
            "  dsv lineage\n"
            "  dsv dashboard --port 8765\n"
            "  dsv train --version v1\n"
            "  dsv train --compare v1 v2\n"
            "  dsv train --all\n"
            "  dsv watch --data dataset_repo/raw/dataset.txt\n"
            "  dsv rollback v1\n"
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"dsv {__version__}",
        help="Show DSV version and exit.",
    )
    parser.add_argument(
        "--repo",
        default="dataset_repo",
        metavar="PATH",
        help="Path to the dataset repository (default: dataset_repo).",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── Repository management ──────────────────────────────────────────────
    sub.add_parser(
        "init",
        help="Initialise a new DSV repository.",
        description="Creates the dataset_repo/ folder structure and sets up DSV tracking.",
    )

    p_add = sub.add_parser(
        "add",
        help="Stage a dataset file and/or preprocessing config for the next commit.",
        description=(
            "Stage changes before committing.  Works like `git add`.\n\n"
            "  dsv add --data dataset_repo/raw/dataset.txt\n"
            "  dsv add --config config.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_add.add_argument("--data",   default=None, metavar="FILE",
                       help="Raw dataset file to stage.")
    p_add.add_argument("--config", default=None, metavar="FILE",
                       help="Preprocessing config JSON to stage.")

    p_commit = sub.add_parser(
        "commit",
        help="Create a new version from staged changes.",
        description=(
            "Commit staged changes as a named version.  Works like `git commit`.\n\n"
            "  dsv commit --name v1 --message 'initial dataset'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_commit.add_argument("--name",    required=True, metavar="NAME",
                          help="Unique version name (e.g. v1, v2-clean).")
    p_commit.add_argument("--message", required=True, metavar="MSG",
                          help="Short description of the changes.")
    p_commit.add_argument("--parent",  default=None,  metavar="VERSION",
                          help="Explicit parent version (auto-detected if omitted).")
    p_commit.add_argument("--step",    default=None,  metavar="DESCRIPTION",
                          help="Transformation step label (e.g. 'remove_stopwords').")

    sub.add_parser(
        "status",
        help="Show staged changes and working tree state.",
    )

    sub.add_parser(
        "list",
        help="List all committed versions in the repository.",
    )

    p_show = sub.add_parser(
        "show",
        help="Show full details of a specific version.",
    )
    p_show.add_argument("version_ref", metavar="VERSION",
                        help="Version name or hash prefix to display.")

    p_compare = sub.add_parser(
        "compare",
        help="Compare metrics and config between two versions.",
    )
    p_compare.add_argument("version_ref_1", metavar="VERSION_A")
    p_compare.add_argument("version_ref_2", metavar="VERSION_B")

    p_lineage = sub.add_parser(
        "lineage",
        help="Show the version lineage DAG in the terminal.",
    )
    p_lineage.add_argument(
        "version_ref", nargs="?", default=None, metavar="VERSION",
        help="Show lineage path to this version (shows full DAG if omitted).",
    )

    p_dash = sub.add_parser(
        "dashboard",
        help="Launch the interactive visual dashboard in your browser.",
    )
    p_dash.add_argument("--port", type=int, default=8765, metavar="PORT",
                        help="HTTP port to serve the dashboard on (default: 8765).")

    # ── Training ────────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Train a Naive Bayes classifier on one or more dataset versions.",
        description=(
            "Train and evaluate a Bag-of-Words Naive Bayes model.\n\n"
            "  dsv train --version v1\n"
            "  dsv train --compare v1 v2\n"
            "  dsv train --all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_group = p_train.add_mutually_exclusive_group(required=True)
    train_group.add_argument(
        "--version", metavar="NAME",
        help="Train on a single dataset version.",
    )
    train_group.add_argument(
        "--compare", nargs=2, metavar=("VERSION_A", "VERSION_B"),
        dest="compare_versions",
        help="Train on two versions and compare their reports.",
    )
    train_group.add_argument(
        "--all", action="store_true", dest="all_versions",
        help="Train on ALL versions in the repository.",
    )

    # ── Watcher ─────────────────────────────────────────────────────────────
    p_watch = sub.add_parser(
        "watch",
        help="Auto-version datasets whenever a raw file changes on disk.",
        description=(
            "Polls raw/ for file changes and commits a new version automatically.\n\n"
            "  dsv watch\n"
            "  dsv watch --data dataset_repo/raw/dataset.txt --interval 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_watch.add_argument("--data",         metavar="FILE", action="append", dest="files",
                         help="File to watch (can be repeated; watches all of raw/ if omitted).")
    p_watch.add_argument("--interval",     type=float, default=2.0, metavar="SECONDS",
                         help="Polling interval in seconds (default: 2).")
    p_watch.add_argument("--no-train",     action="store_true", dest="no_train",
                         help="Skip auto-training after each auto-version.")
    p_watch.add_argument("--no-dashboard", action="store_true", dest="no_dashboard",
                         help="Skip refreshing dashboard data after each auto-version.")

    # ── Legacy / power-user commands ────────────────────────────────────────
    p_create = sub.add_parser(
        "create",
        help="[Legacy] Create a version directly, bypassing the staging area.",
        description=(
            "Direct version creation — skips `dsv add` staging.  Kept for\n"
            "backward compatibility; prefer `dsv add` + `dsv commit` instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_create.add_argument("--data",    required=True, metavar="FILE",
                          help="Dataset file to version.")
    p_create.add_argument("--name",    required=True, metavar="NAME",
                          help="Version name.")
    p_create.add_argument("--message", required=True, metavar="MSG",
                          help="Commit message.")
    p_create.add_argument("--parent",  default=None,  metavar="VERSION")
    p_create.add_argument("--step",    default=None,  metavar="DESCRIPTION")


    # ── Demo ────────────────────────────────────────────────────────────────
    p_demo = sub.add_parser(
        "demo",
        help="One-command demo: creates 5 versions, trains, opens dashboard.",
        description=(
            "Runs a full end-to-end DSV demo in your current repo:\n\n"
            "  1. Resets the repo (unless --no-reset)\n"
            "  2. Creates 5 versions with a branching DAG\n"
            "  3. Trains a Naive Bayes model on each version\n"
            "  4. Opens the interactive dashboard in your browser\n\n"
            "Requires a labeled dataset at dataset_repo/raw/dataset.txt\n"
            "(format: text|label  one row per line)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_demo.add_argument(
        "--no-reset", action="store_true", dest="no_reset",
        help="Keep existing versions — only rebuild dashboard data.",
    )

    # ── AI Insights ─────────────────────────────────────────────────────────
    p_insights = sub.add_parser(
        "insights",
        help="AI-powered analysis of your dataset versions (requires GOOGLE_API_KEY).",
        description=(
            "Use Gemini AI to analyse your datasets, compare versions,\n"
            "and get concrete recommendations for what to try next.\n\n"
            "  dsv insights version v1\n"
            "  dsv insights compare v1 v2\n"
            "  dsv insights suggest\n"
            "  dsv insights report\n\n"
            "Requires: export GOOGLE_API_KEY=your-api-key-here"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    insights_sub = p_insights.add_subparsers(dest="insights_command", metavar="SUBCOMMAND")
    insights_sub.required = True

    p_iv = insights_sub.add_parser(
        "version",
        help="Deep-dive AI analysis of a single version.",
    )
    p_iv.add_argument("version_ref", metavar="VERSION",
                      help="Version name to analyse.")

    p_ic = insights_sub.add_parser(
        "compare",
        help="AI comparison of two versions.",
    )
    p_ic.add_argument("version_ref_1", metavar="VERSION_A")
    p_ic.add_argument("version_ref_2", metavar="VERSION_B")

    insights_sub.add_parser(
        "suggest",
        help="AI recommendations: what preprocessing combos to try next.",
    )

    insights_sub.add_parser(
        "report",
        help="Full AI repository health report across all versions.",
    )

    p_rollback = sub.add_parser(
        "rollback",
        help="Restore a previous version's preprocessing config to config.json.",
        description=(
            "Writes a past version's preprocessing config back to config.json\n"
            "and moves HEAD to that version.  No versions are deleted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_rollback.add_argument("version_ref", metavar="VERSION",
                            help="The version to roll back to.")

    return parser


# ---------------------------------------------------------------------------
# Command dispatch table
# ---------------------------------------------------------------------------

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
    "train":     cmd_train,
    "insights":  cmd_insights,
    "demo":      cmd_demo,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point registered as the ``dsv`` console script.

    Parses command-line arguments and dispatches to the appropriate handler.
    Catches unexpected exceptions at the top level to avoid raw tracebacks
    being shown to users; use ``DSV_DEBUG=1`` to re-enable full tracebacks.
    """
    import os
    debug = os.environ.get("DSV_DEBUG", "").strip() not in ("", "0")

    parser = build_parser()
    args = parser.parse_args()

    try:
        COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print()
        warn("Interrupted.")
        sys.exit(130)
    except SystemExit:
        raise  # let argparse / sys.exit() pass through normally
    except Exception as exc:  # pylint: disable=broad-except
        if debug:
            raise
        err(f"Unexpected error: {exc}")
        info("Set DSV_DEBUG=1 for a full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
