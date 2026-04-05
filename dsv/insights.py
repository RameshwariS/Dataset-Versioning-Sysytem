"""
insights.py — AI-powered analysis for DSV
==========================================
Provides the ``dsv insights`` command family.  Uses the Google Gemini API
(gemini-1.5-flash) to generate natural-language analysis of your
dataset versions, preprocessing decisions, and training results.

Sub-commands
------------
  dsv insights version VERSION      Deep-dive on a single version
  dsv insights compare A B           AI comparison of two versions
  dsv insights suggest               Recommendations for what to try next
  dsv insights report                Full repo health report (all versions)

Environment variable
--------------------
  GOOGLE_API_KEY      Must be set.  Obtain one at https://aistudio.google.com/app/apikey

Dependencies
------------
  google-generativeai   pip install google-generativeai
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Colour helpers (re-used from dsv.py style)
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _c(code, text):
    if os.environ.get("NO_COLOR") or not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
        return text
    return code + text + RESET

def ok(msg):    print(_c(GREEN,  "[ok] ")    + msg)
def err(msg):   print(_c(RED,    "[error] ") + msg)
def info(msg):  print(_c(CYAN,   "[info] ")  + msg)
def warn(msg):  print(_c(YELLOW, "[warn] ")  + msg)


# ---------------------------------------------------------------------------
# Google Gemini client bootstrap
# ---------------------------------------------------------------------------

def _get_client():
    """Return a configured Gemini client, or exit with a helpful message."""
    api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyDqLC6HYhkAVcAkm86CZwbq7vPUxSNAJg8").strip()
    if not api_key:
        err("GOOGLE_API_KEY is not set.")
        print()
        print("  Set it with:")
        print(_c(BOLD, "    export GOOGLE_API_KEY=your-api-key-here"))
        print()
        print("  Get a key at: " + _c(CYAN, "https://aistudio.google.com/app/apikey"))
        sys.exit(1)

    try:
        import google.generativeai as genai
    except ImportError:
        err("The 'google-generativeai' package is not installed.")
        print()
        print("  Install it with:")
        print(_c(BOLD, "    pip install google-generativeai"))
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai


MODEL = "gemini-3-flash-preview"

def _call_gemini(client, prompt: str, system: str = "") -> str:
    """Send a prompt to Gemini and return the text response."""
    try:
        model = client.GenerativeModel(
            model_name=MODEL,
            system_instruction=system if system else None
        )
        response = model.generate_content(prompt)
        # Accessing .text can raise an error if blocked by safety filters
        if response and response.text:
            return response.text
        return "[AI inference returned no text]"
    except Exception as exc:
        return f"[AI inference error: {exc}]"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert ML data engineer and dataset analyst. You analyse NLP \
dataset versions produced by DSV (Dataset Versioning System). \
Be concise, specific, and actionable. Use plain text — no markdown headers \
or bullet symbols. Use short numbered lists where helpful. \
Avoid generic advice; always refer to the specific numbers provided."""


def _load_version_summary(version_ref: str, storage, manager) -> dict:
    """Return a compact dict with all data needed to build a prompt."""
    data = manager.get_version(version_ref)
    meta    = data["meta"]
    metrics = data["metrics"]
    config  = data["config"]

    # Load training report if it exists
    version_id = meta["version_id"]
    version_dir = storage.versions_path / version_id
    # Also check by name-based directory
    name_dir = storage.versions_path / meta.get("name", "")
    training = None
    for vdir in [version_dir, name_dir]:
        tr = vdir / "training_report.json"
        if tr.exists():
            try:
                training = json.loads(tr.read_text(encoding="utf-8"))
            except Exception:
                pass
            break

    return {
        "name":     meta.get("name", version_ref),
        "message":  meta.get("message", ""),
        "created":  meta.get("created_at", ""),
        "parent":   meta.get("parent_name", None),
        "step":     meta.get("transformation_step", None),
        "hash":     version_id[:16] + "...",
        "metrics":  metrics,
        "config":   config,
        "training": training,
    }


def _format_version_block(v: dict) -> str:
    """Render a version summary as plain text for inclusion in a prompt."""
    lines = [
        f"Version name    : {v['name']}",
        f"Commit message  : {v['message']}",
        f"Parent version  : {v['parent'] or 'none (root)'}",
        f"Transformation  : {v['step'] or 'not specified'}",
        f"Created         : {v['created']}",
        "",
        "Dataset metrics:",
        f"  Rows              : {v['metrics']['num_rows']}",
        f"  Vocabulary size   : {v['metrics']['vocabulary_size']} unique tokens",
        f"  Total tokens      : {v['metrics']['total_tokens']}",
        f"  Avg sentence len  : {v['metrics']['avg_sentence_len']} tokens",
        "",
        "Preprocessing config:",
    ]
    for step, enabled in v["config"].items():
        lines.append(f"  {step:<25} : {'ON' if enabled else 'OFF'}")

    if v["training"]:
        t = v["training"]
        tm = t.get("test_metrics", {})
        lines += [
            "",
            "Training results (Naive Bayes, 25% test split):",
            f"  Test accuracy : {tm.get('accuracy', 'n/a')}",
            f"  Macro F1      : {tm.get('macro_f1', 'n/a')}",
        ]
        per_class = tm.get("per_class", {})
        for cls, scores in per_class.items():
            lines.append(
                f"  [{cls}] P={scores.get('precision','?')}  "
                f"R={scores.get('recall','?')}  F1={scores.get('f1','?')}"
            )
        feats = t.get("top_features", {})
        for cls, feat_list in feats.items():
            tokens = [f["token"] for f in (feat_list or [])[:6]]
            if tokens:
                lines.append(f"  Top [{cls}] tokens: {', '.join(tokens)}")
    else:
        lines.append("")
        lines.append("Training results : not available (run: dsv train --version " + v["name"] + ")")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sub-command: insights version
# ---------------------------------------------------------------------------

def cmd_insights_version(args, storage, manager) -> None:
    """Deep-dive AI analysis of a single dataset version."""
    print()
    info(f"Loading version '{args.version_ref}'...")

    try:
        v = _load_version_summary(args.version_ref, storage, manager)
    except (FileNotFoundError, KeyError) as exc:
        err(str(exc)); sys.exit(1)

    client = _get_client()

    prompt = f"""\
Analyse this dataset version and give me:
1. A one-sentence summary of what this version represents.
2. Assessment of the preprocessing choices — what they achieve and any risks.
3. If training results are available, interpret them honestly — is this good or bad for this type of data, and why?
4. Two specific, concrete suggestions to improve this version.

{_format_version_block(v)}"""

    info("Asking Gemini for insights...")
    print()

    response = _call_gemini(client, prompt, system=SYSTEM_PROMPT)

    # Save insights for dashboard
    meta = manager.get_version(args.version_ref)["meta"]
    v_dir = storage.versions_path / meta["name"]
    if not v_dir.exists():
        v_dir = storage.versions_path / meta["version_id"]
    
    if v_dir.exists():
        insights_path = v_dir / "ai_insights.json"
        try:
            insights_data = {
                "version_name": v["name"],
                "generated_at": meta.get("created_at", ""),
                "insights_raw": response,
                "model": MODEL
            }
            insights_path.write_text(json.dumps(insights_data, indent=2), encoding="utf-8")
            info(f"Insights saved to {insights_path.name}")

            # Also refresh dashboard_data.json if it exists in the current directory
            # so the user doesn't have to restart the dashboard.
            try:
                from .dsv import DASHBOARD_DATA, _read_head
                if DASHBOARD_DATA.exists():
                    dag = manager.build_dag()
                    vlist = storage.list_versions()
                    dag["latest_id"] = vlist[-1]["version_id"] if vlist else None
                    dag["head_name"] = _read_head()
                    DASHBOARD_DATA.write_text(json.dumps(dag, indent=2), encoding="utf-8")
                    info("Dashboard data refreshed")
            except Exception:
                pass
        except Exception as e:
            warn(f"Failed to save insights: {e}")

    print(_c(BOLD, f"╔══ AI Insights: {v['name']} ") + _c(DIM, "═" * max(0, 56 - len(v['name']))))
    print()
    for line in response.strip().split("\n"):
        print("  " + line)
    print()
    print(_c(BOLD, "╚" + "═" * 62))
    print()


# ---------------------------------------------------------------------------
# Sub-command: insights compare
# ---------------------------------------------------------------------------

def cmd_insights_compare(args, storage, manager) -> None:
    """AI-powered comparison of two versions."""
    print()
    info(f"Loading versions '{args.version_ref_1}' and '{args.version_ref_2}'...")

    try:
        a = _load_version_summary(args.version_ref_1, storage, manager)
        b = _load_version_summary(args.version_ref_2, storage, manager)
    except (FileNotFoundError, KeyError) as exc:
        err(str(exc)); sys.exit(1)

    client = _get_client()

    prompt = f"""\
Compare these two dataset versions and answer:
1. What preprocessing decisions differ between A and B, and what effect do they have on the data?
2. Which version has better dataset quality and why? Use the metrics.
3. If training results are available for both, which version produces a better model and why?
4. What does the comparison tell us about the best direction for the next version?

VERSION A
---------
{_format_version_block(a)}

VERSION B
---------
{_format_version_block(b)}"""

    info("Asking Gemini to compare...")
    print()

    response = _call_gemini(client, prompt, system=SYSTEM_PROMPT)

    print(_c(BOLD, f"╔══ AI Comparison: {a['name']} vs {b['name']} ") +
          _c(DIM, "═" * max(0, 46 - len(a['name']) - len(b['name']))))
    print()
    for line in response.strip().split("\n"):
        print("  " + line)
    print()
    print(_c(BOLD, "╚" + "═" * 62))
    print()


# ---------------------------------------------------------------------------
# Sub-command: insights suggest
# ---------------------------------------------------------------------------

def cmd_insights_suggest(args, storage, manager) -> None:
    """AI recommendations: what preprocessing steps to try next."""
    print()
    info("Scanning all versions to build recommendations...")

    versions = manager.list_versions()
    if not versions:
        err("No versions found. Create some versions first."); sys.exit(1)

    summaries = []
    for v_entry in versions:
        try:
            v = _load_version_summary(v_entry["name"], storage, manager)
            summaries.append(_format_version_block(v))
        except Exception:
            pass

    client = _get_client()

    all_versions_text = "\n\n---\n\n".join(summaries)

    prompt = f"""\
I have a dataset versioning project with {len(summaries)} version(s). \
Here is the full history of all versions:

{all_versions_text}

Based on this history, give me:
1. What preprocessing combinations have NOT been tried yet that are worth exploring?
2. If training results are available, which version should I build on next and why?
3. Three concrete `dsv add` + `dsv commit` command examples I could run right now to explore promising directions. Use real, plausible version names.
4. One warning: is there anything in this history that looks like a mistake or a dead end?"""

    info("Asking Gemini for recommendations...")
    print()

    response = _call_gemini(client, prompt, system=SYSTEM_PROMPT)

    print(_c(BOLD, "╔══ AI Recommendations ") + _c(DIM, "═" * 40))
    print()
    for line in response.strip().split("\n"):
        print("  " + line)
    print()
    print(_c(BOLD, "╚" + "═" * 62))
    print()


# ---------------------------------------------------------------------------
# Sub-command: insights report
# ---------------------------------------------------------------------------

def cmd_insights_report(args, storage, manager) -> None:
    """Full repo health report across all versions."""
    print()
    info("Building full repository report...")

    versions = manager.list_versions()
    if not versions:
        err("No versions found."); sys.exit(1)

    summaries = []
    for v_entry in versions:
        try:
            v = _load_version_summary(v_entry["name"], storage, manager)
            summaries.append(_format_version_block(v))
        except Exception:
            pass

    client = _get_client()

    all_versions_text = "\n\n---\n\n".join(summaries)

    prompt = f"""\
Write a concise repository health report for a DSV project with \
{len(summaries)} dataset version(s). The report should cover:

1. DATASET EVOLUTION: How has the data changed across versions? \
   Is the preprocessing strategy coherent?
2. MODEL PERFORMANCE TREND: If training reports exist, is accuracy \
   improving, declining, or flat across versions? What explains the trend?
3. BEST VERSION: Which version is currently the strongest and why?
4. RISKS: Any signs of overfitting, data leakage, or preprocessing \
   errors in the history?
5. NEXT STEPS: The single most important action to take next.

All versions:

{all_versions_text}"""

    info("Asking Gemini to generate report...")
    print()

    response = _call_gemini(client, prompt, system=SYSTEM_PROMPT)

    print(_c(BOLD, "╔══ AI Repository Health Report ") + _c(DIM, "═" * 31))
    print(_c(DIM,  f"  {len(summaries)} version(s) analysed"))
    print()
    for line in response.strip().split("\n"):
        print("  " + line)
    print()
    print(_c(BOLD, "╚" + "═" * 62))
    print()


# ---------------------------------------------------------------------------
# Dispatcher called from cli.py
# ---------------------------------------------------------------------------

def cmd_insights(args, storage=None, manager=None) -> None:
    """Route to the correct insights sub-command."""
    from .storage import StorageManager
    from .versioning import DatasetVersionManager

    if storage is None:
        storage = StorageManager(repo_path=args.repo)
    if not storage.is_initialised():
        err("Repository not initialised. Run " + _c(BOLD, "dsv init") + " first.")
        sys.exit(1)
    if manager is None:
        manager = DatasetVersionManager(storage=storage)

    sub = getattr(args, "insights_command", None)
    if sub == "version":
        cmd_insights_version(args, storage, manager)
    elif sub == "compare":
        cmd_insights_compare(args, storage, manager)
    elif sub == "suggest":
        cmd_insights_suggest(args, storage, manager)
    elif sub == "report":
        cmd_insights_report(args, storage, manager)
    else:
        err("Unknown insights sub-command. Use: version, compare, suggest, report")
        sys.exit(1)
