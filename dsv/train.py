"""
train.py
--------
Train a Bag-of-Words Naive Bayes classifier on a stored dataset version,
evaluate it, and save the training report back into the version directory.

Usage
-----
    python train.py --version v1-baseline
    python train.py --version v2-full-clean
    python train.py --compare v1-baseline v2-full-clean

Training report
---------------
Each run writes  training_report.json  into the version folder:

    dataset_repo/versions/<hash>/training_report.json

The report contains:
  - train/test split sizes
  - dataset version metadata
  - vectoriser statistics (vocab size, avg doc length)
  - evaluation metrics (accuracy, macro-F1, per-class P/R/F1)
  - top discriminative features per class
  - explanation of what the metrics mean

Format of labeled rows in dataset.txt
--------------------------------------
    <text>|<label>
    e.g.  "Deep learning is powerful.|positive"

Rows without a "|" separator are skipped with a warning.

Train / test split
------------------
A fixed random seed (42) is used so that results are reproducible across runs
and comparable across dataset versions.  The split is stratified: we preserve
class proportions in both train and test sets so that accuracy is not inflated
by class imbalance.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .storage import StorageManager
from .versioning import DatasetVersionManager
from .model import TFIDFVectorizer, NaiveBayesClassifier, evaluate, top_features


# ── constants ───────────────────────────────────────────────────────────────────

RANDOM_SEED    = 42
TEST_RATIO     = 0.25   # 25 % held out for evaluation
REPORT_FILE    = "training_report.json"
TOP_N_FEATURES = 8


# ── data helpers ─────────────────────────────────────────────────────────────────

def parse_labeled_rows(rows: List[str]) -> Tuple[List[str], List[str]]:
    """
    Parse rows of the form  "<text>|<label>".

    Returns (texts, labels) — parallel lists.
    Rows without a "|" separator are silently skipped (unlabeled data).
    Only prints a warning if SOME rows have labels but others don't.
    """
    texts, labels = [], []
    labeled_count   = 0
    unlabeled_count = 0

    for row in rows:
        row = row.strip()
        if not row:
            continue
        if "|" not in row:
            unlabeled_count += 1
            continue
        sep   = row.rfind("|")
        text  = row[:sep].strip()
        label = row[sep + 1:].strip()
        if text and label:
            texts.append(text)
            labels.append(label)
            labeled_count += 1
        else:
            unlabeled_count += 1

    # Only warn if it looks like the user intended labels but some are missing
    if labeled_count > 0 and unlabeled_count > 0:
        print(f"  [warn] {unlabeled_count} unlabeled rows skipped "
              f"({labeled_count} labeled rows kept).")
    # If NONE have labels — this is an unlabeled dataset, no warning needed
    # (train_and_evaluate will return a clean skip message)
    return texts, labels


def stratified_split(
    texts: List[str],
    labels: List[str],
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Stratified train/test split preserving class proportions.

    Returns
    -------
    train_texts, test_texts, train_labels, test_labels
    """
    rng = random.Random(seed)

    # Group indices by class
    class_indices: Dict[str, List[int]] = {}
    for i, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(i)

    train_idx, test_idx = [], []
    for lbl, indices in class_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])

    # Sort to keep ordering deterministic
    train_idx.sort()
    test_idx.sort()

    return (
        [texts[i]  for i in train_idx],
        [texts[i]  for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )


# ── training pipeline ────────────────────────────────────────────────────────────

def train_and_evaluate(
    version_ref: str,
    storage: StorageManager,
    manager: DatasetVersionManager,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full training pipeline for one dataset version.

    Steps
    -----
    1. Load version from storage.
    2. Parse labeled rows  →  texts + labels.
    3. Stratified train/test split.
    4. Fit TF-IDF vectoriser on train set only (no leakage).
    5. Transform train and test sets.
    6. Train Naive Bayes classifier.
    7. Evaluate on test set.
    8. Extract top discriminative features per class.
    9. Build and return (and persist) the training report.

    Parameters
    ----------
    version_ref : version name or SHA-256 hash
    storage     : StorageManager instance
    manager     : DatasetVersionManager instance
    verbose     : if True, print progress to stdout

    Returns
    -------
    Full training report dict (also saved to training_report.json).
    """
    # ── 1. Load version ──────────────────────────────────────────────────
    data = manager.get_version(version_ref)
    meta    = data["meta"]
    rows    = data["rows"]
    config  = data["config"]
    metrics = data["metrics"]
    version_id = meta.get("version_id", version_ref)

    if verbose:
        print(f"\n{'═' * 62}")
        print(f"  Training on version : {meta.get('name', version_ref)}")
        print(f"  Message             : {meta.get('message', '')}")
        print(f"  Dataset rows        : {len(rows)}")
        print(f"{'═' * 62}")

    # ── 2. Parse labels ──────────────────────────────────────────────────
    texts, labels = parse_labeled_rows(rows)
    if len(texts) < 4:
        print(f"  [warn] Need at least 4 labeled rows to train, got {len(texts)}.")
        print(f"  [warn] Make sure your dataset rows follow the format: text|label")
        print(f"  [warn] Skipping training for this version.")
        return {
            "version_name":    meta.get("name", version_ref),
            "version_id":      version_id,
            "skipped":         True,
            "reason":          f"Only {len(texts)} labeled rows found (need >= 4). "
                               f"Ensure rows follow: text|label format.",
            "dataset_stats":   {"total_labeled_rows": len(texts)},
            "test_metrics":    {"accuracy": None, "macro_f1": None},
            "train_metrics":   {"accuracy": None, "macro_f1": None},
            "top_features":    {},
            "preprocessing_config": config,
        }

    classes = sorted(set(labels))
    class_dist = {c: labels.count(c) for c in classes}

    if verbose:
        print(f"\n  Labeled rows parsed : {len(texts)}")
        print(f"  Classes             : {classes}")
        print(f"  Class distribution  : {class_dist}")

    # ── 3. Stratified split ──────────────────────────────────────────────
    tr_texts, te_texts, tr_labels, te_labels = stratified_split(
        texts, labels, TEST_RATIO, RANDOM_SEED
    )

    if verbose:
        print(f"\n  Train size : {len(tr_texts)} rows")
        print(f"  Test  size : {len(te_texts)} rows  (seed={RANDOM_SEED})")

    # ── 4–5. TF-IDF vectorisation ────────────────────────────────────────
    vectorizer = TFIDFVectorizer()
    tr_vecs = vectorizer.fit_transform(tr_texts)   # fit on TRAIN only
    te_vecs = vectorizer.transform(te_texts)        # apply to TEST

    avg_doc_len = (
        sum(len(t.split()) for t in tr_texts) / len(tr_texts)
        if tr_texts else 0.0
    )

    if verbose:
        print(f"\n  Vocabulary size     : {vectorizer.vocab_size} tokens  "
              f"(from training split)")
        print(f"  Avg doc length      : {avg_doc_len:.2f} tokens")

    # ── 6. Train classifier ──────────────────────────────────────────────
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(tr_vecs, tr_labels, vectorizer.vocab_size)

    # ── 7. Evaluate ──────────────────────────────────────────────────────
    tr_preds = clf.predict(tr_vecs)
    te_preds = clf.predict(te_vecs)

    train_eval = evaluate(tr_labels, tr_preds, classes)
    test_eval  = evaluate(te_labels, te_preds, classes)

    if verbose:
        print(f"\n  {'─' * 40}")
        print(f"  TRAIN accuracy      : {train_eval['accuracy']:.4f}  "
              f"(macro-F1: {train_eval['macro_f1']:.4f})")
        print(f"  TEST  accuracy      : {test_eval['accuracy']:.4f}  "
              f"(macro-F1: {test_eval['macro_f1']:.4f})")
        print(f"  {'─' * 40}")
        for cls in classes:
            pc = test_eval["per_class"][cls]
            print(f"  [{cls:>8}]  P={pc['precision']:.3f}  "
                  f"R={pc['recall']:.3f}  F1={pc['f1']:.3f}  "
                  f"support={pc['support']}")

    # ── 8. Top features ──────────────────────────────────────────────────
    top_feats: Dict[str, List] = {}
    for cls in classes:
        feats = top_features(clf, vectorizer, cls, top_n=TOP_N_FEATURES)
        top_feats[cls] = [
            {"token": tok, "discriminative_score": round(score, 4)}
            for tok, score in feats
        ]

    if verbose:
        print(f"\n  TOP {TOP_N_FEATURES} DISCRIMINATIVE TOKENS PER CLASS")
        for cls in classes:
            tokens_str = ", ".join(f["token"] for f in top_feats[cls])
            print(f"  [{cls:>8}]  {tokens_str}")

    # ── 9. Build report ──────────────────────────────────────────────────
    report = {
        "version_name":    meta.get("name", version_ref),
        "version_id":      version_id,
        "version_message": meta.get("message", ""),
        "preprocessing_config": config,
        "dataset_stats": {
            "total_labeled_rows":   len(texts),
            "class_distribution":   class_dist,
            "train_size":           len(tr_texts),
            "test_size":            len(te_texts),
            "vectorizer_vocab_size": vectorizer.vocab_size,
            "avg_train_doc_len":    round(avg_doc_len, 4),
        },
        "train_metrics": train_eval,
        "test_metrics":  test_eval,
        "top_features":  top_feats,
        "model_config": {
            "model_type":    "MultinomialNaiveBayes",
            "vectorizer":    "TF-IDF (from scratch)",
            "alpha":         1.0,
            "test_ratio":    TEST_RATIO,
            "random_seed":   RANDOM_SEED,
            "split_strategy": "stratified",
        },
        "explanation": _build_explanation(test_eval, train_eval, config, vectorizer),
    }

    # ── Persist report ───────────────────────────────────────────────────
    # Use name-based folder (human readable), fall back to hash if needed
    version_name = meta.get("name", version_ref)
    name_dir  = storage.versions_path / version_name
    hash_dir  = storage.versions_path / version_id
    version_dir = name_dir if name_dir.exists() else hash_dir
    report_path = version_dir / REPORT_FILE
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n  Report saved → {report_path}")

    return report


def _build_explanation(
    test_eval: Dict,
    train_eval: Dict,
    config: Dict,
    vectorizer: TFIDFVectorizer,
) -> Dict[str, str]:
    """
    Build a human-readable explanation of training results and what
    the preprocessing configuration likely caused.
    """
    gap = round(train_eval["accuracy"] - test_eval["accuracy"], 4)
    overfitting_note = (
        "Large gap between train and test accuracy suggests overfitting. "
        "With a small dataset this is expected — train accuracy is inflated."
        if gap > 0.15
        else "Train/test accuracy gap is small, suggesting good generalisation "
             "given the dataset size."
    )

    stopwords_note = (
        "Stopwords were REMOVED. This reduces vocabulary noise: function words "
        "like 'is', 'the', 'a' are excluded so the model focuses on "
        "content-bearing terms, which should improve discriminability."
        if config.get("remove_stopwords")
        else "Stopwords were KEPT. Function words ('is', 'the', 'a') are part of "
             "the vocabulary. They add noise but also retain grammatical signals."
    )

    punct_note = (
        "Punctuation was REMOVED. This means 'learning.' and 'learning' are "
        "treated as the same token, reducing vocabulary fragmentation."
        if config.get("remove_punctuation")
        else "Punctuation was KEPT. Tokens like 'learning.' and 'learning' are "
             "counted separately, which inflates vocabulary size slightly."
    )

    vocab_note = (
        f"Vocabulary has {vectorizer.vocab_size} tokens (from training split). "
        "A larger vocabulary captures more distinctions but also more noise. "
        "A smaller vocabulary (after stopword/punctuation removal) forces the "
        "model to rely on truly informative content words."
    )

    return {
        "overfitting_analysis": overfitting_note,
        "stopwords_effect":     stopwords_note,
        "punctuation_effect":   punct_note,
        "vocabulary_analysis":  vocab_note,
        "overall": (
            f"Test accuracy: {test_eval['accuracy']:.4f}, "
            f"Macro-F1: {test_eval['macro_f1']:.4f}. "
            f"Train/test gap: {gap:.4f}."
        ),
    }


# ── comparison printer ───────────────────────────────────────────────────────────

def compare_reports(report_a: Dict, report_b: Dict) -> None:
    """
    Print a side-by-side comparison of two training reports.
    Highlights which preprocessing choices led to different outcomes.
    """
    na = report_a["version_name"]
    nb = report_b["version_name"]

    print(f"\n{'═' * 70}")
    print(f"  TRAINING COMPARISON")
    print(f"  A : {na}")
    print(f"  B : {nb}")
    print(f"{'═' * 70}")

    # ── Dataset stats ────────────────────────────────────────────────────
    sa = report_a["dataset_stats"]
    sb = report_b["dataset_stats"]
    print(f"\n  {'DATASET STATS':<35}  {'A: ' + na:>18}  {'B: ' + nb:>18}")
    print("  " + "─" * 72)
    rows = [
        ("Labeled rows",      sa["total_labeled_rows"],     sb["total_labeled_rows"]),
        ("Train size",        sa["train_size"],              sb["train_size"]),
        ("Test  size",        sa["test_size"],               sb["test_size"]),
        ("Vectorizer vocab",  sa["vectorizer_vocab_size"],   sb["vectorizer_vocab_size"]),
        ("Avg train doc len", sa["avg_train_doc_len"],       sb["avg_train_doc_len"]),
    ]
    for label, va, vb in rows:
        delta_str = ""
        try:
            delta = round(float(vb) - float(va), 4)
            sign = "+" if delta > 0 else ""
            delta_str = f"  ({sign}{delta})"
        except Exception:
            pass
        print(f"  {label:<35}  {str(va):>18}  {str(vb):>18}{delta_str}")

    # ── Eval metrics ─────────────────────────────────────────────────────
    ma = report_a["test_metrics"]
    mb = report_b["test_metrics"]
    print(f"\n  {'TEST METRICS':<35}  {'A: ' + na:>18}  {'B: ' + nb:>18}")
    print("  " + "─" * 72)
    for key in ["accuracy", "macro_f1"]:
        va, vb = ma[key], mb[key]
        delta = round(vb - va, 4)
        sign = "+" if delta > 0 else ""
        winner = "← B wins" if delta > 0 else ("← A wins" if delta < 0 else "tie")
        print(f"  {key:<35}  {va:>18}  {vb:>18}  ({sign}{delta})  {winner}")

    # Per-class F1
    classes = sorted(ma["per_class"].keys())
    for cls in classes:
        key = f"F1 [{cls}]"
        va = ma["per_class"][cls]["f1"]
        vb = mb["per_class"][cls]["f1"]
        delta = round(vb - va, 4)
        sign = "+" if delta > 0 else ""
        print(f"  {key:<35}  {va:>18}  {vb:>18}  ({sign}{delta})")

    # ── Config diffs ─────────────────────────────────────────────────────
    ca = report_a["preprocessing_config"]
    cb = report_b["preprocessing_config"]
    all_keys = sorted(set(ca) | set(cb))
    changed = [(k, ca.get(k), cb.get(k)) for k in all_keys if ca.get(k) != cb.get(k)]

    print(f"\n  PREPROCESSING DIFFERENCES")
    if not changed:
        print("  (configs are identical)")
    else:
        print(f"  {'Step':<30}  {'A: ' + na:>18}  {'B: ' + nb:>18}")
        print("  " + "─" * 70)
        for key, va, vb in changed:
            print(f"  {key:<30}  {str(va):>18}  {str(vb):>18}")

    # ── Reasoning ────────────────────────────────────────────────────────
    print(f"\n  INTERPRETATION")
    print("  " + "─" * 68)

    vocab_a = sa["vectorizer_vocab_size"]
    vocab_b = sb["vectorizer_vocab_size"]
    acc_a   = ma["accuracy"]
    acc_b   = mb["accuracy"]

    if vocab_b < vocab_a:
        print(f"  • B has a SMALLER vocabulary ({vocab_b} vs {vocab_a}).")
        print(f"    Removing stopwords/punctuation concentrated the vocabulary")
        print(f"    on content words, reducing noise.")
    elif vocab_b > vocab_a:
        print(f"  • B has a LARGER vocabulary ({vocab_b} vs {vocab_a}).")
        print(f"    More tokens = richer features but also more noise.")

    if abs(acc_b - acc_a) < 0.01:
        print(f"  • Accuracy is nearly the same ({acc_a:.4f} vs {acc_b:.4f}).")
        print(f"    With only {sa['test_size']} test samples, differences < 0.05")
        print(f"    may be within random fluctuation.")
    elif acc_b > acc_a:
        print(f"  • B achieves HIGHER test accuracy ({acc_b:.4f} vs {acc_a:.4f}).")
        print(f"    The preprocessing in B likely reduced vocabulary noise,")
        print(f"    helping the model focus on discriminative content words.")
    else:
        print(f"  • A achieves HIGHER test accuracy ({acc_a:.4f} vs {acc_b:.4f}).")
        print(f"    Over-aggressive preprocessing in B may have removed")
        print(f"    tokens that were actually informative for classification.")

    # Top features comparison
    print(f"\n  TOP FEATURES COMPARISON (most discriminative tokens per class)")
    all_classes = sorted(set(report_a["top_features"]) | set(report_b["top_features"]))
    for cls in all_classes:
        feats_a = [f["token"] for f in report_a["top_features"].get(cls, [])]
        feats_b = [f["token"] for f in report_b["top_features"].get(cls, [])]
        only_a  = set(feats_a) - set(feats_b)
        only_b  = set(feats_b) - set(feats_a)
        shared  = set(feats_a) & set(feats_b)
        print(f"\n  [{cls}]")
        print(f"    A tokens : {', '.join(feats_a)}")
        print(f"    B tokens : {', '.join(feats_b)}")
        if shared:
            print(f"    Shared   : {', '.join(sorted(shared))}")
        if only_a:
            print(f"    Only A   : {', '.join(sorted(only_a))}  "
                  f"(dropped by B's preprocessing)")
        if only_b:
            print(f"    Only B   : {', '.join(sorted(only_b))}  "
                  f"(emerged after B's preprocessing)")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────────

def load_existing_report(
    version_ref: str,
    storage: StorageManager,
    manager: DatasetVersionManager,
) -> Optional[Dict]:
    """Load a previously saved training report if it exists."""
    try:
        # Try name-based folder first, then hash-based
        name_dir = storage.versions_path / version_ref
        if name_dir.exists():
            report_path = name_dir / REPORT_FILE
            if report_path.exists():
                return json.loads(report_path.read_text(encoding="utf-8"))
        version_id = manager.resolve_version(version_ref)
        report_path = storage.versions_path / version_id / REPORT_FILE
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description=(
            "Train a Bag-of-Words Naive Bayes classifier on a dataset version "
            "and optionally compare two versions."
        ),
    )
    parser.add_argument(
        "--repo", default="dataset_repo", metavar="PATH",
        help="Path to the dataset repository (default: dataset_repo).",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version", metavar="NAME_OR_HASH",
        help="Train on a single dataset version.",
    )
    group.add_argument(
        "--compare", nargs=2, metavar=("VERSION_A", "VERSION_B"),
        help="Train on two versions and compare their training reports.",
    )
    group.add_argument(
        "--all", action="store_true", dest="all_versions",
        help="Train on ALL versions in the repository.",
    )

    return parser


def _run_or_load(
    version_ref: str,
    storage: StorageManager,
    manager: DatasetVersionManager,
) -> Dict:
    """Train (or reload cached report) for a version."""
    cached = load_existing_report(version_ref, storage, manager)
    if cached:
        print(f"\n  [info] Loaded existing report for '{version_ref}'.")
        return cached
    return train_and_evaluate(version_ref, storage, manager)


def _refresh_dashboard(storage):
    """Regenerate dashboard_data.json after training."""
    try:
        from pathlib import Path
        import json as _json
        from .versioning import DatasetVersionManager as _DVM
        _dag = _DVM(storage=storage).build_dag()
        _versions = storage.list_versions()
        _dag["latest_id"] = _versions[-1]["version_id"] if _versions else None
        _head_file = Path.cwd() / ".dsv_head"
        _dag["head_name"] = _head_file.read_text(encoding="utf-8").strip() if _head_file.exists() else None
        _data_file = Path.cwd() / "dashboard_data.json"
        _data_file.write_text(_json.dumps(_dag, indent=2), encoding="utf-8")
        print("  [ok] dashboard_data.json refreshed.")
    except Exception as e:
        print(f"  [warn] Could not refresh dashboard data: {e}")


def _refresh_dashboard(storage):
    """Regenerate dashboard_data.json after training."""
    try:
        from pathlib import Path
        import json as _json
        from .versioning import DatasetVersionManager as _DVM
        _dag = _DVM(storage=storage).build_dag()
        _versions = storage.list_versions()
        _dag["latest_id"] = _versions[-1]["version_id"] if _versions else None
        _head_file = Path.cwd() / ".dsv_head"
        _dag["head_name"] = _head_file.read_text(encoding="utf-8").strip() if _head_file.exists() else None
        _data_file = Path.cwd() / "dashboard_data.json"
        _data_file.write_text(_json.dumps(_dag, indent=2), encoding="utf-8")
        print("  [ok] dashboard_data.json refreshed.")
    except Exception as e:
        print(f"  [warn] Could not refresh dashboard data: {e}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    storage = StorageManager(repo_path=args.repo)
    if not storage.is_initialised():
        print("[error] Repository not initialised. Run 'python dsv.py init' first.")
        sys.exit(1)

    manager = DatasetVersionManager(storage=storage)

    if args.version:
        try:
            train_and_evaluate(args.version, storage, manager, verbose=True)
            _refresh_dashboard(storage)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[error] {exc}")
            sys.exit(1)

    elif args.compare:
        ref_a, ref_b = args.compare
        try:
            report_a = _run_or_load(ref_a, storage, manager)
            report_b = _run_or_load(ref_b, storage, manager)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[error] {exc}")
            sys.exit(1)
        compare_reports(report_a, report_b)
        _refresh_dashboard(storage)

    elif args.all_versions:
        versions = storage.list_versions()
        if not versions:
            print("[error] No versions found. Create some versions first.")
            sys.exit(1)

        print(f"\n  Training on all {len(versions)} version(s)...\n")
        print("  " + "-" * 60)

        reports = []
        for v in versions:
            name = v["name"]
            try:
                report = _run_or_load(name, storage, manager)
                reports.append(report)
            except Exception as exc:
                print(f"  [warn] Skipped '{name}': {exc}")

        # Summary table
        print("\n" + "=" * 62)
        print("  TRAINING SUMMARY — ALL VERSIONS")
        print("=" * 62)
        print(f"  {'VERSION':<25}  {'ACCURACY':>10}  {'MACRO F1':>10}  {'ROWS':>6}  {'VOC':>6}")
        print("  " + "-" * 62)

        best_acc    = max((r["test_metrics"]["accuracy"] for r in reports), default=0)
        best_name   = ""

        for r in reports:
            acc  = r["test_metrics"]["accuracy"]
            f1   = r["test_metrics"]["macro_f1"]
            ds   = r.get("dataset_stats", {})
            rows = ds.get("total_labeled_rows", ds.get("num_rows", "?"))
            voc  = ds.get("vectorizer_vocab_size", ds.get("vocabulary_size", "?"))
            name = r["version_name"]
            marker = " ← best" if acc == best_acc else ""
            print(f"  {name:<25}  {acc:>10.4f}  {f1:>10.4f}  {rows:>6}  {voc:>6}{marker}")

        print()
        print(f"  Total versions trained : {len(reports)}")
        if reports:
            accs = [r["test_metrics"]["accuracy"] for r in reports]
            print(f"  Best accuracy          : {max(accs):.4f}")
            print(f"  Worst accuracy         : {min(accs):.4f}")
            print(f"  Average accuracy       : {sum(accs)/len(accs):.4f}")
        print()

        _refresh_dashboard(storage)


if __name__ == "__main__":
    main()
