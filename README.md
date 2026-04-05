# DSV — Dataset Versioning System

A **git-style CLI tool** for versioning, comparing, and training on datasets.
Think of it as `git` for your ML training data.

---

## Installation

```bash
# Clone / unzip the project, then install in editable mode:
pip install -e .

# Verify the command is available:
dsv --help
```

---

## Quick Start

```bash
# 1. Initialise a repository in the current directory
dsv init

# 2. Place your raw dataset in dataset_repo/raw/
cp my_data.txt dataset_repo/raw/dataset.txt

# 3. Stage it
dsv add --data dataset_repo/raw/dataset.txt

# 4. Optionally stage a preprocessing config too
dsv add --config config.json

# 5. Commit as a named version
dsv commit --name v1 --message "initial raw dataset"

# 6. List all versions
dsv list

# 7. Show details for a version
dsv show v1

# 8. Train a classifier on it
dsv train --version v1
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `dsv init` | Initialise a new DSV repository |
| `dsv add --data FILE [--config FILE]` | Stage a dataset file and/or config |
| `dsv commit --name NAME --message MSG` | Create a new version from staged changes |
| `dsv status` | Show what is staged vs. committed |
| `dsv list` | List all versions |
| `dsv show VERSION` | Show full details of a version |
| `dsv compare VERSION_A VERSION_B` | Diff metrics and config between two versions |
| `dsv lineage [VERSION]` | Visualise the version lineage DAG |
| `dsv dashboard [--port PORT]` | Open the interactive browser dashboard |
| `dsv train --version NAME` | Train a Naive Bayes classifier on a version |
| `dsv train --compare A B` | Train on two versions and compare reports |
| `dsv train --all` | Train on every version and print a summary |
| `dsv watch [--data FILE]` | Auto-version on every file save |
| `dsv rollback VERSION` | Restore a past version's config |
| `dsv create --data FILE --name NAME --message MSG` | *(Legacy)* Direct version creation |

### Global flags

```
--repo PATH   Path to the dataset repository (default: dataset_repo)
--version     Print DSV version and exit
```

---

## Git-style Workflow

```
dsv init
dsv add --data dataset_repo/raw/dataset.txt
dsv commit --name v1-raw --message "raw data"

# edit dataset or config...

dsv add --data dataset_repo/raw/dataset.txt --config config.json
dsv commit --name v2-clean --message "removed duplicates and stopwords"

dsv compare v1-raw v2-clean
dsv lineage
```

---

## Preprocessing Config

Edit `config.json` to toggle preprocessing steps:

```json
{
  "lowercase":          true,
  "remove_punctuation": true,
  "remove_duplicates":  true,
  "strip_whitespace":   true,
  "tokenize":           false,
  "remove_stopwords":   false
}
```

Stage it with `dsv add --config config.json` before the next commit.

---

## Training

Dataset rows must follow the format `<text>|<label>`:

```
Deep learning is powerful.|positive
This data is noisy and bad.|negative
```

Then:

```bash
dsv train --version v1-raw
dsv train --compare v1-raw v2-clean
dsv train --all
```

---

## Debugging

Set `DSV_DEBUG=1` to get full Python tracebacks on errors:

```bash
DSV_DEBUG=1 dsv commit --name v1 --message "test"
```

---

## Project Structure

```
dsv/
├── __init__.py       Package metadata and version
├── cli.py            Entry point — main() registered as `dsv` console script
├── dsv.py            Core commands (init, add, commit, list, show, compare, …)
├── train.py          Training and evaluation logic
├── watch.py          File-watcher for auto-versioning
├── storage.py        Filesystem storage layer
├── versioning.py     Version creation, lineage, DAG building
├── preprocessing.py  Text preprocessing pipeline
├── metrics.py        Dataset metrics calculation
└── model.py          TF-IDF vectoriser + Naive Bayes classifier
pyproject.toml        Build metadata and console_scripts entry point
setup.py              Legacy editable-install shim
README.md             This file
```
