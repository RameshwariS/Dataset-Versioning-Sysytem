"""
DSV — Dataset Versioning System
================================
A git-style CLI tool for versioning, comparing, and training on datasets.
"""

__version__ = "1.0.0"
__author__ = "DSV Contributors"


def init(repo="dataset_repo", dataset_type="txt"):
    """
    Initialise a DSV repository in the current directory.

    Parameters
    ----------
    repo : str
        Path to the repository directory (default: 'dataset_repo').
    dataset_type : str
        Type of dataset: 'txt' for text datasets (default) or 'numeric' for
        tabular/CSV datasets.

    Example
    -------
        import dsv
        dsv.init()                         # text dataset (default)
        dsv.init(dataset_type="numeric")   # numeric/tabular dataset
    """
    from dsv.storage import StorageManager
    storage = StorageManager(repo_path=repo)
    storage.init(dataset_type=dataset_type)
    print(f"[dsv] Initialised repository at '{repo}/' (type: {dataset_type})")
    print(f"[dsv] Place your dataset in '{repo}/raw/'")
