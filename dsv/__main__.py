"""
__main__.py
-----------
Allows the package to be run directly with ``python -m dsv``.

This is equivalent to running the ``dsv`` console script registered
by setuptools via the entry point ``dsv.cli:main``.

Usage::

    python -m dsv init
    python -m dsv add --data dataset_repo/raw/dataset.txt
    python -m dsv commit --name v1 --message "first version"
    python -m dsv --help
"""

from dsv.cli import main

if __name__ == "__main__":
    main()
