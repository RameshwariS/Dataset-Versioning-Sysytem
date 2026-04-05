"""
setup.py
--------
Legacy setup file kept alongside pyproject.toml for maximum compatibility
with older pip versions and editable installs (``pip install -e .``).

The canonical metadata lives in ``pyproject.toml``.  This file simply
calls ``setup()`` with no arguments so setuptools reads pyproject.toml.

Install in editable / development mode::

    pip install -e .

Then use the CLI from anywhere::

    dsv --help
    dsv init
    dsv create --data my_data.txt --name v1 --message "first version"
"""

from setuptools import setup

# All metadata is declared in pyproject.toml.
# This file only exists so that ``pip install -e .`` works with older pip.
if __name__ == "__main__":
    setup()
