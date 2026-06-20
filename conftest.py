"""Ensure the repository root is first on ``sys.path`` for the test run.

This package is installed editable via PEP 660, which appends a finder to
``sys.meta_path`` *after* the default ``PathFinder`` and registers a namespace
path hook. When pytest runs without the repo root on ``sys.path``, ``import app``
resolves to a namespace package and ``app/__init__.py`` never executes — so
``from app import __version__`` fails. Putting the repo root first makes the real
package (with its ``__init__`` attributes) win.
"""

import os
import sys

_ROOT = os.path.dirname(__file__)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
