"""Pytest configuration: make the repository root importable.

Allows tests to ``import evaluation.literature_checker`` regardless of the directory pytest
is invoked from.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
