"""Pytest configuration for earthdelta tests."""
import sys
from pathlib import Path

# Ensure the package is importable
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
