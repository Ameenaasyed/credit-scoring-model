"""
Pytest configuration and environment fixture setup.
Configures python path so all project modules and pickled pipelines are globally resolvable
when running from either the project root or a parent directory.
"""

import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importing src registers the portable 'feature_engineering' alias via src/__init__.py
import src  # noqa: E402, F401
