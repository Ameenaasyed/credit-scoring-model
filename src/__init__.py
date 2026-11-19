"""
Credit Scoring Model source package.
Ensures custom transformers remain importable across serialized Joblib pipelines
regardless of the working directory.
"""

import sys
from . import feature_engineering

# Register top-level 'feature_engineering' alias so pre-trained Joblib pipelines
# deserialize portably without requiring machine-specific .pth files.
sys.modules.setdefault('feature_engineering', feature_engineering)
