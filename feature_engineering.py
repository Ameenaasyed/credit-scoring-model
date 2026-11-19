"""
Top-level compatibility bridge for serialized Scikit-Learn Joblib pipelines.
Allows loading models directly from the project root directory without
requiring machine-specific site-packages .pth files.
"""

from src.feature_engineering import FinancialFeatureEngineer

__all__ = ["FinancialFeatureEngineer"]
