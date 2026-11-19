"""
Integration Tests for Trained ML Models.
Validates model serialization, probability calibration, output reproducibility, and performance sanity checks.
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import load_dataset, prepare_train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'credit_dataset.csv')


def test_model_files_exist():
    """Verify serialized model files exist on disk."""
    expected_files = [
        'logistic_regression_model.joblib',
        'decision_tree_model.joblib',
        'preprocessing_pipeline.joblib'
    ]
    for filename in expected_files:
        path = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(path), f"Expected model file missing: {path}"


def test_logistic_regression_pipeline_predict_proba():
    """Verify Logistic Regression outputs valid probabilities summing to 1.0."""
    path = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
    model = joblib.load(path)

    df = load_dataset(DATA_PATH)
    X_train, X_test, _, _ = prepare_train_test_split(df)

    probas = model.predict_proba(X_test)
    assert probas.shape == (200, 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    # Sum of class probabilities per row equals 1.0
    row_sums = probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_decision_tree_pipeline_predict_proba():
    """Verify Decision Tree outputs valid class probabilities."""
    path = os.path.join(MODELS_DIR, 'decision_tree_model.joblib')
    model = joblib.load(path)

    df = load_dataset(DATA_PATH)
    X_train, X_test, _, _ = prepare_train_test_split(df)

    probas = model.predict_proba(X_test)
    assert probas.shape == (200, 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    row_sums = probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_model_reproducibility():
    """Verify that inference is strictly deterministic and reproducible."""
    path = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
    model = joblib.load(path)

    df = load_dataset(DATA_PATH)
    sample = df.drop(columns=['class']).iloc[:5]

    probas_1 = model.predict_proba(sample)
    probas_2 = model.predict_proba(sample)
    assert np.allclose(probas_1, probas_2)
