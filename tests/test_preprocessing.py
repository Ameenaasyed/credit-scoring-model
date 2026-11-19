"""
Unit and Integration Tests for Data Preprocessing and Feature Engineering.
Validates dataset loading, target encoding, stratified split, and leakage-free transformations.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import (
    load_dataset,
    encode_target,
    decode_target,
    prepare_train_test_split,
    build_preprocessor_pipeline,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from src.feature_engineering import FinancialFeatureEngineer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'credit_dataset.csv')


def test_dataset_exists_and_loads():
    """Verify that dataset file exists and has exactly 1000 rows and 21 columns."""
    assert os.path.exists(DATA_PATH), f"Dataset missing at {DATA_PATH}"
    df = load_dataset(DATA_PATH)
    assert df.shape == (1000, 21), f"Expected shape (1000, 21), got {df.shape}"
    assert TARGET_COLUMN in df.columns, "Target column 'class' not found in dataset"
    assert df.isnull().sum().sum() == 0, "Raw dataset contains unexpected missing values"


def test_target_encoding_and_decoding():
    """Verify target encoding correctly assigns high-risk/bad to 1 and low-risk/good to 0."""
    sample_targets = pd.Series(['good', 'bad', 'bad', 'good'])
    encoded = encode_target(sample_targets)
    assert list(encoded) == [0, 1, 1, 0]

    decoded = decode_target(encoded.values)
    assert 'Bad Credit / Higher Credit Risk' in decoded[1]
    assert 'Good Credit / Lower Credit Risk' in decoded[0]


def test_target_encoding_invalid_raises():
    """Verify that unrecognized target values trigger an explicit ValueError."""
    invalid_targets = pd.Series(['good', 'unknown_risk'])
    with pytest.raises(ValueError):
        encode_target(invalid_targets)


def test_stratified_train_test_split():
    """Verify that stratified train/test split maintains exact 70/30 class proportions."""
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, test_size=0.2, random_state=42)

    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(y_train) == 800
    assert len(y_test) == 200

    # Verify 30% minority proportion in both train and test
    train_bad_ratio = y_train.mean()
    test_bad_ratio = y_test.mean()
    assert abs(train_bad_ratio - 0.30) < 0.01, f"Train bad ratio {train_bad_ratio} deviates from 0.30"
    assert abs(test_bad_ratio - 0.30) < 0.01, f"Test bad ratio {test_bad_ratio} deviates from 0.30"


def test_feature_engineering_transformer():
    """Verify that FinancialFeatureEngineer generates all 5 engineered features accurately."""
    df = load_dataset(DATA_PATH).drop(columns=[TARGET_COLUMN])
    engineer = FinancialFeatureEngineer()
    df_transformed = engineer.fit_transform(df)

    expected_new_cols = [
        'monthly_payment_burden',
        'credit_to_age_ratio',
        'high_burden_flag',
        'liquidity_buffer_score',
        'stability_composite'
    ]
    for col in expected_new_cols:
        assert col in df_transformed.columns, f"Engineered column '{col}' missing from transformed DataFrame"

    # Verify mathematical correctness
    sample = df_transformed.iloc[0]
    expected_burden = sample['credit_amount'] / sample['duration']
    assert abs(sample['monthly_payment_burden'] - expected_burden) < 1e-5


def test_preprocessor_pipeline_transformation():
    """Verify full Scikit-learn Pipeline produces non-empty numeric matrix without NaNs."""
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    pipeline = build_preprocessor_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    assert isinstance(X_train_trans, np.ndarray)
    assert isinstance(X_test_trans, np.ndarray)
    assert X_train_trans.shape[0] == 800
    assert X_test_trans.shape[0] == 200
    assert X_train_trans.shape[1] == X_test_trans.shape[1]
    assert not np.isnan(X_train_trans).any(), "Transformed training data contains NaNs"
    assert not np.isnan(X_test_trans).any(), "Transformed test data contains NaNs"
