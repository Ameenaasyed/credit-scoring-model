"""
Unit and Integration Tests for Real-Time Prediction and Input Validation.
Tests individual predictions, validation constraints, decision thresholds, and edge cases.
"""

import os
import pytest
import pandas as pd
from src.prediction import (
    predict_credit_risk,
    batch_predict,
    validate_applicant_input,
    DEFAULT_MODEL_PATH
)
from src.data_preprocessing import load_dataset, prepare_train_test_split


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'credit_dataset.csv')
FALLBACK_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'logistic_regression_model.joblib')


@pytest.fixture
def sample_applicant():
    return {
        'checking_status': '<0',
        'duration': 24,
        'credit_history': 'existing paid',
        'purpose': 'new car',
        'credit_amount': 4500,
        'savings_status': '<100',
        'employment': '1<=X<4',
        'installment_commitment': 3,
        'personal_status': 'male single',
        'other_parties': 'none',
        'residence_since': 2,
        'property_magnitude': 'car',
        'age': 30,
        'other_payment_plans': 'none',
        'housing': 'own',
        'existing_credits': 1,
        'job': 'skilled',
        'num_dependents': 1,
        'own_telephone': 'none',
        'foreign_worker': 'yes'
    }


def test_input_validation_valid(sample_applicant):
    """Verify valid applicant data passes validation without errors."""
    is_valid, errors = validate_applicant_input(sample_applicant)
    assert is_valid is True
    assert len(errors) == 0


def test_input_validation_missing_field(sample_applicant):
    """Verify missing field is flagged as validation failure."""
    del sample_applicant['credit_amount']
    is_valid, errors = validate_applicant_input(sample_applicant)
    assert is_valid is False
    assert any("Missing required numerical field: 'credit_amount'" in e for e in errors)


def test_input_validation_out_of_bounds(sample_applicant):
    """Verify numerical field exceeding realistic bounds is rejected."""
    sample_applicant['age'] = 12  # below minimum age 18
    is_valid, errors = validate_applicant_input(sample_applicant)
    assert is_valid is False
    assert any("age" in e and "out of expected bounds" in e for e in errors)


def test_input_validation_invalid_category(sample_applicant):
    """Verify invalid categorical value is rejected."""
    sample_applicant['checking_status'] = 'infinite_money'
    is_valid, errors = validate_applicant_input(sample_applicant)
    assert is_valid is False
    assert any("invalid category" in e for e in errors)


def test_predict_credit_risk_structure(sample_applicant):
    """Verify prediction returns all required fields and correct data types."""
    # If best model doesn't exist yet, test using logistic regression or decision tree
    model_to_use = DEFAULT_MODEL_PATH
    if not os.path.exists(model_to_use):
        model_to_use = FALLBACK_MODEL_PATH
    
    assert os.path.exists(model_to_use), f"Model not found at {model_to_use}"
    result = predict_credit_risk(sample_applicant, model_path=model_to_use)

    assert result['status'] == 'success'
    assert 'predicted_risk_class' in result
    assert result['predicted_risk_class'] in [0, 1]
    assert 'predicted_risk_label' in result
    assert 'Good Credit' in result['predicted_risk_label'] or 'Bad Credit' in result['predicted_risk_label']
    assert 0.0 <= result['high_risk_probability'] <= 1.0
    assert 0.0 <= result['good_credit_probability'] <= 1.0
    assert abs(result['high_risk_probability'] + result['good_credit_probability'] - 1.0) < 1e-3
    assert len(result['key_risk_factors']) > 0
    assert 'limitations_disclaimer' in result


def test_predict_threshold_effect(sample_applicant):
    """Verify that lowering risk threshold catches more high-risk applicants."""
    model_to_use = DEFAULT_MODEL_PATH
    if not os.path.exists(model_to_use):
        model_to_use = FALLBACK_MODEL_PATH

    res_strict = predict_credit_risk(sample_applicant, model_path=model_to_use, threshold=0.10)
    res_lenient = predict_credit_risk(sample_applicant, model_path=model_to_use, threshold=0.90)

    # A very low threshold of 0.10 flags applicant as high-risk much more easily
    assert res_strict['predicted_risk_class'] >= res_lenient['predicted_risk_class']


def test_batch_predict():
    """Verify batch prediction on DataFrame produces augmented output DataFrame."""
    model_to_use = DEFAULT_MODEL_PATH
    if not os.path.exists(model_to_use):
        model_to_use = FALLBACK_MODEL_PATH

    df = load_dataset(DATA_PATH)
    sample_df = df.drop(columns=['class']).head(10)
    results_df = batch_predict(sample_df, model_path=model_to_use)

    assert len(results_df) == 10
    assert 'predicted_risk_class' in results_df.columns
    assert 'predicted_risk_label' in results_df.columns
    assert 'high_risk_probability' in results_df.columns
