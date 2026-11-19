"""
Inference and Real-Time Prediction Module for Credit Scoring Model.
Provides validated prediction functions for single applicant profiles and batch datasets.
Includes explainability indicators, confidence scoring, and regulatory disclaimers.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple

if __package__:
    from . import feature_engineering
else:
    import feature_engineering

sys.modules.setdefault('feature_engineering', feature_engineering)

# Valid categorical categories based on Statlog German Credit dataset
ALLOWED_CATEGORIES = {
    'checking_status': ['<0', '0<=X<200', '>=200', 'no checking'],
    'credit_history': ['critical/other existing credit', 'existing paid', 'delayed previously', 'no credits/all paid', 'all paid'],
    'purpose': ['radio/tv', 'education', 'furniture/equipment', 'new car', 'used car', 'business', 'domestic appliance', 'repairs', 'other', 'retraining'],
    'savings_status': ['<100', '100<=X<500', '500<=X<1000', '>=1000', 'no known savings'],
    'employment': ['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'],
    'personal_status': ['male single', 'female div/dep/mar', 'male div/sep', 'male mar/wid'],
    'other_parties': ['none', 'guarantor', 'co applicant'],
    'property_magnitude': ['real estate', 'life insurance', 'car', 'no known property'],
    'other_payment_plans': ['bank', 'stores', 'none'],
    'housing': ['rent', 'own', 'for free'],
    'job': ['unemp/unskilled non res', 'unskilled resident', 'skilled', 'high qualif/self emp/mgmt'],
    'own_telephone': ['yes', 'none'],
    'foreign_worker': ['yes', 'no']
}

NUMERICAL_RANGES = {
    'duration': (1, 120),            # Months
    'credit_amount': (100, 100000),   # DM / Currency units
    'installment_commitment': (1, 4), # % of disposable income scale
    'residence_since': (1, 10),      # Years
    'age': (18, 100),                # Years
    'existing_credits': (1, 10),     # Count
    'num_dependents': (1, 10)        # Count
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_credit_model.joblib')

DISCLAIMER_TEXT = (
    "DISCLAIMER & LIMITATIONS: This model is an educational machine learning prototype trained "
    "on the historical Statlog German Credit benchmark dataset. It does not constitute a certified "
    "credit rating agency score, financial advice, or an automated lending guarantee. In compliance with "
    "fair lending regulations (e.g., ECOA, FCRA, EU AI Act), real-world credit decisions require strict "
    "auditing for demographic bias, transparency of automated reasoning, and adverse action reporting."
)


def validate_applicant_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates applicant dictionary for presence, data types, and realistic bounds.
    """
    errors = []
    
    # Check numerical fields
    for field, (min_val, max_val) in NUMERICAL_RANGES.items():
        if field not in data:
            errors.append(f"Missing required numerical field: '{field}'")
        else:
            val = data[field]
            if not isinstance(val, (int, float)) or np.isnan(val):
                errors.append(f"Field '{field}' must be a numeric value, got: {type(val).__name__}")
            elif val < min_val or val > max_val:
                errors.append(f"Field '{field}' value {val} out of expected bounds [{min_val}, {max_val}]")

    # Check categorical fields
    for field, allowed in ALLOWED_CATEGORIES.items():
        if field not in data:
            errors.append(f"Missing required categorical field: '{field}'")
        else:
            val = str(data[field]).strip()
            if val not in allowed:
                errors.append(f"Field '{field}' has invalid category '{val}'. Allowed values: {allowed}")

    return len(errors) == 0, errors


def analyze_risk_factors(data: Dict[str, Any], prob_high_risk: float) -> List[str]:
    """
    Generates human-understandable heuristic risk indicators explaining why
    the applicant was classified into their respective risk category.
    """
    factors = []

    # Checking account
    if data.get('checking_status') == '<0':
        factors.append("Critical negative checking account balance (< 0 DM), indicating immediate cash shortfall.")
    elif data.get('checking_status') == 'no checking':
        factors.append("No active checking account relationship on record with the financial institution.")
    elif data.get('checking_status') in ['>=200', '0<=X<200']:
        factors.append("Positive checking account liquidity buffer.")

    # Savings
    if data.get('savings_status') in ['<100', 'no known savings']:
        factors.append("Minimal or undisclosed liquid savings reserve to absorb adverse income events.")
    elif data.get('savings_status') in ['500<=X<1000', '>=1000']:
        factors.append("Strong liquid savings reserve (>= 500 DM).")

    # Credit History
    if data.get('credit_history') in ['all paid', 'delayed previously']:
        factors.append("Previous payment delays or newly established credit track record.")
    elif data.get('credit_history') == 'critical/other existing credit':
        factors.append("Substantial past credit utilization with critical accounts at other institutions.")

    # Duration and Amount
    dur = data.get('duration', 0)
    amt = data.get('credit_amount', 0)
    if dur > 36:
        factors.append(f"Extended loan maturity ({dur} months) elevating macroeconomic default exposure.")
    if amt > 5000:
        factors.append(f"Large requested principal amount ({amt} DM) increasing absolute credit exposure.")

    # Installment Commitment
    if data.get('installment_commitment', 0) >= 4:
        factors.append("High installment commitment (>=35% of net disposable income).")

    # Employment & Age
    if data.get('employment') in ['unemployed', '<1']:
        factors.append("Limited job tenure (< 1 year) or current unemployment.")
    if data.get('age', 100) < 25:
        factors.append("Early career applicant profile (< 25 years old) with limited historical wealth accumulation.")

    if not factors:
        factors.append("Standard financial and credit history parameters within normal variance.")

    return factors


def predict_credit_risk(
    applicant_data: Dict[str, Any],
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predicts creditworthiness for a single applicant.
    
    Parameters:
    - applicant_data: Dictionary containing all 20 required applicant attributes
    - model_path: Path to serialized Scikit-learn Pipeline
    - threshold: Classification probability threshold for positive (High Risk) class (default: 0.5)
    
    Returns:
    - Dictionary with risk label, risk class (0/1), probability, confidence, factors, and disclaimer.
    """
    # 1. Validation
    is_valid, validation_errors = validate_applicant_input(applicant_data)
    if not is_valid:
        return {
            'status': 'error',
            'validation_errors': validation_errors,
            'message': 'Input validation failed. Please provide all required applicant fields with valid values.'
        }

    # 2. Model Loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run train_models.py first.")
    
    model_pipeline = joblib.load(model_path)
    model_step_name = type(model_pipeline.named_steps['classifier']).__name__

    # 3. DataFrame Conversion
    input_df = pd.DataFrame([applicant_data])

    # 4. Inference
    proba_array = model_pipeline.predict_proba(input_df)[0]
    prob_good_risk = float(proba_array[0])
    prob_high_risk = float(proba_array[1])

    # Apply decision threshold
    risk_class = 1 if prob_high_risk >= threshold else 0
    if risk_class == 1:
        risk_label = "Bad Credit / Higher Credit Risk"
        confidence = prob_high_risk
    else:
        risk_label = "Good Credit / Lower Credit Risk"
        confidence = prob_good_risk

    # 5. Risk Factors Explanation
    contributing_factors = analyze_risk_factors(applicant_data, prob_high_risk)

    return {
        'status': 'success',
        'model_name': model_step_name,
        'decision_threshold': threshold,
        'predicted_risk_class': risk_class,
        'predicted_risk_label': risk_label,
        'high_risk_probability': round(prob_high_risk, 4),
        'good_credit_probability': round(prob_good_risk, 4),
        'model_confidence': round(confidence, 4),
        'key_risk_factors': contributing_factors,
        'limitations_disclaimer': DISCLAIMER_TEXT
    }


def batch_predict(
    applicants_df: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Executes batch credit risk prediction on a DataFrame of multiple applicants.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    
    model_pipeline = joblib.load(model_path)
    probas = model_pipeline.predict_proba(applicants_df)[:, 1]
    preds = (probas >= threshold).astype(int)

    results = applicants_df.copy()
    results['predicted_risk_class'] = preds
    results['predicted_risk_label'] = np.where(preds == 1, 'Bad Credit / Higher Credit Risk', 'Good Credit / Lower Credit Risk')
    results['high_risk_probability'] = np.round(probas, 4)
    return results


if __name__ == '__main__':
    # Demonstration sample profiles
    sample_high_risk_applicant = {
        'checking_status': '<0',
        'duration': 48,
        'credit_history': 'delayed previously',
        'purpose': 'business',
        'credit_amount': 9500,
        'savings_status': '<100',
        'employment': '<1',
        'installment_commitment': 4,
        'personal_status': 'male single',
        'other_parties': 'none',
        'residence_since': 1,
        'property_magnitude': 'no known property',
        'age': 22,
        'other_payment_plans': 'bank',
        'housing': 'rent',
        'existing_credits': 2,
        'job': 'unskilled resident',
        'num_dependents': 1,
        'own_telephone': 'none',
        'foreign_worker': 'yes'
    }

    sample_low_risk_applicant = {
        'checking_status': '>=200',
        'duration': 12,
        'credit_history': 'critical/other existing credit',
        'purpose': 'radio/tv',
        'credit_amount': 1500,
        'savings_status': '>=1000',
        'employment': '>=7',
        'installment_commitment': 2,
        'personal_status': 'male single',
        'other_parties': 'none',
        'residence_since': 4,
        'property_magnitude': 'real estate',
        'age': 45,
        'other_payment_plans': 'none',
        'housing': 'own',
        'existing_credits': 1,
        'job': 'skilled',
        'num_dependents': 1,
        'own_telephone': 'yes',
        'foreign_worker': 'yes'
    }

    print("Testing Single Applicant Prediction (High Risk Profile)...")
    res_high = predict_credit_risk(sample_high_risk_applicant)
    print(f"Prediction: {res_high['predicted_risk_label']} (Prob: {res_high['high_risk_probability']})")
    print("Risk Factors:", res_high['key_risk_factors'])

    print("\nTesting Single Applicant Prediction (Low Risk Profile)...")
    res_low = predict_credit_risk(sample_low_risk_applicant)
    print(f"Prediction: {res_low['predicted_risk_label']} (Prob: {res_low['high_risk_probability']})")
    print("Risk Factors:", res_low['key_risk_factors'])
