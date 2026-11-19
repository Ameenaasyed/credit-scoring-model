"""
Feature Engineering Module for Credit Scoring Model.
Defines domain-specific financial features and Scikit-Learn compatible transformers.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that creates domain-specific credit risk indicators
    from existing raw applicant attributes without data leakage.
    
    Engineered features:
    1. monthly_payment_burden: Estimated monthly credit payment (credit_amount / duration)
    2. credit_to_age_ratio: Debt burden relative to applicant life stage (credit_amount / age)
    3. high_burden_flag: Binary flag indicating monthly burden above threshold with high installment commitment
    4. capital_adequacy_score: Composite score combining savings status and checking balance
    5. financial_stability_score: Stability composite combining employment duration and residence tenure
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        # Stateless or purely record-level calculations; nothing to fit to prevent leakage
        return self

    def transform(self, X):
        # Support both pandas DataFrame and numpy array
        if not isinstance(X, pd.DataFrame):
            # If passed as array, X is expected to have standard column names
            raise TypeError("FinancialFeatureEngineer expects a pandas DataFrame input.")

        X_out = X.copy()

        # 1. Monthly Payment Burden: monthly debt obligation proxy
        duration = X_out['duration'].clip(lower=1)
        X_out['monthly_payment_burden'] = X_out['credit_amount'] / duration

        # 2. Credit-to-Age Ratio: financial leverage relative to chronological maturity
        age = X_out['age'].clip(lower=18)
        X_out['credit_to_age_ratio'] = X_out['credit_amount'] / age

        # 3. High Installment Burden Flag: high commitment (>=4) with significant loan
        # Installment commitment is % of disposable income (1-4). 4 is >= 35% of income.
        X_out['high_burden_flag'] = (
            (X_out['installment_commitment'] >= 4) & 
            (X_out['monthly_payment_burden'] > 150)
        ).astype(int)

        # 4. Savings and Checking Liquidity Indicator
        # Higher score implies greater liquidity cushion to absorb adverse shocks.
        checking_scores = {
            'no checking': 0,
            '<0': -1,
            '0<=X<200': 1,
            '>=200': 2
        }
        savings_scores = {
            'no known savings': 0,
            '<100': 1,
            '100<=X<500': 2,
            '500<=X<1000': 3,
            '>=1000': 4
        }
        checking_num = X_out['checking_status'].map(checking_scores).fillna(0)
        savings_num = X_out['savings_status'].map(savings_scores).fillna(0)
        X_out['liquidity_buffer_score'] = checking_num + savings_num

        # 5. Stability Composite: employment tenure and residence stability
        employment_scores = {
            'unemployed': 0,
            '<1': 1,
            '1<=X<4': 2,
            '4<=X<7': 3,
            '>=7': 4
        }
        emp_num = X_out['employment'].map(employment_scores).fillna(1)
        res_num = X_out['residence_since'].fillna(1)
        X_out['stability_composite'] = (emp_num * 1.5) + res_num

        self.feature_names_out_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
