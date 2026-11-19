"""
Model Training & Hyperparameter Tuning Module for Credit Scoring Model.
Trains, cross-validates, and tunes:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
Saves trained pipelines and cross-validation summaries.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__:
    from .data_preprocessing import (
        load_dataset,
        prepare_train_test_split,
        build_preprocessor_pipeline
    )
else:
    from data_preprocessing import (
        load_dataset,
        prepare_train_test_split,
        build_preprocessor_pipeline
    )


def train_and_tune_logistic_regression(
    preprocessor: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains and tunes Logistic Regression using L2 regularization and balanced class weights.
    """
    print("\n--- Training Model 1: Logistic Regression ---")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    param_grid = {
        'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear']
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    print(f"Logistic Regression Best CV ROC-AUC: {best_score:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    return best_pipe, {'best_params': best_params, 'cv_roc_auc': float(best_score)}


def train_and_tune_decision_tree(
    preprocessor: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains and tunes Decision Tree Classifier with regularization constraints on depth and leaf size.
    """
    print("\n--- Training Model 2: Decision Tree Classifier ---")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ])

    param_grid = {
        'classifier__max_depth': [3, 4, 5, 6, 8],
        'classifier__min_samples_split': [10, 20, 30, 50],
        'classifier__min_samples_leaf': [5, 10, 15, 20],
        'classifier__criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    print(f"Decision Tree Best CV ROC-AUC: {best_score:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    return best_pipe, {'best_params': best_params, 'cv_roc_auc': float(best_score)}


def train_and_tune_random_forest(
    preprocessor: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    cv: StratifiedKFold
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains and tunes Random Forest Classifier with ensemble bagging and balanced class weights.
    """
    print("\n--- Training Model 3: Random Forest Classifier ---")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 8, 12, None],
        'classifier__min_samples_split': [5, 10, 20],
        'classifier__min_samples_leaf': [2, 4, 8],
        'classifier__max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    print(f"Random Forest Best CV ROC-AUC: {best_score:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    return best_pipe, {'best_params': best_params, 'cv_roc_auc': float(best_score)}


def run_training_pipeline(
    data_path: str = os.path.join(PROJECT_ROOT, 'data', 'credit_dataset.csv'),
    models_dir: str = os.path.join(PROJECT_ROOT, 'models'),
    results_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'evaluation_results')
) -> Dict[str, Any]:
    """
    Executes end-to-end training, saves trained pipelines, and logs CV summaries.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, test_size=0.2, random_state=42)

    # Base preprocessor pipeline
    preprocessor = build_preprocessor_pipeline()
    # Fit preprocessor on train data and save standalone preprocessor
    fitted_preprocessor = build_preprocessor_pipeline()
    fitted_preprocessor.fit(X_train, y_train)
    joblib.dump(fitted_preprocessor, os.path.join(models_dir, 'preprocessing_pipeline.joblib'))
    print(f"Standalone preprocessing pipeline saved to: {models_dir}/preprocessing_pipeline.joblib")

    # 5-fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Logistic Regression
    lr_pipe, lr_meta = train_and_tune_logistic_regression(build_preprocessor_pipeline(), X_train, y_train, cv)
    joblib.dump(lr_pipe, os.path.join(models_dir, 'logistic_regression_model.joblib'))

    # 2. Decision Tree
    dt_pipe, dt_meta = train_and_tune_decision_tree(build_preprocessor_pipeline(), X_train, y_train, cv)
    joblib.dump(dt_pipe, os.path.join(models_dir, 'decision_tree_model.joblib'))

    # 3. Random Forest
    rf_pipe, rf_meta = train_and_tune_random_forest(build_preprocessor_pipeline(), X_train, y_train, cv)
    joblib.dump(rf_pipe, os.path.join(models_dir, 'random_forest_model.joblib'))

    cv_summary = {
        'Logistic Regression': lr_meta,
        'Decision Tree': dt_meta,
        'Random Forest': rf_meta
    }

    with open(os.path.join(results_dir, 'cv_summary.json'), 'w') as f:
        json.dump(cv_summary, f, indent=4)

    print(f"\nAll models successfully trained and serialized to: {models_dir}")
    return {
        'models': {
            'Logistic Regression': lr_pipe,
            'Decision Tree': dt_pipe,
            'Random Forest': rf_pipe
        },
        'cv_summary': cv_summary,
        'data_splits': (X_train, X_test, y_train, y_test)
    }


if __name__ == '__main__':
    run_training_pipeline()
