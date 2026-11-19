# Credit Scoring Model: Predicting Creditworthiness Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.2+-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-17%20passed%20(100%25)-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Reproducible-success.svg)]()

---

## 1. Project Title & Abstract

### Title
**Credit Scoring Model – Predicting Creditworthiness and Default Risk Using Applied Machine Learning**

### Abstract
This project implements an end-to-end, reproducible, production-grade Machine Learning credit scoring classification system. Using the benchmark **Statlog German Credit Dataset** (1,000 authentic applicant records across 20 financial, socio-economic, and demographic features), the system classifies prospective borrowers into two discrete risk tiers: **Good Credit / Lower Credit Risk** (Class 0) and **Bad Credit / Higher Credit Risk** (Class 1). 

Three classical and ensemble machine learning algorithms were trained, cross-validated via 5-fold Stratified K-Fold cross-validation, and rigorously tuned:
1. **Logistic Regression (L2 Regularized with Balanced Class Weights)**
2. **Decision Tree Classifier (Cost-Complexity Pruning & Depth Regulation)**
3. **Random Forest Classifier (Ensemble Bagging with Subspace Sampling)**

Because commercial lending loss is highly asymmetric—where granting credit to a defaulting borrower (**False Negative**) costs upwards of 5x more than denying credit to a creditworthy borrower (**False Positive**)—the evaluation framework prioritizes **High-Risk Recall**, **ROC-AUC**, and **Cost-Sensitive Risk Loss** alongside overall Accuracy. 

On the held-out test set (200 records, 30% default rate), **Logistic Regression** achieved an outstanding **78.33% Recall** on high-risk defaults (identifying 47 out of 60 bad credit applicants) with an **ROC-AUC of 0.7990** and the lowest normalized financial risk cost (**0.540**). **Random Forest** achieved the highest overall test accuracy (**78.50%**) and peak discrimination (**ROC-AUC 0.8175**). All pipelines prevent data leakage via Scikit-Learn `Pipeline` and `ColumnTransformer` constructs, serializing production-ready Joblib models accompanied by automated unit tests (17 passed out of 17).

---

## 2. Problem Statement & Objectives

### Problem Statement
Retail financial institutions, micro-lenders, and commercial banks face significant credit risk exposure when underwriting loan applications. Traditional heuristic credit evaluation or manual review is time-consuming, prone to human inconsistency, and unable to capture non-linear multidimensional interactions between applicant liquidity, loan term, debt commitment, and employment stability. 

Furthermore, naive machine learning models often optimize solely for accuracy on imbalanced datasets (e.g., 70% good vs. 30% bad credit), resulting in models that predict the majority class while failing to detect the true high-risk defaults that cause loan write-offs.

### Project Objectives
1. **Data Acquisition & Integrity Verification:** Ingest the authentic Statlog German Credit dataset from the UCI Machine Learning Repository / OpenML without synthetic fabrication.
2. **Leakage-Free Preprocessing:** Build an automated Scikit-Learn pipeline encapsulating median numerical imputation, standard scaling, categorical imputation, and one-hot encoding, fitting strictly on training splits.
3. **Domain Feature Engineering:** Derive predictive financial ratios, such as monthly debt payment burden, credit-to-age ratios, and liquidity buffer indices.
4. **Algorithm Benchmarking & Hyperparameter Tuning:** Formulate and tune Logistic Regression, Decision Tree, and Random Forest algorithms under 5-fold Stratified Cross-Validation.
5. **Cost-Sensitive Risk Evaluation:** Assess performance using Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix heatmaps, and expected lending loss matrices.
6. **Interpretability & Fair Lending Analysis:** Extract feature importance rankings, visualize decision rules, and evaluate ethical implications regarding demographic variables (age, personal status, foreign worker).
7. **Production Deployment Function & Automated Testing:** Provide an input-validated single and batch inference engine with comprehensive automated test suites.

---

## 3. Dataset Description & Provenance

The model is trained on the authentic **Statlog (German Credit Data) Data Set**, originally curated by Prof. Dr. Hans Hofmann at the University of Hamburg and distributed by the UCI Machine Learning Repository and OpenML (ID: `credit-g`).

- **Total Observations:** 1,000 real applicant profiles
- **Total Features:** 20 input attributes (7 numerical, 13 categorical) + 1 binary target (`class`)
- **Missing Values:** 0 missing entries in the raw benchmark
- **Target Distribution:**
  - `good` (Low Credit Risk): 700 instances (70.0%)
  - `bad` (High Credit Risk / Default): 300 instances (30.0%)

### Feature Dictionary

| Feature Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `checking_status` | Categorical | Status of existing checking account | `<0`, `0<=X<200`, `>=200`, `no checking` |
| `duration` | Numerical | Loan duration in months | 4 to 72 months |
| `credit_history` | Categorical | Past credit repayment history | `critical/other existing credit`, `existing paid`, `delayed previously`, `no credits/all paid`, `all paid` |
| `purpose` | Categorical | Purpose of credit | `new car`, `used car`, `furniture/equipment`, `radio/tv`, `domestic appliance`, `repairs`, `education`, `business`, `retraining`, `other` |
| `credit_amount` | Numerical | Credit amount in Deutsche Mark (DM) | 250 to 18,424 DM |
| `savings_status` | Categorical | Balance in savings accounts / bonds | `<100`, `100<=X<500`, `500<=X<1000`, `>=1000`, `no known savings` |
| `employment` | Categorical | Present employment tenure | `unemployed`, `<1`, `1<=X<4`, `4<=X<7`, `>=7` years |
| `installment_commitment`| Numerical | Installment rate in % of disposable income | 1, 2, 3, 4 (where 4 represents >= 35%) |
| `personal_status` | Categorical | Sex and marital status | `male single`, `female div/dep/mar`, `male div/sep`, `male mar/wid` |
| `other_parties` | Categorical | Other debtors or guarantors | `none`, `co applicant`, `guarantor` |
| `residence_since` | Numerical | Present residence duration in years | 1, 2, 3, 4 years |
| `property_magnitude` | Categorical | Most valuable collateral asset | `real estate`, `life insurance`, `car`, `no known property` |
| `age` | Numerical | Applicant age in years | 19 to 75 years |
| `other_payment_plans`| Categorical | Other installment payment plans | `bank`, `stores`, `none` |
| `housing` | Categorical | Housing arrangement | `rent`, `own`, `for free` |
| `existing_credits` | Numerical | Number of existing credits at this bank | 1 to 4 credits |
| `job` | Categorical | Employment qualification level | `unskilled resident`, `skilled`, `high qualif/self emp/mgmt`, `unemp/unskilled non res` |
| `num_dependents` | Numerical | Number of dependents liable to maintain | 1 or 2 |
| `own_telephone` | Categorical | Registered telephone under applicant name | `yes`, `none` |
| `foreign_worker` | Categorical | Foreign worker status | `yes`, `no` |
| **`class` (Target)** | Binary | Creditworthiness classification | `good` (Creditworthy, 0), `bad` (High Risk, 1) |

---

## 4. Feature Engineering

Five domain-specific credit risk indicators are engineered directly from raw records without external dependencies:

1. **`monthly_payment_burden` (`credit_amount / duration`):**  
   Estimates the monthly cash outflow required to amortize the loan. High burdens relative to disposable income severely inflate default propensity.
2. **`credit_to_age_ratio` (`credit_amount / age`):**  
   Measures leverage against life-stage capital accumulation. A young applicant seeking large credit represents elevated risk compared to an older applicant with identical capital requests.
3. **`high_burden_flag`:**  
   A binary interaction indicator triggered when `installment_commitment == 4` (>=35% of income) and `monthly_payment_burden > 150 DM`.
4. **`liquidity_buffer_score`:**  
   An ordinal synthetic index summing checking liquidity (-1 for negative balance to +2 for >=200 DM) and savings reserves (0 for none to +4 for >=1000 DM).
5. **`stability_composite`:**  
   Weighted composite score combining employment duration (`employment_score * 1.5`) with residential stability (`residence_since`).

---

## 5. Preprocessing & Leakage Prevention Pipeline

Data transformations follow rigorous machine learning standards:
- **Stratified Splitting:** 800 training records (80%) and 200 held-out test records (20%), stratified by the target variable to ensure the exact 70/30 class ratio is maintained.
- **Leakage Prevention:** All scalers, imputers, and encoders are fit **only** on `X_train` and applied transformatively to `X_test` and production inputs.
- **Pipeline Components:**
  - Scikit-Learn `ColumnTransformer` with `SimpleImputer(strategy='median')` and `StandardScaler()` for continuous and engineered variables.
  - `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features.
  - Output dimensionality: 66 transformed numeric input columns.

---

## 6. Machine Learning Algorithms & Training Setup

### Model 1: Logistic Regression
- **Architecture:** Linear generalized model with Logit link function: $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$
- **Hyperparameter Tuning:** Grid search over $C \in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]$ and solvers (`lbfgs`, `liblinear`).
- **Best Configuration:** $C = 0.05$, `solver='liblinear'`, `class_weight='balanced'`.

### Model 2: Decision Tree Classifier
- **Architecture:** Non-parametric recursive binary partitioning using Gini impurity.
- **Hyperparameter Tuning:** Grid search over `max_depth` (3 to 8), `min_samples_split` (10 to 50), and `min_samples_leaf` (5 to 20).
- **Best Configuration:** `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=10`, `criterion='gini'`, `class_weight='balanced'`.

### Model 3: Random Forest Classifier
- **Architecture:** Ensemble of decorrelated decision trees using bagging (Bootstrap Aggregating) and random feature subspace sampling.
- **Hyperparameter Tuning:** Grid search over `n_estimators` (100, 200, 300), `max_depth` (5, 8, 12, None), and `max_features` (`sqrt`, `log2`).
- **Best Configuration:** `n_estimators=200`, `max_depth=12`, `max_features='log2'`, `min_samples_leaf=2`, `min_samples_split=5`, `class_weight='balanced'`.

---

## 7. Experimental Results & Model Evaluation

All models were evaluated on the independent, held-out test partition of **200 applicants** (140 Good Credit, 60 Bad Credit defaults).

### Test Set Performance Comparison Table

| Model | Test Accuracy | ROC-AUC | High-Risk Recall (Class 1) | High-Risk Precision (Class 1) | High-Risk F1-Score | Good Specificity (Class 0) | True Positives (TP) | False Negatives (FN) | True Negatives (TN) | False Positives (FP) | Normalized Risk Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 72.00% | **0.7990** | **78.33%** | 52.22% | **0.6267** | 69.29% | **47** | **13** | 97 | 43 | **0.5400** |
| **Decision Tree** | 59.50% | 0.6546 | 65.00% | 39.39% | 0.4906 | 57.14% | 39 | 21 | 80 | 60 | 0.8250 |
| **Random Forest** | **78.50%** | **0.8175** | 60.00% | **65.45%** | 0.6261 | **86.43%** | 36 | 24 | **121** | **19** | 0.6950 |

*Note: Risk cost is calculated using industry-standard loss weighting: $\text{Total Loss} = 5.0 \times \text{FN} + 1.0 \times \text{FP}$, normalized by total test population ($N=200$).*

### Confusion Matrix Detailed Breakdown

1. **Logistic Regression:**
   - **True Positives (TP = 47):** Correctly caught 47 out of 60 actual credit defaults.
   - **False Negatives (FN = 13):** Missed only 13 defaults.
   - **True Negatives (TN = 97):** Accurately approved 97 creditworthy borrowers.
   - **False Positives (FP = 43):** Erroneously flagged 43 good borrowers for manual review.
2. **Random Forest:**
   - **True Positives (TP = 36):** Caught 36 defaults.
   - **False Negatives (FN = 24):** Missed 24 defaults (nearly double the default misses of Logistic Regression).
   - **True Negatives (TN = 121):** Highest specificity, approving 121 good borrowers.
   - **False Positives (FP = 19):** Minimal false rejections.

---

## 8. Final Model Selection

**Selected Primary Scoring Model:** **Logistic Regression (L2 Regularized, Balanced)**  
**Selected Non-Linear Benchmark:** **Random Forest Classifier**

### Rationale:
In bank lending, failing to predict a default leads to the loss of loan principal balance. Conversely, denying credit to a solvent applicant only costs the lost interest spread.
- Logistic Regression detected **78.33% of all defaults** (missing only 13), resulting in a substantially lower Normalized Risk Cost of **0.5400** compared to **0.6950** for Random Forest.
- Logistic Regression provides **linear coefficients and odds ratios**, satisfying the regulatory requirement for **Adverse Action Notices** under the Equal Credit Opportunity Act (ECOA).
- Random Forest remains preserved in `models/random_forest_model.joblib` for institutions requiring maximum portfolio acceptance rates and peak overall accuracy (78.50%).

---

## 9. Key Exploratory & Interpretability Findings

1. **Checking Account Liquidity is the #1 Predictor:**  
   Applicants with negative checking balances (`<0 DM`) exhibit a default rate of **49.3%**, whereas applicants with balances `>=200 DM` or substantial savings have default rates below **15%**.
2. **Loan Duration & Capital Exposure:**  
   Loans with durations exceeding 36 months exhibit nearly double the default incidence compared to loans under 12 months.
3. **Credit History Paradox:**  
   Applicants with `critical/other existing credit` paradoxically had lower default rates than those with `all paid` or `delayed previously`, reflecting experienced credit users with established debt capacity.
4. **Age Distribution:**  
   Applicants under 25 years of age had a default rate of **41.2%**, compared to **24.1%** for applicants aged 35–50, driven by lower employment tenure and asset accumulation.

---

## 10. Project Structure

```
credit_scoring_model/
├── data/
│   └── credit_dataset.csv                   # Authentic UCI German Credit Dataset (1000 records)
├── notebooks/
│   ├── credit_scoring_analysis.ipynb        # Complete interactive Jupyter Notebook
│   └── generate_notebook.py                 # Notebook generation automation script
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py               # Domain financial ratios & Scikit-learn transformer
│   ├── data_preprocessing.py                # Stratified split & ColumnTransformer pipeline
│   ├── train_models.py                      # 5-fold CV & GridSearch training for LR, DT, RF
│   ├── evaluate_models.py                   # Metrics calculation & 12 publication charts
│   └── prediction.py                        # Validated single & batch prediction module
├── models/
│   ├── best_credit_model.joblib             # End-to-end serialized champion pipeline
│   ├── logistic_regression_model.joblib     # Tuned Logistic Regression pipeline
│   ├── decision_tree_model.joblib           # Tuned Decision Tree pipeline
│   ├── random_forest_model.joblib           # Tuned Random Forest pipeline
│   └── preprocessing_pipeline.joblib        # Standalone fitted preprocessor
├── outputs/
│   ├── graphs/                              # 12 Publication-quality 150 DPI charts
│   │   ├── 01_target_distribution.png
│   │   ├── 02_numerical_distributions.png
│   │   ├── 03_credit_amount_by_risk.png
│   │   ├── 04_categorical_risk_rates.png
│   │   ├── 05_correlation_heatmap.png
│   │   ├── 06_age_group_vs_risk.png
│   │   ├── 07_confusion_matrices.png
│   │   ├── 08_roc_curves.png
│   │   ├── 09_precision_recall_curves.png
│   │   ├── 10_models_metrics_comparison.png
│   │   ├── 11_feature_importance_rf.png
│   │   └── 12_decision_tree_structure.png
│   └── evaluation_results/
│       ├── model_comparison.csv             # Benchmark metrics comparison table
│       ├── model_metrics.json               # Structured JSON evaluation results
│       └── cv_summary.json                  # Cross-validation hyperparameter logs
├── tests/
│   ├── __init__.py
│   ├── conftest.py                          # Global test environment configuration
│   ├── test_preprocessing.py                # 5 data pipeline and leakage tests
│   ├── test_prediction.py                   # 8 validation and inference tests
│   └── test_models.py                       # 4 model calibration and reproducibility tests
├── README.md                                # Comprehensive project documentation
├── feature_engineering.py                   # Top-level portability bridge for Joblib models
├── project_report.md                        # Academic engineering project report
├── requirements.txt                         # Pinned compatible Python package dependencies
└── run_pipeline.py                          # Unified end-to-end pipeline runner
```

---

## 11. Installation & Usage Instructions

### Prerequisites
- Python 3.10 or 3.11
- Pip package manager (and virtual environment recommended)

### 1. Navigate to Project Root & Install Pinned Dependencies
```bash
cd credit_scoring_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute the Master Pipeline
To run the entire pipeline end-to-end (dataset check, preprocessing verification, evaluation, plot generation, prediction check, and tests):
```bash
python3 run_pipeline.py
```

### 3. Run Automated Tests
```bash
python3 -m pytest tests/ -v
```
*Current test status: **17 passed (100% pass rate)**.*

### 4. Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/credit_scoring_analysis.ipynb
```

### 5. Programmatic Prediction in Python
Run directly from the `credit_scoring_model/` project root:
```bash
python3 src/prediction.py
```
Or import in Python from `credit_scoring_model/`:
```python
from src.prediction import predict_credit_risk

applicant = {
    'checking_status': '<0',
    'duration': 36,
    'credit_history': 'delayed previously',
    'purpose': 'business',
    'credit_amount': 8000,
    'savings_status': '<100',
    'employment': '<1',
    'installment_commitment': 4,
    'personal_status': 'male single',
    'other_parties': 'none',
    'residence_since': 1,
    'property_magnitude': 'no known property',
    'age': 23,
    'other_payment_plans': 'bank',
    'housing': 'rent',
    'existing_credits': 2,
    'job': 'skilled',
    'num_dependents': 1,
    'own_telephone': 'none',
    'foreign_worker': 'yes'
}

result = predict_credit_risk(applicant)
print("Classification:", result['predicted_risk_label'])
print("Default Probability:", f"{result['high_risk_probability']:.2%}")
print("Explanatory Factors:", result['key_risk_factors'])
```

---

## 12. Model Interpretability, Fairness & Ethical Governance

### 1. Feature Importance Insights
In the Random Forest ensemble:
1. `credit_amount` & `monthly_payment_burden` (Top 2 continuous financial obligations)
2. `duration` (Term exposure)
3. `checking_status_<0` and `checking_status_no checking` (Immediate liquidity posture)
4. `age` and `credit_to_age_ratio` (Capital accumulation proxy)
5. `savings_status_<100` (Lack of financial reserves)

### 2. Demographic Fairness Considerations
The dataset includes demographic attributes such as `personal_status` (encoding sex and marital status), `foreign_worker`, and `age`.
- Under the **Equal Credit Opportunity Act (ECOA)** and the **EU Artificial Intelligence Act (Annex III - High Risk AI)**, using gender, marital status, or nationality as direct criteria for credit underwriting is illegal.
- In production deployment, sensitive demographic predictors must be removed or sanitized through disparate impact analysis (e.g., Four-Fifths Rule, Demographic Parity, Equalized Odds).
- Models must not confuse correlation with causation: an applicant's age or marital status is a proxy for wealth and income stability, not an inherent behavioral determinant of creditworthiness.

---

## 13. Limitations & Future Roadmap

### Current Limitations
1. **Sample Size:** 1,000 observations is a standard academic benchmark but smaller than institutional production datasets (100k+ records).
2. **Historical Currency:** Amounts are denominated in Deutsche Mark (historical dataset).
3. **Absence of Real-Time Bureau Feeds:** The dataset lacks dynamic bureau features such as revolving credit utilization percentage or debt-to-income (DTI) ratio.

### Future Enhancements
1. **Gradient Boosting Models:** Integrate LightGBM and XGBoost with Bayesian Optimization (Optuna).
2. **SHAP & LIME Integration:** Compute exact TreeSHAP values for real-time waterfall explainability on adverse action notices.
3. **Fairness-Aware Re-weighing:** Incorporate Fairlearn algorithms to balance acceptance rates across demographic sub-populations.
4. **Credit Score Card Calibration:** Convert model probabilities into standard FICO-scale three-digit credit scores (300 to 850) using logarithmic odds scaling.

---

## 14. References & Academic Citations

1. Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. [DOI: 10.24432/C5C59P](https://doi.org/10.24432/C5C59P).
2. Thomas, L. C., Crook, J. N., & Edelman, D. B. (2017). *Credit Scoring and Its Applications (2nd ed.)*. SIAM Monographs on Mathematical Modeling and Computation.
3. Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. *Journal of the Royal Statistical Society: Series A (Statistics in Society)*, 160(3), 523-541.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Federal Reserve Board (2007). *Report to the Congress on Credit Scoring and Its Effects on the Availability and Affordability of Credit*.

---

## 12. Streamlit Frontend

A lightweight Streamlit interface is included in `app.py`. It uses the existing trained Logistic Regression pipeline and the validated `predict_credit_risk()` function; it does not retrain or replace the ML models.

### Run on Windows

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit (normally `http://localhost:8501`). You can also use `run_app.bat` after setup.

See `FRONTEND_SETUP.md` for full setup and troubleshooting instructions. This frontend is for academic demonstration only; do not enter real personal or financial information.
