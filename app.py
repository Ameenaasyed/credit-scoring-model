"""
Credit Scoring Model — Streamlit interface.
Uses the existing serialized champion pipeline and validated prediction function.
Educational demonstration only; not for real lending decisions.
"""
import streamlit as st
import pandas as pd

from src.prediction import (
    ALLOWED_CATEGORIES,
    DEFAULT_MODEL_PATH,
    NUMERICAL_RANGES,
    predict_credit_risk,
)

st.set_page_config(
    page_title="CreditRisk | Credit Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #101d35; }
    [data-testid="stSidebar"] * { color: #eef3ff !important; }
    .hero {
      background: linear-gradient(120deg,#142746 0%,#203e68 58%,#176b70 100%);
      padding: 2rem 2.2rem; border-radius: 20px; color: white;
      margin-bottom: 1.2rem; box-shadow: 0 12px 30px rgba(20,39,70,.14);
    }
    .hero h1 { color:white; font-family:Manrope,sans-serif; font-size:2.1rem; margin:0; }
    .hero p { color:#d8e6f7; margin:.45rem 0 0 0; font-size:1rem; }
    .eyebrow { color:#8fe0d4; font-weight:700; letter-spacing:.13em; font-size:.72rem; }
    .panel {
      background:white; border:1px solid #e5eaf2; border-radius:16px;
      padding:1.2rem 1.35rem; box-shadow:0 4px 16px rgba(20,39,70,.04);
    }
    .risk-high { background:#fff1f0; border:1px solid #ffd2cd; color:#a52d27;
      border-radius:14px; padding:1rem 1.2rem; font-weight:700; }
    .risk-low { background:#eaf8f2; border:1px solid #c6ead9; color:#16734d;
      border-radius:14px; padding:1rem 1.2rem; font-weight:700; }
    div.stButton > button[kind="primary"] {
      background:linear-gradient(90deg,#176b70,#238b83); border:0;
      border-radius:10px; padding:.65rem 1.2rem; font-weight:700;
    }
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6 {
      font-family:Manrope,sans-serif; color:#142746;
    }
    .main .block-container [data-testid="stMarkdownContainer"] p,
    .main .block-container [data-testid="stMarkdownContainer"] li {
      color:#334155;
    }
    .main .block-container .hero h1 { color:white; }
    .main .block-container .hero p { color:#d8e6f7; }
    .main .block-container [data-testid="stCaptionContainer"],
    .main .block-container [data-testid="stCaptionContainer"] p,
    .main .block-container [data-testid="stWidgetLabel"],
    .main .block-container [data-testid="stWidgetLabel"] p {
      color:#475569;
    }
    .main .block-container [data-testid="stMetricLabel"],
    .main .block-container [data-testid="stMetricLabel"] p {
      color:#475569;
    }
    .main .block-container [data-testid="stMetricValue"] {
      color:#142746;
    }
    .main .block-container [data-testid="stExpander"] summary {
      color:#142746;
    }
    .main .block-container [data-testid="stCode"] pre,
    .main .block-container [data-testid="stJson"] * {
      color:#334155 !important;
    }
    .main .block-container [data-testid="stAlert"]:has([data-testid="stIconWarning"]) {
      background:#fff8db; border:1px solid #f1d27a;
    }
    .main .block-container [data-testid="stAlert"]:has([data-testid="stIconWarning"])
    [data-testid="stMarkdownContainer"] p {
      color:#713f12;
    }
    .muted { color:#68758b; font-size:.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 💳 CreditRisk")
    st.caption("Machine Learning • Creditworthiness")
    page = st.radio("NAVIGATION", ["Predict credit risk", "Model overview", "About & limitations"])
    st.divider()
    st.markdown("**Champion model**")
    st.markdown("Logistic Regression")
    st.caption("Loaded from models/best_credit_model.joblib")
    st.divider()
    st.caption("Academic / portfolio prototype")

if page == "Predict credit risk":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">CREDIT INTELLIGENCE · ML DEMONSTRATION</div>
          <h1>Credit Risk Assessment</h1>
          <p>Enter an applicant profile to estimate the probability of higher credit risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Complete all 20 fields. The model uses the existing trained Logistic Regression pipeline; this interface does not retrain it.")

    with st.form("applicant_form"):
        st.markdown("### 01 · Account & credit history")
        c1, c2, c3 = st.columns(3)
        with c1:
            checking_status = st.selectbox("Checking account status", ALLOWED_CATEGORIES["checking_status"], index=0)
            credit_history = st.selectbox("Credit history", ALLOWED_CATEGORIES["credit_history"], index=0)
            purpose = st.selectbox("Loan purpose", ALLOWED_CATEGORIES["purpose"], index=0)
            other_payment_plans = st.selectbox("Other payment plans", ALLOWED_CATEGORIES["other_payment_plans"], index=2)
        with c2:
            savings_status = st.selectbox("Savings status", ALLOWED_CATEGORIES["savings_status"], index=0)
            duration = st.number_input("Loan duration (months)", min_value=1, max_value=120, value=24, step=1)
            credit_amount = st.number_input("Credit amount (DM)", min_value=100, max_value=100000, value=5000, step=100)
            installment_commitment = st.selectbox("Installment commitment (scale 1–4)", [1, 2, 3, 4], index=2)
        with c3:
            existing_credits = st.number_input("Existing credits", min_value=1, max_value=10, value=1, step=1)
            other_parties = st.selectbox("Other parties", ALLOWED_CATEGORIES["other_parties"])
            property_magnitude = st.selectbox("Property / collateral", ALLOWED_CATEGORIES["property_magnitude"])
            foreign_worker = st.selectbox("Foreign worker", ALLOWED_CATEGORIES["foreign_worker"], index=1)

        st.markdown("### 02 · Personal & employment profile")
        c4, c5, c6 = st.columns(3)
        with c4:
            employment = st.selectbox("Employment duration", ALLOWED_CATEGORIES["employment"], index=2)
            job = st.selectbox("Job category", ALLOWED_CATEGORIES["job"], index=2)
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
        with c5:
            personal_status = st.selectbox("Personal status", ALLOWED_CATEGORIES["personal_status"], index=0)
            housing = st.selectbox("Housing", ALLOWED_CATEGORIES["housing"], index=1)
            residence_since = st.number_input("Years at current residence", min_value=1, max_value=10, value=3, step=1)
        with c6:
            num_dependents = st.number_input("Number of dependents", min_value=1, max_value=10, value=1, step=1)
            own_telephone = st.selectbox("Telephone registered", ALLOWED_CATEGORIES["own_telephone"], index=1)
            st.markdown("<div class='muted'>Check that all entries reflect the applicant profile you intend to demonstrate.</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("✨  Predict credit risk", type="primary", use_container_width=True)

    if submitted:
        applicant = {
            "checking_status": checking_status,
            "duration": int(duration),
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": int(credit_amount),
            "savings_status": savings_status,
            "employment": employment,
            "installment_commitment": int(installment_commitment),
            "personal_status": personal_status,
            "other_parties": other_parties,
            "residence_since": int(residence_since),
            "property_magnitude": property_magnitude,
            "age": int(age),
            "other_payment_plans": other_payment_plans,
            "housing": housing,
            "existing_credits": int(existing_credits),
            "job": job,
            "num_dependents": int(num_dependents),
            "own_telephone": own_telephone,
            "foreign_worker": foreign_worker,
        }
        try:
            result = predict_credit_risk(applicant)
            if result.get("status") != "success":
                st.error("Input validation failed.")
                for error in result.get("validation_errors", []):
                    st.write(f"- {error}")
            else:
                st.markdown("## Prediction result")
                high = result["predicted_risk_class"] == 1
                css_class = "risk-high" if high else "risk-low"
                st.markdown(
                    f'<div class="{css_class}">{"⚠️" if high else "✓"} &nbsp; {result["predicted_risk_label"]}</div>',
                    unsafe_allow_html=True,
                )
                a, b, c = st.columns(3)
                a.metric("Higher-risk probability", f'{result["high_risk_probability"]:.2%}')
                b.metric("Lower-risk probability", f'{result["good_credit_probability"]:.2%}')
                c.metric("Decision threshold", f'{result["decision_threshold"]:.0%}')
                st.markdown("### Probability distribution")
                st.progress(float(result["high_risk_probability"]))
                st.caption("Progress bar represents the model-estimated probability of the higher-risk class.")
                st.markdown("### Factors to review")
                for factor in result["key_risk_factors"]:
                    st.markdown(f"- {factor}")
                with st.expander("Technical details"):
                    st.write("Classifier:", result["model_name"])
                    st.write("Predicted class:", result["predicted_risk_class"])
                    st.json({"high_risk_probability": result["high_risk_probability"],
                              "good_credit_probability": result["good_credit_probability"]})
                st.warning(result["limitations_disclaimer"])
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

elif page == "Model overview":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">MODEL PERFORMANCE</div>
          <h1>Inside the scoring engine</h1>
          <p>Three classifiers were compared on a held-out test set of 200 records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.7200, "ROC-AUC": 0.7990, "High-risk recall": 0.7833, "Risk cost": 0.5400},
        {"Model": "Decision Tree", "Accuracy": 0.5950, "ROC-AUC": 0.6546, "High-risk recall": 0.6500, "Risk cost": 0.8250},
        {"Model": "Random Forest", "Accuracy": 0.7850, "ROC-AUC": 0.8175, "High-risk recall": 0.6000, "Risk cost": 0.6950},
    ])
    st.dataframe(
        metrics.style.format({"Accuracy":"{:.1%}","ROC-AUC":"{:.4f}","High-risk recall":"{:.1%}","Risk cost":"{:.4f}"}),
        use_container_width=True, hide_index=True
    )
    x1, x2, x3 = st.columns(3)
    x1.metric("Dataset records", "1,000")
    x2.metric("Good / bad records", "700 / 300")
    x3.metric("Automated tests", "17 / 17 passed*")
    st.markdown("### Why Logistic Regression was selected")
    st.write("It achieved the highest high-risk recall (78.33%) and lowest normalized risk cost (0.5400). Random Forest had the highest accuracy (78.50%) and ROC-AUC (0.8175). The selected model reflects the project's stated preference for reducing missed high-risk cases.")
    st.caption("*17/17 tests passed in the user's local Python 3.10.11 environment, as reported during setup.")
    st.markdown("### Saved model artifacts")
    st.code("models/best_credit_model.joblib\nmodels/logistic_regression_model.joblib\nmodels/decision_tree_model.joblib\nmodels/random_forest_model.joblib")

else:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">PROJECT INFORMATION</div>
          <h1>About this project</h1>
          <p>Machine learning exploration of creditworthiness using the Statlog German Credit benchmark.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Dataset")
    st.write("The project documentation describes 1,000 historical records with 20 input attributes: 700 labelled good credit and 300 labelled bad credit.")
    st.markdown("### Algorithms")
    st.write("- Logistic Regression\n- Decision Tree Classifier\n- Random Forest Classifier")
    st.markdown("### Important limitations")
    st.warning(
        "This is an educational prototype trained on a small historical benchmark dataset. "
        "It is not a certified credit score, financial advice, or an automated lending guarantee. "
        "The probability outputs have not been established as calibrated real-world default probabilities. "
        "Real lending use would require extensive validation, privacy controls, fairness and bias audits, "
        "explainability review, legal compliance, and human oversight."
    )
    st.caption("The interface displays heuristic risk indicators from the existing prediction module; these are not causal explanations of the model.")
