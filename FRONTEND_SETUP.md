# Running the Credit Scoring Model frontend

## Requirements
- Windows 10/11
- Python 3.10 (recommended; the saved model artifacts were serialized with scikit-learn 1.2.1)
- Internet access for the first dependency installation

## First-time setup (Command Prompt)
Open Command Prompt in this folder (`credit_scoring_model`) and run:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the frontend
With `.venv` active, run:

```cmd
python -m streamlit run app.py
```

Or double-click `run_app.bat` after the virtual environment and requirements have been installed.

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## What the interface does
- Collects the 20 input attributes required by `src/prediction.py`.
- Calls the existing `predict_credit_risk()` function.
- Loads `models/best_credit_model.joblib`.
- Displays the predicted class, estimated class probabilities, heuristic risk indicators, and limitations.
- Shows the model comparison metrics documented in the project.

The frontend does not retrain models or modify the saved model files.

## Run the existing tests
```cmd
python -m pytest -v
```

## Troubleshooting
- If `streamlit` is not recognized, use `python -m streamlit run app.py`.
- If a model-loading warning/error occurs, confirm that the pinned versions in `requirements.txt` were installed in the active `.venv`.
- If port 8501 is occupied, Streamlit may offer another local port in the terminal.
- If the browser does not open automatically, copy the `Local URL` printed in the terminal.

## Educational-use notice
This is an academic prototype based on the historical Statlog German Credit dataset. It is not a production lending system, certified credit score, financial advice, or a guarantee of default. Do not enter real personal or financial information.
