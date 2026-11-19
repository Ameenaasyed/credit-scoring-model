#!/usr/bin/env python3
"""
Master Pipeline Runner for Credit Scoring Model Project.
Executes dataset verification, preprocessing, model training, evaluation,
plotting, and automated testing from the project root directory.
"""

import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def run_step(step_name, command, cwd=None):
    print(f"\n=======================================================")
    print(f"STEP: {step_name}")
    print(f"COMMAND: {command}")
    print(f"=======================================================")
    res = subprocess.run(command, shell=True, cwd=cwd or PROJECT_DIR, capture_output=False)
    if res.returncode != 0:
        print(f"ERROR: Step '{step_name}' failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"SUCCESS: Step '{step_name}' completed successfully.")


def main():
    print("=================================================================")
    print("STARTING END-TO-END CREDIT SCORING ML PIPELINE EXECUTION")
    print("=================================================================")

    # 1. Dataset Verification
    csv_path = os.path.join(PROJECT_DIR, 'data', 'credit_dataset.csv')
    if not os.path.exists(csv_path):
        run_step(
            "Acquiring Authentic Dataset",
            f'"{sys.executable}" -c "from sklearn.datasets import fetch_openml; df = fetch_openml(\'credit-g\', version=1, as_frame=True, parser=\'auto\').frame; df.to_csv(\'data/credit_dataset.csv\', index=False)"'
        )
    else:
        print(f"\nAuthentic dataset verified at: {csv_path}")

    # 2. Data Preprocessing Verification
    run_step(
        "Data Preprocessing & Split Verification",
        f'"{sys.executable}" src/data_preprocessing.py'
    )

    # 3. Model Training & Tuning (if models don't exist yet)
    models_dir = os.path.join(PROJECT_DIR, 'models')
    best_model_path = os.path.join(models_dir, 'best_credit_model.joblib')
    if not os.path.exists(best_model_path):
        run_step(
            "Training & Tuning ML Models (LR, DT, RF)",
            f'"{sys.executable}" src/train_models.py'
        )
    else:
        print(f"\nTrained models already exist at: {models_dir}")

    # 4. Model Evaluation & Publication Graphs Generation
    run_step(
        "Model Evaluation & Plot Generation",
        f'"{sys.executable}" src/evaluate_models.py'
    )

    # 5. Real-Time Prediction Demonstration
    run_step(
        "Running Real-Time Prediction Verification",
        f'"{sys.executable}" src/prediction.py'
    )

    # 6. Run Automated Test Suite
    run_step(
        "Executing Automated Pytest Suite",
        f'"{sys.executable}" -m pytest tests/ -v'
    )

    print("\n=================================================================")
    print("ALL PIPELINE STAGES COMPLETED WITH 100% SUCCESS!")
    print("=================================================================")


if __name__ == '__main__':
    main()
