"""
Model Evaluation and Visualization Module for Credit Scoring Model.
Generates comprehensive EDA plots, calculates rigorous test metrics,
creates confusion matrices, ROC curves, feature importances, and selects the best model.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, average_precision_score, classification_report
)
from sklearn.tree import plot_tree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__:
    from .data_preprocessing import (
        load_dataset,
        prepare_train_test_split,
        ALL_NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        decode_target
    )
else:
    from data_preprocessing import (
        load_dataset,
        prepare_train_test_split,
        ALL_NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        decode_target
    )


def setup_plotting_style():
    """Configures academic publication styling for matplotlib and seaborn."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150
    })


def generate_eda_visualizations(df: pd.DataFrame, graphs_dir: str):
    """
    Generates academic-quality EDA visualizations explaining distributions,
    feature relationships, and risk concentrations.
    """
    os.makedirs(graphs_dir, exist_ok=True)
    setup_plotting_style()

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['class'].value_counts()
    colors = ['#2b5c8f', '#d95f02']
    bars = ax.bar(counts.index.str.capitalize(), counts.values, color=colors, width=0.5, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval} ({yval/len(df)*100:.1f}%)", 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_title("Credit Risk Target Class Distribution\n(German Credit Dataset: 70% Good vs 30% Bad)", pad=15)
    ax.set_ylabel("Number of Applicants")
    ax.set_ylim(0, 800)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "01_target_distribution.png"))
    plt.close()

    # 2. Numerical Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    num_cols = ['duration', 'credit_amount', 'age', 'installment_commitment']
    titles = [
        'Loan Duration (Months)', 
        'Credit Amount (DM)', 
        'Applicant Age (Years)', 
        'Installment Rate (% of Disposable Income)'
    ]
    for idx, (col, title) in enumerate(zip(num_cols, titles)):
        r, c = idx // 2, idx % 2
        sns.histplot(df[col], kde=True, ax=axes[r, c], color='#1f77b4', edgecolor='black')
        axes[r, c].set_title(title, fontweight='bold')
        axes[r, c].set_xlabel("")
        axes[r, c].set_ylabel("Frequency")
    plt.suptitle("Distributions of Key Numerical Financial Attributes", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "02_numerical_distributions.png"))
    plt.close()

    # 3. Credit Amount and Duration Stratified by Risk
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(x='class', y='credit_amount', data=df, ax=axes[0], palette=['#4575b4', '#d73027'])
    axes[0].set_title("Credit Amount by Risk Class", fontweight='bold')
    axes[0].set_xlabel("Credit Class")
    axes[0].set_ylabel("Credit Amount (DM)")
    
    sns.boxplot(x='class', y='duration', data=df, ax=axes[1], palette=['#4575b4', '#d73027'])
    axes[1].set_title("Loan Duration by Risk Class", fontweight='bold')
    axes[1].set_xlabel("Credit Class")
    axes[1].set_ylabel("Duration (Months)")
    plt.suptitle("Financial Exposure Across Good vs Bad Credit Risk Applicants", y=1.03, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "03_credit_amount_by_risk.png"))
    plt.close()

    # 4. Categorical Default Rates
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cat_focus = ['checking_status', 'credit_history', 'savings_status', 'purpose']
    cat_titles = [
        'Checking Account Status', 
        'Credit History Standing', 
        'Savings Account Balance', 
        'Loan Purpose'
    ]
    for idx, (col, title) in enumerate(zip(cat_focus, cat_titles)):
        r, c = idx // 2, idx % 2
        # Calculate bad credit rate per group
        risk_rate = df.groupby(col)['class'].apply(lambda s: (s == 'bad').mean() * 100).sort_values(ascending=False)
        sns.barplot(x=risk_rate.values, y=risk_rate.index, ax=axes[r, c], palette='Reds_r', edgecolor='black')
        axes[r, c].set_title(f"Default Risk Rate by {title}", fontweight='bold')
        axes[r, c].set_xlabel("High Risk Rate (%)")
        axes[r, c].set_xlim(0, 100)
        for p in axes[r, c].patches:
            width = p.get_width()
            axes[r, c].text(width + 1.5, p.get_y() + p.get_height()/2.0, f"{width:.1f}%", 
                            va='center', fontsize=9, fontweight='bold')
    plt.suptitle("Bad Credit Risk Concentration Across Categorical Predictors", y=1.02, fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "04_categorical_risk_rates.png"))
    plt.close()

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    num_df = df[['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents']].copy()
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Correlation Heatmap of Numerical Financial Attributes", pad=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "05_correlation_heatmap.png"))
    plt.close()

    # 6. Age Group Risk Analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    age_bins = [18, 25, 35, 50, 80]
    age_labels = ['<25 (Young)', '25-35 (Early Career)', '35-50 (Mid Career)', '50+ (Senior)']
    df_temp = df.copy()
    df_temp['age_bracket'] = pd.cut(df_temp['age'], bins=age_bins, labels=age_labels)
    age_risk = df_temp.groupby('age_bracket')['class'].apply(lambda s: (s == 'bad').mean() * 100)
    bars = ax.bar(age_risk.index, age_risk.values, color='#4a7c59', edgecolor='black', width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", 
                ha='center', va='bottom', fontweight='bold')
    ax.set_title("Bad Credit Risk Rate Across Age Brackets", pad=15, fontweight='bold')
    ax.set_ylabel("High Risk Proportion (%)")
    ax.set_ylim(0, 50)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "06_age_group_vs_risk.png"))
    plt.close()

    print(f"All 6 exploratory data visualizations generated in: {graphs_dir}")


def evaluate_models_and_generate_plots(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    graphs_dir: str,
    results_dir: str
) -> pd.DataFrame:
    """
    Evaluates Logistic Regression, Decision Tree, and Random Forest on held-out test data.
    Generates Confusion Matrices, ROC curves, PR curves, and comparison charts.
    """
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    setup_plotting_style()

    metrics_list = []
    roc_data = {}
    pr_data = {}
    confusion_matrices = {}
    predictions_dict = {}

    # Cost-sensitive lending weights:
    # False Negative (FN: lending to a high-risk applicant who defaults) is typically 5x more costly
    # than False Positive (FP: rejecting a good applicant, lost interest margin).
    COST_FN = 5.0
    COST_FP = 1.0

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        # Risk-focused metrics: Positive class = 1 (Bad Credit / High Risk)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Expected credit loss metric (normalized)
        total_credit_loss = (fn * COST_FN) + (fp * COST_FP)

        metrics_list.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'ROC-AUC': round(roc_auc, 4),
            'High-Risk Recall': round(rec, 4),
            'High-Risk Precision': round(prec, 4),
            'High-Risk F1-Score': round(f1, 4),
            'Good-Credit Specificity': round(spec, 4),
            'Average Precision': round(avg_prec, 4),
            'True Positives (TP)': int(tp),
            'False Negatives (FN)': int(fn),
            'True Negatives (TN)': int(tn),
            'False Positives (FP)': int(fp),
            'Normalized Risk Cost': round(total_credit_loss / len(y_test), 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, roc_auc)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_data[name] = (recall_curve, precision_curve, avg_prec)

        confusion_matrices[name] = cm
        predictions_dict[name] = y_pred

    comparison_df = pd.DataFrame(metrics_list)
    comparison_csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel Comparison Table saved to: {comparison_csv_path}")
    print(comparison_df[['Model', 'Accuracy', 'ROC-AUC', 'High-Risk Recall', 'High-Risk Precision', 'High-Risk F1-Score', 'Normalized Risk Cost']].to_string(index=False))

    # Save detailed JSON metrics
    with open(os.path.join(results_dir, "model_metrics.json"), "w") as f:
        json.dump(comparison_df.to_dict(orient='records'), f, indent=4)

    # 7. Confusion Matrices (Side-by-side)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        labels = [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}", f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=['Good (0)', 'Bad (1)'],
                    yticklabels=['Good (0)', 'Bad (1)'])
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontweight='bold')
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("Actual Label")
    plt.suptitle("Test Set Confusion Matrix Comparison (High Risk = Positive Class 1)", y=1.05, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "07_confusion_matrices.png"))
    plt.close()

    # 8. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Logistic Regression': '#1f77b4', 'Decision Tree': '#ff7f0e', 'Random Forest': '#2ca02c'}
    for name, (fpr, tpr, auc_val) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors.get(name, '#333'), lw=2.2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.500)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (High-Risk Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight='bold', pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "08_roc_curves.png"))
    plt.close()

    # 9. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = (y_test == 1).mean()
    for name, (rec_c, prec_c, ap) in pr_data.items():
        ax.plot(rec_c, prec_c, label=f"{name} (Avg Prec = {ap:.3f})", color=colors.get(name, '#333'), lw=2.2)
    ax.axhline(baseline, color='k', linestyle='--', label=f'Baseline Rate ({baseline:.2f})')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (High-Risk Applicants Identified)")
    ax.set_ylabel("Precision (Accuracy of High-Risk Flags)")
    ax.set_title("Precision-Recall Curves for High-Risk Detection", fontweight='bold', pad=12)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "09_precision_recall_curves.png"))
    plt.close()

    # 10. Model Performance Metrics Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    melted = comparison_df.melt(
        id_vars=['Model'],
        value_vars=['Accuracy', 'ROC-AUC', 'High-Risk Recall', 'High-Risk Precision', 'High-Risk F1-Score'],
        var_name='Metric',
        value_name='Score'
    )
    sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='tab10', ax=ax, edgecolor='black')
    ax.set_title("Cross-Model Performance Metric Benchmark on Held-Out Test Set", fontweight='bold', pad=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "10_models_metrics_comparison.png"))
    plt.close()

    # 11. Feature Importance for Random Forest
    rf_pipe = models.get('Random Forest')
    if rf_pipe and hasattr(rf_pipe.named_steps['classifier'], 'feature_importances_'):
        col_transformer = rf_pipe.named_steps['preprocessor'].named_steps['transformer']
        cat_encoder = col_transformer.named_transformers_['cat'].named_steps['encoder']
        cat_features_encoded = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_feature_names = ALL_NUMERICAL_FEATURES + cat_features_encoded

        rf_clf = rf_pipe.named_steps['classifier']
        importances = rf_clf.feature_importances_
        fi_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis', ax=ax, edgecolor='black')
        ax.set_title("Top 15 Most Important Features – Random Forest Classifier", fontweight='bold', pad=12)
        ax.set_xlabel("Gini Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "11_feature_importance_rf.png"))
        plt.close()

    # 12. Decision Tree Structure Visualization
    dt_pipe = models.get('Decision Tree')
    if dt_pipe and hasattr(dt_pipe.named_steps['classifier'], 'tree_'):
        dt_clf = dt_pipe.named_steps['classifier']
        col_transformer = dt_pipe.named_steps['preprocessor'].named_steps['transformer']
        cat_encoder = col_transformer.named_transformers_['cat'].named_steps['encoder']
        cat_features_encoded = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_feature_names = ALL_NUMERICAL_FEATURES + cat_features_encoded

        fig, ax = plt.subplots(figsize=(18, 9))
        plot_tree(
            dt_clf,
            feature_names=all_feature_names,
            class_names=['Good Credit', 'Bad Credit'],
            filled=True,
            rounded=True,
            max_depth=3,  # Pruned for visual interpretability
            fontsize=9,
            ax=ax
        )
        ax.set_title("Decision Tree Decision Logic Structure (Top 3 Levels)", fontweight='bold', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "12_decision_tree_structure.png"))
        plt.close()

    print(f"All 6 model evaluation visualizations generated in: {graphs_dir}")
    return comparison_df


def select_best_model(
    models: Dict[str, Any],
    comparison_df: pd.DataFrame,
    models_dir: str = os.path.join(PROJECT_ROOT, 'models')
) -> Tuple[str, Any]:
    """
    Selects the optimal model balancing ROC-AUC, High-Risk Recall, and Credit Loss Cost.
    Saves the winner as best_credit_model.joblib.
    """
    # Criteria: Models in banking credit risk must catch high-risk defaults without tanking precision
    # Composite score: 0.5 * ROC-AUC + 0.3 * High-Risk Recall + 0.2 * High-Risk F1
    df = comparison_df.copy()
    df['Composite_Score'] = (
        0.50 * df['ROC-AUC'] +
        0.30 * df['High-Risk Recall'] +
        0.20 * df['High-Risk F1-Score']
    )
    best_row = df.sort_values(by='Composite_Score', ascending=False).iloc[0]
    best_name = best_row['Model']
    best_model = models[best_name]

    dest_path = os.path.join(models_dir, 'best_credit_model.joblib')
    joblib.dump(best_model, dest_path)

    print(f"\n==========================================")
    print(f"WINNER: {best_name}")
    print(f"Test ROC-AUC:           {best_row['ROC-AUC']}")
    print(f"High-Risk Recall:       {best_row['High-Risk Recall']}")
    print(f"High-Risk Precision:    {best_row['High-Risk Precision']}")
    print(f"High-Risk F1-Score:     {best_row['High-Risk F1-Score']}")
    print(f"Normalized Risk Cost:   {best_row['Normalized Risk Cost']}")
    print(f"Composite Score:        {best_row['Composite_Score']:.4f}")
    print(f"Saved best model to:    {dest_path}")
    print(f"==========================================\n")

    return best_name, best_model


def run_full_evaluation():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'credit_dataset.csv')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    graphs_dir = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
    results_dir = os.path.join(PROJECT_ROOT, 'outputs', 'evaluation_results')

    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    print("Generating Exploratory Data Analysis (EDA) visualizations...")
    generate_eda_visualizations(df, graphs_dir)

    print("\nLoading trained model pipelines...")
    lr_pipe = joblib.load(os.path.join(models_dir, 'logistic_regression_model.joblib'))
    dt_pipe = joblib.load(os.path.join(models_dir, 'decision_tree_model.joblib'))
    rf_pipe = joblib.load(os.path.join(models_dir, 'random_forest_model.joblib'))

    models = {
        'Logistic Regression': lr_pipe,
        'Decision Tree': dt_pipe,
        'Random Forest': rf_pipe
    }

    print("\nEvaluating all 3 models on held-out test data (200 records)...")
    comparison_df = evaluate_models_and_generate_plots(models, X_test, y_test, graphs_dir, results_dir)

    best_name, best_model = select_best_model(models, comparison_df, models_dir)
    return comparison_df


if __name__ == '__main__':
    run_full_evaluation()
