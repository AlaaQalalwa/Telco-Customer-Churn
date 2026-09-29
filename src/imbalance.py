"""
Class imbalance handling module for Task 8.
Compares:
  1. Baseline (no imbalance intervention)
  2. Cost-sensitive learning / Class weighting
  3. Oversampling via SMOTE strictly inside training folds using imblearn.pipeline.Pipeline
Analyzes the precision vs. recall trade-off across all models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import RANDOM_STATE, TABLES_DIR, FIGURES_DIR, COST_FN, COST_FP
from src.preprocessing import build_model_pipeline
from src.models import evaluate_pipeline_cv

def get_imbalance_configurations():
    """
    Returns models instantiated under 3 conditions:
      1. None (Default)
      2. Class Weighted / Scale Pos Weight
      3. SMOTE
    """
    # Negative to positive ratio in training data is approximately 5174/1869 ~ 2.77
    scale_weight = 5174.0 / 1869.0

    models = {
        "Logistic Regression": {
            "none": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "weighted": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            "smote": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        },
        "Random Forest": {
            "none": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "weighted": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
            "smote": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        },
        "XGBoost": {
            "none": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
            "weighted": XGBClassifier(scale_pos_weight=scale_weight, eval_metric="logloss", random_state=RANDOM_STATE),
            "smote": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
        },
        "LightGBM": {
            "none": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            "weighted": LGBMClassifier(scale_pos_weight=scale_weight, random_state=RANDOM_STATE, verbose=-1),
            "smote": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        }
    }
    return models

def run_imbalance_experiments(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Runs cross-validation experiments comparing No Handling vs. Class Weighting vs. SMOTE.
    """
    configs = get_imbalance_configurations()
    results = []

    for model_name, strategies in configs.items():
        for strategy, estimator in strategies.items():
            use_smote = (strategy == "smote")
            pipeline = build_model_pipeline(estimator, use_smote=use_smote)
            
            print(f"Evaluating {model_name} with Imbalance Strategy: {strategy.upper()} (SMOTE={use_smote})...")
            metrics = evaluate_pipeline_cv(pipeline, X_train, y_train, return_confusion_counts=True)
            
            total_fn = metrics["Total_FN"]
            total_fp = metrics["Total_FP"]
            expected_cost = float((COST_FN * total_fn) + (COST_FP * total_fp))

            results.append({
                "Model": model_name,
                "Imbalance_Strategy": "No Handling" if strategy == "none" else strategy.capitalize(),
                "Accuracy": metrics["Accuracy_Mean"],
                "Precision": metrics["Precision_Mean"],
                "Recall": metrics["Recall_Mean"],
                "F1_Score": metrics["F1_Mean"],
                "ROC_AUC": metrics["ROC_AUC_Mean"],
                "Total_FN": int(total_fn),
                "Total_FP": int(total_fp),
                "Expected_Cost": float(expected_cost)
            })

    df_imbalance = pd.DataFrame(results)
    df_imbalance.to_csv(TABLES_DIR / "task8_imbalance_handling_results.csv", index=False)

    # Visualization: Grouped bar chart of Recall, Precision, and F1 across strategies
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = [("Recall", "Recall (Minority Sensitivity)"),
                       ("Precision", "Precision (Positive Predictive Value)"),
                       ("F1_Score", "F1-Score (Harmonic Balance)")]

    models_list = list(configs.keys())
    x = np.arange(len(models_list))
    width = 0.25

    for idx, (m_col, m_title) in enumerate(metrics_to_plot):
        none_vals = df_imbalance[df_imbalance["Imbalance_Strategy"] == "No Handling"][m_col].values
        weighted_vals = df_imbalance[df_imbalance["Imbalance_Strategy"] == "Weighted"][m_col].values
        smote_vals = df_imbalance[df_imbalance["Imbalance_Strategy"] == "Smote"][m_col].values

        axes[idx].bar(x - width, none_vals, width, label="No Handling", color="#2b5c8f", edgecolor="black")
        axes[idx].bar(x, weighted_vals, width, label="Class Weighted", color="#d95f02", edgecolor="black")
        axes[idx].bar(x + width, smote_vals, width, label="SMOTE (Fold-safe)", color="#7570b3", edgecolor="black")

        axes[idx].set_title(m_title)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(models_list, rotation=15)
        axes[idx].set_ylim(0, 1.0)
        axes[idx].legend(loc="lower right" if m_col != "Precision" else "upper right")
        axes[idx].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task8_imbalance_comparison.png")
    plt.close(fig)

    print("Task 8 Imbalance Handling experiments completed.")
    return df_imbalance
