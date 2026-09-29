"""
Hyperparameter tuning module for Task 7.
Applies RandomizedSearchCV using Stratified 5-Fold CV on training data only.
Focuses on 2-3 essential hyperparameters per model to maintain computational tractability.
"""
import pandas as pd
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import RANDOM_STATE, CV_SPLITS, TABLES_DIR
from src.preprocessing import build_model_pipeline
from src.models import evaluate_pipeline_cv

def get_tuning_distributions():
    """
    Returns parameter distributions restricted to 2-3 key hyperparameters per model.
    """
    param_grids = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "params": {
                "classifier__C": loguniform(0.01, 50.0),
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs", "saga"]
            },
            "n_iter": 15
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=RANDOM_STATE),
            "params": {
                "classifier__C": [0.1, 1.0, 5.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto", 0.01, 0.1]
            },
            "n_iter": 12  # SVM with probability=True is computationally heavier
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__max_depth": [5, 8, 12, None],
                "classifier__min_samples_split": [2, 5, 10]
            },
            "n_iter": 15
        },
        "XGBoost": {
            "estimator": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
            "params": {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.15],
                "classifier__max_depth": [3, 4, 6]
            },
            "n_iter": 15
        },
        "LightGBM": {
            "estimator": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            "params": {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__num_leaves": [15, 25, 31, 50]
            },
            "n_iter": 15
        }
    }
    return param_grids

def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    default_results_df: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """
    Executes RandomizedSearchCV for all 5 models, logs the best parameters,
    and constructs a side-by-side comparison table of default vs. tuned performance.
    """
    tuning_configs = get_tuning_distributions()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    best_pipelines = {}
    comparison_rows = []

    for name, config in tuning_configs.items():
        print(f"--- Running RandomizedSearchCV for {name} ({config['n_iter']} iterations) ---")
        base_pipe = build_model_pipeline(config["estimator"])

        search = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        best_pipelines[name] = best_pipe
        best_params_clean = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        print(f"  Best ROC-AUC: {search.best_score_:.4f} with params: {best_params_clean}")

        # Compute full 5 metrics on tuned pipeline
        tuned_metrics = evaluate_pipeline_cv(best_pipe, X_train, y_train)

        # Retrieve default metrics
        default_row = default_results_df[default_results_df["Model"].str.contains(name.split()[0])].iloc[0]

        comparison_rows.append({
            "Model": name,
            "Best_Parameters": str(best_params_clean),
            "Default_Accuracy": default_row["Accuracy_Mean"],
            "Tuned_Accuracy": tuned_metrics["Accuracy_Mean"],
            "Default_Recall": default_row["Recall_Mean"],
            "Tuned_Recall": tuned_metrics["Recall_Mean"],
            "Default_Precision": default_row["Precision_Mean"],
            "Tuned_Precision": tuned_metrics["Precision_Mean"],
            "Default_F1": default_row["F1_Mean"],
            "Tuned_F1": tuned_metrics["F1_Mean"],
            "Default_ROC_AUC": default_row["ROC_AUC_Mean"],
            "Tuned_ROC_AUC": tuned_metrics["ROC_AUC_Mean"],
            "ROC_AUC_Gain": tuned_metrics["ROC_AUC_Mean"] - default_row["ROC_AUC_Mean"]
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(TABLES_DIR / "task7_tuning_comparison.csv", index=False)
    return best_pipelines, comparison_df
