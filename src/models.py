"""
Classification models module for Tasks 5 and 6.
Defines candidate algorithms and provides a comprehensive cross-validation evaluation suite.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.config import RANDOM_STATE, CV_SPLITS, TABLES_DIR
from src.preprocessing import build_model_pipeline

def get_baseline_models():
    """Returns baseline models (Task 5)."""
    return {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    }

def get_default_models():
    """Returns the 5 primary classifiers with default hyperparameters (Task 6)."""
    return {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "SVM (RBF Kernel)": SVC(
            probability=True,
            kernel="rbf",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1
        )
    }

def evaluate_pipeline_cv(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
    return_confusion_counts: bool = False
) -> dict:
    """
    Evaluates an arbitrary (imblearn or sklearn) pipeline using Stratified K-Fold CV.
    Computes Accuracy, Precision, Recall, F1, and ROC-AUC on the out-of-fold validation sets.
    Optionally returns summed out-of-fold confusion matrix counts (Total_TN, Total_FP, Total_FN, Total_TP).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_rocs = []

    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_fold_train, y_fold_train)
        y_val_pred = pipeline.predict(X_fold_val)

        # Get probabilities for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            y_val_prob = pipeline.predict_proba(X_fold_val)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_val_prob = pipeline.decision_function(X_fold_val)
        else:
            y_val_prob = y_val_pred

        fold_accuracies.append(accuracy_score(y_fold_val, y_val_pred))
        fold_precisions.append(precision_score(y_fold_val, y_val_pred, zero_division=0))
        fold_recalls.append(recall_score(y_fold_val, y_val_pred, zero_division=0))
        fold_f1s.append(f1_score(y_fold_val, y_val_pred, zero_division=0))
        try:
            fold_rocs.append(roc_auc_score(y_fold_val, y_val_prob))
        except Exception:
            fold_rocs.append(0.5)

        if return_confusion_counts:
            tn, fp, fn, tp = confusion_matrix(y_fold_val, y_val_pred, labels=[0, 1]).ravel()
            total_tn += int(tn)
            total_fp += int(fp)
            total_fn += int(fn)
            total_tp += int(tp)

    res = {
        "Accuracy_Mean": float(np.mean(fold_accuracies)),
        "Accuracy_Std": float(np.std(fold_accuracies)),
        "Precision_Mean": float(np.mean(fold_precisions)),
        "Precision_Std": float(np.std(fold_precisions)),
        "Recall_Mean": float(np.mean(fold_recalls)),
        "Recall_Std": float(np.std(fold_recalls)),
        "F1_Mean": float(np.mean(fold_f1s)),
        "F1_Std": float(np.std(fold_f1s)),
        "ROC_AUC_Mean": float(np.mean(fold_rocs)),
        "ROC_AUC_Std": float(np.std(fold_rocs))
    }
    if return_confusion_counts:
        res["Total_TN"] = total_tn
        res["Total_FP"] = total_fp
        res["Total_FN"] = total_fn
        res["Total_TP"] = total_tp

    return res

def run_models_benchmark(models_dict: dict, X_train: pd.DataFrame, y_train: pd.Series, output_filename: str) -> pd.DataFrame:
    """
    Benchmarks multiple models across Stratified 5-Fold CV and writes results to CSV.
    """
    rows = []
    for model_name, model_inst in models_dict.items():
        pipe = build_model_pipeline(model_inst)
        metrics = evaluate_pipeline_cv(pipe, X_train, y_train)
        metrics["Model"] = model_name
        rows.append(metrics)

    df_results = pd.DataFrame(rows)
    cols_order = [
        "Model",
        "Accuracy_Mean", "Accuracy_Std",
        "Precision_Mean", "Precision_Std",
        "Recall_Mean", "Recall_Std",
        "F1_Mean", "F1_Std",
        "ROC_AUC_Mean", "ROC_AUC_Std"
    ]
    df_results = df_results[cols_order].sort_values("ROC_AUC_Mean", ascending=False)
    df_results.to_csv(TABLES_DIR / output_filename, index=False)
    return df_results
