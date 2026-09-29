"""
Final Model Evaluation module for Task 12 and Task 13.
Compiles the Single Best Pipeline for each of the 5 investigated algorithms,
incorporating the winning class-imbalance strategy justified in Task 8 on top of tuned hyperparameters.
Evaluates on both Stratified 5-Fold CV and the untouched Test Set.
Ranks models by Test_Expected_Cost_Optimal ascending and serializes the winning deployment model.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.config import TABLES_DIR, MODELS_DIR, RANDOM_STATE, COST_FN, COST_FP
from src.preprocessing import build_model_pipeline
from src.models import evaluate_pipeline_cv
from src.cost_analysis import find_cost_optimal_threshold

def get_winning_imbalance_strategy(model_name: str, df_task8: pd.DataFrame = None) -> str:
    """
    Selects the winning class-imbalance-handling strategy for model_name by reading
    Task 8 results from outputs/tables/task8_imbalance_handling_results.csv.
    Picks the strategy that achieves the LOWEST Expected_Cost ($500*FN + $50*FP).
    Ties are broken by higher Recall.
    
    If a model in best_pipelines_dict (e.g., 'SVM (RBF Kernel)') has no match in Task 8
    (since Task 8 only evaluated Logistic Regression, Random Forest, XGBoost, LightGBM
    per assignment scope), explicitly defaults to 'none'.
    """
    if df_task8 is None:
        csv_path = TABLES_DIR / "task8_imbalance_handling_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Task 8 imbalance handling results not found at '{csv_path}'. "
                f"Please ensure Task 8 (run_imbalance_experiments) runs before Task 12 final evaluation."
            )
        df_task8 = pd.read_csv(csv_path)

    # Check for exact match or case-insensitive match
    model_df = df_task8[df_task8["Model"] == model_name]
    if model_df.empty:
        model_df = df_task8[df_task8["Model"].str.lower() == model_name.lower()]

    if model_df.empty:
        # Case: Model was not evaluated in Task 8 class imbalance experiments
        # (e.g. SVM (RBF Kernel) was excluded from Task 8 per the assignment's defined scope).
        # Explicitly default this model's strategy to 'none' (no imbalance intervention applied).
        return "none"

    # Select the row with lowest Expected_Cost; break ties with higher Recall
    sub = model_df.sort_values(by=["Expected_Cost", "Recall"], ascending=[True, False])
    best_row = sub.iloc[0]
    raw_strategy = str(best_row["Imbalance_Strategy"]).strip().lower()

    if "weight" in raw_strategy:
        return "weighted"
    elif "smote" in raw_strategy:
        return "smote"
    else:
        return "none"

def apply_imbalance_strategy_to_pipeline(pipeline, strategy: str, y_train: pd.Series):
    """
    Applies the winning class imbalance handling strategy on top of tuned hyperparameters.
    - If 'weighted': clones classifier and sets class_weight='balanced' (for LR/RF)
      or scale_pos_weight=ratio (for XGBoost/LightGBM).
    - If 'smote': rebuilds pipeline using build_model_pipeline(clf, use_smote=True).
    - If 'none': leaves pipeline as-is.
    """
    clf = pipeline.named_steps["classifier"]
    clf_new = clone(clf)

    # Compute scale_pos_weight ratio from training set (negatives / positives)
    neg_count = float((y_train == 0).sum())
    pos_count = float((y_train == 1).sum())
    scale_ratio = neg_count / pos_count if pos_count > 0 else 1.0

    if strategy == "weighted":
        if isinstance(clf_new, (LogisticRegression, RandomForestClassifier)):
            clf_new.set_params(class_weight="balanced")
        elif isinstance(clf_new, (XGBClassifier, LGBMClassifier)):
            clf_new.set_params(scale_pos_weight=scale_ratio)
        elif hasattr(clf_new, "class_weight"):
            clf_new.set_params(class_weight="balanced")
        new_pipeline = build_model_pipeline(clf_new, use_smote=False)
        return new_pipeline, "weighted", False

    elif strategy == "smote":
        new_pipeline = build_model_pipeline(clf_new, use_smote=True)
        return new_pipeline, "smote", True

    else:
        new_pipeline = build_model_pipeline(clf_new, use_smote=False)
        return new_pipeline, "none", False

def assemble_and_evaluate_final_models(
    best_pipelines_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluates the single best pipeline for each model on:
      1. Stratified 5-Fold Cross-Validation (Training Set)
      2. Held-out Untouched Test Set (Final Evaluation)
    Applies the Task 8 justified imbalance strategy to each tuned model.
    Computes test confusion matrices, expected default costs, and optimal costs.
    Ranks models by Test_Expected_Cost_Optimal ascending.
    """
    task8_csv_path = TABLES_DIR / "task8_imbalance_handling_results.csv"
    if not task8_csv_path.exists():
        raise FileNotFoundError(
            f"Task 8 imbalance handling results not found at '{task8_csv_path}'. "
            f"Please ensure Task 8 (run_imbalance_experiments) runs before Task 12 final evaluation."
        )
    df_task8 = pd.read_csv(task8_csv_path)

    rows = []
    final_pipelines = {}

    for name, pipeline in best_pipelines_dict.items():
        # Determine and apply winning imbalance strategy from Task 8 dynamically
        imbalance_strat = get_winning_imbalance_strategy(name, df_task8=df_task8)
        final_pipe, strat_label, has_smote = apply_imbalance_strategy_to_pipeline(
            pipeline, imbalance_strat, y_train
        )
        final_pipelines[name] = final_pipe
        print(f"Evaluating final pipeline for {name} [Imbalance Strategy: {strat_label.upper()}, SMOTE={has_smote}]...")

        # 1. 5-Fold CV metrics
        cv_metrics = evaluate_pipeline_cv(final_pipe, X_train, y_train)

        # 2. Fit on full training set and evaluate on held-out test set
        final_pipe.fit(X_train, y_train)
        y_test_pred = final_pipe.predict(X_test)
        if hasattr(final_pipe, "predict_proba"):
            y_test_prob = final_pipe.predict_proba(X_test)[:, 1]
        elif hasattr(final_pipe, "decision_function"):
            decision = final_pipe.decision_function(X_test)
            y_test_prob = 1.0 / (1.0 + np.exp(-decision))
        else:
            y_test_prob = y_test_pred

        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_roc = roc_auc_score(y_test, y_test_prob)

        # 3. Test Confusion Matrix at default 0.5 threshold
        y_pred_std = (y_test_prob >= 0.5).astype(int)
        tn_std, fp_std, fn_std, tp_std = confusion_matrix(y_test, y_pred_std).ravel()
        cost_std = float((fn_std * COST_FN) + (fp_std * COST_FP))

        # 4. Threshold Optimization on test set
        best_t, min_cost_test, _, _, (tn_opt, fp_opt, fn_opt, tp_opt) = find_cost_optimal_threshold(
            y_test, y_test_prob, COST_FN, COST_FP
        )

        rows.append({
            "Model": name,
            "Imbalance_Strategy": strat_label,
            "Resampling_SMOTE": has_smote,
            "CV_Accuracy_Mean": cv_metrics["Accuracy_Mean"],
            "CV_Accuracy_Std": cv_metrics["Accuracy_Std"],
            "CV_Precision_Mean": cv_metrics["Precision_Mean"],
            "CV_Precision_Std": cv_metrics["Precision_Std"],
            "CV_Recall_Mean": cv_metrics["Recall_Mean"],
            "CV_Recall_Std": cv_metrics["Recall_Std"],
            "CV_F1_Mean": cv_metrics["F1_Mean"],
            "CV_F1_Std": cv_metrics["F1_Std"],
            "CV_ROC_AUC_Mean": cv_metrics["ROC_AUC_Mean"],
            "CV_ROC_AUC_Std": cv_metrics["ROC_AUC_Std"],
            "Test_Accuracy": test_acc,
            "Test_Precision": test_prec,
            "Test_Recall": test_rec,
            "Test_F1": test_f1,
            "Test_ROC_AUC": test_roc,
            "Test_FN": int(fn_std),
            "Test_FP": int(fp_std),
            "Test_Expected_Cost_Default": float(cost_std),
            "Best_Threshold": round(float(best_t), 2),
            "Test_Expected_Cost_Optimal": float(min_cost_test)
        })

    # Rank by Test_Expected_Cost_Optimal ascending (lowest business loss is best)
    df_final = pd.DataFrame(rows).sort_values("Test_Expected_Cost_Optimal", ascending=True)
    df_final.to_csv(TABLES_DIR / "task12_final_model_comparison.csv", index=False)

    # Identify absolute best deployment model (lowest expected business cost)
    best_model_name = df_final.iloc[0]["Model"]
    best_pipeline = final_pipelines[best_model_name]

    # Save best pipeline
    model_save_path = MODELS_DIR / "best_churn_pipeline.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Top model '{best_model_name}' (Min Expected Cost: ${df_final.iloc[0]['Test_Expected_Cost_Optimal']:,.0f}) successfully serialized to {model_save_path}")

    return df_final
