"""
Cost Analysis and ROC/Confusion Matrix module for Task 10.
Analyzes top-3 models on out-of-fold cross-validation or test data,
generates combined ROC curves, confusion matrices, and performs financial cost threshold optimization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from src.config import FIGURES_DIR, TABLES_DIR, COST_FN, COST_FP, RANDOM_STATE, CV_SPLITS

def get_out_of_fold_predictions(pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Computes unbiased out-of-fold predicted probabilities using Stratified 5-Fold CV.
    """
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]

        pipeline.fit(X_fold_train, y_fold_train)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_fold_val)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function(X_fold_val)
            probs = 1.0 / (1.0 + np.exp(-decision))
        else:
            probs = pipeline.predict(X_fold_val)
            
        oof_probs[val_idx] = probs

    return oof_probs

def find_cost_optimal_threshold(y_true, y_probs, cost_fn: float = COST_FN, cost_fp: float = COST_FP):
    """
    Sweeps decision thresholds in [0.01, 0.99] to find the cost-optimal decision threshold,
    minimizing: Total Cost = (FN * cost_fn) + (FP * cost_fp).
    Returns:
      best_threshold (float), min_cost (float), thresholds (np.ndarray), costs (list),
      (tn_opt, fp_opt, fn_opt, tp_opt)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_t).ravel()
        total_cost_t = (fn_t * cost_fn) + (fp_t * cost_fp)
        costs.append(total_cost_t)

    best_idx = np.argmin(costs)
    best_threshold = float(thresholds[best_idx])
    min_cost = float(costs[best_idx])

    y_pred_opt = (y_probs >= best_threshold).astype(int)
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, y_pred_opt).ravel()

    return best_threshold, min_cost, thresholds, costs, (tn_opt, fp_opt, fn_opt, tp_opt)

def run_cost_and_roc_analysis(top_3_pipelines: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Executes Task 10 requirements:
      - ROC Curves plotted on ONE figure
      - Confusion matrices for top-3 models
      - Business cost threshold optimization ($500 FN, $50 FP)
    """
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    colors = ["#2b5c8f", "#d95f02", "#2ca02c"]

    oof_predictions = {}
    threshold_cost_results = []
    matrix_summaries = []

    for idx, (model_name, pipe) in enumerate(top_3_pipelines.items()):
        print(f"Computing out-of-fold probabilities for Top-3 Model: {model_name}...")
        y_probs = get_out_of_fold_predictions(pipe, X_train, y_train)
        oof_predictions[model_name] = y_probs

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_train, y_probs)
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(
            fpr, tpr, color=colors[idx], lw=2.2,
            label=f"{model_name} (AUC = {roc_auc_val:.4f})"
        )

        # 2. Confusion Matrix at Standard Threshold (0.5)
        y_pred_std = (y_probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred_std).ravel()
        cost_std = (fn * COST_FN) + (fp * COST_FP)

        # 3. Threshold Optimization ($500 FN vs $50 FP)
        best_threshold, min_cost, thresholds, costs, (tn_opt, fp_opt, fn_opt, tp_opt) = find_cost_optimal_threshold(
            y_train, y_probs, COST_FN, COST_FP
        )

        matrix_summaries.append({
            "Model": model_name,
            "Threshold_Type": "Default (0.50)",
            "Threshold": 0.50,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Recall": tp / (tp + fn),
            "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "Total_Cost_$": cost_std
        })
        matrix_summaries.append({
            "Model": model_name,
            "Threshold_Type": "Cost-Optimal",
            "Threshold": round(best_threshold, 2),
            "TN": tn_opt,
            "FP": fp_opt,
            "FN": fn_opt,
            "TP": tp_opt,
            "Recall": tp_opt / (tp_opt + fn_opt),
            "Precision": tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0.0,
            "Total_Cost_$": min_cost
        })

        threshold_cost_results.append({
            "model": model_name,
            "thresholds": thresholds,
            "costs": costs,
            "best_t": best_threshold,
            "min_cost": min_cost,
            "default_cost": cost_std
        })

    # Finalize ROC plot
    ax_roc.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax_roc.set_title("Cross-Validated ROC Curves Comparison for Top 3 Models")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(FIGURES_DIR / "task10_roc_curves_top3.png")
    plt.close(fig_roc)

    # 4. Confusion Matrices Subplot (Default 0.5 vs. Optimal Threshold)
    fig_cm, axes_cm = plt.subplots(len(top_3_pipelines), 2, figsize=(10, 4 * len(top_3_pipelines)))
    if len(top_3_pipelines) == 1:
        axes_cm = np.expand_dims(axes_cm, axis=0)

    for i, (model_name, y_probs) in enumerate(oof_predictions.items()):
        # Standard 0.5
        cm_std = confusion_matrix(y_train, (y_probs >= 0.5).astype(int))
        sns.heatmap(cm_std, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes_cm[i, 0],
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        axes_cm[i, 0].set_title(f"{model_name}\nDefault Threshold (0.50)")
        axes_cm[i, 0].set_ylabel("Actual Label")
        axes_cm[i, 0].set_xlabel("Predicted Label")

        # Optimal
        best_t = threshold_cost_results[i]["best_t"]
        cm_opt = confusion_matrix(y_train, (y_probs >= best_t).astype(int))
        sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes_cm[i, 1],
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        axes_cm[i, 1].set_title(f"{model_name}\nCost-Optimal Threshold ({best_t:.2f})")
        axes_cm[i, 1].set_ylabel("Actual Label")
        axes_cm[i, 1].set_xlabel("Predicted Label")

    fig_cm.tight_layout()
    fig_cm.savefig(FIGURES_DIR / "task10_confusion_matrices_top3.png")
    plt.close(fig_cm)

    # 5. Business Cost Curve vs Threshold Plot
    fig_cost, ax_cost = plt.subplots(figsize=(9, 5.5))
    for i, res in enumerate(threshold_cost_results):
        ax_cost.plot(res["thresholds"], res["costs"], color=colors[i], lw=2.2, label=f"{res['model']} Cost Curve")
        ax_cost.scatter(res["best_t"], res["min_cost"], color=colors[i], s=80, zorder=5)
        ax_cost.axvline(res["best_t"], color=colors[i], linestyle=":", alpha=0.6)
        savings = res["default_cost"] - res["min_cost"]
        pct_savings = (savings / res["default_cost"]) * 100
        print(f"{res['model']}: Default Cost=${res['default_cost']:,.0f} -> Min Cost=${res['min_cost']:,.0f} at t={res['best_t']:.2f} (Savings: ${savings:,.0f} or {pct_savings:.1f}%)")

    ax_cost.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="Default Threshold (0.50)")
    ax_cost.set_xlabel("Decision Threshold (Classification Probability)")
    ax_cost.set_ylabel("Total Business Cost ($)")
    ax_cost.set_title(f"Total Business Cost vs Classification Threshold\n(Cost: False Negative = ${COST_FN:,.0f}, False Positive = ${COST_FP:,.0f})")
    ax_cost.legend(loc="upper right")
    fig_cost.tight_layout()
    fig_cost.savefig(FIGURES_DIR / "task10_cost_threshold_analysis.png")
    plt.close(fig_cost)

    df_matrices = pd.DataFrame(matrix_summaries)
    df_matrices.to_csv(TABLES_DIR / "task10_confusion_matrices_and_costs.csv", index=False)

    print("Task 10 ROC, Confusion Matrices, and Cost Analysis complete.")
    return df_matrices, threshold_cost_results
