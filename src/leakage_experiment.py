"""
Data Leakage Challenge module for Task 11.
Empirically demonstrates the critical danger of data leakage by comparing:
  Experiment A: Correct, leakage-free pipeline (SMOTE inside fold via imblearn Pipeline)
  Experiment B: Intentionally leaked pipeline (SMOTE applied globally before CV)
Quantifies performance inflation on CV and verifies against true held-out test data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE, CV_SPLITS, TABLES_DIR, FIGURES_DIR
from src.preprocessing import build_column_transformer, build_model_pipeline

def run_leakage_challenge_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Executes the controlled leakage experiment comparing fold-safe resampling
    against pre-CV global resampling on Random Forest.
    """
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    # -------------------------------------------------------------
    # Experiment A: Correct Leakage-Free Pipeline
    # Preprocessor and SMOTE executed strictly within each CV fold
    # -------------------------------------------------------------
    correct_pipe = build_model_pipeline(clf, use_smote=True)
    correct_cv_acc, correct_cv_prec, correct_cv_rec, correct_cv_f1, correct_cv_roc = [], [], [], [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        correct_pipe.fit(X_tr, y_tr)
        y_pred = correct_pipe.predict(X_val)
        y_prob = correct_pipe.predict_proba(X_val)[:, 1]

        correct_cv_acc.append(accuracy_score(y_val, y_pred))
        correct_cv_prec.append(precision_score(y_val, y_pred, zero_division=0))
        correct_cv_rec.append(recall_score(y_val, y_pred, zero_division=0))
        correct_cv_f1.append(f1_score(y_val, y_pred, zero_division=0))
        correct_cv_roc.append(roc_auc_score(y_val, y_prob))

    # Evaluate Correct Pipeline on Test Set
    correct_pipe.fit(X_train, y_train)
    test_pred_correct = correct_pipe.predict(X_test)
    test_prob_correct = correct_pipe.predict_proba(X_test)[:, 1]

    correct_test_acc = accuracy_score(y_test, test_pred_correct)
    correct_test_prec = precision_score(y_test, test_pred_correct, zero_division=0)
    correct_test_rec = recall_score(y_test, test_pred_correct, zero_division=0)
    correct_test_f1 = f1_score(y_test, test_pred_correct, zero_division=0)
    correct_test_roc = roc_auc_score(y_test, test_prob_correct)

    # -------------------------------------------------------------
    # Experiment B: Intentionally Leaked Pipeline
    # Preprocessing and SMOTE applied to ENTIRE training set BEFORE CV
    # -------------------------------------------------------------
    preprocessor_leaked = build_column_transformer()
    X_train_trans = preprocessor_leaked.fit_transform(X_train)

    # LEAKAGE HAPPENS HERE: SMOTE generates synthetic neighbors across fold boundaries
    smote_leaked = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote_leaked.fit_resample(X_train_trans, y_train)

    # Cross-validation on pre-resampled dataset (leaked evaluation)
    skf_leaked = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    leaked_cv_acc, leaked_cv_prec, leaked_cv_rec, leaked_cv_f1, leaked_cv_roc = [], [], [], [], []

    for tr_idx, v_idx in skf_leaked.split(X_train_resampled, y_train_resampled):
        X_tr_l, X_val_l = X_train_resampled[tr_idx], X_train_resampled[v_idx]
        y_tr_l, y_val_l = y_train_resampled.iloc[tr_idx], y_train_resampled.iloc[v_idx]

        clf_fold = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        clf_fold.fit(X_tr_l, y_tr_l)
        y_pred_l = clf_fold.predict(X_val_l)
        y_prob_l = clf_fold.predict_proba(X_val_l)[:, 1]

        leaked_cv_acc.append(accuracy_score(y_val_l, y_pred_l))
        leaked_cv_prec.append(precision_score(y_val_l, y_pred_l, zero_division=0))
        leaked_cv_rec.append(recall_score(y_val_l, y_pred_l, zero_division=0))
        leaked_cv_f1.append(f1_score(y_val_l, y_pred_l, zero_division=0))
        leaked_cv_roc.append(roc_auc_score(y_val_l, y_prob_l))

    # Evaluate Leaked Model on Test Set
    clf_leaked_full = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf_leaked_full.fit(X_train_resampled, y_train_resampled)
    X_test_trans = preprocessor_leaked.transform(X_test)
    test_pred_leaked = clf_leaked_full.predict(X_test_trans)
    test_prob_leaked = clf_leaked_full.predict_proba(X_test_trans)[:, 1]

    leaked_test_acc = accuracy_score(y_test, test_pred_leaked)
    leaked_test_prec = precision_score(y_test, test_pred_leaked, zero_division=0)
    leaked_test_rec = recall_score(y_test, test_pred_leaked, zero_division=0)
    leaked_test_f1 = f1_score(y_test, test_pred_leaked, zero_division=0)
    leaked_test_roc = roc_auc_score(y_test, test_prob_leaked)

    # Tabulate Results
    comparison_data = [
        {
            "Pipeline_Condition": "Correct (Leakage-Free Fold-safe)",
            "CV_Accuracy": np.mean(correct_cv_acc),
            "CV_Precision": np.mean(correct_cv_prec),
            "CV_Recall": np.mean(correct_cv_rec),
            "CV_F1": np.mean(correct_cv_f1),
            "CV_ROC_AUC": np.mean(correct_cv_roc),
            "Test_Accuracy": correct_test_acc,
            "Test_Precision": correct_test_prec,
            "Test_Recall": correct_test_rec,
            "Test_F1": correct_test_f1,
            "Test_ROC_AUC": correct_test_roc
        },
        {
            "Pipeline_Condition": "Intentionally Leaked (Pre-CV SMOTE)",
            "CV_Accuracy": np.mean(leaked_cv_acc),
            "CV_Precision": np.mean(leaked_cv_prec),
            "CV_Recall": np.mean(leaked_cv_rec),
            "CV_F1": np.mean(leaked_cv_f1),
            "CV_ROC_AUC": np.mean(leaked_cv_roc),
            "Test_Accuracy": leaked_test_acc,
            "Test_Precision": leaked_test_prec,
            "Test_Recall": leaked_test_rec,
            "Test_F1": leaked_test_f1,
            "Test_ROC_AUC": leaked_test_roc
        }
    ]

    df_leakage = pd.DataFrame(comparison_data)
    df_leakage.to_csv(TABLES_DIR / "task11_leakage_experiment_results.csv", index=False)

    # Visualization: Contrast CV vs Test Performance under Correct vs Leaked regimes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

    correct_cv_vals = [np.mean(correct_cv_acc), np.mean(correct_cv_prec), np.mean(correct_cv_rec), np.mean(correct_cv_f1), np.mean(correct_cv_roc)]
    leaked_cv_vals = [np.mean(leaked_cv_acc), np.mean(leaked_cv_prec), np.mean(leaked_cv_rec), np.mean(leaked_cv_f1), np.mean(leaked_cv_roc)]
    correct_test_vals = [correct_test_acc, correct_test_prec, correct_test_rec, correct_test_f1, correct_test_roc]
    leaked_test_vals = [leaked_test_acc, leaked_test_prec, leaked_test_rec, leaked_test_f1, leaked_test_roc]

    x = np.arange(len(metrics_names))
    width = 0.35

    # Subplot 1: Cross-Validation Comparison (The Illusion)
    axes[0].bar(x - width/2, correct_cv_vals, width, label="Correct CV", color="#2b5c8f", edgecolor="black")
    axes[0].bar(x + width/2, leaked_cv_vals, width, label="Leaked CV (Artificially Inflated)", color="#d95f02", edgecolor="black")
    axes[0].set_title("Cross-Validation Scores: The Illusion of Performance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Subplot 2: Generalization Gap (CV vs Test)
    gap_correct = np.array(correct_cv_vals) - np.array(correct_test_vals)
    gap_leaked = np.array(leaked_cv_vals) - np.array(leaked_test_vals)

    axes[1].bar(x - width/2, gap_correct, width, label="Correct Gap (CV - Test)", color="#2ca02c", edgecolor="black")
    axes[1].bar(x + width/2, gap_leaked, width, label="Leaked Gap (Severe Overoptimism)", color="#d62728", edgecolor="black")
    axes[1].set_title("Generalization Drop: (CV Score - Real Test Score)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task11_leakage_comparison.png")
    plt.close(fig)

    print("Task 11 Leakage Challenge completed.")
    return df_leakage
