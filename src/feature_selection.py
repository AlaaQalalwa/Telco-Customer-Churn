"""
Feature selection module for Task 4.
Implements Mutual Information and Random Forest feature importance techniques
strictly fitted on the training set, comparing feature rankings and downstream CV impact.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.config import FIGURES_DIR, TABLES_DIR, RANDOM_STATE
from src.preprocessing import build_column_transformer, get_transformed_feature_names, build_model_pipeline

def run_feature_selection_analysis(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Computes Mutual Information and Random Forest feature importance on the training data.
    Evaluates whether selecting top features improves cross-validated ROC-AUC.
    """
    # 1. Transform training features using preprocessing pipeline fit only on X_train
    preprocessor = build_column_transformer()
    X_train_trans = preprocessor.fit_transform(X_train)
    feature_names = get_transformed_feature_names(preprocessor)

    # 2. Method A: Mutual Information
    mi_scores = mutual_info_classif(
        X_train_trans, y_train, random_state=RANDOM_STATE
    )
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

    # 3. Method B: Random Forest Gini Importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_selector.fit(X_train_trans, y_train)
    rf_series = pd.Series(rf_selector.feature_importances_, index=feature_names).sort_values(ascending=False)

    # Combined Rankings DataFrame
    importance_df = pd.DataFrame({
        "Mutual_Information": mi_series,
        "MI_Rank": mi_series.rank(ascending=False, method="min").astype(int),
        "RF_Importance": rf_series,
        "RF_Rank": rf_series.rank(ascending=False, method="min").astype(int)
    }).sort_values("Mutual_Information", ascending=False)
    importance_df.to_csv(TABLES_DIR / "task4_feature_rankings.csv")

    # 4. Visualization: Top 15 Features by Both Methods
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    top_mi = mi_series.head(15)
    axes[0].barh(top_mi.index[::-1], top_mi.values[::-1], color="#2b5c8f", edgecolor="black")
    axes[0].set_title("Top 15 Features by Mutual Information")
    axes[0].set_xlabel("Mutual Information Score")

    top_rf = rf_series.head(15)
    axes[1].barh(top_rf.index[::-1], top_rf.values[::-1], color="#d95f02", edgecolor="black")
    axes[1].set_title("Top 15 Features by Random Forest Importance (MDI)")
    axes[1].set_xlabel("Mean Decrease in Impurity")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task4_feature_importance_comparison.png")
    plt.close(fig)

    # 5. Evaluate CV performance impact: Full Features vs. Selected Top-K
    # We test on Logistic Regression and Random Forest using 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    cv_comparison = []
    for k in [10, 15, len(feature_names)]:
        label = "Full Feature Set (All 30)" if k == len(feature_names) else f"Top {k} Features (MI)"
        
        # Pipeline with SelectKBest inside fold
        if k == len(feature_names):
            pipeline_lr = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
            pipeline_rf = build_model_pipeline(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))
        else:
            # Selector inserted inside pipeline to ensure zero leakage
            def make_sel_pipeline(clf):
                prep = build_column_transformer()
                sel = SelectKBest(score_func=mutual_info_classif, k=k)
                from imblearn.pipeline import Pipeline as ImbPipe
                return ImbPipe([
                    ("preprocessor", prep),
                    ("selector", sel),
                    ("classifier", clf)
                ])
            pipeline_lr = make_sel_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
            pipeline_rf = make_sel_pipeline(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))

        scores_lr = cross_val_score(pipeline_lr, X_train, y_train, cv=cv, scoring="roc_auc")
        scores_rf = cross_val_score(pipeline_rf, X_train, y_train, cv=cv, scoring="roc_auc")

        cv_comparison.append({
            "Feature_Set": label,
            "Num_Features": k,
            "LR_ROC_AUC_Mean": np.mean(scores_lr),
            "LR_ROC_AUC_Std": np.std(scores_lr),
            "RF_ROC_AUC_Mean": np.mean(scores_rf),
            "RF_ROC_AUC_Std": np.std(scores_rf),
        })

    cv_comp_df = pd.DataFrame(cv_comparison)
    cv_comp_df.to_csv(TABLES_DIR / "task4_feature_selection_cv_comparison.csv", index=False)

    return importance_df, cv_comp_df
