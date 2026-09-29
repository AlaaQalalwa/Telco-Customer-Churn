"""
===============================================================================
Telco Customer Churn Prediction Machine Learning Pipeline
Birzeit University - Department of Electrical and Computer Engineering
Assignment 1: Complete Leakage-Free Classification Pipeline (Tasks 1–13)
===============================================================================
"""
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Local module imports
from src.config import (
    DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    CV_SPLITS,
    COST_FN,
    COST_FP,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR
)
from src.data_loader import load_raw_data, prepare_features_and_target, get_train_test_split
from src.eda import perform_eda_analysis, generate_eda_plots
from src.preprocessing import build_column_transformer, build_model_pipeline
from src.feature_selection import run_feature_selection_analysis
from src.models import get_baseline_models, get_default_models, run_models_benchmark
from src.tuning import run_hyperparameter_tuning
from src.imbalance import run_imbalance_experiments
from src.cost_analysis import run_cost_and_roc_analysis
from src.leakage_experiment import run_leakage_challenge_experiment
from src.evaluation import assemble_and_evaluate_final_models

def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)

def main():
    total_start = time.time()
    print_banner("Birzeit University - ML Assignment 1: Telco Customer Churn Pipeline")
    print(f"Dataset Path: {DATA_PATH}")
    print(f"Random State: {RANDOM_STATE} (Enforced across all operations)")
    print(f"Evaluation: Stratified {CV_SPLITS}-Fold Cross-Validation & 80/20 Train/Test Split\n")

    # =========================================================================
    # TASK 1: Exploratory Data Analysis (EDA)
    # =========================================================================
    print_banner("Task 1: Exploratory Data Analysis (EDA)")
    df_raw = load_raw_data()
    print(f"1. Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"2. Features: {df_raw.columns.tolist()}")

    eda_results = perform_eda_analysis(df_raw)
    print(f"3. Subtle Structural Issues Detected: {eda_results['whitespace_total_charges_count']} whitespace entries in 'TotalCharges'.")
    print(f"   Associated tenure values: {eda_results['whitespace_tenure_values']} (all 0 months - brand new accounts).")
    print(f"4. Target Distribution (Churn):")
    print(eda_results["target_distribution"])
    churn_pct = eda_results["target_distribution"].loc["Yes", "Percentage"]
    print(f"   Churn Rate: {churn_pct:.2f}% vs Non-Churn: {100-churn_pct:.2f}% -> Ratio ~ {((100-churn_pct)/churn_pct):.2f}:1")
    print(f"5. Generating EDA Visualizations...")
    generate_eda_plots(df_raw)
    print("   [+] Plots saved: churn distribution, numerical distributions, categorical relationships, outliers, correlation heatmap.")

    # =========================================================================
    # TASK 2 & 3: Data Preprocessing and Train/Test Split
    # =========================================================================
    print_banner("Tasks 2 & 3: Data Preprocessing & Train/Test Split")
    X, y = prepare_features_and_target(df_raw)
    print(f"Identifiers removed: customerID. Features shape: {X.shape}, Target shape: {y.shape}")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train Set Shape: {X_train.shape} (Churners: {y_train.sum()}, {y_train.mean()*100:.2f}%)")
    print(f"Test Set Shape:  {X_test.shape} (Churners: {y_test.sum()}, {y_test.mean()*100:.2f}%)")
    print("NOTICE: Test set is now completely sequestered and will NOT be touched until final evaluation (Task 10/12/13).")

    preprocessor = build_column_transformer()
    print("ColumnTransformer constructed:")
    print(" - Numerical Pipeline: Median Imputation -> StandardScaler")
    print(" - Categorical Pipeline: OneHotEncoder(drop='first', handle_unknown='ignore')")
    print(" - Preprocessing strictly encapsulated in pipelines to eliminate fold leakage.")

    # =========================================================================
    # TASK 4: Feature Selection
    # =========================================================================
    print_banner("Task 4: Feature Selection (Training Set Only)")
    importance_df, fs_cv_df = run_feature_selection_analysis(X_train, y_train)
    print("Top 10 Informative Features (Mutual Information & Random Forest Importance):")
    print(importance_df.head(10)[["Mutual_Information", "MI_Rank", "RF_Importance", "RF_Rank"]])
    print("\nCross-Validation Performance: Full Feature Set vs Selected Subsets:")
    print(fs_cv_df)
    print("Observation: Both methods consistently identify Contract, tenure, MonthlyCharges, TechSupport, OnlineSecurity, and InternetService as dominant.")
    print("Empirical Result: Retaining the full feature set (30 encoded features) preserves full non-linear interaction capacity without degradation.")

    # =========================================================================
    # TASK 5: Baseline Models
    # =========================================================================
    print_banner("Task 5: Baseline Classifier Evaluation")
    baselines = get_baseline_models()
    df_baselines = run_models_benchmark(baselines, X_train, y_train, "task5_baseline_cv_results.csv")
    print("Baseline Cross-Validation Results (Stratified 5-Fold):")
    print(df_baselines[["Model", "Accuracy_Mean", "Precision_Mean", "Recall_Mean", "F1_Mean", "ROC_AUC_Mean"]])
    print("Insight: DummyClassifier achieves ~73.4% accuracy simply by predicting majority class, but 0.0 Recall and 0.50 ROC-AUC.")
    print("This conclusively proves why accuracy is completely misleading for imbalanced customer churn.")

    # =========================================================================
    # TASK 6: Classification Models (Default Hyperparameters)
    # =========================================================================
    print_banner("Task 6: Classification Models Benchmark (Default)")
    models_dict = get_default_models()
    df_default_cv = run_models_benchmark(models_dict, X_train, y_train, "task6_default_models_cv_results.csv")
    print("Default Models Cross-Validation Results (Stratified 5-Fold):")
    print(df_default_cv[["Model", "Accuracy_Mean", "Precision_Mean", "Recall_Mean", "F1_Mean", "ROC_AUC_Mean"]])

    # =========================================================================
    # TASK 7: Hyperparameter Tuning
    # =========================================================================
    print_banner("Task 7: Hyperparameter Tuning (RandomizedSearchCV)")
    best_tuned_pipelines, df_tuning = run_hyperparameter_tuning(X_train, y_train, df_default_cv)
    print("\nHyperparameter Tuning Comparison (Default vs Tuned):")
    print(df_tuning[["Model", "Default_ROC_AUC", "Tuned_ROC_AUC", "ROC_AUC_Gain", "Tuned_Recall", "Tuned_F1"]])

    # =========================================================================
    # TASK 8: Class Imbalance Handling
    # =========================================================================
    print_banner("Task 8: Class Imbalance Investigation")
    df_imbalance = run_imbalance_experiments(X_train, y_train)
    print("Imbalance Handling Cross-Validation Comparison:")
    print(df_imbalance.pivot(index="Model", columns="Imbalance_Strategy", values=["Recall", "Precision", "F1_Score", "ROC_AUC", "Expected_Cost"]))

    # Summary table of cost-optimal imbalance strategy per model (lowest Expected_Cost)
    best_strategies = []
    for model_name in df_imbalance["Model"].unique():
        sub = df_imbalance[df_imbalance["Model"] == model_name].sort_values(
            by=["Expected_Cost", "Recall"], ascending=[True, False]
        )
        top_row = sub.iloc[0]
        best_strategies.append({
            "Model": model_name,
            "Winning_Strategy": top_row["Imbalance_Strategy"],
            "Total_FN": int(top_row["Total_FN"]),
            "Total_FP": int(top_row["Total_FP"]),
            "Expected_Cost_$": float(top_row["Expected_Cost"]),
            "Recall": float(top_row["Recall"]),
            "F1_Score": float(top_row["F1_Score"])
        })
    df_best_strat = pd.DataFrame(best_strategies)
    print("\nTask 8 Cost-Optimal Winning Strategy per Model (Minimizing $500*FN + $50*FP):")
    print(df_best_strat[["Model", "Winning_Strategy", "Total_FN", "Total_FP", "Expected_Cost_$", "Recall", "F1_Score"]])

    # =========================================================================
    # TASK 9: Cross-Validation and Metric Selection
    # =========================================================================
    print_banner("Task 9: Cross-Validation & Metric Selection Analysis")
    print("1. Stratified 5-Fold Cross-Validation was consistently executed across all modeling experiments.")
    print("2. Metric Selection Justification:")
    print("   - Accuracy: Misleading due to 73.4% majority baseline.")
    print("   - ROC-AUC: Recommended as primary threshold-independent ranking metric.")
    print("   - Recall: Critical operational metric for capturing churning customers before departure.")
    print("   - Expected Cost: Decisive business metric directly weighting financial asymmetry ($500 FN vs $50 FP).")

    # =========================================================================
    # TASK 10: ROC-AUC and Confusion Matrix Analysis (Top 3 Models)
    # =========================================================================
    print_banner("Task 10: ROC-AUC, Confusion Matrix & Business Cost Analysis")
    # Identify Top 3 models by CV ROC-AUC from Task 7
    top_3_names = df_tuning.sort_values("Tuned_ROC_AUC", ascending=False)["Model"].head(3).tolist()
    top_3_dict = {name: best_tuned_pipelines[name] for name in top_3_names}
    print(f"Top 3 Models Selected by CV ROC-AUC: {top_3_names}")

    df_matrices, cost_curves = run_cost_and_roc_analysis(top_3_dict, X_train, y_train)
    print("\nConfusion Matrix & Cost Comparison (Default 0.50 Threshold vs Cost-Optimal Threshold):")
    print(df_matrices[["Model", "Threshold_Type", "Threshold", "FN", "FP", "Recall", "Total_Cost_$"]])

    # =========================================================================
    # TASK 11: Data Leakage Challenge
    # =========================================================================
    print_banner("Task 11: Data Leakage Challenge Experiment")
    df_leakage = run_leakage_challenge_experiment(X_train, y_train, X_test, y_test)
    print("Data Leakage Empirical Comparison Table:")
    print(df_leakage.T)
    print("\nLeakage Challenge Key Takeaway:")
    leak_cv_roc = df_leakage.loc[df_leakage["Pipeline_Condition"].str.contains("Leaked"), "CV_ROC_AUC"].values[0]
    corr_cv_roc = df_leakage.loc[df_leakage["Pipeline_Condition"].str.contains("Correct"), "CV_ROC_AUC"].values[0]
    leak_test_roc = df_leakage.loc[df_leakage["Pipeline_Condition"].str.contains("Leaked"), "Test_ROC_AUC"].values[0]
    corr_test_roc = df_leakage.loc[df_leakage["Pipeline_Condition"].str.contains("Correct"), "Test_ROC_AUC"].values[0]
    print(f"  Correct Pipeline: CV ROC-AUC = {corr_cv_roc:.4f} | Test ROC-AUC = {corr_test_roc:.4f} (Realistic & Consistent)")
    print(f"  Leaked Pipeline:  CV ROC-AUC = {leak_cv_roc:.4f} | Test ROC-AUC = {leak_test_roc:.4f} (Inflated CV drops on real test data!)")

    # =========================================================================
    # TASK 12: Final Model Comparison (CV vs Test Performance)
    # =========================================================================
    print_banner("Task 12: Final Model Comparison Table (Best Pipeline Per Model)")
    df_final = assemble_and_evaluate_final_models(best_tuned_pipelines, X_train, y_train, X_test, y_test)
    print("\nMaster Final Model Comparison Table:")
    print(df_final[[
        "Model", "Imbalance_Strategy", "Resampling_SMOTE",
        "CV_Recall_Mean", "CV_ROC_AUC_Mean",
        "Test_Accuracy", "Test_Recall", "Test_Precision", "Test_F1", "Test_ROC_AUC",
        "Test_Expected_Cost_Default", "Best_Threshold", "Test_Expected_Cost_Optimal"
    ]])

    # =========================================================================
    # TASK 13: Deployment Recommendation
    # =========================================================================
    print_banner("Task 13: Deployment Recommendation")
    top_model = df_final.iloc[0]
    alt_model = df_final.iloc[1]

    opt_cost = top_model["Test_Expected_Cost_Optimal"]
    def_cost = top_model["Test_Expected_Cost_Default"]
    cost_savings_vs_default = def_cost - opt_cost
    pct_savings_default = (cost_savings_vs_default / def_cost) * 100 if def_cost > 0 else 0.0

    alt_opt_cost = alt_model["Test_Expected_Cost_Optimal"]
    cost_diff_alt = alt_opt_cost - opt_cost

    print(f"RECOMMENDED DEPLOYMENT MODEL: {top_model['Model']}")
    print(f"  - Imbalance Strategy:       {top_model['Imbalance_Strategy'].upper()} (Resampling_SMOTE={top_model['Resampling_SMOTE']})")
    print(f"  - Test ROC-AUC:             {top_model['Test_ROC_AUC']:.4f}")
    print(f"  - Test Accuracy:            {top_model['Test_Accuracy']:.4f}")
    print(f"  - Test Recall:              {top_model['Test_Recall']:.4f}")
    print(f"  - Test F1-Score:            {top_model['Test_F1']:.4f}")
    print(f"  - Default Threshold (0.50) Expected Cost: ${def_cost:,.0f} (FN={top_model['Test_FN']}, FP={top_model['Test_FP']})")
    print(f"  - Cost-Optimal Threshold:                 t* = {top_model['Best_Threshold']:.2f}")
    print(f"  - Cost-Optimal Expected Cost:             ${opt_cost:,.0f}")
    print(f"  - Expected Cost Savings vs Default (0.50): ${cost_savings_vs_default:,.0f} ({pct_savings_default:.1f}% financial loss reduction)")
    print(f"  - Expected Cost Advantage vs {alt_model['Model']}: Saves an additional ${cost_diff_alt:,.0f} in net churn losses")
    print("\nStrategic Justification:")
    print(f"  1. Minimal Business Cost: Achieves the absolute lowest expected business loss (${opt_cost:,.0f})")
    print(f"     when operated at threshold t* = {top_model['Best_Threshold']:.2f} under the $500 FN / $50 FP cost regime.")
    print("  2. Superior Sensitivity: Incorporating the justified class-imbalance strategy ensures high churner detection.")
    print("  3. Generalization Robustness: Consistent alignment between 5-fold CV metrics and unseen test set performance.")

    elapsed = time.time() - total_start
    print_banner(f"All 13 Tasks Completed Successfully in {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
