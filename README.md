# Telco Customer Churn Prediction Pipeline
**Birzeit University — Department of Electrical and Computer Engineering**  
*Assignment 1: Machine Learning Classification Pipeline (Summer Semester 2025/2026)*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Reproducibility](https://img.shields.io/badge/seed-42-brightgreen.svg)]()

---

## 1. Project Overview

This repository provides an end-to-end, reproducible, leakage-free Machine Learning system designed to predict customer churn for a telecommunications provider. The pipeline strictly follows the official University Assignment Specification and covers all 13 Core Tasks:

1. **Task 1 — Exploratory Data Analysis (EDA)**: Comprehensive structural, distributional, and correlation audit.
2. **Task 2 — Data Preprocessing**: Leakage-free `ColumnTransformer` (median imputation, standard scaling, one-hot encoding).
3. **Task 3 — Stratified Train/Test Split**: 80/20 train/test split with strict test set isolation (`random_state=42`).
4. **Task 4 — Feature Selection**: Mutual Information vs. Random Forest Gini importance evaluated inside CV.
5. **Task 5 — Baseline Model**: Dummy Classifier (most frequent) and Logistic Regression baseline.
6. **Task 6 — Classification Models**: Benchmark of 5 algorithms (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM) via Stratified 5-Fold Cross-Validation.
7. **Task 7 — Hyperparameter Tuning**: `RandomizedSearchCV` on 2–3 key parameters per model.
8. **Task 8 — Class Imbalance Handling**: No handling vs. Class Weighting vs. fold-safe SMOTE (`imblearn.pipeline.Pipeline`).
9. **Task 9 — Cross-Validation & Metric Selection**: Stratified 5-Fold CV analysis and metric selection rationale.
10. **Task 10 — ROC-AUC, Confusion Matrices & Business Cost Optimization**: Composite ROC curves, confusion matrices, and financial loss minimization with asymmetric error costs ($500 FN vs. $50 FP).
11. **Task 11 — Data Leakage Challenge**: Empirical comparison between a leakage-free pipeline and an intentionally leaked pre-CV resampling pipeline.
12. **Task 12 — Final Model Comparison**: Master table comparing the single best pipeline per model across 5-fold CV and the untouched test set.
13. **Task 13 — Deployment Recommendation**: Multi-criteria production decision and comparison against alternatives.

---

## 2. Directory Structure

```
Telco Customer Churn/
├── data/
│   └── TelcoCustomerChurn.csv          # Official dataset (7,043 rows x 21 cols)
├── outputs/
│   ├── figures/                        # 11 High-resolution figures (300 DPI)
│   │   ├── task1_churn_distribution.png
│   │   ├── task1_numerical_distributions.png
│   │   ├── task1_categorical_relationships.png
│   │   ├── task1_outliers_boxplots.png
│   │   ├── task1_correlation_matrix.png
│   │   ├── task4_feature_importance_comparison.png
│   │   ├── task8_imbalance_comparison.png
│   │   ├── task10_roc_curves_top3.png
│   │   ├── task10_confusion_matrices_top3.png
│   │   ├── task10_cost_threshold_analysis.png
│   │   └── task11_leakage_comparison.png
│   ├── tables/                         # 14 Detailed CSV output tables
│   │   ├── task1_target_distribution.csv
│   │   ├── task1_numerical_descriptive_stats.csv
│   │   ├── task1_categorical_summary.csv
│   │   ├── task1_outlier_summary.csv
│   │   ├── task1_correlation_matrix.csv
│   │   ├── task4_feature_rankings.csv
│   │   ├── task4_feature_selection_cv_comparison.csv
│   │   ├── task5_baseline_cv_results.csv
│   │   ├── task6_default_models_cv_results.csv
│   │   ├── task7_tuning_comparison.csv
│   │   ├── task8_imbalance_handling_results.csv
│   │   ├── task10_confusion_matrices_and_costs.csv
│   │   ├── task11_leakage_experiment_results.csv
│   │   └── task12_final_model_comparison.csv
│   └── models/
│       └── best_churn_pipeline.joblib  # Serialized production pipeline
├── src/                                # Modular architecture
│   ├── __init__.py
│   ├── config.py                       # Global constants, paths, seed=42, costs
│   ├── data_loader.py                  # Loading, TotalCharges repair, stratified splitting
│   ├── eda.py                          # Statistical summaries & visualizations
│   ├── preprocessing.py                # Leakage-safe ColumnTransformer & pipelines
│   ├── feature_selection.py            # MI & RF feature ranking and CV impact
│   ├── models.py                       # Estimator factories & 5-fold CV evaluation
│   ├── tuning.py                       # RandomizedSearchCV implementation
│   ├── imbalance.py                    # Class weighting & fold-safe SMOTE
│   ├── cost_analysis.py                # Expected cost curves & threshold optimizer
│   ├── leakage_experiment.py           # Controlled leakage demonstration
│   └── evaluation.py                   # Master evaluation & model serialization
├── main.py                             # Executable driver script running Tasks 1–13
├── telco_churn_pipeline.ipynb          # Step-by-step interactive Jupyter notebook
├── requirements.txt                    # Project dependencies
├── REPORT.md                           # Comprehensive Technical Report (8–12 pages equivalent)
└── README.md                           # Documentation & execution guide
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git (optional)

### Setup Environment
```bash
# Clone or navigate to the project directory
cd "Telco Customer Churn"

# (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Running the Project

### Option A: Command-Line Execution (End-to-End)
To run the full pipeline from Task 1 through Task 13, execute `main.py`:
```bash
python main.py
```
This script will:
- Audit and clean the data.
- Generate all 11 plots in `outputs/figures/`.
- Export all 14 empirical evaluation tables to `outputs/tables/`.
- Run Stratified 5-Fold Cross-Validation, hyperparameter tuning, imbalance tests, and leakage benchmarks.
- Serialize the final deployment pipeline to `outputs/models/best_churn_pipeline.joblib`.

### Option B: Interactive Jupyter Notebook
To run and inspect the code interactively cell-by-cell:
```bash
jupyter notebook telco_churn_pipeline.ipynb
```
or open the notebook directly in VS Code / Cursor / JupyterLab.

---

## 5. Summary of Empirical Results

| Model | Pipeline Configuration | 5-Fold CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall | Test F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Recommended)** | Tuned (`n_est=100, max_depth=3, lr=0.05`) | **0.8488 ± 0.0120** | **0.8454** | **0.7991** | **0.5027** | **0.5706** |
| **LightGBM** | Tuned (`n_est=200, num_leaves=25, lr=0.01`) | 0.8460 ± 0.0129 | 0.8445 | 0.8055 | 0.4973 | 0.5759 |
| **Random Forest** | Tuned (`n_est=150, max_depth=8, min_samples=5`) | 0.8460 ± 0.0114 | 0.8444 | 0.8048 | 0.4973 | 0.5750 |
| **Logistic Regression** | Tuned (`C=12.0, penalty=l2, solver=saga`) | 0.8463 ± 0.0128 | 0.8411 | 0.8048 | 0.5588 | 0.6032 |
| **SVM (RBF)** | Tuned (`C=1.0, gamma=0.01, kernel=rbf`) | 0.8367 ± 0.0133 | 0.8298 | 0.7991 | 0.4866 | 0.5626 |
| **Dummy Classifier** | Most Frequent Baseline | 0.5000 ± 0.0000 | 0.5000 | 0.7346 | 0.0000 | 0.0000 |

### Key Findings:
1. **Asymmetric Error Costs**: With False Negatives costing $500 and False Positives costing $50, shifting the decision threshold from the standard $0.50$ to the cost-optimal cutoff ($t^* \approx 0.07 - 0.25$) reduces total financial losses from **$388,350 to $142,400 (a 63.3% cost reduction)**, capturing over 97% of churners.
2. **Data Leakage Proof**: Applying SMOTE globally before cross-validation inflated CV ROC-AUC to an artificial **0.9279**, yet collapsed back to **0.8235** on the real test set. The fold-safe pipeline reported an honest CV of **0.8217**, matching the test set exactly and proving that leakage produces false confidence.
3. **Winning Deployment Architecture**: **Tuned XGBoost** deployed with an operational decision threshold between $0.20$ and $0.30$ provides the highest discriminatory power, best calibration, minimal generalization drop, and lowest total business cost.
