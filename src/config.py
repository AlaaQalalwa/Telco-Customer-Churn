"""
Configuration module defining constants, paths, features, and reproducibility settings.
"""
from pathlib import Path

# Reproducibility seed required across all operations
RANDOM_STATE = 42

# Train / Test split ratio
TEST_SIZE = 0.20
CV_SPLITS = 5

# Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "TelcoCustomerChurn.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Business Cost Matrix (Task 10 Scenario)
# False Negative (churner missed): $500 loss in future revenue
# False Positive (non-churner offered retention discount): $50 cost
COST_FN = 500.0
COST_FP = 50.0

# Dataset Feature Definitions
ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]
