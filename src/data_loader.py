"""
Data loader module for loading, cleaning initial structural issues, and splitting data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_PATH,
    ID_COL,
    TARGET_COL,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TEST_SIZE,
    RANDOM_STATE
)

def load_raw_data(filepath=None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV dataset."""
    path = filepath if filepath is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df

def clean_structural_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Address subtle structural issues such as whitespace strings in TotalCharges.
    Converts whitespace strings to NaN and casts TotalCharges to numeric float64.
    """
    df_clean = df.copy()
    
    # Identify whitespace strings in TotalCharges
    if df_clean["TotalCharges"].dtype == object or pd.api.types.is_string_dtype(df_clean["TotalCharges"]):
        is_whitespace = df_clean["TotalCharges"].str.strip() == ""
        whitespace_count = is_whitespace.sum()
        if whitespace_count > 0:
            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"].str.strip(), errors="coerce"
            )
    else:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        
    return df_clean

def prepare_features_and_target(df: pd.DataFrame):
    """
    Drops customerID and encodes target variable Churn (Yes: 1, No: 0).
    Returns feature dataframe X and binary target Series y.
    """
    df_work = clean_structural_issues(df)
    
    # Drop customerID identifier
    if ID_COL in df_work.columns:
        X = df_work.drop(columns=[ID_COL, TARGET_COL])
    else:
        X = df_work.drop(columns=[TARGET_COL])
        
    # Map target binary variable
    target_mapping = {"Yes": 1, "No": 0}
    if df_work[TARGET_COL].dtype == object or pd.api.types.is_string_dtype(df_work[TARGET_COL]):
        y = df_work[TARGET_COL].map(target_mapping).astype(int)
    else:
        y = df_work[TARGET_COL].astype(int)
        
    return X, y

def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Performs stratified train/test split.
    Ensures test set is completely held out and untouched.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def load_and_split_data(filepath=None):
    """Convenience pipeline function to load data and produce train/test sets."""
    df_raw = load_raw_data(filepath)
    X, y = prepare_features_and_target(df_raw)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    return df_raw, X_train, X_test, y_train, y_test
