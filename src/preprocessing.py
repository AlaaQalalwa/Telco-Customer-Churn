"""
Preprocessing module implementing leakage-free transformers and modular pipelines.
All data-dependent parameters (means, scales, imputation values, one-hot categories)
are strictly fit on training data only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS, RANDOM_STATE

def build_column_transformer(
    numerical_cols=None,
    categorical_cols=None,
    scale_numeric=True
) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for numerical and categorical pipelines.
    - Numerical: Median imputation followed by optional Standard Scaling.
    - Categorical: OneHotEncoder with drop='first' and handle_unknown='ignore'.
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = ImbPipeline(steps=num_steps)

    cat_pipeline = ImbPipeline(steps=[
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def build_model_pipeline(
    estimator,
    preprocessor=None,
    use_smote=False,
    smote_random_state=RANDOM_STATE
) -> ImbPipeline:
    """
    Constructs an end-to-end imblearn Pipeline.
    If use_smote is True, SMOTE is inserted AFTER the preprocessor and BEFORE the classifier.
    This guarantees that oversampling happens strictly on training folds during CV.
    """
    if preprocessor is None:
        preprocessor = build_column_transformer()

    steps = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(("resampler", SMOTE(random_state=smote_random_state)))

    steps.append(("classifier", estimator))

    return ImbPipeline(steps=steps)

def get_transformed_feature_names(fitted_column_transformer, numerical_cols=None, categorical_cols=None):
    """
    Extracts explicit feature names output by the ColumnTransformer.
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    # Numerical feature names
    feature_names = list(numerical_cols)

    # Categorical encoded names
    cat_encoder = fitted_column_transformer.named_transformers_["cat"].named_steps["encoder"]
    cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names.extend(cat_names)

    return feature_names
