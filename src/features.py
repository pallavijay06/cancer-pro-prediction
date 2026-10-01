import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset.csv"
)


# =========================================================
# FINAL FEATURE GROUPS
# =========================================================
#
# IMPORTANT:
# These are the ONLY columns allowed to enter the model.
#
# Total = 14 features
#
# demographics       = 5
# disease            = 3
# treatment          = 3
# psychological      = 1
# baseline_qol       = 2
#
# Total              = 14
#
# Radiotherapy is intentionally excluded from the primary
# feature matrix because its timing relative to chemotherapy
# initiation is uncertain.
#
# QLQ_ID is an identifier and is never a feature.
#
# Overall_EORTC_baseline is retained only as an outcome-source
# variable in dataset.csv and is not used as a predictor.
# =========================================================

FEATURE_GROUPS = {

    "demographics": [
        "Age",
        "Educational_level",
        "Occupational_status",
        "Children",
        "Marital_status",
    ],

    "disease": [
        "TNM_stage",
        "HER2",
        "ECOG",
    ],

    "treatment": [
        "Surgery",
        "Axillary_lymphadenectomy",
        "Chemotherapy_regimen",
    ],

    "psychological": [
        "Perceived_risk_recurrence",
    ],

    "baseline_qol": [
        "EORTC_baseline",
        "PH_baseline",
    ],
}


# =========================================================
# GET FINAL FEATURE LIST
# =========================================================

def get_feature_columns():
    """
    Return the complete list of approved model features.

    This is the single source of truth for X.
    """

    feature_columns = []

    for group_features in FEATURE_GROUPS.values():
        feature_columns.extend(group_features)

    # Remove accidental duplicates while preserving order
    feature_columns = list(
        dict.fromkeys(feature_columns)
    )

    return feature_columns


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(outcome="y_threshold"):
    """
    Load processed dataset and return:

        X = approved baseline features only
        y = selected target

    The model feature list is explicitly defined by
    FEATURE_GROUPS.

    This prevents:
        - patient IDs
        - end-of-treatment variables
        - outcome variables
        - excluded clinical variables

    from accidentally entering X.
    """

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # -----------------------------------------------------
    # Check target
    # -----------------------------------------------------

    valid_outcomes = [
        "y_drop",
        "y_threshold",
    ]

    if outcome not in valid_outcomes:

        raise ValueError(
            f"Invalid outcome '{outcome}'. "
            f"Choose from: {valid_outcomes}"
        )

    if outcome not in df.columns:

        raise KeyError(
            f"Target '{outcome}' not found in dataset."
        )

    # -----------------------------------------------------
    # Get ONLY approved features
    # -----------------------------------------------------

    feature_columns = get_feature_columns()

    # -----------------------------------------------------
    # Verify all features exist
    # -----------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:

        raise KeyError(
            "The following required model features are "
            "missing from dataset.csv:\n"
            f"{missing_features}"
        )

    # -----------------------------------------------------
    # X
    # -----------------------------------------------------

    X = df[
        feature_columns
    ].copy()

    # -----------------------------------------------------
    # y
    # -----------------------------------------------------

    y = df[
        outcome
    ].copy()

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    forbidden_features = {
        "QLQ_ID",
        "Radiotherapy",
        "Overall_EORTC_baseline",
        "Overall_EORTC_end",
        "Health_status_end",
        "PH_end",
        "y_drop_change",
        "y_drop",
        "y_threshold",
    }

    leakage_columns_present = (
        set(X.columns)
        & forbidden_features
    )

    if leakage_columns_present:

        raise RuntimeError(
            "Forbidden columns detected in X:\n"
            f"{sorted(leakage_columns_present)}"
        )

    # -----------------------------------------------------
    # Final feature count check
    # -----------------------------------------------------

    expected_feature_count = 14

    if X.shape[1] != expected_feature_count:

        raise RuntimeError(
            f"Expected {expected_feature_count} model "
            f"features, but found {X.shape[1]}:\n"
            f"{list(X.columns)}"
        )

    return X, y


# =========================================================
# GET FEATURE GROUP MAPPING
# =========================================================

def get_feature_groups():
    """
    Return:

        feature -> group

    Example:

        Age -> demographics
        TNM_stage -> disease
        Surgery -> treatment
        EORTC_baseline -> baseline_qol
    """

    mapping = {}

    for group, features in FEATURE_GROUPS.items():

        for feature in features:

            if feature in mapping:

                raise ValueError(
                    f"Feature '{feature}' appears in "
                    "multiple feature groups."
                )

            mapping[feature] = group

    return mapping


# =========================================================
# GET COLUMNS BY TYPE
# =========================================================

def get_feature_types():
    """
    Return numerical and categorical feature lists.

    Numeric:
        Age
        EORTC_baseline
        PH_baseline
        ECOG

    Categorical:
        Everything else.
    """

    numerical_features = [
        "Age",
        "EORTC_baseline",
        "PH_baseline",
        "ECOG",
    ]

    all_features = get_feature_columns()

    categorical_features = [
        feature
        for feature in all_features
        if feature not in numerical_features
    ]

    return (
        numerical_features,
        categorical_features
    )


# =========================================================
# M1 PREPROCESSOR
# =========================================================

def get_preprocessor_M1():
    """
    M1:
        Numerical:
            KNN imputation
            StandardScaler

        Categorical:
            Most-frequent imputation
            One-hot encoding
    """

    numerical_features, categorical_features = (
        get_feature_types()
    )

    numerical_pipeline = Pipeline([
        (
            "imputer",
            KNNImputer(
                n_neighbors=5
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ],
        remainder="drop"
    )

    return preprocessor


# =========================================================
# M2 PREPROCESSOR
# =========================================================

def get_preprocessor_M2():
    """
    M2:
        Numerical:
            Median imputation
            Missing-value indicators
            StandardScaler

        Categorical:
            Most-frequent imputation
            One-hot encoding
    """

    numerical_features, categorical_features = (
        get_feature_types()
    )

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ],
        remainder="drop"
    )

    return preprocessor


# =========================================================
# GENERIC PREPROCESSOR ACCESS
# =========================================================

def get_preprocessor(matrix="M1", variant=None):
    """
    Return the requested preprocessing pipeline.

    Parameters
    ----------
    matrix : str
        "M1" -> KNN imputation
        "M2" -> median imputation + missing indicators

    variant : str, optional
        Alias for matrix. This is supported for compatibility
        with the Experiment 1 runner.

    If both matrix and variant are supplied, variant takes priority.
    """

    if variant is not None:
        matrix = variant

    matrix = matrix.upper()

    if matrix == "M1":
        return get_preprocessor_M1()

    elif matrix == "M2":
        return get_preprocessor_M2()

    else:
        raise ValueError(
            "Unknown preprocessing matrix. "
            "Use 'M1' or 'M2'."
        )

# =========================================================
# SELF-TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FEATURE PIPELINE SELF-TEST")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nDataset path:")
    print(DATA_PATH)

    # -----------------------------------------------------
    # Feature list
    # -----------------------------------------------------

    feature_columns = get_feature_columns()

    print("\nApproved model features:")
    for i, feature in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"{i:2}. {feature}"
        )

    print(
        f"\nTotal features: "
        f"{len(feature_columns)}"
    )

    # -----------------------------------------------------
    # Feature groups
    # -----------------------------------------------------

    print("\nFeature groups:")

    print(
        get_feature_groups()
    )

    # -----------------------------------------------------
    # Feature types
    # -----------------------------------------------------

    numerical, categorical = (
        get_feature_types()
    )

    print("\nNumerical features:")
    print(numerical)

    print("\nCategorical features:")
    print(categorical)

    # -----------------------------------------------------
    # Load y_drop
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("Testing y_drop")
    print("-" * 70)

    X_drop, y_drop = load_dataset(
        "y_drop"
    )

    print(
        "X shape:",
        X_drop.shape
    )

    print(
        "y shape:",
        y_drop.shape
    )

    # -----------------------------------------------------
    # Load y_threshold
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("Testing y_threshold")
    print("-" * 70)

    X_threshold, y_threshold = (
        load_dataset("y_threshold")
    )

    print(
        "X shape:",
        X_threshold.shape
    )

    print(
        "y shape:",
        y_threshold.shape
    )

    # -----------------------------------------------------
    # Preprocessors
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("Testing preprocessors")
    print("-" * 70)

    m1 = get_preprocessor("M1")
    m2 = get_preprocessor("M2")

    print(
        "M1 type:",
        type(m1)
    )

    print(
        "M2 type:",
        type(m2)
    )

    print("\nSELF-TEST COMPLETE")