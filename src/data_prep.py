"""
Complete reproducible data-preparation pipeline.

Run from the project root:

    python src/data_prep.py

Pipeline:
    1. Build dataset from raw XLSX files
    2. Clean the generated dataset
    3. Run final validation
"""

from pathlib import Path
import subprocess
import sys


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"


# =========================================================
# RUN SCRIPT
# =========================================================

def run_script(script_name):
    """
    Run another project script using the same Python
    interpreter/environment.
    """

    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required script not found:\n{script_path}"
        )

    print()
    print("=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code "
            f"{result.returncode}."
        )


# =========================================================
# FINAL DATASET VALIDATION
# =========================================================

def validate_dataset():

    import pandas as pd

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "dataset.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset was not created:\n{dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    print()
    print("=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    print("Dataset path:")
    print(dataset_path)

    print()
    print("Shape:")
    print(df.shape)

    # -----------------------------------------------------
    # Required target columns
    # -----------------------------------------------------

    required_targets = [
        "y_drop",
        "y_threshold",
    ]

    for target in required_targets:

        if target not in df.columns:
            raise AssertionError(
                f"Missing target column: {target}"
            )

    # -----------------------------------------------------
    # Required baseline variables
    # -----------------------------------------------------

    required_features = [
        "Age",
        "Educational_level",
        "Occupational_status",
        "Children",
        "Marital_status",
        "TNM_stage",
        "HER2",
        "ECOG",
        "Surgery",
        "Axillary_lymphadenectomy",
        "Chemotherapy_regimen",
        "EORTC_baseline",
        "PH_baseline",
        "Perceived_risk_recurrence",
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:
        raise AssertionError(
            "Missing required features:\n"
            f"{missing_features}"
        )

    # -----------------------------------------------------
    # Target validation
    # -----------------------------------------------------

    for target in required_targets:

        values = set(
            df[target].dropna().unique()
        )

        if not values.issubset({0, 1}):

            raise AssertionError(
                f"{target} contains unexpected values: "
                f"{values}"
            )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print()
    print("Missing values:")

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:
        print("None")

    else:
        print(missing)

    # -----------------------------------------------------
    # Target distributions
    # -----------------------------------------------------

    print()
    print("Target distributions:")

    for target in required_targets:

        print()
        print(target)

        counts = (
            df[target]
            .value_counts()
            .sort_index()
        )

        percentages = (
            counts / len(df) * 100
        ).round(2)

        print(
            pd.DataFrame({
                "count": counts,
                "percent": percentages
            })
        )

    # -----------------------------------------------------
    # Leakage columns must NOT be features
    #
    # They may remain in dataset.csv for outcome
    # construction/audit.
    # -----------------------------------------------------

    print()
    print("Outcome/source columns retained for audit:")

    audit_columns = [
        "Overall_EORTC_baseline",
        "Overall_EORTC_end",
        "Health_status_end",
        "y_drop_change",
        "y_drop",
        "y_threshold",
    ]

    for column in audit_columns:

        if column in df.columns:
            print(f"  {column}")

    print()
    print("FINAL VALIDATION PASSED")


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("REPRODUCIBLE DATA PREPARATION PIPELINE")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    # -----------------------------------------------------
    # 1. Build dataset from raw files
    # -----------------------------------------------------

    run_script(
        "build_dataset.py"
    )

    # -----------------------------------------------------
    # 2. Clean dataset
    # -----------------------------------------------------

    run_script(
        "clean_data.py"
    )

    # -----------------------------------------------------
    # 3. Validate final dataset
    # -----------------------------------------------------

    validate_dataset()

    print()
    print("=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()