import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SCORES_PATH = RAW_DIR / "breast_scores.xlsx"
QLQ_PATH = RAW_DIR / "breast_qlq_c30.xlsx"
CPPO_PATH = RAW_DIR / "breast_cppo.xlsx"

OUTPUT_PATH = PROCESSED_DIR / "dataset.csv"
AUDIT_PATH = PROCESSED_DIR / "linkage_audit.csv"


# =========================================================
# SETTINGS
# =========================================================

MATCH_DECIMALS = 2

DROP_THRESHOLD = 10
HEALTH_STATUS_THRESHOLD = 50


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_surgery(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    # CPPO uses BSC, QLQ uses BCS
    if value == "BSC":
        return "BCS"

    return value


def rounded_value(value):

    if pd.isna(value):
        return np.nan

    return round(float(value), MATCH_DECIMALS)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    scores = pd.read_excel(SCORES_PATH)
    qlq = pd.read_excel(QLQ_PATH)
    cppo = pd.read_excel(CPPO_PATH)

    print(f"breast_scores.xlsx : {scores.shape}")
    print(f"breast_qlq_c30.xlsx: {qlq.shape}")
    print(f"breast_cppo.xlsx   : {cppo.shape}")

    return scores, qlq, cppo


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

def clean_columns(df):

    df = df.copy()

    df.columns = df.columns.str.strip()

    return df


# =========================================================
# PREPARE MATCHING
# =========================================================

def prepare_matching(qlq, cppo):

    qlq = qlq.copy()
    cppo = cppo.copy()

    # Normalize surgery
    qlq["Surgery_norm"] = qlq["Surgery"].apply(
        normalize_surgery
    )

    cppo["Surgery_norm"] = cppo["Surgery"].apply(
        normalize_surgery
    )

    # Matching keys
    qlq["baseline_match"] = (
        qlq["Overall_EORTC_baseline"]
        .apply(rounded_value)
    )

    qlq["end_match"] = (
        qlq["HE_end"]
        .apply(rounded_value)
    )

    cppo["baseline_match"] = (
        cppo["EORTC_baseline"]
        .apply(rounded_value)
    )

    cppo["end_match"] = (
        cppo["Health_status_end"]
        .apply(rounded_value)
    )

    return qlq, cppo


# =========================================================
# MATCH CPPO -> QLQ
# =========================================================

def match_patients(qlq, cppo):

    print()
    print("=" * 70)
    print("CPPO -> QLQ PATIENT MATCHING")
    print("=" * 70)

    audit_rows = []
    matches = []

    for cppo_index, cppo_row in cppo.iterrows():

        candidates = qlq[
            (qlq["Surgery_norm"] == cppo_row["Surgery_norm"])
            &
            (
                qlq["baseline_match"]
                == cppo_row["baseline_match"]
            )
            &
            (
                qlq["end_match"]
                == cppo_row["end_match"]
            )
        ]

        candidate_indices = candidates.index.tolist()

        # -------------------------------------------------
        # NO MATCH
        # -------------------------------------------------

        if len(candidate_indices) == 0:

            audit_rows.append({
                "CPPO_index": cppo_index,
                "match_status": "NO_MATCH",
                "candidate_count": 0,
                "QLQ_indices": ""
            })

            continue

        # -------------------------------------------------
        # AMBIGUOUS
        # -------------------------------------------------

        if len(candidate_indices) > 1:

            audit_rows.append({
                "CPPO_index": cppo_index,
                "match_status": "AMBIGUOUS",
                "candidate_count": len(candidate_indices),
                "QLQ_indices": ",".join(
                    map(str, candidate_indices)
                )
            })

            continue

        # -------------------------------------------------
        # UNIQUE
        # -------------------------------------------------

        qlq_index = candidate_indices[0]

        audit_rows.append({
            "CPPO_index": cppo_index,
            "match_status": "UNIQUE",
            "candidate_count": 1,
            "QLQ_indices": str(qlq_index)
        })

        matches.append({
            "CPPO_index": cppo_index,
            "QLQ_index": qlq_index
        })

    audit = pd.DataFrame(audit_rows)
    matches = pd.DataFrame(matches)

    print()
    print("Candidate distribution:")
    print(
        audit["candidate_count"]
        .value_counts()
        .sort_index()
    )

    print()
    print("-" * 70)

    print(f"Total CPPO patients : {len(audit)}")

    print(
        "No match            :",
        (audit["match_status"] == "NO_MATCH").sum()
    )

    print(
        "Unique match        :",
        (audit["match_status"] == "UNIQUE").sum()
    )

    print(
        "Ambiguous match     :",
        (audit["match_status"] == "AMBIGUOUS").sum()
    )

    print("-" * 70)

    return matches, audit


# =========================================================
# BUILD DATASET
# =========================================================

def build_dataset(cppo, qlq, matches):

    print()
    print("=" * 70)
    print("CREATING MODELING DATASET")
    print("=" * 70)

    if len(matches) == 0:
        raise RuntimeError("No unique matches found.")

    # -----------------------------------------------------
    # Check that QLQ records are not reused
    # -----------------------------------------------------

    if matches["QLQ_index"].duplicated().any():

        duplicated = matches[
            matches["QLQ_index"].duplicated(
                keep=False
            )
        ]

        raise RuntimeError(
            "A QLQ record is being matched to "
            "multiple CPPO records.\n"
            f"{duplicated}"
        )

    # -----------------------------------------------------
    # Retrieve matched rows
    # -----------------------------------------------------

    cppo_matched = (
        cppo.loc[matches["CPPO_index"]]
        .reset_index(drop=True)
    )

    qlq_matched = (
        qlq.loc[matches["QLQ_index"]]
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Create final dataset
    # -----------------------------------------------------

    dataset = pd.DataFrame()

    # =====================================================
    # CLINICAL / DEMOGRAPHIC FEATURES
    # =====================================================

    clinical_features = [
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
        "Radiotherapy",
        "EORTC_baseline",
        "Perceived_risk_recurrence",
    ]

    for column in clinical_features:

        if column not in cppo_matched.columns:

            raise KeyError(
                f"Missing CPPO column: {column}"
            )

        dataset[column] = (
            cppo_matched[column].values
        )

    # =====================================================
    # BASELINE QLQ FEATURES
    # =====================================================

    # Keep PH_baseline as one of the baseline QoL variables.
    dataset["PH_baseline"] = (
        qlq_matched["PH_baseline"].values
    )

    # =====================================================
    # OUTCOME SOURCE VARIABLES
    # =====================================================

    dataset["Overall_EORTC_baseline"] = (
        qlq_matched["Overall_EORTC_baseline"].values
    )

    dataset["Overall_EORTC_end"] = (
        qlq_matched["Overall_EORTC_end"].values
    )

    dataset["Health_status_end"] = (
        cppo_matched["Health_status_end"].values
    )

    # =====================================================
    # OUTCOME 1: DROP
    # =====================================================

    dataset["y_drop_change"] = (
        dataset["Overall_EORTC_end"]
        - dataset["Overall_EORTC_baseline"]
    )

    dataset["y_drop"] = (
        dataset["y_drop_change"] <= -DROP_THRESHOLD
    ).astype(int)

    # =====================================================
    # OUTCOME 2: THRESHOLD
    # =====================================================

    dataset["y_threshold"] = (
        dataset["Health_status_end"]
        <= HEALTH_STATUS_THRESHOLD
    ).astype(int)

    # =====================================================
    # QLQ ID
    # =====================================================

    dataset.insert(
        0,
        "QLQ_ID",
        qlq_matched["ID"].values
    )

    # =====================================================
    # COLUMN ORDER
    # =====================================================

    column_order = [
        "QLQ_ID",

        # Demographics
        "Age",
        "Educational_level",
        "Occupational_status",
        "Children",
        "Marital_status",

        # Disease
        "TNM_stage",
        "HER2",
        "ECOG",

        # Treatment
        "Surgery",
        "Axillary_lymphadenectomy",
        "Chemotherapy_regimen",
        "Radiotherapy",

        # Baseline QoL
        "EORTC_baseline",
        "PH_baseline",
        "Perceived_risk_recurrence",

        # Outcome source variables
        "Overall_EORTC_baseline",
        "Overall_EORTC_end",
        "Health_status_end",

        # Outcome calculations
        "y_drop_change",
        "y_drop",
        "y_threshold",
    ]

    dataset = dataset[column_order]

    return dataset


# =========================================================
# REPORT
# =========================================================

def report(dataset, audit):

    print()
    print("=" * 70)
    print("FINAL DATASET AUDIT")
    print("=" * 70)

    print(
        f"Patients: {len(dataset)}"
    )

    print(
        f"Columns : {len(dataset.columns)}"
    )

    # -----------------------------------------------------
    # Outcome 1
    # -----------------------------------------------------

    print()
    print("-" * 70)
    print("OUTCOME 1: y_drop")
    print("-" * 70)

    print(
        "Definition:"
    )

    print(
        "Overall_EORTC_end - Overall_EORTC_baseline <= -10"
    )

    print()

    print(
        dataset["y_drop"]
        .value_counts()
        .sort_index()
    )

    total = len(dataset)

    for value, count in (
        dataset["y_drop"]
        .value_counts()
        .sort_index()
        .items()
    ):

        print(
            f"Class {value}: "
            f"{count} "
            f"({100 * count / total:.2f}%)"
        )

    # -----------------------------------------------------
    # Outcome 2
    # -----------------------------------------------------

    print()
    print("-" * 70)
    print("OUTCOME 2: y_threshold")
    print("-" * 70)

    print(
        "Definition:"
    )

    print(
        "Health_status_end <= 50"
    )

    print()

    print(
        dataset["y_threshold"]
        .value_counts()
        .sort_index()
    )

    for value, count in (
        dataset["y_threshold"]
        .value_counts()
        .sort_index()
        .items()
    ):

        print(
            f"Class {value}: "
            f"{count} "
            f"({100 * count / total:.2f}%)"
        )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print()
    print("-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = dataset.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")

    else:
        print(missing)

    # -----------------------------------------------------
    # Continuous change
    # -----------------------------------------------------

    print()
    print("-" * 70)
    print("OVERALL EORTC CHANGE")
    print("-" * 70)

    print(
        dataset["y_drop_change"].describe()
    )


# =========================================================
# MAIN
# =========================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load
    scores, qlq, cppo = load_data()

    # Clean column names
    scores = clean_columns(scores)
    qlq = clean_columns(qlq)
    cppo = clean_columns(cppo)

    # Prepare matching
    qlq, cppo = prepare_matching(
        qlq,
        cppo
    )

    # Match
    matches, audit = match_patients(
        qlq,
        cppo
    )

    # Save audit
    audit.to_csv(
        AUDIT_PATH,
        index=False
    )

    print()
    print(
        f"Linkage audit saved to:\n{AUDIT_PATH}"
    )

    # Build dataset
    dataset = build_dataset(
        cppo,
        qlq,
        matches
    )

    # Report
    report(
        dataset,
        audit
    )

    # Save
    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )

    print()
    print(
        "Dataset shape:",
        dataset.shape
    )


if __name__ == "__main__":
    main()