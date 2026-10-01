from pathlib import Path

import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CPPO_PATH = RAW_DATA_DIR / "breast_cppo.xlsx"
QLQ_PATH = RAW_DATA_DIR / "breast_qlq_c30.xlsx"
SCORES_PATH = RAW_DATA_DIR / "breast_scores.xlsx"

MATCH_PATH = (
    PROCESSED_DATA_DIR /
    "cppo_qlq_match_candidates.csv"
)


# =========================================================
# HELPERS
# =========================================================

def normalize_surgery(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    mapping = {
        "BCS": "BSC",
        "BSC": "BSC",
        "MASTECTOMY": "MASTECTOMY",
    }

    return mapping.get(
        value,
        value
    )


def num(value):

    if pd.isna(value):
        return np.nan

    try:
        return round(
            float(value),
            2
        )
    except:
        return np.nan


# =========================================================
# LOAD
# =========================================================

cppo = pd.read_excel(
    CPPO_PATH
)

qlq = pd.read_excel(
    QLQ_PATH
)

scores = pd.read_excel(
    SCORES_PATH
)


# =========================================================
# NORMALIZE
# =========================================================

cppo["Surgery_norm"] = (
    cppo["Surgery"].apply(
        normalize_surgery
    )
)

qlq["Surgery_norm"] = (
    qlq["Surgery"].apply(
        normalize_surgery
    )
)

scores["Surgery_norm"] = (
    scores["SURGERY"].apply(
        normalize_surgery
    )
)


# =========================================================
# NUMERIC MATCHING COLUMNS
# =========================================================

cppo["baseline_match"] = (
    cppo["EORTC_baseline"].apply(num)
)

cppo["health_end_match"] = (
    cppo["Health_status_end"].apply(num)
)

qlq["baseline_match"] = (
    qlq["Overall_EORTC_baseline"].apply(num)
)

qlq["health_end_match"] = (
    qlq["HE_end"].apply(num)
)


# =========================================================
# LOAD EXISTING CANDIDATES
# =========================================================

candidates = pd.read_csv(
    MATCH_PATH
)


# =========================================================
# PROBLEMATIC ROWS
# =========================================================

problematic = candidates[
    candidates["num_candidates"] != 1
].copy()


print("\n")
print("=" * 80)
print("PROBLEMATIC MATCHES")
print("=" * 80)

print(
    f"Total problematic CPPO rows: {len(problematic)}"
)


# =========================================================
# SCORE MATCHING FUNCTION
# =========================================================

score_baseline_cols = [
    "HE_BASELINE",
    "PH_BASELINE",
    "RO_BASELINE",
    "EF_BASELINE",
    "COG_BASELINE",
    "SO_BASELINE",
]

qlq_baseline_cols = [
    "HE_baseline",
    "PH_baseline",
    "RO_baseline",
    "EF_baseline",
    "COG_baseline",
    "SO_baseline",
]

score_end_cols = [
    "HE6",
    "PH6",
    "RO6",
    "EF6",
    "SO6",
    "COG6",
]

qlq_end_cols = [
    "HE_end",
    "PH_end",
    "RO_end",
    "EF_end",
    "SO_end",
    "COG_end",
]


def score_matches_qlq(
    qlq_row,
    score_row
):

    # -----------------------------------------------------
    # Surgery
    # -----------------------------------------------------

    if (
        qlq_row["Surgery_norm"]
        != score_row["Surgery_norm"]
    ):
        return False

    # -----------------------------------------------------
    # Baseline six dimensions
    # -----------------------------------------------------

    for q_col, s_col in zip(
        qlq_baseline_cols,
        score_baseline_cols
    ):

        q = num(
            qlq_row[q_col]
        )

        s = num(
            score_row[s_col]
        )

        if pd.isna(q) or pd.isna(s):
            return False

        if q != s:
            return False

    # -----------------------------------------------------
    # End six dimensions
    # -----------------------------------------------------

    for q_col, s_col in zip(
        qlq_end_cols,
        score_end_cols
    ):

        q = num(
            qlq_row[q_col]
        )

        s = num(
            score_row[s_col]
        )

        if pd.isna(q) or pd.isna(s):
            return False

        if q != s:
            return False

    return True


# =========================================================
# DIAGNOSE EACH PROBLEM
# =========================================================

diagnostic_rows = []


for _, problem in problematic.iterrows():

    cppo_idx = int(
        problem["cppo_index"]
    )

    candidate_indices = eval(
        problem["candidate_indices"]
    )

    cppo_row = cppo.iloc[
        cppo_idx
    ]

    print("\n" + "-" * 80)

    print(
        f"CPPO index: {cppo_idx}"
    )

    print(
        "CPPO values:"
    )

    print(
        "  Surgery:",
        cppo_row["Surgery"]
    )

    print(
        "  EORTC baseline:",
        cppo_row["EORTC_baseline"]
    )

    print(
        "  Health status end:",
        cppo_row["Health_status_end"]
    )

    print(
        "\nQLQ candidates:"
    )

    for q_idx in candidate_indices:

        q_row = qlq.loc[
            q_idx
        ]

        print(
            f"\n  QLQ index: {q_idx}"
        )

        print(
            f"    ID: {q_row['ID']}"
        )

        print(
            f"    Surgery: {q_row['Surgery']}"
        )

        print(
            f"    Overall baseline: "
            f"{q_row['Overall_EORTC_baseline']}"
        )

        print(
            f"    HE_end: {q_row['HE_end']}"
        )

        # -------------------------------------------------
        # Check score dataset
        # -------------------------------------------------

        matching_score_indices = []

        for s_idx, s_row in scores.iterrows():

            if score_matches_qlq(
                q_row,
                s_row
            ):
                matching_score_indices.append(
                    s_idx
                )

        print(
            f"    Matching score rows: "
            f"{matching_score_indices}"
        )

        diagnostic_rows.append({
            "cppo_index": cppo_idx,
            "qlq_index": q_idx,
            "qlq_id": q_row["ID"],
            "matching_score_indices":
                str(matching_score_indices),
        })


# =========================================================
# SAVE DIAGNOSTIC FILE
# =========================================================

diagnostic_df = pd.DataFrame(
    diagnostic_rows
)

output_path = (
    PROCESSED_DATA_DIR /
    "problematic_match_diagnostics.csv"
)

diagnostic_df.to_csv(
    output_path,
    index=False
)


print("\n")
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print(
    f"Saved to:\n{output_path}"
)