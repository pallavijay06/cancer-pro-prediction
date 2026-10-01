from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

CPPO_PATH = RAW / "breast_cppo.xlsx"
QLQ_PATH = RAW / "breast_qlq_c30.xlsx"


# =========================================================
# HELPERS
# =========================================================

def surgery_normalize(x):

    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()

    mapping = {
        "BCS": "BSC",
        "BSC": "BSC",
        "MASTECTOMY": "MASTECTOMY",
    }

    return mapping.get(x, x)


def numeric(x):

    if pd.isna(x):
        return np.nan

    try:
        return float(x)
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


# =========================================================
# NORMALIZE
# =========================================================

cppo["Surgery_norm"] = (
    cppo["Surgery"].apply(
        surgery_normalize
    )
)

qlq["Surgery_norm"] = (
    qlq["Surgery"].apply(
        surgery_normalize
    )
)


# =========================================================
# LOAD CURRENT MATCH RESULTS
# =========================================================

matches = pd.read_csv(
    PROCESSED /
    "cppo_qlq_match_candidates.csv"
)

unmatched = matches[
    matches["num_candidates"] == 0
]


# =========================================================
# FIND NEAREST QLQ RECORDS
# =========================================================

results = []


for _, m in unmatched.iterrows():

    cppo_index = int(
        m["cppo_index"]
    )

    row = cppo.iloc[
        cppo_index
    ]

    surgery = row[
        "Surgery_norm"
    ]

    cppo_baseline = numeric(
        row["EORTC_baseline"]
    )

    cppo_end = numeric(
        row["Health_status_end"]
    )

    # -----------------------------------------------------
    # Restrict candidates to same surgery
    # -----------------------------------------------------

    candidates = qlq[
        qlq["Surgery_norm"] == surgery
    ].copy()

    if candidates.empty:

        print(
            f"\nCPPO {cppo_index}: "
            f"NO QLQ RECORD WITH SAME SURGERY"
        )

        continue

    # -----------------------------------------------------
    # Calculate differences
    # -----------------------------------------------------

    candidates["baseline_diff"] = (
        candidates[
            "Overall_EORTC_baseline"
        ].apply(numeric)
        - cppo_baseline
    ).abs()

    candidates["end_diff"] = (
        candidates[
            "HE_end"
        ].apply(numeric)
        - cppo_end
    ).abs()

    candidates["total_diff"] = (
        candidates["baseline_diff"]
        +
        candidates["end_diff"]
    )

    # -----------------------------------------------------
    # Sort by closeness
    # -----------------------------------------------------

    candidates = candidates.sort_values(
        "total_diff"
    )

    best = candidates.head(3)

    print("\n" + "=" * 80)

    print(
        f"CPPO INDEX: {cppo_index}"
    )

    print(
        f"CPPO surgery       : {row['Surgery']}"
    )

    print(
        f"CPPO baseline      : {cppo_baseline}"
    )

    print(
        f"CPPO health end    : {cppo_end}"
    )

    print("\nClosest QLQ records:")

    for q_index, q in best.iterrows():

        print(
            f"\nQLQ index: {q_index}"
        )

        print(
            f"QLQ ID: {q['ID']}"
        )

        print(
            f"QLQ surgery: {q['Surgery']}"
        )

        print(
            f"QLQ baseline: "
            f"{q['Overall_EORTC_baseline']}"
        )

        print(
            f"QLQ HE_end: "
            f"{q['HE_end']}"
        )

        print(
            f"Baseline difference: "
            f"{q['baseline_diff']:.4f}"
        )

        print(
            f"End difference: "
            f"{q['end_diff']:.4f}"
        )

        print(
            f"Total difference: "
            f"{q['total_diff']:.4f}"
        )

        results.append({
            "cppo_index": cppo_index,
            "qlq_index": q_index,
            "qlq_id": q["ID"],
            "cppo_baseline": cppo_baseline,
            "qlq_baseline": q[
                "Overall_EORTC_baseline"
            ],
            "baseline_diff": q[
                "baseline_diff"
            ],
            "cppo_end": cppo_end,
            "qlq_end": q[
                "HE_end"
            ],
            "end_diff": q[
                "end_diff"
            ],
            "total_diff": q[
                "total_diff"
            ],
        })


# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(
    results
)

output = (
    PROCESSED /
    "unmatched_nearest_qlq.csv"
)

results_df.to_csv(
    output,
    index=False
)

print("\n" + "=" * 80)

print(
    "UNMATCHED INVESTIGATION COMPLETE"
)

print(
    f"Saved to:\n{output}"
)