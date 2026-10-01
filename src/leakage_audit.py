import pandas as pd
from pathlib import Path


# =========================================================
# PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset.csv"
)


# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(DATA_PATH)


# =========================================================
# FEATURES TO AUDIT
# =========================================================

features_to_audit = [
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
    "PH_baseline",
    "Perceived_risk_recurrence",
]


# =========================================================
# AUDIT TABLE
# =========================================================

audit = pd.DataFrame({
    "feature": features_to_audit
})


# These are deliberately left as manual-review fields.
# We should NOT invent the timing from the spreadsheet alone.

audit["available_before_or_at_treatment_start"] = "REVIEW"

audit["reason"] = ""

audit.loc[
    audit["feature"] == "Age",
    "reason"
] = "Demographic characteristic."

audit.loc[
    audit["feature"] == "Educational_level",
    "reason"
] = "Baseline demographic characteristic."

audit.loc[
    audit["feature"] == "Occupational_status",
    "reason"
] = "Baseline demographic characteristic."

audit.loc[
    audit["feature"] == "Children",
    "reason"
] = "Baseline demographic/family characteristic."

audit.loc[
    audit["feature"] == "Marital_status",
    "reason"
] = "Baseline demographic characteristic."

audit.loc[
    audit["feature"] == "TNM_stage",
    "reason"
] = "Disease staging information."

audit.loc[
    audit["feature"] == "HER2",
    "reason"
] = "Tumor biomarker information."

audit.loc[
    audit["feature"] == "ECOG",
    "reason"
] = "Clinical performance status."

audit.loc[
    audit["feature"] == "Surgery",
    "reason"
] = "Treatment-related variable; exact timing must be verified."

audit.loc[
    audit["feature"] == "Axillary_lymphadenectomy",
    "reason"
] = "Treatment/surgical variable; exact timing must be verified."

audit.loc[
    audit["feature"] == "Chemotherapy_regimen",
    "reason"
] = "Treatment assignment/regimen."

audit.loc[
    audit["feature"] == "Radiotherapy",
    "reason"
] = (
    "Requires special review: radiotherapy may occur after "
    "chemotherapy and could therefore violate the prediction-time constraint."
)

audit.loc[
    audit["feature"] == "EORTC_baseline",
    "reason"
] = "Baseline patient-reported outcome."

audit.loc[
    audit["feature"] == "PH_baseline",
    "reason"
] = "Baseline QLQ-C30 physical functioning score."

audit.loc[
    audit["feature"] == "Perceived_risk_recurrence",
    "reason"
] = (
    "Requires review to determine when the risk perception "
    "was recorded relative to treatment."
)


# =========================================================
# PRINT
# =========================================================

print("=" * 80)
print("LEAKAGE AUDIT")
print("=" * 80)

print(
    audit.to_string(index=False)
)


# =========================================================
# EXPLICIT LEAKAGE CHECK
# =========================================================

print()
print("=" * 80)
print("OUTCOME / END-OF-TREATMENT VARIABLES")
print("=" * 80)

outcome_columns = [
    "Overall_EORTC_end",
    "Health_status_end",
    "y_drop_change",
    "y_drop",
    "y_threshold",
]

for column in outcome_columns:

    if column in df.columns:

        print(f"FOUND: {column}")

print()
print(
    "These columns must NOT be supplied as model features."
)


# =========================================================
# SAVE
# =========================================================

output_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leakage_audit.csv"
)

audit.to_csv(
    output_path,
    index=False
)

print()
print("=" * 80)
print("AUDIT SAVED")
print("=" * 80)

print(output_path)