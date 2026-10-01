import pandas as pd
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset.csv"
)


# =========================================================
# LOAD
# =========================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print(f"Input shape: {df.shape}")


# =========================================================
# 1. CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
)


# =========================================================
# 2. CLEAN STRING VALUES
# =========================================================

categorical_columns = [
    "Educational_level",
    "Occupational_status",
    "Children",
    "Marital_status",
    "TNM_stage",
    "HER2",
    "Surgery",
    "Axillary_lymphadenectomy",
    "Chemotherapy_regimen",
    "Radiotherapy",
    "Perceived_risk_recurrence",
]


for column in categorical_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )


# =========================================================
# 3. STANDARDIZE SURGERY
# =========================================================

df["Surgery"] = df["Surgery"].replace({
    "BSC": "BCS"
})


# =========================================================
# 4. STANDARDIZE CHILDREN
# =========================================================

# Convert all values to a consistent representation.
#
# 0 -> 0
# 1 -> 1
# 2 -> 2
# > 2 children -> 3+
#
# We keep the missing value as NaN.

children_mapping = {
    "0": "0",
    "1": "1",
    "2": "2",
    "> 2 children": "3+"
}

df["Children"] = (
    df["Children"]
    .replace(children_mapping)
)


# =========================================================
# 5. REPORT MISSING VALUES
# =========================================================

print()
print("=" * 70)
print("MISSING VALUES AFTER CLEANING")
print("=" * 70)

missing = df.isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")

else:
    print(missing)


# =========================================================
# 6. REPORT UNIQUE VALUES
# =========================================================

print()
print("=" * 70)
print("CLEANED CATEGORICAL VALUES")
print("=" * 70)

for column in categorical_columns:

    if column not in df.columns:
        continue

    print()
    print(f"{column}:")
    print(df[column].value_counts(dropna=False))


# =========================================================
# 7. CHECK FOR REMAINING WHITESPACE
# =========================================================

print()
print("=" * 70)
print("WHITESPACE CHECK")
print("=" * 70)

whitespace_found = False

for column in categorical_columns:

    if column not in df.columns:
        continue

    values = df[column].dropna().astype(str)

    bad_values = values[
        values != values.str.strip()
    ].unique()

    if len(bad_values) > 0:

        whitespace_found = True

        print(f"{column}:")
        for value in bad_values:
            print(repr(value))


if not whitespace_found:
    print("No leading/trailing whitespace found.")


# =========================================================
# 8. SAVE CLEAN DATASET
# =========================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(f"Output shape: {df.shape}")

print()
print("Saved to:")

print(OUTPUT_PATH)