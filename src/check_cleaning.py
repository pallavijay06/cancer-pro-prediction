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

print("=" * 70)
print("DATASET CLEANING INSPECTION")
print("=" * 70)

print(f"Shape: {df.shape}")

# =========================================================
# COLUMNS
# =========================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print(repr(column))


# =========================================================
# CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    "Educational_level",
    "Occupational_status",
    "Marital_status",
    "TNM_stage",
    "HER2",
    "ECOG",
    "Surgery",
    "Axillary_lymphadenectomy",
    "Chemotherapy_regimen",
    "Radiotherapy",
    "Perceived_risk_recurrence",
    "Children",
]


for column in categorical_columns:

    if column not in df.columns:
        print(f"\nWARNING: {column} not found")
        continue

    print("\n" + "-" * 70)
    print(f"{column}")
    print("-" * 70)

    print("Unique values:")

    for value in df[column].drop_duplicates().tolist():
        print(f"  {repr(value)}")

    print("\nValue counts:")

    print(
        df[column]
        .value_counts(dropna=False)
    )


# =========================================================
# MISSING VALUES
# =========================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
)


# =========================================================
# WHITESPACE CHECK
# =========================================================

print("\n" + "=" * 70)
print("VALUES WITH LEADING/TRAILING WHITESPACE")
print("=" * 70)

found_whitespace = False

for column in categorical_columns:

    if column not in df.columns:
        continue

    values = df[column].dropna().astype(str)

    bad_values = values[
        values != values.str.strip()
    ].unique()

    if len(bad_values) > 0:

        found_whitespace = True

        print(f"\n{column}:")

        for value in bad_values:
            print(f"  {repr(value)}")


if not found_whitespace:
    print("No categorical values with leading/trailing whitespace.")


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)