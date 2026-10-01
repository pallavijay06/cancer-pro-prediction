"""
Experiment 2: Feature-group ablation with bootstrap AUROC 95% CI.

Experiment 2 is run on the best configuration selected from Experiment 1:

    best model + best outcome + best preprocessing matrix

Ablation conditions:
    1. remove_demographics
    2. remove_disease
    3. remove_treatment
    4. remove_psychological
    5. remove_baseline_qol
    6. baseline_qol_only
    7. everything_except_baseline_qol

Run from the repository root.

Example:
    python src/run_exp2.py \
        --model lasso \
        --outcome y_drop \
        --variant M1

A smaller smoke test can be run with:
    python src/run_exp2.py \
        --model lasso \
        --outcome y_drop \
        --variant M1 \
        --repeats 1 \
        --smoke

Results are saved to:
    results/exp2_ablation_results.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Make imports work when running:
#     python src/run_exp2.py
# ---------------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from evaluate import nested_cv, bootstrap_auc_ci
from models import get_model
from features import load_dataset, get_preprocessor, get_feature_groups


SEED = 42


# ---------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------

ABLATION_NAMES = [
    "remove_demographics",
    "remove_disease",
    "remove_treatment",
    "remove_psychological",
    "remove_baseline_qol",
    "baseline_qol_only",
    "everything_except_baseline_qol",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_ablation_features(X, feature_groups, ablation_name):
    """
    Return a DataFrame containing the raw features required for one
    Experiment 2 ablation condition.

    Features are removed BEFORE preprocessing.

    This is important because preprocessing must only see the feature
    subset used by that ablation condition.
    """

    all_features = []

    for group_features in feature_groups.values():
        all_features.extend(group_features)

    # Preserve original feature order and remove duplicates.
    all_features = list(dict.fromkeys(all_features))

    # Verify that all expected features exist.
    missing = [col for col in all_features if col not in X.columns]

    if missing:
        raise ValueError(
            "The following feature-group columns are missing from X:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    if ablation_name == "remove_demographics":
        remove = set(feature_groups["demographics"])

    elif ablation_name == "remove_disease":
        remove = set(feature_groups["disease"])

    elif ablation_name == "remove_treatment":
        remove = set(feature_groups["treatment"])

    elif ablation_name == "remove_psychological":
        remove = set(feature_groups["psychological"])

    elif ablation_name == "remove_baseline_qol":
        remove = set(feature_groups["baseline_qol"])

    elif ablation_name == "baseline_qol_only":
        selected = list(feature_groups["baseline_qol"])
        return X[selected].copy()

    elif ablation_name == "everything_except_baseline_qol":
        remove = set(feature_groups["baseline_qol"])

    else:
        raise ValueError(
            f"Unknown ablation condition: {ablation_name}"
        )

    selected = [col for col in all_features if col not in remove]

    return X[selected].copy()


def build_preprocessor_for_columns(
    variant,
    feature_columns,
    feature_groups,
):
    """
    Build M1/M2 preprocessing for the exact feature subset used by
    an ablation condition.

    We do not reuse a preprocessor configured for all 14 original
    features because an ablation must remove those features before
    preprocessing.
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Determine numeric and categorical columns from the actual
    # feature subset.

    # Numeric features in the current project.
    numeric_candidates = {
        "Age",
        "EORTC_baseline",
        "PH_baseline",
        "ECOG",
    }

    numeric_columns = [
        col for col in feature_columns
        if col in numeric_candidates
    ]

    categorical_columns = [
        col for col in feature_columns
        if col not in numeric_candidates
    ]

    transformers = []

    if numeric_columns:

        if variant == "M1":
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        KNNImputer()
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    ),
                ]
            )

        elif variant == "M2":
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    ),
                ]
            )

        else:
            raise ValueError(
                f"Unknown preprocessing variant: {variant}"
            )

        transformers.append(
            (
                "num",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    )
                ),
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def run_one_ablation(
    X,
    y,
    feature_groups,
    ablation_name,
    model_name,
    variant,
    repeats,
):
    """
    Run nested CV for one ablation configuration.
    """

    X_ablation = build_ablation_features(
        X,
        feature_groups,
        ablation_name,
    )

    feature_columns = list(X_ablation.columns)

    preprocessor = build_preprocessor_for_columns(
        variant=variant,
        feature_columns=feature_columns,
        feature_groups=feature_groups,
    )

    estimator, param_grid = get_model(
        model_name,
        preprocessor,
    )

    result = nested_cv(
        estimator,
        param_grid,
        X_ablation,
        y,
        n_splits=5,
        n_repeats=repeats,
        inner_splits=5,
        seed=SEED,
        n_jobs=-1,
    )

    ci_lo, ci_hi = bootstrap_auc_ci(
        result["y"],
        result["oof_proba"],
        n_boot=2000,
        alpha=0.05,
        seed=SEED,
    )

    return {
        "ablation": ablation_name,
        "model": model_name,
        "variant": variant,
        "outcome": "",
        "n_features": len(feature_columns),
        "features": feature_columns,
        "auroc_mean": result["auc_mean"],
        "auroc_sd": result["auc_sd"],
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "pr_auc_mean": result["ap_mean"],
        "pr_auc_sd": result["ap_sd"],
        "n_patients": len(y),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Experiment 2 feature-group ablation."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Best model selected from Experiment 1.",
    )

    parser.add_argument(
        "--outcome",
        required=True,
        choices=["y_drop", "y_threshold"],
        help="Best outcome selected from Experiment 1.",
    )

    parser.add_argument(
        "--variant",
        required=True,
        choices=["M1", "M2"],
        help="Best preprocessing matrix selected from Experiment 1.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of outer CV repeats. Default: 3.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only one ablation configuration for a quick test.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 2: FEATURE-GROUP ABLATION")
    print("=" * 70)

    print()
    print("Selected Experiment 1 configuration:")
    print(f"  Model   : {args.model}")
    print(f"  Outcome : {args.outcome}")
    print(f"  Variant : {args.variant}")
    print(f"  Repeats : {args.repeats}")

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    X, y = load_dataset(
        outcome=args.outcome
    )

    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    # ---------------------------------------------------------------
    # Load feature groups
    # ---------------------------------------------------------------

    feature_groups_raw = get_feature_groups()

    # get_feature_groups() returns:
    #     feature_name -> group_name
    #
    # Experiment 2 needs:
    #     group_name -> [feature_name, ...]

    feature_groups = {}

    for feature_name, group_name in feature_groups_raw.items():
        feature_groups.setdefault(group_name, []).append(feature_name)

    print()
    print("Feature groups:")

    for group_name, columns in feature_groups.items():
        print(
            f"  {group_name:<20} "
            f"{len(columns)} features"
        )

    # ---------------------------------------------------------------
    # Select ablations
    # ---------------------------------------------------------------

    if args.smoke:
        ablations = ["remove_demographics"]
        print()
        print("SMOKE TEST MODE")
        print("Running only:")
        print("  remove_demographics")
    else:
        ablations = ABLATION_NAMES

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        results_dir,
        "exp2_ablation_results.csv",
    )

    # ---------------------------------------------------------------
    # Run ablations
    # ---------------------------------------------------------------

    rows = []

    for i, ablation_name in enumerate(
        ablations,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"ABLATION {i}/{len(ablations)}: "
            f"{ablation_name}"
        )
        print("=" * 70)

        result = run_one_ablation(
            X=X,
            y=y,
            feature_groups=feature_groups,
            ablation_name=ablation_name,
            model_name=args.model,
            variant=args.variant,
            repeats=args.repeats,
        )

        result["outcome"] = args.outcome

        rows.append(result)

        print()
        print(
            f"Features : {result['n_features']}"
        )

        print(
            f"AUROC    : "
            f"{result['auroc_mean']:.4f} "
            f"+/- {result['auroc_sd']:.4f}"
        )

        print(
            f"95% CI   : "
            f"[{result['bootstrap_ci_low']:.4f}, "
            f"{result['bootstrap_ci_high']:.4f}]"
        )

        print(
            f"PR-AUC   : "
            f"{result['pr_auc_mean']:.4f} "
            f"+/- {result['pr_auc_sd']:.4f}"
        )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(rows)

    # Store feature lists as readable strings.
    results_df["features"] = results_df[
        "features"
    ].apply(
        lambda x: ", ".join(x)
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 70)

    print()
    print("Results:")
    print(
        results_df[
            [
                "ablation",
                "n_features",
                "auroc_mean",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "pr_auc_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved to:")
    print(output_path)


if __name__ == "__main__":
    main()