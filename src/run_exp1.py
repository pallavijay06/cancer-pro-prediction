"""Experiment 1: compare models with nested CV.

Real data (once your teammate's features.py exists):
    python src/run_exp1.py --variant M1 --outcome y_drop
    python src/run_exp1.py --variant M2 --outcome y_threshold --models lasso ridge ensemble
Test with the stand-in dataset:
    python src/run_exp1.py --dummy --models lasso ridge --repeats 1
Run from the repo root. Results are appended to results/exp1_results.csv
and out-of-fold probabilities saved to results/oof/ (for ROC curves).
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import bootstrap_auc_ci, nested_cv          # noqa: E402
from models import MODEL_NAMES, get_model                  # noqa: E402

warnings.filterwarnings("ignore")


def stub_preprocessor():
    """Stand-in for features.get_preprocessor() (dummy data only)."""
    from sklearn.compose import ColumnTransformer, make_column_selector as sel
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), sel(dtype_include="number")),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]),
         sel(dtype_exclude="number")),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="M1", choices=["M1", "M2"])
    ap.add_argument("--outcome", default="y_drop", choices=["y_drop", "y_threshold"])
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dummy", action="store_true", help="use dummy data + stub preprocessor")
    a = ap.parse_args()

    if a.dummy:
        df = pd.read_csv("data/processed/dummy_dataset.csv")
        y, X = df[a.outcome], df.drop(columns=["y_drop", "y_threshold"])
        prep = stub_preprocessor()
    else:
        from features import get_preprocessor, load_dataset   # teammate's module
        X, y = load_dataset(outcome=a.outcome)
        prep = get_preprocessor(variant=a.variant)

    os.makedirs("results/oof", exist_ok=True)
    rows = []
    for name in a.models:
        est, grid = get_model(name, prep)
        res = nested_cv(est, grid, X, y, n_repeats=a.repeats)
        lo, hi = bootstrap_auc_ci(res["y"], res["oof_proba"])
        print(f"{name:<14} AUROC {res['auc_mean']:.3f} +/- {res['auc_sd']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  PR-AUC {res['ap_mean']:.3f}", flush=True)
        rows.append(dict(model=name, variant="dummy" if a.dummy else a.variant,
                         outcome=a.outcome, auc_mean=res["auc_mean"], auc_sd=res["auc_sd"],
                         ci_lo=lo, ci_hi=hi, ap_mean=res["ap_mean"], n=len(y)))
        tag = f"{name}_{'dummy' if a.dummy else a.variant}_{a.outcome}"
        np.savez(f"results/oof/{tag}.npz", y=res["y"], p=res["oof_proba"])

    out = "results/exp1_results.csv"
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()
