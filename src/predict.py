"""Demo: train the chosen model on all data, save it, and predict risk for a patient.

    python src/predict.py train --model ensemble --variant M1 --outcome y_drop
    python src/predict.py predict --example 5            # use patient #5 of the dataset
    python src/predict.py predict --template             # writes patient_template.json
    python src/predict.py predict --input patient.json   # fill in what you know; the rest = missing
Add --dummy to try everything on the stand-in dataset.

Training uses the same pipeline + grid as Experiment 1 (inner 5-fold tuning on ALL data).
The saved model is for the demo; honest performance numbers come from nested CV (Experiment 1).
Educational project - not a clinical tool.
"""
import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from models import get_model  # noqa: E402

MODEL_PATH = "models/best_model.joblib"


def load_data(outcome, variant, dummy):
    if dummy:
        from run_exp1 import stub_preprocessor
        df = pd.read_csv("data/processed/dummy_dataset.csv")
        return df.drop(columns=["y_drop", "y_threshold"]), df[outcome], stub_preprocessor()
    from features import get_preprocessor, load_dataset
    X, y = load_dataset(outcome=outcome)
    return X, y, get_preprocessor(variant=variant)


def youden_threshold(model, variant, outcome):
    """Decision threshold maximising sensitivity+specificity-1 on out-of-fold predictions."""
    f = f"results/oof/{model}_{variant}_{outcome}.npz"
    if not os.path.exists(f):
        return 0.5
    from sklearn.metrics import roc_curve
    z = np.load(f, allow_pickle=True)
    kp = next(k for k in ("p", "proba", "oof_proba", "y_prob") if k in z.files)
    ky = next(k for k in ("y", "y_true") if k in z.files)
    fpr, tpr, thr = roc_curve(z[ky], z[kp])
    return float(thr[np.argmax(tpr - fpr)])


def train(a):
    X, y, prep = load_data(a.outcome, a.variant, a.dummy)
    est, grid = get_model(a.model, prep)
    if grid:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)
        model, best = search.best_estimator_, search.best_params_
    else:
        model, best = est.fit(X, y), {}
    tag = "dummy" if a.dummy else a.variant
    bundle = dict(model=model, features=list(X.columns),
                  dtypes={c: str(t) for c, t in X.dtypes.items()},
                  outcome=a.outcome, variant=tag, model_name=a.model, best_params=best,
                  threshold=youden_threshold(a.model, tag, a.outcome),
                  prevalence=float(np.mean(y)), n_train=len(y))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Trained {a.model} ({tag}) for {a.outcome} on {len(y)} patients -> {MODEL_PATH}")
    print(f"Best params: {best}\nDecision threshold (Youden, from out-of-fold predictions): {bundle['threshold']:.3f}")


def to_frame(bundle, values):
    """Build a one-row DataFrame with the training columns/dtypes; unspecified features = NaN."""
    row = {}
    for c in bundle["features"]:
        v = values.get(c, None)
        if "float" in bundle["dtypes"][c] or "int" in bundle["dtypes"][c]:
            row[c] = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        else:
            row[c] = np.nan if v is None else str(v)
    df = pd.DataFrame([row], columns=bundle["features"])
    for c in bundle["features"]:
        if not ("float" in bundle["dtypes"][c] or "int" in bundle["dtypes"][c]):
            df[c] = df[c].astype(object)
    return df


def report(bundle, X_row, truth=None):
    p = float(bundle["model"].predict_proba(X_row)[0, 1])
    flag = "HIGH" if p >= bundle["threshold"] else "LOWER"
    what = {"y_drop": "quality-of-life drop of >= 10 points",
            "y_threshold": "end-of-treatment health status <= 50"}.get(bundle["outcome"], bundle["outcome"])
    print(f"\nModel: {bundle['model_name']} ({bundle['variant']}) | Outcome: {what}")
    print(f"Predicted probability: {p:.1%}   (cohort base rate {bundle['prevalence']:.1%})")
    print(f"Risk flag: {flag}  (threshold {bundle['threshold']:.2f})")
    if truth is not None:
        print(f"Actual outcome for this patient: {int(truth)}  [in-sample demo, not a performance estimate]")
    print("Educational project - not for clinical use.")


def predict(a):
    if not os.path.exists(MODEL_PATH):
        sys.exit("No saved model. Run: python src/predict.py train --model ... first")
    bundle = joblib.load(MODEL_PATH)
    if a.template:
        json.dump({c: None for c in bundle["features"]}, open("patient_template.json", "w"), indent=2)
        print("Wrote patient_template.json - fill in values (leave null for unknown) and use --input")
    elif a.input:
        report(bundle, to_frame(bundle, json.load(open(a.input))))
    elif a.example is not None:
        X, y, _ = load_data(bundle["outcome"], bundle["variant"], bundle["variant"] == "dummy")
        report(bundle, X.iloc[[a.example]], y.iloc[a.example])
    else:
        sys.exit("Use --example N, --input file.json or --template")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default="ensemble")
    t.add_argument("--variant", default="M1", choices=["M1", "M2"])
    t.add_argument("--outcome", default="y_drop", choices=["y_drop", "y_threshold"])
    t.add_argument("--dummy", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--example", type=int)
    p.add_argument("--input")
    p.add_argument("--template", action="store_true")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)


if __name__ == "__main__":
    main()
