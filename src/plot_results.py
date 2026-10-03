"""Figures + summary table from Experiment 1 results.

    python src/plot_results.py
    python src/plot_results.py --results results/exp1_results.csv --oof results/oof --out results/figures

Outputs (in --out):
    exp1_roc_curves.png      ROC curves of the top-3 models per outcome (pooled out-of-fold predictions)
    exp1_model_comparison.png  AUROC (mean +/- SD over outer folds) per model, M1 vs M2
    exp1_summary_table.csv / .md   model x (outcome, variant) table for the write-up
Also prints the best model per outcome (use it as the base for the ablation and predict.py).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ALIASES = {
    "model": ["model", "model_name", "name", "estimator"],
    "variant": ["variant", "feature_set", "features", "preprocessing", "prep"],
    "outcome": ["outcome", "target", "label"],
    "auc_mean": ["auc_mean", "auroc_mean", "auroc", "auc", "mean_auc", "roc_auc"],
    "auc_sd": ["auc_sd", "auroc_sd", "auc_std", "std", "sd"],
}
LABELS = {"lasso": "LASSO", "ridge": "Ridge", "elasticnet": "Elastic Net", "knn": "KNN",
          "random_forest": "Random Forest", "boosting": "Boosting", "svm": "SVM",
          "mlp": "Neural Net (MLP)", "ensemble": "Ensemble Voting"}
OUTCOME_TITLES = {"y_drop": "Drop outcome (QoL fell >= 10 pts)",
                  "y_threshold": "Threshold outcome (end health status <= 50)"}


def load_results(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    for std, cands in ALIASES.items():
        hit = next((c for c in cands if c in df.columns), None)
        if hit is None and std != "auc_sd":
            sys.exit(f"Could not find a '{std}' column in {path}. Columns present: {list(df.columns)}\n"
                     f"-> send this list to your assistant/teammate so the aliases can be extended.")
        if hit:
            rename[hit] = std
    df = df.rename(columns=rename)
    if "auc_sd" not in df.columns:
        df["auc_sd"] = np.nan
    before = len(df)
    df = df.drop_duplicates(["model", "variant", "outcome"], keep="last")
    if len(df) < before:
        print(f"[note] dropped {before - len(df)} duplicate rows (kept the latest run of each combination)")
    return df


def load_oof(oof_dir, model, variant, outcome):
    path = os.path.join(oof_dir, f"{model}_{variant}_{outcome}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    ky = next((k for k in ("y", "y_true") if k in z.files), None)
    kp = next((k for k in ("p", "proba", "oof_proba", "y_prob", "y_score") if k in z.files), None)
    if ky is None or kp is None:
        print(f"[warn] unrecognised keys in {path}: {z.files}")
        return None
    return z[ky], z[kp]


def roc_figure(df, oof_dir, out):
    outcomes = [o for o in OUTCOME_TITLES if o in set(df["outcome"])] or sorted(df["outcome"].unique())
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 5), squeeze=False)
    for ax, outcome in zip(axes[0], outcomes):
        top = df[df["outcome"] == outcome].sort_values("auc_mean", ascending=False).head(3)
        for rank, (_, r) in enumerate(top.iterrows()):
            data = load_oof(oof_dir, r["model"], r["variant"], outcome)
            if data is None:
                print(f"[warn] no OOF file for {r['model']}/{r['variant']}/{outcome}; skipped in ROC plot")
                continue
            y, p = data
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, lw=2.5 if rank == 0 else 1.5,
                    label=f"{LABELS.get(r['model'], r['model'])} ({r['variant']}), AUC={roc_auc_score(y, p):.3f}")
        ax.plot([0, 1], [0, 1], "r--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(OUTCOME_TITLES.get(outcome, outcome), fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("ROC curves, top-3 models (pooled out-of-fold predictions)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exp1_roc_curves.png"), dpi=200)
    plt.close(fig)


def comparison_figure(df, out):
    outcomes = [o for o in OUTCOME_TITLES if o in set(df["outcome"])] or sorted(df["outcome"].unique())
    variants = sorted(df["variant"].unique())
    fig, axes = plt.subplots(1, len(outcomes), figsize=(7 * len(outcomes), 5), squeeze=False)
    for ax, outcome in zip(axes[0], outcomes):
        d = df[df["outcome"] == outcome]
        order = d.groupby("model")["auc_mean"].mean().sort_values(ascending=False).index.tolist()
        w = 0.8 / max(len(variants), 1)
        for j, v in enumerate(variants):
            dv = d[d["variant"] == v].set_index("model").reindex(order)
            ax.bar(np.arange(len(order)) + j * w, dv["auc_mean"], w, yerr=dv["auc_sd"],
                   capsize=3, label=v)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xticks(np.arange(len(order)) + w * (len(variants) - 1) / 2)
        ax.set_xticklabels([LABELS.get(m, m) for m in order], rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUROC (mean +/- SD over outer folds)")
        ax.set_title(OUTCOME_TITLES.get(outcome, outcome), fontsize=10)
        ax.legend(title="Feature matrix")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Experiment 1: model comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exp1_model_comparison.png"), dpi=200)
    plt.close(fig)


def summary_table(df, out):
    df = df.copy()
    sd = df["auc_sd"].fillna(0)
    df["cell"] = df["auc_mean"].map("{:.3f}".format) + " ± " + sd.map("{:.3f}".format)
    tab = df.pivot_table(index="model", columns=["outcome", "variant"], values="cell", aggfunc="first")
    tab.index = [LABELS.get(m, m) for m in tab.index]
    tab.to_csv(os.path.join(out, "exp1_summary_table.csv"))
    cols = [f"{o} / {v}" for o, v in tab.columns]
    lines = ["| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in tab.iterrows():
        lines.append(f"| {name} | " + " | ".join("" if pd.isna(x) else x for x in row) + " |")
    with open(os.path.join(out, "exp1_summary_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/exp1_results.csv")
    ap.add_argument("--oof", default="results/oof")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_results(a.results)
    n_models = df["model"].nunique()
    print(f"{len(df)} result rows | {n_models} models | variants {sorted(df['variant'].unique())} "
          f"| outcomes {sorted(df['outcome'].unique())}")
    if "ensemble" not in set(df["model"]):
        print("[warn] no 'ensemble' rows found - the paper's best model is missing from Experiment 1")
    roc_figure(df, a.oof, a.out)
    comparison_figure(df, a.out)
    summary_table(df, a.out)
    print("\nBest model per outcome (by mean outer-fold AUROC):")
    for outcome, d in df.groupby("outcome"):
        b = d.sort_values("auc_mean", ascending=False).iloc[0]
        print(f"  {outcome:<12} {b['model']} / {b['variant']}  AUROC {b['auc_mean']:.3f}")
    print(f"\nSaved figures + table to {a.out}/")


if __name__ == "__main__":
    main()
