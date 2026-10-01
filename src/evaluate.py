"""Evaluation utilities: nested repeated CV, AUROC / PR-AUC, bootstrap CI.

Protocol (frozen at CP0):
  outer: RepeatedStratifiedKFold (5 folds x 3 repeats) -> unbiased performance
  inner: StratifiedKFold (5 folds) + GridSearchCV      -> hyperparameter tuning
Imputation / scaling live INSIDE the estimator (sklearn Pipeline), so they are
re-fit on the training part of every fold and nothing leaks from the test fold.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)

SEED = 42


def nested_cv(estimator, param_grid, X, y, n_splits=5, n_repeats=3,
              inner_splits=5, seed=SEED, n_jobs=-1):
    """Run nested CV. X must be a DataFrame, y a 1-D array/Series of 0/1.

    Returns a dict with per-fold scores, their mean/SD, and out-of-fold
    probabilities averaged over repeats (use these for ROC curves / bootstrap CI).
    """
    y = np.asarray(y).astype(int)
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=seed)
    fold_auc, fold_ap, best_params = [], [], []
    oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))

    for i, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + i)
        search = GridSearchCV(clone(estimator), param_grid, scoring="roc_auc",
                              cv=inner, n_jobs=n_jobs, refit=True)
        search.fit(X.iloc[tr], y[tr])
        p = search.predict_proba(X.iloc[te])[:, 1]
        fold_auc.append(roc_auc_score(y[te], p))
        fold_ap.append(average_precision_score(y[te], p))
        best_params.append(search.best_params_)
        oof_sum[te] += p
        oof_cnt[te] += 1

    oof = oof_sum / oof_cnt
    return {
        "auc_mean": float(np.mean(fold_auc)), "auc_sd": float(np.std(fold_auc)),
        "ap_mean": float(np.mean(fold_ap)), "ap_sd": float(np.std(fold_ap)),
        "fold_auc": fold_auc, "best_params": best_params,
        "oof_proba": oof, "y": y,
    }


def bootstrap_auc_ci(y, p, n_boot=2000, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for AUROC over patients."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(name, res):
    """One-line summary + bootstrap CI on the pooled out-of-fold predictions."""
    lo, hi = bootstrap_auc_ci(res["y"], res["oof_proba"])
    return (f"{name:<14} AUROC {res['auc_mean']:.3f} +/- {res['auc_sd']:.3f} "
            f"| OOF 95% CI [{lo:.3f}, {hi:.3f}] | PR-AUC {res['ap_mean']:.3f}")
