"""The 9 models from the paper, each with a hyperparameter grid.

Usage:
    from models import get_model, MODEL_NAMES
    estimator, grid = get_model("lasso", preprocessor)     # preprocessor from features.py
    res = nested_cv(estimator, grid, X, y)                 # evaluate.py

For single models, `estimator` is Pipeline([prep, clf]) and grid keys are prefixed
with "clf__". For the ensemble, grid is {} because each member tunes itself
(see build_ensemble) -> nested_cv then skips the inner search.
"""
import warnings

import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

SEED = 42
_SK_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def _logreg(kind, **kw):
    """Version-robust logistic regression. 'lasso' | 'ridge' | 'elasticnet'.
    (scikit-learn >= 1.8 deprecates `penalty` in favour of `l1_ratio`.)"""
    base = dict(max_iter=5000, random_state=SEED)
    if kind == "ridge":                       # default penalty is L2
        return LogisticRegression(**base, **kw)
    if _SK_VERSION >= (1, 8):
        l1 = 1.0 if kind == "lasso" else 0.5
        return LogisticRegression(l1_ratio=l1, solver="saga", **base, **kw)
    penalty = "l1" if kind == "lasso" else "elasticnet"
    extra = {} if kind == "lasso" else {"l1_ratio": 0.5}
    return LogisticRegression(penalty=penalty, solver="saga", **base, **extra, **kw)


def _boosting():
    """LightGBM if installed (as in the paper), else sklearn GradientBoosting."""
    try:
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(n_estimators=200, min_child_samples=10, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8,
                             random_state=SEED, verbose=-1, n_jobs=1)
        grid = {"learning_rate": [0.01, 0.05, 0.1], "num_leaves": [4, 8, 16]}
    except ImportError:
        est = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
        grid = {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 4]}
    return est, grid


# name -> (raw classifier, grid WITHOUT pipeline prefix)
def _specs():
    gbm, gbm_grid = _boosting()
    C = [0.001, 0.01, 0.1, 1, 10]
    return {
        "lasso": (_logreg("lasso"), {"C": C}),
        "ridge": (_logreg("ridge"), {"C": C}),
        "elasticnet": (_logreg("elasticnet"),
                       {"C": C, "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
        "knn": (KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 11, 21, 31], "weights": ["uniform", "distance"]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
                          {"min_samples_split": [2, 5, 10, 20], "max_features": ["sqrt", None]}),
        "boosting": (gbm, gbm_grid),
        "svm": (SVC(probability=True, random_state=SEED),
                {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "mlp": (MLPClassifier(max_iter=1000, early_stopping=True, n_iter_no_change=10,
                              random_state=SEED),
                {"hidden_layer_sizes": [(32,), (64, 16)], "alpha": [1e-3, 1e-2, 1e-1, 1.0]}),
    }


# Ensemble members, as in the paper: 3 logistic + 2 tree-based + 1 SVM
ENSEMBLE_MEMBERS = ["lasso", "ridge", "elasticnet", "random_forest", "boosting", "svm"]
BASE_MODELS = list(_specs().keys())
MODEL_NAMES = BASE_MODELS + ["ensemble"]


def _pipe(clf, preprocessor):
    return Pipeline([("prep", clone(preprocessor)), ("clf", clf)])


def _prefixed(grid):
    return {f"clf__{k}": v for k, v in grid.items()}


def build_ensemble(preprocessor, inner_splits=5, seed=SEED):
    """Soft-voting ensemble: averages predicted probabilities of 6 tuned members.
    Each member is a GridSearchCV over its own full pipeline (preprocessing included),
    so tuning happens only on the training data it receives -> no leakage.
    Composition is fixed (no 'pick the best 3' step) -> no selection leakage."""
    specs = _specs()
    members = []
    for name in ENSEMBLE_MEMBERS:
        clf, grid = specs[name]
        cv = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
        members.append((name, GridSearchCV(_pipe(clf, preprocessor), _prefixed(grid),
                                           scoring="roc_auc", cv=cv, n_jobs=1)))
    return VotingClassifier(members, voting="soft", n_jobs=1)


def get_model(name, preprocessor):
    """Return (estimator, param_grid) ready for evaluate.nested_cv."""
    if name == "ensemble":
        return build_ensemble(preprocessor), {}
    clf, grid = _specs()[name]
    return _pipe(clf, preprocessor), _prefixed(grid)


if __name__ == "__main__":
    print("Models:", MODEL_NAMES)
