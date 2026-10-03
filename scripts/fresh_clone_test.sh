#!/usr/bin/env bash
# Phase 8: simulate a grader/TA cloning the repo and following the README.
#   bash scripts/fresh_clone_test.sh https://github.com/<owner>/cancer-pro-prediction.git [branch]
# Private repo: your normal git credentials (HTTPS login / SSH key) must work.
set -euo pipefail
REPO_URL="${1:?usage: bash scripts/fresh_clone_test.sh <repo-url> [branch]}"
BRANCH="${2:-main}"
WORK="$(mktemp -d)"
echo "== cloning $BRANCH into $WORK"
git clone --branch "$BRANCH" "$REPO_URL" "$WORK/repo"
cd "$WORK/repo"

echo "== creating venv + installing requirements"
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

echo "== files the demo needs"
for f in README.md requirements.txt src/features.py src/models.py src/evaluate.py \
         src/run_exp1.py src/run_exp2.py src/plot_results.py src/predict.py \
         data/processed/dataset.csv; do
  [ -f "$f" ] && echo "  ok      $f" || { echo "  MISSING $f"; MISSING=1; }
done
[ -z "${MISSING:-}" ] || { echo "FAIL: add the missing files to git (check .gitignore!)"; exit 1; }

export PYTHONPATH=src
echo "== data + preprocessors load"
python - <<'PY'
from features import load_dataset, get_preprocessor
X, y = load_dataset("y_drop")
print("rows", len(X), "| features", X.shape[1], "| positives", round(float(y.mean()), 3))
assert not [c for c in X.columns if c.lower().endswith("_end")], "leakage: _end column in features"
for v in ("M1", "M2"):
    get_preprocessor(v).fit_transform(X, y)
print("preprocessors M1/M2 OK")
PY

echo "== quick experiment (2 models, 1 repeat)"
python src/run_exp1.py --variant M1 --outcome y_drop --models lasso ridge --repeats 1

echo "== plots"
python src/plot_results.py
ls results/figures

echo "== demo"
python src/predict.py train --model lasso --variant M1 --outcome y_drop
python src/predict.py predict --example 0

echo
echo "PASS - a fresh clone works. (temp copy: $WORK/repo)"
