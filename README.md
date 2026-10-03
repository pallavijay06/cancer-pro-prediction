# Predicting Decline in Patient-Reported Outcomes for Cancer Patients on Chemotherapy

## 1. Project Overview

This project investigates whether machine learning can be used to predict a decline in patient-reported outcomes (PROs) among breast cancer patients receiving chemotherapy.

The project follows a two-stage experimental framework:

- **Experiment 1:** Compare multiple machine learning models across two preprocessing matrices and two prediction outcomes.
- **Experiment 2:** Perform feature-group ablation on the best-performing Experiment 1 configuration.

The project emphasizes reproducible data preparation, leakage-aware preprocessing, patient-level feature groups, and cross-validation-based model evaluation.

---

## 2. Dataset

The project uses three Excel files:

- `breast_qlq_c30.xlsx`
- `breast_cppo.xlsx`
- `breast_scores.xlsx`

The QLQ-C30 file contains patient-reported quality-of-life measurements, while the CPPO file contains clinical, demographic, treatment, and related patient information.

`breast_scores.xlsx` contains overlapping QLQ-related measurements and is therefore not required for the final modeling dataset.

### Patient Linkage

The CPPO and QLQ records were linked using a conservative matching procedure based on:

- normalized surgery information
- baseline EORTC score
- end health-status / corresponding QLQ information

Only **unique matches** were accepted.

The final linkage produced:

| Category | Patients |
|---|---:|
| Total CPPO patients | 219 |
| Unique matches | 186 |
| Ambiguous matches | 16 |
| Unmatched | 17 |

Ambiguous and unmatched records were not forced into the dataset in order to avoid introducing potentially incorrect patient linkages.

A linkage audit is saved in:

```text
data/processed/linkage_audit.csv
```

## Modeling & Evaluation (Part B)

Predicts whether a breast cancer patient's quality of life declines during treatment, using only
baseline information (adapted from Ostberg & Peterson, CS229 2020).

### Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src                                   # Windows PowerShell: $env:PYTHONPATH="src"
```
macOS note: LightGBM needs `brew install libomp`. Without it the code automatically falls back to
scikit-learn's GradientBoosting.

### Run
```bash
# Experiment 1: 9 models x {M1, M2} x {y_drop, y_threshold}   (nested CV, AUROC)
for v in M1 M2; do for o in y_drop y_threshold; do
  python src/run_exp1.py --variant $v --outcome $o --repeats 3
done; done
python src/plot_results.py            # figures + summary table -> results/figures/

# Experiment 2: feature-group ablation  (see Part A section for arguments)
python src/run_exp2.py

# Demo: train the chosen model on all data, then predict
python src/predict.py train --model lasso --variant M1 --outcome y_drop
python src/predict.py predict --example 5
python src/predict.py predict --template        # writes patient_template.json to fill in
```

### Method in brief
- Outer: repeated stratified 5-fold CV (3 repeats). Inner: 5-fold grid search. All preprocessing
  (imputation, scaling, encoding) lives inside the sklearn Pipeline, so it is re-fit in every fold.
- Models: LASSO, Ridge, Elastic Net, KNN, Random Forest, Gradient Boosting (LightGBM), SVM, MLP,
  and a soft-voting ensemble of six tuned members (3 logistic, RF, boosting, SVM).
- Metrics: AUROC (mean +/- SD over outer folds; bootstrap 95% CI on pooled out-of-fold predictions), PR-AUC.
- Seed 42 everywhere. Educational project, not a clinical tool.
