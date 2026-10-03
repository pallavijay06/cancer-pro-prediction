# Predicting a Decline in Patient-Reported Quality of Life for Breast Cancer Patients on Treatment
**UE24CS352A Machine Learning, Mini-Project** | Team: Pallavi J (PES1UG24CS313), Shristi Saha (PES1UG24CS903) | Repo: https://github.com/pallavijay06/cancer-pro-prediction

## 1. Problem statement
Chemotherapy can lower patients' quality of life (QoL) in ways clinicians cannot easily anticipate. Following
Ostberg & Peterson (CS229, 2020), who predicted declines in PROMIS physical-health scores for chemotherapy patients
using Stanford EHR data, we ask: **using only information available at the start of treatment, can we predict which
breast cancer patients will report a clinically meaningful decline in QoL?** This is a binary classification task with two
targets: **drop** (overall EORTC QLQ-C30 score falls by ≥ 10 points from baseline to end of follow-up) and
**threshold** (end-of-treatment global health status ≤ 50). Early identification could allow targeted supportive care.

## 2. Dataset
The original Stanford data is private, so we use a publicly available breast cancer cohort with the EORTC QLQ-C30
questionnaire. The dataset source/citation is documented separately in the project materials. The clinical file contains **219 CPPO patients**, while
the QLQ-C30 records were linked to the clinical records using a conservative patient-matching procedure based on
normalized surgery information, baseline EORTC score, and end health-status / corresponding QLQ information. Only
unique matches were accepted, producing an analysis cohort of **186 patients**.

The final modeling dataset contains **14 baseline features** across five groups: demographics, disease, treatment,
psychological, and baseline QoL scores. The drop target has **30.1% positives** and the threshold target has
**29.6% positives**. **Leakage control:** no end-of-treatment column is used as a feature. In particular,
`Overall_EORTC_end`, `Health_status_end`, and other outcome/source columns are retained only for target construction
or audit purposes and excluded from the predictor matrix.

## 3. Approach
We replicate the paper's methodology on this dataset. (i) *Preprocessing*: two feature matrices, **M1** (KNN
imputation) and **M2** (median imputation + missing-value indicators), mirroring the paper's two lab-featurization
schemes; one-hot encoding and scaling. (ii) *Models*: LASSO, Ridge and Elastic Net logistic regression, KNN, Random Forest, gradient boosting
(LightGBM when installed, otherwise scikit-learn GradientBoosting), SVM, a neural network (MLP), and a
**voting ensemble** averaging six tuned models (LASSO, Ridge, Elastic Net, Random Forest, boosting, SVM).
(iii) *Evaluation*: because the cohort is small, we use **nested, repeated stratified cross-validation** (outer
5-fold × 3 repeats for performance; inner 5-fold grid search for hyperparameters) with **AUROC** as the main metric,
plus PR-AUC and bootstrap 95% CIs. Unlike the paper, we never tune or early-stop on test data. (iv) *Ablation*:
we remove one feature group at a time from the best model.

## 4. Implementation overview
Python 3, scikit-learn **1.9.1**, and gradient boosting with the project's LightGBM/scikit-learn fallback.
`src/features.py` builds the dataset and preprocessors; all preprocessing is inside an sklearn `Pipeline`, so
imputation and scaling are re-fit within every fold. `src/models.py` defines the nine models and grids;
`src/evaluate.py` implements nested CV and bootstrap CIs; `run_exp1.py` / `run_exp2.py` run the experiments;
`plot_results.py` produces figures; `predict.py` trains the chosen model and scores a patient (demo).
Seed 42; everything is reproducible from the README.

## 5. Results
**Experiment 1 (model comparison).** Across the final nested-CV results, LASSO with either M1 or M2 gave the highest
AUROC for the drop target: **0.679 ± 0.058** (95% CI **[0.600, 0.758]**). For the threshold target, KNN with either
M1 or M2 gave the highest AUROC: **0.616 ± 0.102** (95% CI **[0.525, 0.705]**). M1 and M2 produced identical
AUROC values for all model/target combinations in the final results. The voting ensemble reached AUROC
**0.658 ± 0.048** for drop and **0.583 ± 0.073** for threshold, below the best single-model results; its CIs
overlap those of the corresponding best models. For drop, KNN and MLP had the lowest mean AUROC (0.620 and
0.615), while for threshold the ensemble and LASSO were among the lower-performing models (0.583 and 0.590).
Figure: `results/figures/exp1_roc_curves.png` (ROC curves, pooled out-of-fold predictions).

**Experiment 2 (ablation).** Using LASSO with M2 for the drop target, removing demographics increased AUROC
from the full-model value of **0.679 to 0.710**, while removing disease reduced it to **0.596**, removing treatment
to **0.611**, and removing psychological features to **0.672**. Removing baseline QoL produced the largest
decrease, from **0.679 to 0.590**. A baseline-QoL-only model achieved AUROC **0.607**, while removing baseline
QoL and retaining the other feature groups also gave **0.590**. Several ablation confidence intervals overlap
with the full-model interval, so these differences should be interpreted cautiously.

## 6. Conclusions
Using only baseline information, the best drop-target model achieved an AUROC of **0.679 ± 0.058**
(95% CI **[0.600, 0.758]**), while the best threshold-target model achieved **0.616 ± 0.102**
(95% CI **[0.525, 0.705]**). These values are below the paper's reported AUROCs of 0.77 for the drop
target and 0.75 for the threshold target. In our ablation study, removing baseline QoL features reduced
AUROC from **0.679 to 0.590**, while the baseline-QoL-only model achieved **0.607**, indicating that
baseline QoL features contain substantial predictive signal in this cohort. However, several confidence
intervals overlap, so differences between models should be interpreted cautiously.

**Limitations:** small analysis cohort (186 patients), a single cohort, targets defined by cutoffs, a
different questionnaire (QLQ-C30 rather than PROMIS), and confidence intervals that overlap between several
models. **Future work:** a larger and external validation cohort, calibration, and repeated ablations over
random feature subsets.

**Reference.** N. Ostberg and D. Peterson, "Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy," CS 229 Final Project, Stanford, 2020.
