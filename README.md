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