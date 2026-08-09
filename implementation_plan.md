# Implementation Plan - Scientific Foundation Repair & Comprehensive Experiments (COMPLETED)

We have resolved all critical scientific and data integrity bottlenecks identified in the audit. This includes refactoring the preprocessing pipeline to eliminate data leakage, replacing the heuristic DP accountant with a mathematically rigorous Rényi Differential Privacy (RDP) composition algorithm, renaming the MSKCC synthetic evaluation to a controlled covariate/concept shift simulation, re-running all baselines under leakage-free splits, and executing a comprehensive experimental suite.

---

## Completed Proposed Changes

### 1. Data Preprocessing & Leakage Fix
*   **Modified** [`src/preprocessing.py`](file:///D:/Mini_project_JP/src/preprocessing.py): Separated fit/transform logic. Preprocessors now accept fitted states to transform test/shifted cohorts.
*   **Modified** [`src/app.py`](file:///D:/Mini_project_JP/src/app.py): Partitioned raw datasets into train/test splits *before* running scaling or PCA.

### 2. Rényi DP Accountant & Calibration
*   **Modified** [`src/federated.py`](file:///D:/Mini_project_JP/src/federated.py): Implemented exact RDP step calculations, composition over order grids, and dynamic binary-search noise calibration.
*   **Modified** [`src/logistic_numpy.py`](file:///D:/Mini_project_JP/src/logistic_numpy.py) & [`src/cox_numpy.py`](file:///D:/Mini_project_JP/src/cox_numpy.py): Enabled passing pre-calibrated noise multipliers directly.

### 3. Domain-Shift Generalizability Study
*   **Modified** [`src/preprocessing.py`](file:///D:/Mini_project_JP/src/preprocessing.py): Replaced `load_mskcc_validation_cohort` with `generate_domain_shifted_cohort`. Supports covariate shift and concept shift.
*   **Modified** [`src/app.py`](file:///D:/Mini_project_JP/src/app.py): Updated Streamlit section to "Controlled Synthetic Domain-Shift Generalizability Study". Exposes slider controls for severity and options for shift types.

### 4. Experimental Suite Strengthening
*   **Created** [`src/statistical_analysis.py`](file:///D:/Mini_project_JP/src/statistical_analysis.py): Holds routines for bootstrap confidence intervals, Shapley stability across 5 seeds, and Dirichlet sweeps.
*   **Modified** [`src/app.py`](file:///D:/Mini_project_JP/src/app.py): Integrated Dirichlet sweeps, weight-drift plots, bootstrap CIs, and Shapley seed stability analysis.

---

## Verification Results

### Automated Verification Script
*   Created and ran [`scratch/verify_scientific_foundation.py`](file:///D:/Mini_project_JP/scratch/verify_scientific_foundation.py):
    *   Zero data leakage check: **PASS**
    *   RDP accountant monotonicity & calibration: **PASS**
    *   Domain-shift covariate/concept generation: **PASS**
    *   All checks passed successfully.
