# Task Checklist - Scientific Foundation Repair & Experiments

- [x] **Phase 1: Fix Data Leakage**
  - [x] Modify [`src/preprocessing.py`](file:///D:/Mini_project_JP/src/preprocessing.py) to enable separate fitting and transforming for preprocessing (StandardScaler, OneHotEncoder, PCA).
  - [x] Update [`src/app.py`](file:///D:/Mini_project_JP/src/app.py) to split datasets into train and test splits **before** fitting scalers, encoders, and PCA.
  - [x] Ensure that client preprocessing is partitioned correctly without cross-client leakage.

- [x] **Phase 2: Fix MSKCC (Synthetic Domain-Shift Study)**
  - [x] Rename `load_mskcc_validation_cohort` in [`src/preprocessing.py`](file:///D:/Mini_project_JP/src/preprocessing.py) to `generate_domain_shifted_cohort`.
  - [x] Implement realistic domain perturbations (covariate shift) rather than pure random normal noise.
  - [x] Rename the Streamlit section in [`src/app.py`](file:///D:/Mini_project_JP/src/app.py) to "Controlled Synthetic Domain-Shift Generalizability Study". Remove all claims representing it as genuine MSKCC patient data.

- [x] **Phase 3: Fix DP/RDP Privacy Accountant**
  - [x] Implement exact analytical Rényi Differential Privacy (RDP) step calculations in [`src/federated.py`](file:///D:/Mini_project_JP/src/federated.py).
  - [x] Add `compose_rdp_to_dp` using order grid optimization to return standard $(\epsilon, \delta)$-DP bounds.
  - [x] Add dynamic noise calibration `calibrate_noise_multiplier` and test its validity.
  - [x] Define the privacy unit (sample-level DP) and adjacency definitions in methodology.
  - [x] Integrate the accountant into `fedavg_train`, `fedprox_train`, and `fedavg_cox_train`.

- [x] **Phase 4 & 5: Run and Strengthen Experiments**
  - [x] Re-run centralized classification baselines (Logistic, L1, L2, RF, MLP) on clinical, protein, and multi-modal features with leakage-free pipeline.
  - [x] Implement client-drift tracking (L2 norms) in [`src/fedprox_experiments.py`](file:///D:/Mini_project_JP/src/fedprox_experiments.py) for all Dirichlet $\alpha \in \{10, 1.0, 0.5, 0.1\}$.
  - [x] Implement baselines for Cox survival model (Centralized, Local, FedAvg, FedProx).
  - [x] Sweep privacy-utility curves ($\epsilon$ vs. AUC).
  - [x] Run Shapley value valuations across 5 seeds and report stability ($\phi_k = \mu_k \pm \sigma_k$).
  - [x] Add bootstrap confidence intervals for AUC and C-index in [`src/experiments.py`](file:///D:/Mini_project_JP/src/experiments.py).
  - [x] Update Streamlit dashboard elements to display the corrected metrics and plots.

- [x] **Phase 6 & 7: Frame SMPC & Economic Model**
  - [x] Frame SMPC explicitly as "proposed SMPC-compatible secure aggregation design".
  - [x] Frame economic billing explicitly as a "proposed incentive mechanism evaluated through simulation".

- [x] **Phase 8: Automated Verification Script**
  - [x] Create [`scratch/verify_scientific_foundation.py`](file:///D:/Mini_project_JP/scratch/verify_scientific_foundation.py) to verify data integrity, RDP monotonicity, federated weight drift, and bootstrap statistics.
  - [x] Verify execution runs to completion and exits with code 0.

- [x] **Phase 9: Independent Validation & Empirical Execution**
  - [x] Create [`scratch/validate_rdp_math.py`](file:///D:/Mini_project_JP/scratch/validate_rdp_math.py) to validate RDP accountant bounds and calibrate values.
  - [x] Verify unique patient barcodes in the clinical dataset to establish the sample-level privacy unit.
  - [x] Create and run [`scratch/run_all_experiments.py`](file:///D:/Mini_project_JP/scratch/run_all_experiments.py) to execute all sweeps and save results to [`reports/empirical_results.json`](file:///D:/Mini_project_JP/reports/empirical_results.json).

- [x] **Phase 10: 9.5/10 Advanced Interaction Study & Manuscript Compilation**
  - [x] Create and execute [`scratch/comprehensive_9_5_experiments.py`](file:///D:/Mini_project_JP/scratch/comprehensive_9_5_experiments.py) to sweep the 2D privacy-heterogeneity matrix, evaluate personalization profiles, audit Shapley value shifts under privacy noise, and test empirical MIA resistance.
  - [x] Overwrite the manuscript draft at [`Paper/research_paper_latex_code.txt`](file:///D:/Mini_project_JP/Paper/research_paper_latex_code.txt) using correct patient/sample counts, qualified RDP mechanism parameters ($q=1.0$), and the revised 9.5/10 trade-off thesis framework.
