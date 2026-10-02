# Fixes and validation (October 2026)

This file records what was wrong in the earlier pipeline, what was changed, and how the
new results are validated. All numbers quoted in the app and in `Paper/research_paper.tex`
are generated from `results/final/*.json`.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py                    # the showcase app (old app kept as app_legacy.py)
python run_all.py                       # re-run every experiment (~25 min on 2 cores)
python -m fl_study.validate             # 22 automated correctness checks
python Paper/generate_paper_assets.py   # regenerate paper numbers, tables and figures
```

## Problems found and fixed

| # | Problem | Fix | Where |
|---|---------|-----|-------|
| 1 | ~1,650 one-hot features built from per-patient identifiers (`submitter_id.samples`, `sample_id.samples`, `pathology_report_uuid.samples`, treatment IDs, timestamps) plus outcome/follow-up columns were used as model inputs | Explicit feature whitelist; identifier/timestamp/outcome columns excluded and tested | `fl_study/data.py`; `src/preprocessing.py` and the package copy now also drop them |
| 2 | Reports claimed clinical + protein "significantly outperforms" clinical-only | Re-tested with 5x5 repeated CV: adding protein PCs lowers AUC by 0.159 [0.130, 0.188] | `results/final/centralized.json` |
| 3 | Gleason grade and pathologic N come from the same surgical report as the label | Kept in the main model, disclosed; pre-operative-only ablation reported (AUC ~0.65) | app, paper |
| 4 | Top-contributor stability averaged P(rank = 1) over clients, so it was always 1/K | TS = P(no-DP top contributor still ranked first under DP) | `fl_study/valuation.py`, `scratch/client_scaling_experiments.py` |
| 5 | Client drift: FedAvg silently recorded 0.0; FedProx measured global-model change | Both record mean_k ‖w_k − w_global‖ (paper definition) | `fl_study/fl.py`, `src/federated.py`, `src/fedprox_experiments.py`, package copy |
| 6 | Paper: "12 deaths (9 train, 0 test)" in the 347-patient cohort | 9 deaths (8 train / 1 test for seed 42); 12 is over all 572 records | paper |
| 7 | Per-site comparisons pooled predictions of different local models (rewards learning each site's base rate) | Per-site / per-client macro AUC | `fl_study/experiments.py` |
| 8 | NumPy logistic regression had no intercept | Intercept added | `fl_study/fl.py` |
| 9 | PCA count reported as 129 in one report and 115 in another | 115 when fitted on the training split (correct); 129 only if fitted on all rows | validation check |
| 10 | Base paper (Kazlouski et al.) was described as using LOO valuation | It does not value contributions; LOO is our added baseline | app, paper table |
| 11 | pandas 3 stores text as `str`, which `select_dtypes(include=['object'])` misses | Non-numeric columns treated as categorical | `src/preprocessing.py` |

## Conclusions that changed after the fixes (20 seeds, paired 95% CIs)

* **MIA (H7):** the attack is at chance even without DP (AUC 0.511 [0.489, 0.533]); the old
  0.568 → 0.480 drop came from the identifier features. H7 is not supported.
* **FedProx under DP (H3):** FedProx beats FedAvg at ε ≤ 2, but a FedAvg with half the
  learning rate matches it, so the gain is a step-size (noise-damping) effect.
* **Drift (H1):** FedProx reduces drift by 32–36% (old: 4.2–4.3%); without DP it does not change AUC.
* **Proteomics:** reduces discrimination.
* **Shapley under DP (H5):** confirmed and strengthened with exact Shapley values for K = 3, 5, 10.
  Scaling is now K ≤ 10 (exact enumeration), not N = 20 (Monte-Carlo).
* **Base-paper replication (new):** on the 9 real TCGA sites, FL − LOC = +0.015 [−0.007, +0.037]
  (no significant local gain) while the free-rider penalty is only −0.008 [−0.013, −0.003].

## Validation

`python -m fl_study.validate` checks: raw/cohort/split counts, survival events, absence of
identifier features, pre-op set purity, train-only preprocessing, NumPy LR vs scikit-learn,
FedAvg identities, drift recording, Dirichlet partition, RDP accountant vs closed form,
noise calibration, empirical noise scale, clipping bound, vectorised vs per-coalition Shapley
training, Shapley axioms, fixed TS metric, MIA positive control, and reproducibility.
The app's *Validation* page shows the stored results and can re-run the checks live.

`Paper/research_paper_before_fixes.tex` is the manuscript as it was before these changes.
