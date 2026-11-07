# Fixes and validation (October 2026)

This file records what was wrong in the earlier pipeline, what was changed, and how the
new results are validated. All numbers quoted in the app and in `Paper/research_paper.tex`
are generated from `results/final/*.json`.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py                    # the showcase app (old app kept as archive/app_legacy.py)
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

The manuscript as it was before these changes (`Paper/research_paper_before_fixes.tex`) is in the git history under the tag `pre-cleanup-2026-11`.

---

## Upgrade: the base paper's own 19-hospital data (PCa-19)

**Why.** TCGA-PRAD is small (347–493 patients), its "hospitals" are biobank contributors, its strongest predictors are
post-surgical and it has no PSA. Reviewers at strong journals would treat that as a toy setting. The base paper
(Kazlouski et al., 2025) released its full patient-level data in its public repository
(github.com/phaseivai/Hospital-Participation-in-FL, MIT licence).

**What was added**

| Item | Where |
| --- | --- |
| 5,610 patients, 19 hospitals, 8 countries; label ISUP ≥ 2 at biopsy; age, PSA, prostate volume, PI-RADS | `datasets/PCa19/` (CSV + provenance + licence) |
| Line-by-line re-implementation of the authors' protocol (sklearn LR, one-shot FedAvg, Ettala/Noh variables, their Monte-Carlo seeding) | `fl_study/multisite.py: reproduce_base_paper` |
| Full DP-FPS framework on the real hospitals: strategies, learning curve, privacy grid with the step-size control, drift, personalisation, Shapley under DP, membership inference | `fl_study/multisite.py`, `run_pca19.py` |
| External validation: leave-one-country-out, calibration (O/E, slope, Brier), decision curves, local recalibration | `multisite.loco` |
| Exact Shapley for the 10 largest hospitals + stratified-sampling Shapley for all 19 (sizes 0 and K−1 exact) | `multisite.coalition_utilities_masks`, `stratified_shapley` |
| 8 new validation checks (30 in total) | `fl_study/validate.py`, group "PCa-19" |
| New app page "🌍 19 hospitals (PCa-19)"; PCa-19 verdicts on the Final-results page | `app.py`, `fl_study/claims.py` |
| New paper section, 6 tables, 4 figures; abstract, introduction, discussion and conclusion rewritten | `Paper/sec_pca19.tex`, `Paper/generate_paper_assets.py` (old version: `research_paper_before_pca19.tex`) |

**Things found along the way**

* The base paper's text says "nine countries"; its Table I and released data contain **eight** (Finland and Turkey are
  in its download script but not in the data). The validation check expects 8.
* Plain permutation-sampling Shapley converges slowly here (max error 0.014 AUC with 400 permutations on 6 hospitals),
  because a hospital placed first always gains ~0.35 AUC over a random model. Stratified sampling by coalition size,
  with the two dominant strata computed exactly, cuts the error to 0.003 at similar cost.
* Leave-one-country-out confidence intervals over seeds are almost zero-width (the held-out country is fully used, so
  only the training draw varies). Inference therefore uses the 8 countries as independent units.

---

## Revision for Q1 submission

| Item | What was done | Result |
| --- | --- | --- |
| New method: OSA-Shapley | `fl_study/private_valuation.py`; each hospital releases one DP-trained model; coalition models by size-weighted averaging (post-processing); exact Shapley over all coalitions. Local steps tuned on seeds 0-3, evaluated on seeds 20-39. | Retraining each coalition at "eps = 1 per run" really costs eps ~ 33 per hospital (10 hospitals). At an honest budget retraining loses the ranking (Spearman 0.10 at eps = 1); OSA-Shapley keeps it (0.80 at eps = 1, 0.89 at eps = 2) and is not significantly worse than per-run retraining that spends 28-42x more privacy. Exact 19-hospital valuation (524,287 coalitions) in about a minute. |
| Subsampled DP accounting | Poisson-subsampled DP-SGD + sampled-Gaussian RDP (`fl.rdp_sampled_gaussian`) | Accountant matches numerical integration to 1e-13 and Abadi et al.'s eps = 1.26. Lower noise multipliers (49 -> 15.6 at eps = 1) but no utility gain on held-out seeds (TCGA -0.009, PCa-19 0.000). |
| Stronger baselines | SCAFFOLD, FedAdam, federated MLP, converged FedAvg (400 steps), central gradient boosting / random forest / MLP; FedNova = FedAvg here | All federated variants within 0.006 AUC of FedAvg; SCAFFOLD removes 65% of drift with no AUC change; non-linear models do not transport better to unseen countries. Converged training changes no PCa-19 conclusion. |
| IEEE template | Official IEEEtran.cls v1.8b and IEEEtran.bst v1.14 (two independent GitHub copies, identical to the official release) in `Paper/`; journal mode (IEEE JBHI style), main paper + supplement | Main paper 12 pages, supplement 8 pages; clean build. |
| Reference check | Every BibTeX entry checked against Crossref, DBLP, arXiv and publisher pages (Google Scholar blocks automated queries) | Of 52 original entries: 13 correct, 28 corrected (authors, DOIs, titles, venues), 9 fabricated (removed or replaced by the real work), 2 pointing to a different paper (replaced). 16 new references added and verified. Old file kept as `Paper/references_before_verification.bib`. |
