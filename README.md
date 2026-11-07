# The Hidden Privacy Cost of Valuing Hospitals in Federated Learning

**Accounting, a Decision Rule, and Evidence from Real Federations**
Ponnaluri Jeyanth, Pudota Jaideep, M. V. P. Chandra Sekhara Rao — R.V.R. & J.C. College of Engineering, Guntur, India
(manuscript submitted to IEEE Journal of Biomedical and Health Informatics)

Hospitals that train a model together need to know what each one contributes. Under differential privacy, the usual
way of computing Shapley values (retraining every coalition) secretly multiplies the privacy cost: with ten hospitals a
reported per-run ε = 1 really costs ε ≈ 33. We make that cost explicit, compare every valuation that pays it only once
(OSA-Shapley, reconstruction from the training run, a single-model score) with cheap proxies such as hospital size on
two real federations, and test a decision rule for choosing between them in an internally pre-registered experiment. The paper is an empirical
study: its propositions are short consequences of standard DP facts, OSA-Shapley is a simple baseline that wins under DP
only for low-dimensional convex models, and all models studied are low-dimensional tabular models.

## Interactive showcase

```bash
pip install -r requirements.txt     # Streamlit >= 1.40
streamlit run app.py
```

| Section | Pages |
|---|---|
| Start | Overview with the headline result and guided tours |
| Data and setting | The two federations · Federated training & privacy |
| Contributions | The hidden privacy cost (calculator) · Valuation methods compared · Second federation: heart · Decision rule (advisor and pre-registered test) |
| Try it | Valuation lab: value the hospitals live, train a federation, risk-calculator demo |
| Evidence | Claims & verdicts · Reproducibility (41 checks) · Paper & citation |
| Archive | Earlier work (TCGA-PRAD, the project's first phase; not in the paper) |

## Data (public, used as released)

* **PCa-19** — 19 prostate-cancer cohorts, 5,610 men, 8 countries (Kazlouski et al., IEEE EMBC 2025;
  [phaseivai/Hospital-Participation-in-FL](https://github.com/phaseivai/Hospital-Participation-in-FL), MIT). `datasets/PCa19/`
* **UCI Heart Disease** — 4 hospitals, 920 patients (Janosi et al. 1988, doi:10.24432/C52P4X, CC BY 4.0). `datasets/UCI_HeartDisease/`

## Reproduce

```bash
python run_pca19.py                    # every experiment -> results/final/
python -m fl_study.validate            # 41 automated checks
python Paper/generate_paper_assets.py  # every number, table and figure of the paper
```

The pre-registration of the decision-rule test is `results/preregistration_nu_sweep.md`
(SHA-256 `3d5dfcdf…68a1`, written before the test was run; internal, not externally registered).

## Repository

* `app.py`, `showcase/` — the Streamlit showcase
* `fl_study/` — tested research code (`private_valuation.py` = OSA-Shapley and the reconstruction baselines,
  `multisite.py` = PCa-19 experiments, `heart.py`, `nu_sweep.py`, `validate.py`)
* `results/final/` — every result (20 held-out splits, paired 95% CIs)
* `Paper/` — LaTeX sources and PDFs of the paper and supplement
* `archive/` — the project's first phase (TCGA-PRAD pipeline in `archive/src/`, earlier apps, reports and notebook);
  kept for the record, superseded by `fl_study/` and not used by the paper. Earlier drafts and superseded notes are in
  the git history (tag `pre-cleanup-2026-11`)
* `docs/README_project_history.md` — the earlier stages of the project (including the TCGA-PRAD work, which is not
  part of the paper)
