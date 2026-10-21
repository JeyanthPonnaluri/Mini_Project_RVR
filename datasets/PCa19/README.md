# PCa-19: 19-hospital pre-biopsy prostate cancer cohort

`pca19.csv` contains 5,610 patients from 19 hospitals (8 countries). It is the exact patient-level
collection used by the base paper:

> A. Kazlouski, I. Montoya Perez, T. Pahikkala, A. Airola. *Hospital Participation in Federated
> Learning: Evaluating Sustainability and Clinical Utility*, Proc. IEEE EMBC 2025, doi:10.1109/EMBC58623.2025.11252903.

**Where it comes from**

- Repository: https://github.com/phaseivai/Hospital-Participation-in-FL (MIT licence, see
  `LICENSE_base_repo_MIT.txt`).
- Commit: see `base_repo_commit.txt`.
- File: `data/data.pkl`. It was converted to CSV without changing any values; a `hospital` column
  was added.

The authors harmonised the data from public supplementary files of peer-reviewed studies, which
are listed in Table I of the base paper and fetched by `download_datasets.sh` in their repository.

| Hospital code | Country | n |
| --- | --- | --- |
| de1, de2 | Germany | 763, 162 |
| es1–es8 | Spain | 79, 16, 30, 200, 200, 346, 711, 58 |
| kr1, kr2 | South Korea | 406, 590 |
| ch1, ch2 | China | 312, 530 |
| us1, us2 | USA | 280, 310 |
| nl | Netherlands | 266 |
| it | Italy | 218 |
| gb | United Kingdom | 133 |

**Columns**

- `sig_cancer`: label. 1 = clinically significant prostate cancer (ISUP grade group ≥ 2 at
  biopsy).
- `age`: age in years.
- `PSA`: prostate-specific antigen, in ng/mL.
- `PV`: prostate volume, in mL.
- `PIRADS`: highest PI-RADS score on MRI, from 1 to 5.
- `5ARI`: 5-alpha-reductase inhibitor use. It is 0 for every patient, so it is not used as a
  feature.

Note: the base paper's text says "nine countries", but its Table I and the released data contain
8 countries (Finland and Turkey appear in its download script but not in `data.pkl`).

All columns are available before biopsy, so models trained on this cohort are pre-biopsy risk
calculators. There are no missing values; the authors discarded incomplete records.
