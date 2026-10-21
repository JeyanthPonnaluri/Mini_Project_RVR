# Pre-registered test of the decision rule (written before any sweep result exists)

Date written: 2026-10-17, before the code for this sweep was run on any seed.

## The rule being tested

From four settings (PCa-19 10 largest, PCa-19 10 smallest, UCI Heart Disease, PCa-19 MLP) we derived:
OSA-Shapley beats MR reconstruction under DP when nu = median(training patients per hospital) / sqrt(model parameters)
is about 100 or more, and loses its advantage when nu is about 50 or less.

## Design (fixed in advance)

- Hospitals: the 10 largest PCa-19 hospitals, logistic regression, held-out splits (seeds 20-39), the same
  OSA-Shapley and MR settings as in the paper (OSA: 30 DP-GD steps, lr 1.0; DP-FedAvg: T = 20, E = 5, lr 0.5).
- Reference ("truth") in every condition: exact non-private retraining Shapley values computed in that same condition.
- Reference set for v(S): the full held-out test patients of the 10 hospitals (never subsampled).
- Arm A (fewer patients): keep a random fraction f of every hospital's training patients,
  f in {1.0, 0.6, 0.4, 0.25, 0.15}, p = 7, so nu is about 114, 68, 46, 28, 17.
- Arm B (more parameters): all training patients, add k uninformative standard-normal features per patient,
  k in {21, 50, 100}, so p = 28, 57, 107 and nu is about 57, 40, 29.
- Outcome: paired difference OSA - MR in Spearman rho with the reference, 20 splits, 95% t-interval.
  Primary budget epsilon = 2; secondary epsilon = 1.

## Predictions (epsilon = 2)

- H1. For every condition with nu >= 100, OSA - MR > 0 (95% CI above 0).
- H2. For every condition with nu <= 50, OSA - MR is not positive (95% CI includes 0 or lies below 0).
- H3. Across all conditions, OSA - MR increases with nu (Spearman correlation between nu and the mean OSA - MR > 0).
- H4. The crossover nu* (where the mean OSA - MR changes sign, by linear interpolation in log nu between adjacent
  conditions) lies between 50 and 100. If OSA - MR never changes sign, H4 fails.
- H5. Both arms agree: at matched nu (Arm A f = 0.4, nu ~ 46 vs Arm B k = 50, nu ~ 40), the sign of OSA - MR is the same.

## Secondary prediction (epsilon = 1)

- H6. Noise is about twice as large at epsilon = 1 (sigma 49 vs 25), so the crossover at epsilon = 1 lies at a higher
  nu than at epsilon = 2 (or OSA - MR at epsilon = 1 is lower than at epsilon = 2 in at least 6 of the 8 conditions).

## Reporting commitment

All hypotheses are reported as supported or not supported, whatever the outcome. No condition, seed or method setting
is changed after seeing results.
