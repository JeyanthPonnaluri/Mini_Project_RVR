"""Small statistics helpers used by every experiment (NaN-aware)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def summary(x):
    """mean, sd, n and t-based 95% CI of the mean (NaNs dropped)."""
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    n = len(x)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0, "ci_low": None, "ci_high": None}
    m = float(x.mean())
    if n == 1:
        return {"mean": m, "sd": 0.0, "n": 1, "ci_low": None, "ci_high": None}
    sd = float(x.std(ddof=1))
    h = float(st.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {"mean": m, "sd": sd, "n": n, "ci_low": m - h, "ci_high": m + h}


def paired(a, b):
    """Paired difference a-b: summary + two-sided Wilcoxon signed-rank p-value."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    out = summary(d)
    if len(d) >= 5 and np.any(d != 0):
        out["wilcoxon_p"] = float(st.wilcoxon(d).pvalue)
    else:
        out["wilcoxon_p"] = None
    return out


def verdict(ci_low, ci_high, positive_is_good=True):
    """'supported' if the 95% CI excludes 0 in the expected direction,
    'opposite' if it excludes 0 in the other direction, else 'not significant'."""
    if ci_low is None or ci_high is None:
        return "insufficient data"
    if positive_is_good:
        return "supported" if ci_low > 0 else "opposite" if ci_high < 0 else "not significant"
    return "supported" if ci_high < 0 else "opposite" if ci_low > 0 else "not significant"


def bootstrap_auc_ci(y, p, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
