"""
Pre-registered test of the decision rule (results/preregistration_nu_sweep.md, SHA-256 3d5dfcdf...68a1).
Within the 10 largest PCa-19 hospitals, vary nu = median(training patients per hospital) / sqrt(parameters):
  Arm A: keep a fraction f of every hospital's training patients (p = 7);
  Arm B: all patients, add k uninformative N(0,1) features (p = 7 + k).
In each condition: exact non-private retraining Shapley = reference; OSA-Shapley and MR/OR at epsilon 2 and 1.
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

from . import valuation as V
from .stats import summary, paired

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "cache_nu_sweep")
SEEDS = list(range(20, 40))
CONDS = [("A", 1.0, 0), ("A", 0.6, 0), ("A", 0.4, 0), ("A", 0.25, 0), ("A", 0.15, 0),
         ("B", 1.0, 21), ("B", 1.0, 50), ("B", 1.0, 100)]
EPS = [2.0, 1.0]


def _cond_key(c):
    return f"{c[0]}_f{c[1]}_k{c[2]}"


def _data(seed, cond):
    from . import multisite as M
    arm, f, k = cond
    df, names, tr, te, X, y = M.split(seed)
    h10 = names[:10]
    if k:
        Z = np.random.default_rng(7000 + seed).normal(size=(len(y), k))
        X = np.hstack([X, Z])
    rng = np.random.default_rng(9000 + seed)
    cl = []
    for h in h10:
        idx = tr[h]
        if f < 1.0:
            m = max(4, int(round(f * len(idx))))
            while True:
                sub = np.sort(rng.choice(idx, m, replace=False))
                if 0 < y[sub].sum() < len(sub):
                    break
            idx = sub
        cl.append((X[idx], y[idx].astype(float)))
    ref = np.concatenate([te[h] for h in h10])
    return cl, X[ref], y[ref].astype(float)


def _run(args):
    from . import multisite as M
    seed, cond = args
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"{_cond_key(cond)}_seed{seed}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    cl, Xr, yr = _data(seed, cond)
    truth, _ = M._exact_retrain(cl, Xr, yr, M.cfg(), seed)
    rs = lambda p: float(V.rank_stability(truth, [np.asarray(p, float)])["spearman"][0])
    out = {"sizes": [len(c[1]) for c in cl], "p": int(cl[0][0].shape[1] + 1), "rho": {}}
    for eps in EPS:
        pm = M._post_methods(cl, Xr, yr, eps, seed)
        out["rho"][str(eps)] = {m: rs(v) for m, v in pm.items()}
    json.dump(out, open(fn, "w"))
    return out


def run(seeds=SEEDS, processes=2):
    jobs = [(s, c) for c in CONDS for s in seeds]
    with Pool(processes) as pool:
        res = pool.map(_run, jobs)
    by = {}
    for (s, c), r in zip(jobs, res):
        by.setdefault(_cond_key(c), []).append(r)
    out = {"seeds": seeds, "conditions": []}
    for c in CONDS:
        R = by[_cond_key(c)]
        nu = float(np.median(R[0]["sizes"]) / np.sqrt(R[0]["p"]))
        nus = [float(np.median(r["sizes"]) / np.sqrt(r["p"])) for r in R]
        row = {"arm": c[0], "fraction": c[1], "extra_features": c[2], "p": R[0]["p"], "nu": float(np.mean(nus)),
               "median_train_patients": float(np.mean([np.median(r["sizes"]) for r in R]))}
        for eps in EPS:
            k = str(eps)
            row[k] = {m: summary([r["rho"][k][m] for r in R]) for m in ("osa", "mr", "or")}
            row[k]["osa_minus_mr"] = paired([r["rho"][k]["osa"] for r in R], [r["rho"][k]["mr"] for r in R])
        out["conditions"].append(row)
    return out
