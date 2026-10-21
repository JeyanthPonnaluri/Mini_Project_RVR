"""
Second real federation: the UCI Heart Disease collection (Janosi, Steinbrunn, Pfisterer, Detrano, 1988; CC BY 4.0).
Four real hospitals, used as released: Cleveland Clinic Foundation (USA), Hungarian Institute of Cardiology
(Budapest), University Hospitals of Zurich and Basel (Switzerland), and the VA Medical Center, Long Beach (USA).
Target: angiographic coronary artery disease (num > 0). This is the "Fed-Heart-Disease" benchmark of FLamby.

Preprocessing (no leakage): columns with > 30% missing values overall (slope, ca, thal) are dropped; a cholesterol of 0
is treated as missing (all of Switzerland); remaining missing values are imputed with the median of the pooled
TRAINING rows and flagged with a missingness indicator; chest-pain type and resting ECG are one-hot encoded; numeric
features are standardised with training-row statistics. OSA-Shapley uses the hyper-parameters fixed on PCa-19
(no re-tuning on this cohort).
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import fl
from . import valuation as V
from .stats import summary, paired

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "datasets", "UCI_HeartDisease")
SITES = [("cleveland", "Cleveland (USA)"), ("hungarian", "Budapest (Hungary)"),
         ("switzerland", "Zurich/Basel (Switzerland)"), ("va", "Long Beach VA (USA)")]
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
SEEDS = list(range(20, 40))
EPS = [None, 5.0, 2.0, 1.0]
CACHE = os.path.join(ROOT, "results", "cache_heart")
CORRUPT = 1                       # Budapest (second largest) gets random training labels in the useless-hospital test


def load() -> pd.DataFrame:
    parts = []
    for key, name in SITES:
        d = pd.read_csv(os.path.join(DIR, f"processed.{key}.data"), header=None, names=COLS, na_values="?")
        d["hospital"] = key
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.loc[df.chol == 0, "chol"] = np.nan
    df["y"] = (df.num > 0).astype(int)
    return df


NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BIN = ["sex", "fbs", "exang"]


def features(df, train_idx):
    tr = df.iloc[train_idx]
    out = pd.DataFrame(index=df.index)
    for c in NUM + BIN + ["restecg"]:
        med = tr[c].median()
        if df[c].isna().any():
            out[c + "_missing"] = df[c].isna().astype(float)
        out[c] = df[c].fillna(med)
    for c in NUM:
        mu, sd = out.iloc[train_idx][c].mean(), out.iloc[train_idx][c].std(ddof=0)
        out[c] = (out[c] - mu) / sd
    for v in (2, 3, 4):
        out[f"cp_{v}"] = (df.cp == v).astype(float)
    for v in (1, 2):
        out[f"restecg_{v}"] = (out["restecg"] == v).astype(float)
    out = out.drop(columns=["restecg"])
    return out.values.astype(float)


def split(seed, df=None):
    df = load() if df is None else df
    rng = np.random.RandomState(seed)
    tr, te = {}, {}
    for key, _ in SITES:
        idx = np.where(df.hospital.values == key)[0]
        a, b = train_test_split(idx, test_size=0.2, random_state=rng.randint(1 << 30), stratify=df.y.values[idx])
        tr[key], te[key] = a, b
    X = features(df, np.concatenate([tr[k] for k, _ in SITES]))
    return df, tr, te, X, df.y.values.astype(float)


def _seed(seed):
    from . import multisite as M
    from . import private_valuation as PV
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"seed{seed}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    df, tr, te, X, y = split(seed)
    keys = [k for k, _ in SITES]
    cl = [(X[tr[k]], y[tr[k]]) for k in keys]
    cl_te = [(X[te[k]], y[te[k]]) for k in keys]
    ref = np.concatenate([te[k] for k in keys]); Xr, yr = X[ref], y[ref]
    truth, _ = M._exact_retrain(cl, Xr, yr, M.cfg(), seed)
    out = M._proxy_block(cl, cl_te, Xr, yr, seed, truth, corrupt_index=CORRUPT, heavy=True)
    rs = lambda p: float(V.rank_stability(truth, [np.asarray(p, float)])["spearman"][0])
    for eps in EPS[1:]:
        out["rho"][str(eps)]["retrain_per_run"] = rs(M._exact_retrain(cl, Xr, yr, M.cfg(epsilon=eps), 1000 + seed)[0])
    out["truth"] = truth.tolist()
    out["fed_auc"] = float(M._auc_safe(yr, fl.add_bias(Xr) @ PV.osa_shapley(cl, Xr, yr, None, return_models=True)[1].T
                                      @ (np.array([len(c[1]) for c in cl]) / sum(len(c[1]) for c in cl))))
    json.dump(out, open(fn, "w"))
    return out


def run(seeds=SEEDS, processes=2):
    from . import private_valuation as PV
    with Pool(processes) as pool:
        runs = pool.map(_seed, seeds)
    df = load()
    R0 = runs[0]
    res = {"seeds": seeds, "sites": [{"key": k, "name": n, "n": int((df.hospital == k).sum()),
                                      "prevalence": float(df.y[df.hospital == k].mean())} for k, n in SITES],
           "n": int(len(df)), "n_features": int(split(seeds[0], df)[3].shape[1]),
           "corrupt_hospital": SITES[CORRUPT][1], "rho": {}, "osa_minus": {}, "corrupt": {},
           "truth_mean": [float(np.mean([r["truth"][i] for r in runs])) for i in range(len(SITES))],
           "fed_auc": summary([r["fed_auc"] for r in runs]),
           "true_epsilon": {str(e): float(PV.retraining_privacy_cost(e, len(SITES), 100)) for e in EPS[1:]}}
    for k in R0["rho"]:
        res["rho"][k] = {m: summary([r["rho"][k][m] for r in runs]) for m in R0["rho"][k]}
        res["osa_minus"][k] = {m: paired([r["rho"][k]["osa"] for r in runs], [r["rho"][k][m] for r in runs])
                               for m in R0["rho"][k] if m != "osa"}
    for k in R0["corrupt"]:
        res["corrupt"][k] = {m: summary([r["corrupt"][k][m] for r in runs]) for m in R0["corrupt"][k]}
    return res
