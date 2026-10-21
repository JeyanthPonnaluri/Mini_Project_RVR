"""
PCa-19: the base paper's own 19-hospital, pre-biopsy prostate cancer cohort (5,610 patients).
Task: clinically significant cancer (ISUP grade group >= 2) BEFORE biopsy, using age, PSA,
prostate volume and the MRI PI-RADS score. Source and licence: datasets/PCa19/README.md.

Two protocols
-------------
1) `reproduce_base_paper` - a line-by-line re-implementation of the authors' public code
   (sklearn LogisticRegression on the raw Ettala / Noh variables, stratified 80/20 split per
   hospital, ONE-SHOT FedAvg = average of fully trained local models, free-riding = all other
   hospitals train and the excluded one is tested on 100% of its data, Monte-Carlo random states
   drawn exactly as in their scripts). With the same seed it must give the same numbers as
   their code; validate.py checks this against the stored output of their code.

2) Our protocol (everything else in this file) - the same framework that was used on TCGA:
   standardised features (statistics from training rows only), iterative FedAvg / FedProx with
   gradient descent (T=20 rounds, E=5 epochs), DP-SGD with a Renyi-DP accountant, exact and
   Monte-Carlo Shapley, personalisation, membership inference, 20 seeds and paired 95% CIs.
   Additions that a clinical journal expects: leave-one-COUNTRY-out external validation,
   calibration (observed/expected ratio, calibration slope, Brier score) and decision-curve
   net benefit.
"""
from __future__ import annotations

import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import fl
from . import valuation as V
from .stats import paired, summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCA19_CSV = os.path.join(ROOT, "datasets", "PCa19", "pca19.csv")

COUNTRY = {"de1": "Germany", "de2": "Germany", "kr1": "South Korea", "kr2": "South Korea",
           "ch1": "China", "ch2": "China", "us1": "USA", "us2": "USA", "nl": "Netherlands",
           "it": "Italy", "gb": "United Kingdom", **{f"es{i}": "Spain" for i in range(1, 9)}}

SEEDS = list(range(20))
EPSILONS = [10.0, 5.0, 2.0, 1.0, 0.5]
BASE_CFG = dict(rounds=20, epochs=5, lr=0.5, l2=1e-3, delta=1e-5, clip=1.0)
MU = 0.5
CORE_FEATURES = ["age", "log_psa", "log_pv", "pirads_3", "pirads_4", "pirads_5"]


def cfg(**kw):
    return fl.TrainConfig(**{**BASE_CFG, **kw})


# =============================================================================
# Data
# =============================================================================
def load_pca19() -> pd.DataFrame:
    df = pd.read_csv(PCA19_CSV, float_precision="round_trip")   # bit-exact copy of the authors' data.pkl
    df["country"] = df["hospital"].map(COUNTRY)
    df = df.rename(columns={"sig_cancer": "y"})
    df["y"] = df["y"].astype(int)
    return df


def hospitals_by_size(df=None):
    df = load_pca19() if df is None else df
    c = df["hospital"].value_counts()
    # same order as the authors: largest first (ties keep first appearance, as python's sort)
    names = list(dict.fromkeys(df["hospital"]))
    return sorted(names, key=lambda k: c[k], reverse=True)


def core_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Our feature set: age, log PSA, log prostate volume and PI-RADS 3/4/5 indicators
    (PI-RADS 1-2 = reference). Logs tame the extreme PSA values (up to 1,891 ng/mL in es2)."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float)
    out["log_psa"] = np.log(df["PSA"].astype(float).clip(lower=0.1))
    out["log_pv"] = np.log(df["PV"].astype(float).clip(lower=1.0))
    for k in (3, 4, 5):
        out[f"pirads_{k}"] = (df["PIRADS"].round() == k).astype(float)
    return out[CORE_FEATURES]


class Standardiser:
    """Mean/sd from training rows only. In a federation the hospitals share their sums and
    sums of squares (secure aggregation); the result equals pooled statistics."""

    def fit(self, F: pd.DataFrame):
        self.mu = F.mean().values
        self.sd = F.std(ddof=0).replace(0, 1.0).values
        return self

    def transform(self, F: pd.DataFrame):
        return (F.values - self.mu) / self.sd


# ---- published risk calculators (no training, used as the base-paper "BL") -----------------
def _spline(x, i):
    t = [3.80, 6.60, 9.40, 18.47]
    # np.power, exactly as in the authors' utils.get_spline: the Ettala splines reach ~1e10 for extreme PSA and
    # lbfgs does not fully converge on them, so even last-bit differences in the inputs change the fitted model.
    return (np.power(max(x - t[i], 0), 3) - np.power(max(x - t[2], 0), 3) * (t[3] - t[i]) / (t[3] - t[2])
            + np.power(max(x - t[3], 0), 3) * (t[2] - t[i]) / (t[3] - t[2]))


def ettala_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the authors' apply_spline_transformation (5ARI is 0 for every patient)."""
    out = pd.DataFrame(index=df.index)
    psa = df["PSA"].astype(float) * np.where(df["5ARI"] == 1, 2.0, 1.0)
    pv = df["PV"].astype(float) / np.where(df["5ARI"] == 1, 0.7, 1.0)
    out["age"] = df["age"].astype(float)
    out["PSA"] = psa
    out["PV"] = pv
    out["PSA_SPLINE_2"] = psa.apply(_spline, i=0)
    out["PSA_SPLINE_3"] = psa.apply(_spline, i=1)
    pr = df["PIRADS"].map({0: 0, 1: 0, 2: 0, 3: 3, 4: 4, 5: 5})
    for k in (3, 4, 5):
        out[f"PIRADS_{k}"] = (pr == k).astype(float)
    return out


def noh_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the authors' apply_transformation_noh."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float)
    out["PIRADS_group"] = df["PIRADS"].map({0: 0, 1: 0, 2: 0, 3: 2, 4: 3, 5: 4}).astype(float)
    psad = df["PSA"] / df["PV"]
    out["psad_group"] = pd.cut(psad, bins=[0, 0.06, 0.09, 0.14, 0.19, 0.24, np.inf],
                               labels=[1, 2, 3, 4, 5, 6], include_lowest=True).astype(float)
    return out


def ettala_published(df):
    F = ettala_features(df)
    z = (-6.97314184 + 0.064172722 * F["age"] - 0.008141264 * F["PV"] - 0.182694534 * F["PSA"]
         + 0.006136442 * F["PSA_SPLINE_2"] - 0.013049396 * F["PSA_SPLINE_3"]
         + 1.37637197 * F["PIRADS_3"] + 2.50939431 * F["PIRADS_4"] + 4.07331563 * F["PIRADS_5"])
    return (1 / (1 + np.exp(-z))).values


def noh_published(df):
    F = noh_features(df)
    z = -9.5189 + 0.0511 * F["age"] + 1.1834 * F["PIRADS_group"] + 0.4847 * F["psad_group"]
    return (1 / (1 + np.exp(-z))).values


CALCULATORS = {"ettala": (ettala_features, ettala_published), "noh": (noh_features, noh_published)}


# =============================================================================
# 1) Faithful reproduction of the authors' protocol
# =============================================================================
def _sk_fit(X, y):
    m = LogisticRegression(solver="lbfgs", max_iter=10000)
    m.fit(X, y)
    return m.coef_.ravel(), float(m.intercept_[0]), len(y)


def _sk_auc(X, y, coef, b):
    return roc_auc_score(y, 1 / (1 + np.exp(-(X @ coef + b))))


def reproduce_base_paper(calculator="noh", n_repeats=200, seed=1234):
    """Returns the authors' Figure-3 matrix (rows BL, LOC, CEN, FL, FR; columns = hospitals,
    largest first) computed with their protocol. Random states are drawn exactly as in their
    scripts: np.random.seed(1234); one np.random.randint(0, 10_000_000) per repeat, shared by
    all strategies."""
    df = load_pca19()
    names = hospitals_by_size(df)
    feat, published = CALCULATORS[calculator]
    F = feat(df)
    data = {h: (F[df.hospital == h].values, df.y[df.hospital == h].values) for h in names}
    order = list(dict.fromkeys(df["hospital"]))           # authors' dict order (training order)
    bl = np.array([roc_auc_score(df.y[df.hospital == h], published(df[df.hospital == h])) for h in names])
    np.random.seed(seed)
    states = [np.random.randint(0, 10000000) for _ in range(n_repeats)]
    rows = {k: [] for k in ("LOC", "CEN", "FL", "FR")}
    for rs in states:
        sp = {}
        for h in order:
            X, y = data[h]
            sp[h] = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)
        loc = [_sk_auc(sp[h][1], sp[h][3], *_sk_fit(sp[h][0], sp[h][2])[:2]) for h in names]
        Xc = np.vstack([sp[h][0] for h in order]); yc = np.concatenate([sp[h][2] for h in order])
        c, b, _ = _sk_fit(Xc, yc)
        cen = [_sk_auc(sp[h][1], sp[h][3], c, b) for h in names]
        ups = {h: _sk_fit(sp[h][0], sp[h][2]) for h in order}

        def avg(hs):
            n = sum(ups[h][2] for h in hs)
            return (sum(ups[h][0] * ups[h][2] for h in hs) / n, sum(ups[h][1] * ups[h][2] for h in hs) / n)

        gc, gb = avg(order)
        fl_ = [_sk_auc(sp[h][1], sp[h][3], gc, gb) for h in names]
        fr = []
        for h in names:
            ec, eb = avg([o for o in names if o != h])     # authors' fr.py averages in size order
            fr.append(_sk_auc(data[h][0], data[h][1], ec, eb))
        for k, v in zip(("LOC", "CEN", "FL", "FR"), (loc, cen, fl_, fr)):
            rows[k].append(v)
    sizes = np.array([len(data[h][1]) for h in names])
    out = {"calculator": calculator, "n_repeats": n_repeats, "hospitals": names, "sizes": sizes.tolist(),
           "matrix": {"BL": bl.tolist(), **{k: np.mean(v, 0).tolist() for k, v in rows.items()}}}
    out["AV"] = {k: float(np.mean(v)) for k, v in out["matrix"].items()}
    out["AVW"] = {k: float(np.average(v, weights=sizes)) for k, v in out["matrix"].items()}
    # paired per-repeat differences of hospital-averaged AUC (AV)
    av = {k: np.array(v).mean(1) for k, v in rows.items()}
    out["diff"] = {"FL-LOC": paired(av["FL"], av["LOC"]), "FR-FL": paired(av["FR"], av["FL"]),
                   "CEN-FL": paired(av["CEN"], av["FL"])}
    return out


# =============================================================================
# 2) Our protocol
# =============================================================================
def calibration(y, p):
    """Observed/expected ratio, calibration slope (logistic recalibration) and Brier score."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    if len(np.unique(y)) < 2:
        slope = float("nan")
    else:
        m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
        slope = float(m.coef_[0, 0])
    return {"oe": float(y.mean() / p.mean()), "slope": slope, "brier": float(np.mean((p - y) ** 2))}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibrate(y_cal, p_cal, p_new, method="intercept"):
    """Local recalibration of a transported model on a small sample of the new hospital/country.
    'intercept': re-estimate only the baseline risk (calibration-in-the-large; AUC unchanged).
    'logistic' : re-estimate intercept and slope of the linear predictor (AUC unchanged)."""
    from scipy.optimize import minimize_scalar
    lp_c = _logit(p_cal); lp_n = _logit(p_new); y_cal = np.asarray(y_cal, float)
    if method == "intercept":
        f = lambda a: -np.sum(y_cal * (lp_c + a) - np.log1p(np.exp(lp_c + a)))
        a = minimize_scalar(f, bounds=(-10, 10), method="bounded").x
        return 1 / (1 + np.exp(-(lp_n + a)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp_c[:, None], y_cal)
    return m.predict_proba(lp_n[:, None])[:, 1]


def net_benefit(y, p, thresholds):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    out = []
    for t in thresholds:
        pos = p >= t
        tp = np.sum(pos & (y == 1)); fp = np.sum(pos & (y == 0))
        out.append(float(tp / n - fp / n * t / (1 - t)))
    return out


def split(seed, df=None):
    """Stratified 80/20 split inside every hospital; standardiser fitted on all training rows."""
    df = load_pca19() if df is None else df
    names = hospitals_by_size(df)
    rng = np.random.RandomState(seed)
    tr, te = {}, {}
    for h in names:
        idx = np.where(df.hospital.values == h)[0]
        a, b = train_test_split(idx, test_size=0.2, random_state=rng.randint(1 << 30), stratify=df.y.values[idx])
        tr[h], te[h] = a, b
    raw = core_raw(df)
    st = Standardiser().fit(raw.iloc[np.concatenate([tr[h] for h in names])])
    return df, names, tr, te, st.transform(raw), df.y.values


def _p(X, w):
    return fl.predict_proba(X, w)


# ---- A. strategies, learning curve, calibration --------------------------------------------
def _strategies_seed(arg):
    seed, ov = (arg, {}) if isinstance(arg, (int, np.integer)) else arg
    df, names, tr, te, X, y = split(seed)
    c = cfg(**ov); cp = cfg(mu=MU, **ov)
    clients = [(X[tr[h]], y[tr[h]]) for h in names]
    all_tr = np.concatenate([tr[h] for h in names])
    w_fl = fl.federated_train(clients, c, seed)["w"]
    w_fp = fl.federated_train(clients, cp, seed)["w"]
    w_cen = fl.centralized_train(X[all_tr], y[all_tr], c, seed)
    noh = noh_published(df); ett = ettala_published(df)
    per = {}
    fr_y, fr_p = [], []
    for i, h in enumerate(names):
        t = te[h]
        w_loc = fl.local_only_train(X[tr[h]], y[tr[h]], c, seed)
        w_fr = fl.federated_train([cl for j, cl in enumerate(clients) if j != i], c, seed)["w"]
        all_h = np.concatenate([tr[h], te[h]])
        p_fr_all = _p(X[all_h], w_fr)
        per[h] = {"LOC": fl.safe_auc(y[t], _p(X[t], w_loc)), "FL": fl.safe_auc(y[t], _p(X[t], w_fl)),
                  "FedProx": fl.safe_auc(y[t], _p(X[t], w_fp)), "CEN": fl.safe_auc(y[t], _p(X[t], w_cen)),
                  "FR_test": fl.safe_auc(y[t], _p(X[t], w_fr)),
                  "FR_all": fl.safe_auc(y[all_h], p_fr_all),       # the authors' FR evaluation (100% of data)
                  "BL_noh": fl.safe_auc(y[t], noh[t]), "BL_ettala": fl.safe_auc(y[t], ett[t]),
                  "FR_cal": calibration(y[all_h], p_fr_all)}
        fr_y.append(y[all_h]); fr_p.append(p_fr_all)
    keys = ["LOC", "FL", "FedProx", "CEN", "FR_test", "FR_all", "BL_noh", "BL_ettala"]
    sizes = np.array([len(tr[h]) + len(te[h]) for h in names], float)
    macro = {k: float(np.mean([per[h][k] for h in names])) for k in keys}
    weighted = {k: float(np.average([per[h][k] for h in names], weights=sizes)) for k in keys}
    # learning curve: 5 random participation orders per seed (hospitals chosen at random, no size preference)
    rng = np.random.default_rng(5000 + seed)
    lc = {K: {"FL": [], "FR": []} for K in range(1, len(names))}
    for _ in range(5):
        order = rng.permutation(len(names))
        for K in range(1, len(names)):
            part, non = order[:K], order[K:]
            w = fl.federated_train([clients[j] for j in part], c, seed)["w"]
            lc[K]["FL"].append(np.mean([fl.safe_auc(y[te[names[j]]], _p(X[te[names[j]]], w)) for j in part]))
            nh = [np.concatenate([tr[names[j]], te[names[j]]]) for j in non]
            lc[K]["FR"].append(np.mean([fl.safe_auc(y[a], _p(X[a], w)) for a in nh]))
    return {"seed": seed, "per": per, "macro": macro, "weighted": weighted,
            "lc": {K: {k: float(np.mean(v)) for k, v in d.items()} for K, d in lc.items()},
            "fr_cal_pooled": calibration(np.concatenate(fr_y), np.concatenate(fr_p))}


def strategies(seeds=SEEDS, processes=2, overrides=None):
    df = load_pca19()
    names = hospitals_by_size(df)
    with Pool(processes) as pool:
        runs = pool.map(_strategies_seed, [s if not overrides else (s, overrides) for s in seeds])
    keys = ["LOC", "FL", "FedProx", "CEN", "FR_test", "FR_all", "BL_noh", "BL_ettala"]
    per = []
    for h in names:
        sub = df[df.hospital == h]
        row = {"hospital": h, "country": COUNTRY[h], "n": int(len(sub)), "prevalence": float(sub.y.mean())}
        for k in keys:
            row[k] = summary([r["per"][h][k] for r in runs])["mean"]
        d = paired([r["per"][h]["FL"] for r in runs], [r["per"][h]["LOC"] for r in runs])
        row["FL_minus_LOC"] = d
        row["FR_cal"] = {m: summary([r["per"][h]["FR_cal"][m] for r in runs])["mean"] for m in ("oe", "slope", "brier")}
        per.append(row)
    M = lambda a, b, agg="macro": paired([r[agg][a] for r in runs], [r[agg][b] for r in runs])
    out = {"hospitals": per,
           "macro": {k: summary([r["macro"][k] for r in runs]) for k in keys},
           "weighted": {k: summary([r["weighted"][k] for r in runs]) for k in keys},
           "diff": {"FL-LOC": M("FL", "LOC"), "FR_test-FL": M("FR_test", "FL"), "CEN-FL": M("CEN", "FL"),
                    "FedProx-FL": M("FedProx", "FL"), "FL-BL_noh": M("FL", "BL_noh"),
                    "FL-BL_ettala": M("FL", "BL_ettala"), "FL-LOC_weighted": M("FL", "LOC", "weighted")},
           "learning_curve": [{"K": K, "FL": summary([r["lc"][K]["FL"] for r in runs]),
                               "FR": summary([r["lc"][K]["FR"] for r in runs])} for K in range(1, len(names))],
           "fr_calibration_pooled": {m: summary([r["fr_cal_pooled"][m] for r in runs]) for m in ("oe", "slope", "brier")}}
    # which hospitals gain from joining? (size vs per-hospital FL-LOC)
    from scipy.stats import spearmanr
    out["gain_vs_size_spearman"] = float(spearmanr([p["n"] for p in per], [p["FL_minus_LOC"]["mean"] for p in per]).correlation)
    out["n_gain_sig"] = int(sum(1 for p in per if p["FL_minus_LOC"]["ci_low"] is not None and p["FL_minus_LOC"]["ci_low"] > 0))
    out["n_loss_sig"] = int(sum(1 for p in per if p["FL_minus_LOC"]["ci_high"] is not None and p["FL_minus_LOC"]["ci_high"] < 0))
    out["training"] = cfg(**(overrides or {})).__dict__
    return out


# ---- B. leave-one-country-out external validation -----------------------------------------
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _loco_seed(seed):
    df, names, tr, te, X, y = split(seed)
    countries = sorted(set(COUNTRY.values()))
    noh = noh_published(df)
    out = {}
    sig5 = cfg(epsilon=5.0).sigma(); sig1 = cfg(epsilon=1.0).sigma()
    for ctry in countries:
        held = [h for h in names if COUNTRY[h] == ctry]
        train_h = [h for h in names if COUNTRY[h] != ctry]
        # standardiser re-fitted on the training countries only (no information from the held-out country)
        raw = core_raw(df)
        tr_rows = np.concatenate([tr[h] for h in train_h])
        st = Standardiser().fit(raw.iloc[tr_rows]); Xc = st.transform(raw)
        clients = [(Xc[tr[h]], y[tr[h]]) for h in train_h]
        test = np.concatenate([np.concatenate([tr[h], te[h]]) for h in held])  # 100% of the held-out country
        models = {"FedAvg": fl.federated_train(clients, cfg(), seed)["w"],
                  "FedProx": fl.federated_train(clients, cfg(mu=MU), seed)["w"],
                  "FedProx+DP(eps=5)": fl.federated_train(clients, cfg(mu=MU, epsilon=5.0), seed, sigma=sig5)["w"],
                  "FedProx+DP(eps=1)": fl.federated_train(clients, cfg(mu=MU, epsilon=1.0), seed, sigma=sig1)["w"],
                  "Centralised": fl.centralized_train(Xc[tr_rows], y[tr_rows], cfg(), seed)}
        # local recalibration: 20% of the held-out country (stratified) recalibrates, 80% evaluates
        cal, ev = train_test_split(np.arange(len(test)), test_size=0.8, stratify=y[test],
                                   random_state=np.random.RandomState(seed).randint(1 << 30))
        res = {}
        preds = {k: _p(Xc[test], w) for k, w in models.items()}
        preds["Noh (published)"] = noh[test]
        for k, p in preds.items():
            yt = y[test]
            res[k] = {"auc": fl.safe_auc(yt, p), **calibration(yt, p), "nb": net_benefit(yt, p, THRESHOLDS)}
            rc = {"raw": p[ev], "intercept": recalibrate(yt[cal], p[cal], p[ev], "intercept"),
                  "logistic": recalibrate(yt[cal], p[cal], p[ev], "logistic")}
            res[k]["recal"] = {m: {**calibration(yt[ev], q), "nb": net_benefit(yt[ev], q, THRESHOLDS)} for m, q in rc.items()}
            res[k]["recal"]["n_cal"] = int(len(cal))
        prev = y[test].mean()
        res["treat_all_nb"] = [float(prev - (1 - prev) * t / (1 - t)) for t in THRESHOLDS]
        res["n"] = int(len(test)); res["prevalence"] = float(prev)
        out[ctry] = res
    return out


def loco(seeds=SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_loco_seed, seeds)
    countries = list(runs[0].keys())
    models = [k for k in runs[0][countries[0]] if isinstance(runs[0][countries[0]][k], dict)]
    rows = []
    for c in countries:
        row = {"country": c, "n": runs[0][c]["n"], "prevalence": runs[0][c]["prevalence"]}
        for m in models:
            row[m] = {k: summary([r[c][m][k] for r in runs])["mean"] for k in ("auc", "oe", "slope", "brier")}
        rows.append(row)
    macro = {m: summary([np.mean([r[c][m]["auc"] for c in countries]) for r in runs]) for m in models}
    diffs = {f"{m} - Noh (published)": paired([np.mean([r[c][m]["auc"] for c in countries]) for r in runs],
                                              [np.mean([r[c]["Noh (published)"]["auc"] for c in countries]) for r in runs])
             for m in models if m != "Noh (published)"}
    diffs["FedProx+DP(eps=5) - FedProx"] = paired([np.mean([r[c]["FedProx+DP(eps=5)"]["auc"] for c in countries]) for r in runs],
                                                  [np.mean([r[c]["FedProx"]["auc"] for c in countries]) for r in runs])
    diffs["FedProx - Centralised"] = paired([np.mean([r[c]["FedProx"]["auc"] for c in countries]) for r in runs],
                                            [np.mean([r[c]["Centralised"]["auc"] for c in countries]) for r in runs])
    # decision curve: patient-weighted net benefit over all held-out countries
    dca = {m: [summary([np.average([r[c][m]["nb"][i] for c in countries], weights=[r[c]["n"] for c in countries])
                        for r in runs])["mean"] for i in range(len(THRESHOLDS))] for m in models}
    dca["Biopsy all"] = [float(np.average([runs[0][c]["treat_all_nb"][i] for c in countries],
                                          weights=[runs[0][c]["n"] for c in countries])) for i in range(len(THRESHOLDS))]
    dca["Biopsy none"] = [0.0] * len(THRESHOLDS)
    # country-level inference: each of the 8 countries is one independent external test
    cmean = lambda m, key="auc": [float(np.mean([r[c][m][key] for r in runs])) for c in countries]
    by_country = {f"{m} - Noh (published)": paired(cmean(m), cmean("Noh (published)")) for m in models if m != "Noh (published)"}
    by_country["FedProx+DP(eps=5) - FedProx"] = paired(cmean("FedProx+DP(eps=5)"), cmean("FedProx"))
    by_country["FedProx+DP(eps=1) - FedProx"] = paired(cmean("FedProx+DP(eps=1)"), cmean("FedProx"))
    by_country["FedAvg - Centralised"] = paired(cmean("FedAvg"), cmean("Centralised"))
    slope_dp = {m: summary(cmean(m, "slope")) for m in models}
    # recalibration: |log O/E| and slope, averaged over countries
    recal = {}
    for m in models:
        recal[m] = {}
        for meth in ("raw", "intercept", "logistic"):
            abs_log_oe = [float(np.mean([abs(math.log(r[c][m]["recal"][meth]["oe"])) for r in runs])) for c in countries]
            slope = [float(np.mean([r[c][m]["recal"][meth]["slope"] for r in runs])) for c in countries]
            brier = [float(np.mean([r[c][m]["recal"][meth]["brier"] for r in runs])) for c in countries]
            oe_rng = [float(np.mean([r[c][m]["recal"][meth]["oe"] for r in runs])) for c in countries]
            recal[m][meth] = {"abs_log_oe": summary(abs_log_oe), "slope": summary(slope), "brier": summary(brier),
                              "oe_by_country": dict(zip(countries, oe_rng))}
        recal[m]["n_cal_by_country"] = {c: runs[0][c][m]["recal"]["n_cal"] for c in countries}
        recal[m]["intercept_vs_raw_abs_log_oe"] = paired(
            [float(np.mean([abs(math.log(r[c][m]["recal"]["intercept"]["oe"])) for r in runs])) for c in countries],
            [float(np.mean([abs(math.log(r[c][m]["recal"]["raw"]["oe"])) for r in runs])) for c in countries])
        recal[m]["logistic_vs_raw_abs_slope_err"] = paired(
            [float(np.mean([abs(r[c][m]["recal"]["logistic"]["slope"] - 1) for r in runs])) for c in countries],
            [float(np.mean([abs(r[c][m]["recal"]["raw"]["slope"] - 1) for r in runs])) for c in countries])
    return {"countries": rows, "macro_auc": macro, "diff": diffs, "diff_by_country": by_country,
            "calibration_slope_by_model": slope_dp, "recalibration": recal,
            "thresholds": THRESHOLDS, "dca": dca, "models": models,
            "note": "Seed-level CIs (diff) reflect only training-sample variation; diff_by_country treats the 8 countries as independent external tests (n = 8)."}


# ---- C. drift and privacy on the real hospitals --------------------------------------------
def _privacy_seed(seed):
    df, names, tr, te, X, y = split(seed)
    clients = [(X[tr[h]], y[tr[h]]) for h in names]
    tests = [(X[te[h]], y[te[h]]) for h in names]
    yt = np.concatenate([t[1] for t in tests]); Xt = np.vstack([t[0] for t in tests])
    out = {}
    for eps in [None] + EPSILONS:
        sig = None if eps is None else cfg(epsilon=eps).sigma()
        cell = {}
        for name, c in (("FedAvg", cfg(epsilon=eps)), ("FedProx", cfg(epsilon=eps, mu=MU)),
                        ("FedAvg_half_lr", cfg(epsilon=eps, lr=BASE_CFG["lr"] / 2))):
            r = fl.federated_train(clients, c, seed, sigma=sig)
            w = r["w"]
            per = [fl.safe_auc(t[1], _p(t[0], w)) for t in tests]
            cell[name] = {"macro": float(np.mean(per)), "pooled": fl.safe_auc(yt, _p(Xt, w)), "per": per,
                          "drift": float(np.mean(r["drift_history"]))}
        out["none" if eps is None else str(eps)] = cell
    return out


def privacy(seeds=SEEDS, processes=2):
    names = hospitals_by_size()
    with Pool(processes) as pool:
        runs = pool.map(_privacy_seed, seeds)
    keys = ["none"] + [str(e) for e in EPSILONS]
    algs = ["FedAvg", "FedProx", "FedAvg_half_lr"]
    cells = []
    for k in keys:
        row = {"epsilon": None if k == "none" else float(k)}
        for a in algs:
            row[a] = {"macro": summary([r[k][a]["macro"] for r in runs]), "pooled": summary([r[k][a]["pooled"] for r in runs]),
                      "drift": summary([r[k][a]["drift"] for r in runs])}
        row["FedProx-FedAvg"] = paired([r[k]["FedProx"]["macro"] for r in runs], [r[k]["FedAvg"]["macro"] for r in runs])
        row["FedProx-FedAvg_half_lr"] = paired([r[k]["FedProx"]["macro"] for r in runs], [r[k]["FedAvg_half_lr"]["macro"] for r in runs])
        if k != "none":
            row["FedAvg-noDP"] = paired([r[k]["FedAvg"]["macro"] for r in runs], [r["none"]["FedAvg"]["macro"] for r in runs])
            row["FedProx-noDP"] = paired([r[k]["FedProx"]["macro"] for r in runs], [r["none"]["FedProx"]["macro"] for r in runs])
        cells.append(row)
    drift = paired([r["none"]["FedProx"]["drift"] for r in runs], [r["none"]["FedAvg"]["drift"] for r in runs])
    red = [100 * (1 - r["none"]["FedProx"]["drift"] / r["none"]["FedAvg"]["drift"]) for r in runs]
    # who pays for privacy? per-hospital AUC loss at eps=1 (FedProx) vs hospital size
    from scipy.stats import spearmanr
    sizes = load_pca19().hospital.value_counts()[names].values
    loss = [float(np.nanmean([r["1.0"]["FedProx"]["per"][i] - r["none"]["FedProx"]["per"][i] for r in runs])) for i in range(len(names))]
    return {"cells": cells, "sigma": {str(e): cfg(epsilon=e).sigma() for e in EPSILONS},
            "steps": cfg().steps, "drift_FedProx_minus_FedAvg": drift, "drift_reduction_pct": summary(red),
            "per_hospital_loss_eps1": dict(zip(names, loss)),
            "loss_vs_size_spearman": float(spearmanr(sizes, loss).correlation)}


# ---- D. personalisation on the real hospitals ---------------------------------------------
def _pfl_seed(seed):
    df, names, tr, te, X, y = split(seed)
    clients = [(X[tr[h]], y[tr[h]]) for h in names]
    c = cfg()
    w_g = fl.federated_train(clients, cfg(mu=MU), seed)["w"]
    out = {}
    for h, (Xk, yk) in zip(names, clients):
        Xt, yt = X[te[h]], y[te[h]]
        out[h] = {"Local": fl.safe_auc(yt, _p(Xt, fl.local_only_train(Xk, yk, c, seed))),
                  "FedProx": fl.safe_auc(yt, _p(Xt, w_g)),
                  "PFL": fl.safe_auc(yt, _p(Xt, fl.personalise(Xk, yk, w_g)))}
    return out


def personalisation(seeds=SEEDS, processes=2):
    names = hospitals_by_size()
    with Pool(processes) as pool:
        runs = pool.map(_pfl_seed, seeds)
    sizes = load_pca19().hospital.value_counts()
    groups = {"small (< 100)": [h for h in names if sizes[h] < 100],
              "medium (100-399)": [h for h in names if 100 <= sizes[h] < 400],
              "large (>= 400)": [h for h in names if sizes[h] >= 400]}
    out = {"per_hospital": {h: {m: summary([r[h][m] for r in runs])["mean"] for m in ("Local", "FedProx", "PFL")} for h in names},
           "groups": {}}
    for g, hs in list(groups.items()) + [("all", names)]:
        agg = {m: [np.mean([r[h][m] for h in hs]) for r in runs] for m in ("Local", "FedProx", "PFL")}
        out["groups"][g] = {"hospitals": hs, **{m: summary(v) for m, v in agg.items()},
                            "PFL-Local": paired(agg["PFL"], agg["Local"]),
                            "PFL-FedProx": paired(agg["PFL"], agg["FedProx"]),
                            "FedProx-Local": paired(agg["FedProx"], agg["Local"])}
    return out


# ---- E. contribution valuation under DP ---------------------------------------------------
def coalition_utilities_masks(clients, X_ref, y_ref, cfg_: fl.TrainConfig, masks, seed=0, sigma=None):
    """Vectorised FedAvg/FedProx training of an arbitrary list of coalitions (rows of `masks`).
    Same mathematics as valuation.all_coalition_utilities, which is the special case of all
    2^K - 1 masks. Returns an array of utilities (reference-set AUC; 0.5 if undefined)."""
    from .fl import add_bias, sigmoid
    masks = np.asarray(masks, float)
    if sigma is None and cfg_.epsilon is not None:
        sigma = cfg_.sigma()
    rng = np.random.default_rng(seed)
    sizes = np.array([len(c[1]) for c in clients], float)
    wts = masks * sizes
    wts = wts / wts.sum(1, keepdims=True)
    d = clients[0][0].shape[1] + 1
    W = np.zeros((len(masks), d))
    Xbs = [add_bias(Xk) for Xk, _ in clients]
    for _ in range(cfg_.rounds):
        newW = np.zeros_like(W)
        for k, (Xb, (_, yk)) in enumerate(zip(Xbs, clients)):
            idx = np.where(masks[:, k] > 0)[0]
            if len(idx) == 0:
                continue
            Wk = W[idx].copy(); anchor = W[idx]; n = len(yk)
            for _e in range(cfg_.epochs):
                R = sigmoid(Xb @ Wk.T) - yk[:, None]
                if sigma is None:
                    G = (R.T @ Xb) / n
                else:
                    per = R.T[:, :, None] * Xb[None, :, :]
                    nr = np.maximum(np.linalg.norm(per, axis=2, keepdims=True), 1e-12)
                    per = per * np.minimum(1.0, cfg_.clip / nr)
                    G = (per.sum(1) + rng.normal(0.0, sigma * cfg_.clip, size=Wk.shape)) / n
                G = G + cfg_.l2 * Wk
                if cfg_.mu > 0:
                    G = G + cfg_.mu * (Wk - anchor)
                Wk = Wk - cfg_.lr * G
            newW[idx] += wts[idx, k][:, None] * Wk
        W = newW
    P = sigmoid(add_bias(X_ref) @ W.T)
    u = np.array([fl.safe_auc(y_ref, P[:, i]) for i in range(len(masks))])
    return np.where(np.isnan(u), 0.5, u)


def mc_shapley(clients, X_ref, y_ref, cfg_, n_perm=200, seed=0, sigma=None, noise_seed=None):
    """Permutation-sampling (Monte-Carlo) Shapley for many hospitals. All prefix coalitions of
    all sampled permutations are trained in one vectorised batch. `seed` fixes the permutations
    (shared between the DP and no-DP runs); `noise_seed` fixes the DP noise."""
    K = len(clients)
    rng = np.random.default_rng(10_000 + seed)
    perms = [rng.permutation(K) for _ in range(n_perm)]
    keyset = {}
    for p in perms:
        for j in range(1, K + 1):
            keyset.setdefault(tuple(sorted(p[:j].tolist())), None)
    keys = list(keyset)
    masks = np.zeros((len(keys), K))
    for i, S in enumerate(keys):
        masks[i, list(S)] = 1
    u = dict(zip(keys, coalition_utilities_masks(clients, X_ref, y_ref, cfg_, masks,
                                                   seed if noise_seed is None else noise_seed, sigma)))
    u[()] = 0.5
    phi = np.zeros(K)
    for p in perms:
        prev = ()
        for j in range(K):
            cur = tuple(sorted(p[:j + 1].tolist()))
            phi[p[j]] += u[cur] - u[prev]
            prev = cur
    return phi / n_perm, len(keys)


def stratified_shapley(clients, X_ref, y_ref, cfg_, m=3, seed=0, sigma=None, noise_seed=None):
    """Stratified-sampling Shapley (Maleki et al., 2013). phi_k = (1/K) * sum over coalition
    sizes s of E[v(S + k) - v(S)] for |S| = s. Sizes 0 (hospital alone) and K-1 (leave-one-out)
    contain one coalition each and are computed exactly - these strata carry most of the variance
    of plain permutation sampling because v(empty) = 0.5. The other strata use `m` random
    coalitions per hospital and size. `seed` fixes the sampled coalitions (shared between DP and
    no-DP runs); `noise_seed` fixes the DP noise. All coalitions are trained in one batch."""
    K = len(clients)
    rng = np.random.default_rng(20_000 + seed)
    plan = {}
    for k in range(K):
        others = [i for i in range(K) if i != k]
        for s in range(K):
            if s == 0:
                samples = [()]
            elif s == K - 1:
                samples = [tuple(others)]
            else:
                samples = [tuple(sorted(rng.choice(others, s, replace=False).tolist())) for _ in range(m)]
            plan[(k, s)] = samples
    keyset = {}
    for (k, s), Ss in plan.items():
        for S in Ss:
            keyset.setdefault(S, None); keyset.setdefault(tuple(sorted(S + (k,))), None)
    keyset.pop((), None)
    keys = list(keyset)
    masks = np.zeros((len(keys), K))
    for i, S in enumerate(keys):
        masks[i, list(S)] = 1
    u = dict(zip(keys, coalition_utilities_masks(clients, X_ref, y_ref, cfg_, masks,
                                                 seed if noise_seed is None else noise_seed, sigma)))
    u[()] = 0.5
    phi = np.zeros(K)
    for (k, s), Ss in plan.items():
        phi[k] += np.mean([u[tuple(sorted(S + (k,)))] - u[S] for S in Ss]) / K
    return phi, len(keys)


def _shap_seed(args):
    seed, K10, n_perm, R = args
    df, names, tr, te, X, y = split(seed)
    out = {}
    # (i) exact Shapley over the 10 largest hospitals
    h10 = names[:K10]
    cl = [(X[tr[h]], y[tr[h]]) for h in h10]
    ref = np.concatenate([te[h] for h in h10])
    masks = np.array([[(m >> k) & 1 for k in range(K10)] for m in range(1, 2 ** K10)], float)
    keys = [tuple(np.where(m > 0)[0].tolist()) for m in masks]

    def exact(eps, s):
        c = cfg(epsilon=eps)
        u = dict(zip(keys, coalition_utilities_masks(cl, X[ref], y[ref], c, masks, s)))
        u[()] = 0.5
        return V.shapley_fast(u, K10), V.loo_from_utilities(u, K10)

    phi0, loo0 = exact(None, seed)
    out["exact10"] = {"hospitals": h10, "phi_nodp": phi0.tolist(), "loo_nodp": loo0.tolist(), "dp": {}}
    for eps in EPSILONS:
        phis = [exact(eps, 1000 * (r + 1) + seed)[0] for r in range(R)]
        out["exact10"]["dp"][str(eps)] = V.rank_stability(phi0, phis)
    # (ii) Monte-Carlo Shapley over all 19 hospitals
    cl19 = [(X[tr[h]], y[tr[h]]) for h in names]
    ref19 = np.concatenate([te[h] for h in names])
    m0, nco = stratified_shapley(cl19, X[ref19], y[ref19], cfg(), n_perm, seed)
    out["mc19"] = {"hospitals": names, "phi_nodp": m0.tolist(), "coalitions": nco, "dp": {}}
    for eps in (5.0, 1.0):
        m_dp, _ = stratified_shapley(cl19, X[ref19], y[ref19], cfg(epsilon=eps), n_perm, seed, noise_seed=7000 + seed)
        out["mc19"]["dp"][str(eps)] = V.rank_stability(m0, [m_dp])
    return out


def shapley(seeds=SEEDS, processes=2, K10=10, n_perm=3, R=1):
    from scipy.stats import spearmanr
    with Pool(processes) as pool:
        runs = pool.map(_shap_seed, [(s, K10, n_perm, R) for s in seeds])
    h10 = runs[0]["exact10"]["hospitals"]; names = runs[0]["mc19"]["hospitals"]
    sizes = load_pca19().hospital.value_counts()
    ex = {"hospitals": h10, "phi_nodp": {h: summary([r["exact10"]["phi_nodp"][i] for r in runs]) for i, h in enumerate(h10)},
          "loo_nodp": {h: summary([r["exact10"]["loo_nodp"][i] for r in runs]) for i, h in enumerate(h10)},
          "spearman_shapley_vs_loo": summary([spearmanr(r["exact10"]["phi_nodp"], r["exact10"]["loo_nodp"]).correlation for r in runs]),
          "spearman_shapley_vs_size": float(spearmanr([sizes[h] for h in h10],
                                                      [np.mean([r["exact10"]["phi_nodp"][i] for r in runs]) for i in range(len(h10))]).correlation),
          "per_epsilon": []}
    for eps in EPSILONS:
        k = str(eps)
        ex["per_epsilon"].append({"epsilon": eps, **{m: summary([np.mean(r["exact10"]["dp"][k][m]) for r in runs])
                                                     for m in ("spearman", "rank_reversal", "top_contributor_stability")}})
    mc = {"hospitals": names, "coalitions_per_seed": runs[0]["mc19"]["coalitions"], "samples_per_stratum": n_perm,
          "method": "stratified sampling by coalition size (sizes 0 and K-1 exact)",
          "phi_nodp": {h: summary([r["mc19"]["phi_nodp"][i] for r in runs]) for i, h in enumerate(names)},
          "spearman_shapley_vs_size": float(spearmanr([sizes[h] for h in names],
                                                      [np.mean([r["mc19"]["phi_nodp"][i] for r in runs]) for i in range(len(names))]).correlation),
          "per_epsilon": [{"epsilon": float(k), **{m: summary([r["mc19"]["dp"][k][m][0] for r in runs])
                                                   for m in ("spearman", "rank_reversal", "top_contributor_stability")}}
                          for k in ("5.0", "1.0")]}
    # efficiency check: sum of exact Shapley = v(all) - v(empty)
    return {"exact10": ex, "mc19": mc, "K10": K10, "realisations": R}


# ---- F. membership inference ---------------------------------------------------------------
def _mia_seed(seed):
    df, names, tr, te, X, y = split(seed)
    clients = [(X[tr[h]], y[tr[h]]) for h in names]
    Xm = np.vstack([c[0] for c in clients]); ym = np.concatenate([c[1] for c in clients])
    te_all = np.concatenate([te[h] for h in names])
    out = {}
    for eps in [None] + EPSILONS:
        w = fl.federated_train(clients, cfg(mu=MU, epsilon=eps), seed)["w"]
        out["none" if eps is None else str(eps)] = fl.mia_confidence_attack(w, Xm, ym, X[te_all], y[te_all], seed)[0]
    # positive control: a model that can memorise (60 training patients + 200 random noise features)
    rng = np.random.default_rng(seed)
    i_m = rng.choice(len(ym), 60, replace=False); i_n = rng.choice(len(te_all), 60, replace=False)
    Z = rng.normal(size=(len(y), 200))
    Xa = np.hstack([X, Z])
    m = LogisticRegression(C=1e4, max_iter=5000).fit(np.vstack([Xa[np.concatenate([tr[h] for h in names])][i_m]]), ym[i_m])
    w = np.r_[m.coef_.ravel(), m.intercept_]          # bias is the last weight in fl.add_bias
    out["control"] = fl.mia_confidence_attack(w, Xa[np.concatenate([tr[h] for h in names])][i_m], ym[i_m],
                                              Xa[te_all][i_n], y[te_all][i_n], seed)[0]
    return out


def mia(seeds=SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_mia_seed, seeds)
    keys = ["none"] + [str(e) for e in EPSILONS]
    return {"rows": [{"epsilon": None if k == "none" else float(k), "attack_auc": summary([r[k] for r in runs])} for k in keys],
            "control": summary([r["control"] for r in runs])}


# ---- cohort facts --------------------------------------------------------------------------
def cohort():
    df = load_pca19()
    names = hospitals_by_size(df)
    rows = []
    for h in names:
        s = df[df.hospital == h]
        rows.append({"hospital": h, "country": COUNTRY[h], "n": int(len(s)), "prevalence": float(s.y.mean()),
                     "age_median": float(s.age.median()), "psa_median": float(s.PSA.median()),
                     "pv_median": float(s.PV.median()), "pirads_ge4": float((s.PIRADS >= 4).mean())})
    return {"n": int(len(df)), "n_pos": int(df.y.sum()), "hospitals": rows, "n_hospitals": len(names),
            "n_countries": int(df.country.nunique()), "features": CORE_FEATURES,
            "prevalence_range": [float(min(r["prevalence"] for r in rows)), float(max(r["prevalence"] for r in rows))],
            "source": "github.com/phaseivai/Hospital-Participation-in-FL (MIT), data/data.pkl"}


# ---- G. privacy-honest valuation: OSA-Shapley vs retraining ------------------------------
VAL_SEEDS = list(range(20, 40))      # disjoint from the seeds used to tune OSA_STEPS (0-3)


def _valuation_compare(cl, Xr, yr, seed, eps_list, K_tag):
    """Compare valuation methods against the exact no-DP retraining Shapley values ('truth')."""
    from . import private_valuation as PV
    K = len(cl)
    masks = np.array([[(m >> k) & 1 for k in range(K)] for m in range(1, 2 ** K)], float)
    keys = [tuple(np.where(m > 0)[0].tolist()) for m in masks]

    def shap_retrain(c_, s_, sigma=None):
        u = dict(zip(keys, coalition_utilities_masks(cl, Xr, yr, c_, masks, s_, sigma)))
        u[()] = 0.5
        return V.shapley_fast(u, K)

    truth = shap_retrain(cfg(), seed)
    out = {"truth": truth.tolist(), "osa_nodp": V.rank_stability(truth, [PV.osa_shapley(cl, Xr, yr, None)]), "eps": {}}
    for eps in eps_list:
        c_ = cfg(epsilon=eps)
        sig_bm = fl.calibrate_sigma(eps, c_.steps * 2 ** (K - 1), c_.delta)
        rng = np.random.default_rng(5000 + seed)
        out["eps"][str(eps)] = {
            "retrain_per_run": V.rank_stability(truth, [shap_retrain(c_, 1000 + seed)]),
            "retrain_budget_matched": V.rank_stability(truth, [shap_retrain(c_, 2000 + seed, sigma=sig_bm)]),
            "osa": V.rank_stability(truth, [PV.osa_shapley(cl, Xr, yr, eps, rng=rng)]),
            "retrain_true_epsilon": PV.retraining_privacy_cost(eps, K, c_.steps),
        }
    return out


def _pv_seed(seed):
    from . import private_valuation as PV
    df, names, tr, te, X, y = split(seed)
    h10 = names[:10]
    cl = [(X[tr[h]], y[tr[h]]) for h in h10]
    ref = np.concatenate([te[h] for h in h10])
    res = {"k10": _valuation_compare(cl, X[ref], y[ref], seed, EPSILONS, "PCa19-10")}
    # all 19 hospitals: exact OSA-Shapley (2^19 - 1 coalitions); no exact retraining truth exists here
    cl19 = [(X[tr[h]], y[tr[h]]) for h in names]
    ref19 = np.concatenate([te[h] for h in names])
    phi0 = PV.osa_shapley(cl19, X[ref19], y[ref19], None)
    res["k19"] = {"phi_nodp": phi0.tolist(), "eps": {}}
    for eps in (5.0, 1.0):
        phi = PV.osa_shapley(cl19, X[ref19], y[ref19], eps, rng=np.random.default_rng(9000 + seed))
        res["k19"]["eps"][str(eps)] = V.rank_stability(phi0, [phi])
    return res


def private_valuation(seeds=VAL_SEEDS, processes=2):
    from scipy.stats import spearmanr
    from . import private_valuation as PV
    with Pool(processes) as pool:
        runs = pool.map(_pv_seed, seeds)
    return _summarise_valuation([r["k10"] for r in runs], EPSILONS) | {
        "k19": {"hospitals": hospitals_by_size(),
                "phi_nodp": {h: summary([r["k19"]["phi_nodp"][i] for r in runs]) for i, h in enumerate(hospitals_by_size())},
                "per_epsilon": [{"epsilon": float(k), **{m: summary([r["k19"]["eps"][k][m][0] for r in runs])
                                                         for m in ("spearman", "rank_reversal", "top_contributor_stability")}}
                                for k in ("5.0", "1.0")],
                "spearman_vs_size": float(spearmanr(load_pca19().hospital.value_counts()[hospitals_by_size()].values,
                                                    [np.mean([r["k19"]["phi_nodp"][i] for r in runs]) for i in range(19)]).correlation)},
        "seeds": seeds, "osa_steps": PV.OSA_STEPS, "osa_lr": PV.OSA_LR}


def _summarise_valuation(runs, eps_list):
    methods = ["retrain_per_run", "retrain_budget_matched", "osa"]
    rows = []
    for eps in eps_list:
        k = str(eps)
        row = {"epsilon": eps, "retrain_true_epsilon": runs[0]["eps"][k]["retrain_true_epsilon"]}
        for m in methods:
            row[m] = {met: summary([r["eps"][k][m][met][0] for r in runs])
                      for met in ("spearman", "rank_reversal", "top_contributor_stability")}
        row["osa_minus_budget_matched"] = paired([r["eps"][k]["osa"]["spearman"][0] for r in runs],
                                                 [r["eps"][k]["retrain_budget_matched"]["spearman"][0] for r in runs])
        row["osa_minus_per_run"] = paired([r["eps"][k]["osa"]["spearman"][0] for r in runs],
                                          [r["eps"][k]["retrain_per_run"]["spearman"][0] for r in runs])
        rows.append(row)
    return {"rows": rows, "osa_nodp": {met: summary([r["osa_nodp"][met][0] for r in runs])
                                       for met in ("spearman", "rank_reversal", "top_contributor_stability")}}


def _pv_tcga_seed(seed):
    """Same comparison on TCGA-PRAD: 10 Dirichlet hospitals (alpha = 0.5), matched cohort."""
    from . import experiments as E
    Xtr, ytr, Xte, yte, _ = E.matched_split(seed)
    cl, _ = E.dirichlet_clients(Xtr, ytr, Xte, yte, 10, 0.5, seed)
    return _valuation_compare(cl, Xte, yte, seed, EPSILONS, "TCGA-10")


def private_valuation_tcga(seeds=SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_pv_tcga_seed, seeds)
    return _summarise_valuation(runs, EPSILONS) | {"seeds": seeds}


# ---- G2. valuation, extended: post-processing competitors, corrupted hospital,
#          removal, sensitivity ------------------------------------------------------------------
VP_EPS = [5.0, 2.0, 1.0, 0.5]
CORRUPT_INDEX = 5          # hospital (by size rank among the 10 largest) given random training labels


def _exact_retrain(cl, Xr, yr, c_, seed, sigma=None):
    K = len(cl)
    masks = np.array([[(m >> k) & 1 for k in range(K)] for m in range(1, 2 ** K)], float)
    keys = [tuple(np.where(m > 0)[0].tolist()) for m in masks]
    u = dict(zip(keys, coalition_utilities_masks(cl, Xr, yr, c_, masks, seed, sigma)))
    u[()] = 0.5
    return V.shapley_fast(u, K), u


def _post_methods(cl, Xr, yr, eps, seed, steps=None):
    """phi of the three post-processing valuations that cost epsilon once: OSA, MR, OR."""
    from . import private_valuation as PV
    sizes = [len(c[1]) for c in cl]
    osa = PV.osa_shapley(cl, Xr, yr, eps, rng=np.random.default_rng(5000 + seed),
                         **({} if steps is None else {"steps": steps}))
    trace, _ = PV.fl_trace(cl, cfg(epsilon=eps), 3000 + seed)
    return {"osa": osa, "mr": PV.mr_shapley(trace, sizes, Xr, yr), "or": PV.or_shapley(trace, sizes, Xr, yr)}


def _vp_seed(args):
    seed, heavy = args
    from . import private_valuation as PV
    df, names, tr, te, X, y = split(seed)
    h10 = names[:10]
    cl = [(X[tr[h]], y[tr[h]]) for h in h10]
    ref = np.concatenate([te[h] for h in h10])
    Xr, yr = X[ref], y[ref]
    truth, u_true = _exact_retrain(cl, Xr, yr, cfg(), seed)
    out = {"truth": truth.tolist(), "methods": {}, "removal": {}, "sens": {}, "corrupt": {}}
    rs = lambda phi, ref_=truth: float(V.rank_stability(ref_, [phi])["spearman"][0])
    # (1) post-processing competitors, all at a true per-hospital cost of epsilon
    nod = _post_methods(cl, Xr, yr, None, seed)
    out["methods"]["none"] = {m: rs(p) for m, p in nod.items()}
    full = tuple(range(10))
    for eps in VP_EPS:
        ph = _post_methods(cl, Xr, yr, eps, seed)
        out["methods"][str(eps)] = {m: rs(p) for m, p in ph.items()}
        # (2) removal: AUC of the federation after removing the top-3 hospitals by each method's ranking
        if eps in (2.0, 1.0):
            rem = {}
            for m, p in list(ph.items()) + [("truth", truth)]:
                top = set(np.argsort(-np.asarray(p))[:3].tolist())
                rem[m] = u_true[tuple(i for i in full if i not in top)]
            rr = np.random.default_rng(seed)
            rem["random"] = float(np.mean([u_true[tuple(i for i in full if i not in set(rr.choice(10, 3, replace=False).tolist()))]
                                           for _ in range(50)]))
            rem["all"] = u_true[full]
            out["removal"][str(eps)] = rem
    # (4) sensitivity: local steps and reference-set size (OSA)
    for steps in (10, 30, 60):
        for eps in (2.0, 1.0):
            phi = PV.osa_shapley(cl, Xr, yr, eps, rng=np.random.default_rng(5000 + seed), steps=steps)
            out["sens"][f"steps={steps},eps={eps}"] = rs(phi)
    rr = np.random.default_rng(100 + seed)
    for frac in (0.25, 0.5):
        idx = np.sort(rr.choice(len(yr), int(frac * len(yr)), replace=False))
        for eps in (2.0, 1.0):
            phi = PV.osa_shapley(cl, Xr[idx], yr[idx], eps, rng=np.random.default_rng(5000 + seed))
            out["sens"][f"ref={frac},eps={eps}"] = rs(phi)
    # (5) a useless hospital: its training labels are replaced by random labels at its own prevalence
    #     (design fixed on tuning seeds 0-3; a 30% flip rate was too weak for even the no-DP truth to detect)
    ycor = cl[CORRUPT_INDEX][1].copy()
    ycor = np.random.default_rng(777 + seed).binomial(1, ycor.mean(), size=len(ycor)).astype(float)
    clc = list(cl); clc[CORRUPT_INDEX] = (cl[CORRUPT_INDEX][0], ycor)
    tc, _ = _exact_retrain(clc, Xr, yr, cfg(), seed)
    def det(phi):
        r = np.argsort(np.argsort(np.asarray(phi)))      # 0 = lowest value
        return {"last": float(r[CORRUPT_INDEX] == 0), "bottom2": float(r[CORRUPT_INDEX] <= 1), "rank": int(r[CORRUPT_INDEX])}
    out["corrupt"]["truth"] = det(tc)
    out["corrupt"]["none"] = {m: det(p) for m, p in _post_methods(clc, Xr, yr, None, seed).items()}
    for eps in (2.0, 1.0):
        res = {m: det(p) for m, p in _post_methods(clc, Xr, yr, eps, seed).items()}
        if heavy:
            c_ = cfg(epsilon=eps)
            res["retrain_per_run"] = det(_exact_retrain(clc, Xr, yr, c_, 1000 + seed)[0])
            sig_bm = fl.calibrate_sigma(eps, c_.steps * 2 ** 9, c_.delta)
            res["retrain_budget_matched"] = det(_exact_retrain(clc, Xr, yr, c_, 2000 + seed, sigma=sig_bm)[0])
        out["corrupt"][str(eps)] = res
    return out


def _vp_tcga_seed(seed):
    from . import experiments as E
    Xtr, ytr, Xte, yte, _ = E.matched_split(seed)
    cl, _ = E.dirichlet_clients(Xtr, ytr, Xte, yte, 10, 0.5, seed)
    truth, _ = _exact_retrain(cl, Xte, yte, cfg(), seed)
    out = {}
    for eps in [None] + VP_EPS:
        ph = _post_methods(cl, Xte, yte, eps, seed)
        out["none" if eps is None else str(eps)] = {m: float(V.rank_stability(truth, [p])["spearman"][0]) for m, p in ph.items()}
    return out


def valuation_plus(seeds=VAL_SEEDS, processes=2, heavy=True, tcga_seeds=SEEDS, runs=None):
    """PCa-19 part on the held-out valuation seeds; TCGA part on the same seeds as private_valuation_tcga."""
    with Pool(processes) as pool:
        if runs is None:
            runs = pool.map(_vp_seed, [(s, heavy) for s in seeds])
        truns = pool.map(_vp_tcga_seed, tcga_seeds)
        truns2 = pool.map(_vp_tcga_seed, seeds)
    keys = ["none"] + [str(e) for e in VP_EPS]
    meth = ["osa", "mr", "or"]
    res = {"seeds": seeds, "tcga_seeds": tcga_seeds, "corrupt_index": CORRUPT_INDEX, "corrupt_hospital": hospitals_by_size()[CORRUPT_INDEX],
           "methods": {k: {m: summary([r["methods"][k][m] for r in runs]) for m in meth} for k in keys},
           "osa_minus_mr": {k: paired([r["methods"][k]["osa"] for r in runs], [r["methods"][k]["mr"] for r in runs]) for k in keys},
           "osa_minus_or": {k: paired([r["methods"][k]["osa"] for r in runs], [r["methods"][k]["or"] for r in runs]) for k in keys},
           "tcga": {k: {m: summary([r[k][m] for r in truns]) for m in meth} for k in keys},
           "tcga_osa_minus_mr": {k: paired([r[k]["osa"] for r in truns], [r[k]["mr"] for r in truns]) for k in keys},
           "tcga_heldout_seeds": {"seeds": seeds,
                                  "methods": {k: {m: summary([r[k][m] for r in truns2]) for m in meth} for k in keys},
                                  "osa_minus_mr": {k: paired([r[k]["osa"] for r in truns2], [r[k]["mr"] for r in truns2]) for k in keys}},
           "removal": {k: {m: summary([r["removal"][k][m] for r in runs]) for m in runs[0]["removal"][k]} for k in ("2.0", "1.0")},
           "sensitivity": {k: summary([r["sens"][k] for r in runs]) for k in runs[0]["sens"]},
           "corrupt": {"truth": {m: summary([r["corrupt"]["truth"][m] for r in runs]) for m in ("last", "bottom2", "rank")}}}
    for k in ["none", "2.0", "1.0"]:
        ms = runs[0]["corrupt"][k].keys()
        res["corrupt"][k] = {m: {s: summary([r["corrupt"][k][m][s] for r in runs]) for s in ("last", "bottom2", "rank")} for m in ms}
    return res


# ---- G3. valuation with a non-convex model (federated MLP) ------------------------------------
MLP_EPS = [5.0, 2.0, 1.0, 0.5]
MLP_HEAVY_SEEDS = 5          # budget-matched DP retraining of all MLP coalitions is run on the first 5 seeds


VMLP_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cache_valuation_mlp")


def _vmlp_seed(args):
    """Cached per seed (and per heavy component) so that an interrupted run resumes."""
    import json as _json
    seed, heavy = args
    os.makedirs(VMLP_CACHE, exist_ok=True)
    f = os.path.join(VMLP_CACHE, f"seed{seed}.json")
    if os.path.exists(f):
        out = _json.load(open(f))
    else:
        out = _vmlp_core(seed)
        _json.dump(out, open(f, "w"))
    if heavy:
        from . import valuation_mlp as VM
        df, names, tr, te, X, y = split(seed)
        cl = [(X[tr[h]], y[tr[h]]) for h in names[:10]]
        ref = np.concatenate([te[h] for h in names[:10]])
        truth = np.array(out["truth"])
        for eps in (2.0, 1.0):
            g = os.path.join(VMLP_CACHE, f"seed{seed}_bm{eps}.json")
            if not os.path.exists(g):
                _, s_bm = VM.retrain_sigmas(eps, 10)
                phi = VM.exact_shapley_retrain(cl, X[ref], y[ref], 2000 + seed, sigma=s_bm, chunk=64)
                _json.dump({"rho": float(V.rank_stability(truth, [phi])["spearman"][0])}, open(g, "w"))
            out["rho"][str(eps)]["retrain_budget_matched"] = _json.load(open(g))["rho"]
    return out


def _vmlp_core(seed):
    from . import valuation_mlp as VM
    df, names, tr, te, X, y = split(seed)
    cl = [(X[tr[h]], y[tr[h]]) for h in names[:10]]
    ref = np.concatenate([te[h] for h in names[:10]])
    Xr, yr = X[ref], y[ref]
    truth = VM.exact_shapley_retrain(cl, Xr, yr, seed)
    rs = lambda phi: float(V.rank_stability(truth, [phi])["spearman"][0])
    out = {"truth": truth.tolist(), "rho": {}}
    for eps in [None] + MLP_EPS:
        k = "none" if eps is None else str(eps)
        out["rho"][k] = {"osa_param": rs(VM.osa_shapley_mlp(cl, Xr, yr, eps, seed, mode="param")),
                         "osa_ensemble": rs(VM.osa_shapley_mlp(cl, Xr, yr, eps, seed, mode="ensemble")),
                         "mr": rs(VM.mr_shapley_mlp(cl, Xr, yr, eps, seed))}
    return out


def valuation_mlp(seeds=VAL_SEEDS, processes=2):
    from . import valuation_mlp as VM
    with Pool(processes) as pool:
        light = pool.map(_vmlp_seed, [(s, False) for s in seeds], chunksize=1)        # cheap part first, all seeds
        heavy = pool.map(_vmlp_seed, [(s, True) for s in seeds[:MLP_HEAVY_SEEDS]], chunksize=1)
    runs = heavy + light[MLP_HEAVY_SEEDS:]
    keys = ["none"] + [str(e) for e in MLP_EPS]
    meth = ["osa_param", "osa_ensemble", "mr"]
    res = {"seeds": seeds, "osa_cfg": VM.OSA_MLP_CFG, "hidden": VM.H,
           "rho": {k: {m: summary([r["rho"][k][m] for r in runs]) for m in meth} for k in keys},
           "osa_param_minus_mr": {k: paired([r["rho"][k]["osa_param"] for r in runs], [r["rho"][k]["mr"] for r in runs]) for k in keys},
           "osa_ensemble_minus_mr": {k: paired([r["rho"][k]["osa_ensemble"] for r in runs], [r["rho"][k]["mr"] for r in runs]) for k in keys},
           "budget_matched": {}, "true_eps_per_run": {}}
    hv = [r for r in runs if "retrain_budget_matched" in r["rho"]["2.0"]]
    for k in ("2.0", "1.0"):
        res["budget_matched"][k] = {"n_seeds": len(hv),
                                    "retrain_budget_matched": summary([r["rho"][k]["retrain_budget_matched"] for r in hv]),
                                    "osa_ensemble_same_seeds": summary([r["rho"][k]["osa_ensemble"] for r in hv]),
                                    "mr_same_seeds": summary([r["rho"][k]["mr"] for r in hv]),
                                    "osa_ensemble_minus_bm": paired([r["rho"][k]["osa_ensemble"] for r in hv],
                                                                    [r["rho"][k]["retrain_budget_matched"] for r in hv])}
        from . import private_valuation as PV
        N = VM.TUNED["mlp_rounds"] * VM.TUNED["mlp_epochs"]
        res["true_eps_per_run"][k] = float(PV.retraining_privacy_cost(float(k), 10, N))
    return res


# ---- H. stronger baselines ----------------------------------------------------------------
FED_NAMES = ["FedAvg", "FedAvg (400 steps)", "FedProx", "SCAFFOLD", "FedAdam", "Federated MLP"]
CEN_NAMES = ["Central LR", "Central gradient boosting", "Central random forest", "Central MLP"]


def _predictors(clients, Xtr_pooled, ytr_pooled, seed, epsilon=None):
    from . import baselines as B
    preds = B.federated_models(clients, cfg(epsilon=epsilon), seed)
    if epsilon is None:
        for k, mdl in B.central_models(Xtr_pooled, ytr_pooled, seed).items():
            preds[k] = (lambda X, m=mdl: m.predict_proba(X)[:, 1])
    return preds


def _baselines_seed(seed):
    df, names, tr, te, X, y = split(seed)
    cl = [(X[tr[h]], y[tr[h]]) for h in names]
    all_tr = np.concatenate([tr[h] for h in names])
    sizes = np.array([len(tr[h]) + len(te[h]) for h in names], float)
    out = {"within": {}, "within_dp1": {}, "loco": {}}
    for k, f in _predictors(cl, X[all_tr], y[all_tr], seed).items():
        per = [fl.safe_auc(y[te[h]], f(X[te[h]])) for h in names]
        out["within"][k] = {"macro": float(np.mean(per)), "weighted": float(np.average(per, weights=sizes))}
    for k, f in _predictors(cl, None, None, seed, epsilon=1.0).items():
        out["within_dp1"][k] = float(np.mean([fl.safe_auc(y[te[h]], f(X[te[h]])) for h in names]))
    from . import fl as _fl
    out["drift"] = {"FedAvg": float(np.mean(_fl.federated_train(cl, cfg(), seed)["drift_history"])),
                    "FedProx": float(np.mean(_fl.federated_train(cl, cfg(mu=MU), seed)["drift_history"])),
                    "SCAFFOLD": float(np.mean(_fl.federated_train_scaffold(cl, cfg(), seed)["drift_history"]))}
    raw = core_raw(df)
    for ctry in sorted(set(COUNTRY.values())):
        held = [h for h in names if COUNTRY[h] == ctry]; trn = [h for h in names if COUNTRY[h] != ctry]
        rows = np.concatenate([tr[h] for h in trn])
        Xc = Standardiser().fit(raw.iloc[rows]).transform(raw)
        clc = [(Xc[tr[h]], y[tr[h]]) for h in trn]
        test = np.concatenate([np.concatenate([tr[h], te[h]]) for h in held])
        out["loco"][ctry] = {}
        for k, f in _predictors(clc, Xc[rows], y[rows], seed).items():
            p = f(Xc[test])
            out["loco"][ctry][k] = {"auc": fl.safe_auc(y[test], p), "slope": calibration(y[test], p)["slope"]}
    return out


def baselines(seeds=VAL_SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_baselines_seed, seeds)
    models = FED_NAMES + CEN_NAMES
    countries = list(runs[0]["loco"].keys())
    within = {m: {a: summary([r["within"][m][a] for r in runs]) for a in ("macro", "weighted")} for m in models}
    diff_within = {m: paired([r["within"][m]["macro"] for r in runs], [r["within"]["FedAvg"]["macro"] for r in runs])
                   for m in models if m != "FedAvg"}
    dp1 = {m: summary([r["within_dp1"][m] for r in runs]) for m in ["FedAvg", "FedProx", "SCAFFOLD", "FedAdam"]}
    dp1_diff = {m: paired([r["within_dp1"][m] for r in runs], [r["within_dp1"]["FedAvg"] for r in runs])
                for m in ["FedProx", "SCAFFOLD", "FedAdam"]}
    cm = lambda m, key: [float(np.mean([r["loco"][c][m][key] for r in runs])) for c in countries]
    loco = {m: {"auc": summary(cm(m, "auc")), "slope": summary(cm(m, "slope")),
                "auc_by_country": dict(zip(countries, cm(m, "auc")))} for m in models}
    loco_diff = {m: paired(cm(m, "auc"), cm("FedAvg", "auc")) for m in models if m != "FedAvg"}
    drift = {k: summary([r["drift"][k] for r in runs]) for k in ("FedAvg", "FedProx", "SCAFFOLD")}
    from . import baselines as B
    return {"models": models, "within": within, "within_diff_vs_FedAvg": diff_within, "within_dp_eps1": dp1,
            "within_dp_eps1_diff_vs_FedAvg": dp1_diff, "loco": loco, "loco_diff_vs_FedAvg_by_country": loco_diff,
            "drift": drift, "seeds": seeds, "tuned": B.TUNED,
            "note": "FedNova equals FedAvg here (equal local steps at every hospital)."}


def _baselines_tcga_seed(seed):
    from . import experiments as E, baselines as B
    Xtr, ytr, Xte, yte, _ = E.matched_split(seed)
    out = {}
    for alpha in (1.0, 0.1):
        cl, _ = E.dirichlet_clients(Xtr, ytr, Xte, yte, 10, alpha, seed)
        cl = [c for c in cl if len(c[1]) > 0]
        preds = B.federated_models(cl, cfg(), seed)
        out[str(alpha)] = {k: fl.safe_auc(yte, f(Xte)) for k, f in preds.items()}
    for k, mdl in B.central_models(Xtr, ytr, seed).items():
        out.setdefault("central", {})[k] = fl.safe_auc(yte, mdl.predict_proba(Xte)[:, 1])
    return out


def baselines_tcga(seeds=SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_baselines_tcga_seed, seeds)
    res = {}
    for a in ("1.0", "0.1"):
        res[a] = {m: summary([r[a][m] for r in runs]) for m in FED_NAMES}
        res[a + "_diff_vs_FedAvg"] = {m: paired([r[a][m] for r in runs], [r[a]["FedAvg"] for r in runs]) for m in FED_NAMES[1:]}
    res["central"] = {m: summary([r["central"][m] for r in runs]) for m in CEN_NAMES}
    res["seeds"] = seeds
    return res


# ---- I. Poisson subsampling + sampled-Gaussian RDP accounting ---------------------------------
SUBSAMPLE_CONFIGS = [("full batch (q=1)", 1.0, 0.5), ("q=0.3", 0.3, 0.15), ("q=0.1", 0.1, 0.05)]
# learning rates for q < 1 chosen on tuning seeds 0-3 (TCGA, FedProx, eps in {1, 0.5}); evaluated on seeds 20-39


def _subsample_seed(seed):
    from . import experiments as E
    df, names, tr, te, X, y = split(seed)
    cl9 = [(X[tr[h]], y[tr[h]]) for h in names]
    Xtr, ytr, Xte, yte, _ = E.matched_split(seed)
    clt, _ = E.dirichlet_clients(Xtr, ytr, Xte, yte, 3, 10.0, seed)
    te_all = np.concatenate([te[h] for h in names])
    out = {}
    for name, q, lr in SUBSAMPLE_CONFIGS:
        for eps in [None] + EPSILONS:
            c = cfg(mu=MU, epsilon=eps, q=q, lr=lr)
            sig = c.sigma()
            w9 = fl.federated_train(cl9, c, seed, sigma=sig)["w"]
            wt = fl.federated_train(clt, c, seed, sigma=sig)["w"]
            out[f"{name}|{eps}"] = {
                "pca19_macro": float(np.mean([fl.safe_auc(y[te[h]], _p(X[te[h]], w9)) for h in names])),
                "pca19_slope": calibration(y[te_all], _p(X[te_all], w9))["slope"],
                "tcga_auc": fl.safe_auc(yte, _p(Xte, wt)), "sigma": None if sig is None else float(sig),
                "steps": c.steps}
    return out


def subsampling(seeds=VAL_SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_subsample_seed, seeds)
    rows = []
    for name, q, lr in SUBSAMPLE_CONFIGS:
        for eps in [None] + EPSILONS:
            k = f"{name}|{eps}"
            row = {"config": name, "q": q, "lr": lr, "epsilon": eps, "sigma": runs[0][k]["sigma"], "steps": runs[0][k]["steps"]}
            for met in ("pca19_macro", "pca19_slope", "tcga_auc"):
                row[met] = summary([r[k][met] for r in runs])
                if q < 1.0:
                    row[met + "_vs_full"] = paired([r[k][met] for r in runs], [r[f"full batch (q=1)|{eps}"][met] for r in runs])
            rows.append(row)
    # accounting illustration: epsilon at the full-batch noise level if the same steps were subsampled
    return {"rows": rows, "seeds": seeds, "configs": SUBSAMPLE_CONFIGS}


# ---- G4. practicality: OSA-Shapley vs cheap contribution proxies; small real silos -------------
# Answers "why not just share sizes or summary statistics?" Every proxy is compared with the same
# reference as OSA (exact non-private retraining Shapley). Proxies that use a model use the SAME DP
# releases as OSA, so they cost exactly the same epsilon.
PROXY_EPS = [None, 5.0, 2.0, 1.0]
PROXY_CACHE = os.path.join(ROOT, "results", "cache_valuation_proxies")


def _auc_safe(y, s):
    from . import private_valuation as PV
    y = np.asarray(y)
    if y.min() == y.max():
        return 0.5
    return float(PV.auc_columns(y, np.asarray(s)[:, None])[0])


def _proxy_values(cl, cl_te, Xr, yr, eps, seed):
    """Contribution scores of every proxy and of OSA-Shapley, from one set of DP releases."""
    from . import private_valuation as PV
    K = len(cl)
    phi, W = PV.osa_shapley(cl, Xr, yr, eps, rng=np.random.default_rng(5000 + seed), return_models=True)
    sizes = np.array([len(c[1]) for c in cl], float)
    pos = np.array([c[1].sum() for c in cl], float)
    # summary statistics: closeness of each hospital's feature means and prevalence to the pooled ones
    pooled = np.concatenate([c[0] for c in cl]); ypool = np.concatenate([c[1] for c in cl])
    ref_stat = np.r_[pooled.mean(0), ypool.mean()]
    stat = np.array([-np.linalg.norm(np.r_[c[0].mean(0), c[1].mean()] - ref_stat) for c in cl])
    own_test = np.array([_auc_safe(yt, fl.add_bias(Xt) @ W[k]) for k, (Xt, yt) in enumerate(cl_te)])
    ref_auc = np.array([_auc_safe(yr, fl.add_bias(Xr) @ W[k]) for k in range(K)])
    masks, _ = PV.all_masks(K)
    u = PV.osa_utilities(W, sizes, masks, Xr, yr)
    full = u[-1]
    loo = np.array([full - u[(2 ** K - 1) - (1 << k) - 1] for k in range(K)])
    return {"size": sizes, "positives": pos, "statistics": stat, "own_test_auc": own_test,
            "ref_auc": ref_auc, "osa_loo": loo, "osa": phi}


PROXY_NAMES = ["size", "positives", "statistics", "own_test_auc", "ref_auc", "osa_loo", "osa"]


def _proxy_block(cl, cl_te, Xr, yr, seed, truth, corrupt_index=None, heavy=False):
    from . import private_valuation as PV
    rs = lambda p: float(V.rank_stability(truth, [np.asarray(p, float)])["spearman"][0])
    out = {"rho": {}, "corrupt": {}}
    for eps in PROXY_EPS:
        k = "none" if eps is None else str(eps)
        pv = _proxy_values(cl, cl_te, Xr, yr, eps, seed)
        out["rho"][k] = {m: rs(v) for m, v in pv.items()}
        pm = _post_methods(cl, Xr, yr, eps, seed)
        out["rho"][k].update({"mr": rs(pm["mr"]), "or": rs(pm["or"])})
        if eps is not None:
            if heavy:
                c_ = cfg(epsilon=eps)
                sig_bm = fl.calibrate_sigma(eps, c_.steps * 2 ** (len(cl) - 1), c_.delta)
                out["rho"][k]["retrain_budget_matched"] = rs(_exact_retrain(cl, Xr, yr, c_, 2000 + seed, sigma=sig_bm)[0])
    if corrupt_index is not None:
        i = corrupt_index
        ycor = np.random.default_rng(777 + seed).binomial(1, cl[i][1].mean(), size=len(cl[i][1])).astype(float)
        clc = list(cl); clc[i] = (cl[i][0], ycor)
        for eps in PROXY_EPS:
            k = "none" if eps is None else str(eps)
            pv = _proxy_values(clc, cl_te, Xr, yr, eps, seed)
            r = {m: float(np.argsort(np.argsort(np.asarray(v, float), kind="stable"), kind="stable")[i] == 0) for m, v in pv.items()}
            # ties at the bottom (identical size or prevalence) do not count as a detection
            for m, v in pv.items():
                v = np.asarray(v, float)
                if np.sum(v == v[i]) > 1 and v[i] == v.min():
                    r[m] = 0.0
            out["corrupt"][k] = r
    return out


def _vproxy_seed(seed):
    import json as _json
    os.makedirs(PROXY_CACHE, exist_ok=True)
    fn = os.path.join(PROXY_CACHE, f"seed{seed}.json")
    if os.path.exists(fn):
        return _json.load(open(fn))
    df, names, tr, te, X, y = split(seed)
    res = {}
    for tag, hs, corrupt in (("large10", names[:10], CORRUPT_INDEX), ("small10", names[-10:], None)):
        cl = [(X[tr[h]], y[tr[h]]) for h in hs]
        cl_te = [(X[te[h]], y[te[h]]) for h in hs]
        ref = np.concatenate([te[h] for h in hs])
        Xr, yr = X[ref], y[ref]
        truth, _ = _exact_retrain(cl, Xr, yr, cfg(), seed)
        block = _proxy_block(cl, cl_te, Xr, yr, seed, truth, corrupt, heavy=(tag == "small10"))
        block["hospitals"] = list(hs); block["sizes"] = [int(len(tr[h]) + len(te[h])) for h in hs]
        block["truth"] = truth.tolist()
        res[tag] = block
    _json.dump(res, open(fn, "w"))
    return res


def valuation_proxies(seeds=VAL_SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_vproxy_seed, seeds)
    out = {"seeds": seeds, "proxies": PROXY_NAMES}
    for tag in ("large10", "small10"):
        R0 = runs[0][tag]
        blk = {"hospitals": R0["hospitals"], "sizes": R0["sizes"], "rho": {}, "osa_minus": {}}
        for k in R0["rho"]:
            blk["rho"][k] = {m: summary([r[tag]["rho"][k][m] for r in runs]) for m in R0["rho"][k]}
            blk["osa_minus"][k] = {m: paired([r[tag]["rho"][k]["osa"] for r in runs], [r[tag]["rho"][k][m] for r in runs])
                                   for m in R0["rho"][k] if m != "osa"}
        if R0["corrupt"]:
            blk["corrupt"] = {k: {m: summary([r[tag]["corrupt"][k][m] for r in runs]) for m in R0["corrupt"][k]}
                              for k in R0["corrupt"]}
            blk["corrupt_hospital"] = R0["hospitals"][CORRUPT_INDEX]
        out[tag] = blk
    return out
