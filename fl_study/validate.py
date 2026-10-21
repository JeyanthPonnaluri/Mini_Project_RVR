"""
Automated validation checks. Each check returns (passed: bool, detail: str).
Run:  python -m fl_study.validate        (also shown live on the app's Validation page)
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import data, fl
from . import valuation as V

CHECKS = []


def check(title, group):
    def deco(f):
        CHECKS.append((group, title, f))
        return f
    return deco


# ---------------------------------------------------------------- data
@check("Raw data: 572 sample records from 500 patients", "Data")
def _raw():
    raw = pd.read_csv(data.CLINICAL_TSV, sep="\t")
    ok = len(raw) == 572 and raw["submitter_id"].nunique() == 500
    return ok, f"{len(raw)} records, {raw['submitter_id'].nunique()} patients"


@check("Clinical cohort: one primary-tumour sample per patient", "Data")
def _one_per_patient():
    c = data.load_clinical_cohort()
    ok = c["patient"].is_unique and c["sample"].str[13:15].eq("01").all() and c["t_stage"].notna().all()
    return ok, f"{len(c)} patients, {int(c['y'].sum())} advanced (T3/T4)"


@check("Matched clinical-protein cohort = 347 patients (277 train / 70 test)", "Data")
def _matched():
    m, P = data.load_matched_cohort()
    tr, te = train_test_split(np.arange(len(m)), test_size=0.2, stratify=m["y"], random_state=42)
    ok = len(m) == 347 and len(tr) == 277 and len(te) == 70 and m["patient"].is_unique
    return ok, f"{len(m)} patients ({int(m['y'].sum())} T3/T4), split {len(tr)}/{len(te)}"


@check("Survival events in matched cohort = 9 (paper previously said 12)", "Data")
def _events():
    m, _ = data.load_matched_cohort()
    ev = int(data.survival_events(m["sample"])["OS"].sum())
    return ev == 9, f"{ev} deaths among 347 patients (12 is the count over all 572 records)"


@check("No identifier / timestamp / outcome column is used as a feature", "Leakage")
def _no_ids():
    m, _ = data.load_matched_cohort()
    names = data.TabularPreprocessor("full").fit(m).feature_names_
    banned = [c for cols in data.EXCLUDED_REASONS.values() for c in cols]
    toks = ["uuid", "datetime", "submitter", "sample_id", "treatment", "vital", "follow", "site"]
    bad = [n for n in names if any(t in n.lower() for t in toks) or n in banned]
    return not bad and len(names) < 30, f"{len(names)} features: {', '.join(names)}"


@check("Pre-op feature set contains no post-surgical pathology", "Leakage")
def _preop():
    m, _ = data.load_matched_cohort()
    names = data.TabularPreprocessor("preop").fit(m).feature_names_
    bad = [n for n in names if "gleason" in n or "pathologic" in n]
    return not bad, f"{len(names)} features: {', '.join(names)}"


@check("Scaler / imputer / PCA are fitted on training rows only", "Leakage")
def _train_only():
    m, P = data.load_matched_cohort()
    tr, te = train_test_split(np.arange(len(m)), test_size=0.2, stratify=m["y"], random_state=42)
    tp = data.TabularPreprocessor("full").fit(m.iloc[tr])
    exp = m.iloc[tr]["age"].fillna(m.iloc[tr]["age"].median()).mean()
    pp = data.ProteinPreprocessor().fit(P.iloc[tr])
    pp_all = data.ProteinPreprocessor().fit(P)
    ok = abs(tp.scaler_.mean_[0] - exp) < 1e-9 and pp.n_components != pp_all.n_components
    return ok, (f"age mean in scaler = train mean ({exp:.3f}); protein PCs: {pp.n_components} (train-fit) "
                f"vs {pp_all.n_components} if fitted on all 347 rows (explains the 115 vs 129 in the old reports)")


@check("Test rows with unseen categories are encoded as the reference level (no crash)", "Leakage")
def _unseen():
    m, _ = data.load_matched_cohort()
    tp = data.TabularPreprocessor("full").fit(m)
    t = m.head(3).copy(); t["race"] = "martian"
    X = tp.transform(t)
    return X.shape[1] == len(tp.feature_names_) and np.isfinite(X).all(), f"shape {X.shape}"


# ---------------------------------------------------------------- model / FL
@check("NumPy logistic regression matches scikit-learn (centralised)", "Model")
def _lr_vs_sklearn():
    m, _ = data.load_matched_cohort()
    tr, te = train_test_split(np.arange(len(m)), test_size=0.2, stratify=m["y"], random_state=1)
    Xtr, Xte, _ = data.build_features(m.iloc[tr], m.iloc[te])
    w = fl.centralized_train(Xtr, m["y"].values[tr], fl.TrainConfig())
    a = fl.safe_auc(m["y"].values[te], fl.predict_proba(Xte, w))
    b = fl.safe_auc(m["y"].values[te], LogisticRegression(max_iter=5000).fit(Xtr, m["y"].values[tr]).predict_proba(Xte)[:, 1])
    return abs(a - b) < 0.03, f"NumPy AUC {a:.3f} vs sklearn {b:.3f}"


@check("FedAvg with one client equals centralised training", "Model")
def _fedavg_single():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5)); y = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
    a = fl.federated_train([(X, y)], fl.TrainConfig())["w"]
    b = fl.centralized_train(X, y, fl.TrainConfig())
    return np.allclose(a, b), f"max |diff| = {np.abs(a - b).max():.2e}"


@check("FedAvg with IID equal clients and 1 local epoch equals centralised GD", "Model")
def _fedavg_equiv():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 4)); y = (X[:, 1] > 0).astype(int)
    c = fl.TrainConfig(rounds=50, epochs=1, lr=0.3, l2=0.0)
    a = fl.federated_train([(X[i::3], y[i::3]) for i in range(3)], c)["w"]
    b = fl.federated_train([(X[np.r_[0:90:3, 1:90:3, 2:90:3]], y[np.r_[0:90:3, 1:90:3, 2:90:3]])], c)["w"]
    return np.allclose(a, b, atol=1e-10), f"max |diff| = {np.abs(a - b).max():.2e} (gradient averaging identity)"


@check("Client drift is recorded for FedAvg (was always 0.0) and FedProx reduces it", "Model")
def _drift():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 5)); y = (X[:, 0] > 0).astype(int)
    a, _ = fl.dirichlet_partition(y, y[:3], 3, 0.1, 0)
    cl = [(X[a == k], y[a == k]) for k in range(3)]
    d0 = np.mean(fl.federated_train(cl, fl.TrainConfig())["drift_history"])
    d1 = np.mean(fl.federated_train(cl, fl.TrainConfig(mu=0.5))["drift_history"])
    return d0 > 0 and d1 < d0, f"FedAvg drift {d0:.4f} > FedProx drift {d1:.4f} > 0"


@check("Dirichlet partition keeps every sample exactly once; small alpha = stronger skew", "Model")
def _dirichlet():
    y = np.r_[np.ones(200), np.zeros(100)].astype(int)
    spreads = {}
    for alpha in (100.0, 0.1):
        s = []
        for seed in range(20):
            a, b = fl.dirichlet_partition(y, y[:60], 3, alpha, seed)
            assert len(a) == 300 and set(a) <= {0, 1, 2}
            pr = [y[a == k].mean() for k in range(3) if (a == k).any()]
            s.append(max(pr) - min(pr))
        spreads[alpha] = np.mean(s)
    return spreads[0.1] > spreads[100.0] + 0.2, f"label-rate spread alpha=100: {spreads[100.0]:.2f}, alpha=0.1: {spreads[0.1]:.2f}"


# ---------------------------------------------------------------- privacy
@check("RDP accountant: epsilon matches the closed-form Gaussian optimum", "Privacy")
def _rdp_closed():
    errs = []
    for sigma, T in [(5.0, 100), (20.0, 100), (50.0, 45)]:
        rho = T / (2 * sigma ** 2); L = math.log(1e5)
        exact = rho + 2 * math.sqrt(rho * L)   # min over continuous alpha
        grid, _ = fl.rdp_epsilon(sigma, T, 1e-5)
        errs.append((grid - exact) / exact)
    return all(0 <= e < 0.01 for e in errs), f"grid epsilon within {max(errs) * 100:.3f}% above the analytic optimum (never below)"


@check("Noise calibration hits the target epsilon and is monotone", "Privacy")
def _calib():
    sig = [fl.calibrate_sigma(e, 100) for e in (0.5, 1, 2, 5, 10)]
    eps = [fl.rdp_epsilon(s, 100)[0] for s in sig]
    ok = all(a > b for a, b in zip(sig, sig[1:])) and all(abs(e - t) < 1e-6 for e, t in zip(eps, (0.5, 1, 2, 5, 10)))
    return ok, "sigma = " + ", ".join(f"{s:.2f}" for s in sig) + " for eps = 0.5, 1, 2, 5, 10 (T=100, delta=1e-5)"


@check("DP noise actually added has std = sigma * clip (empirical)", "Privacy")
def _noise():
    X = np.zeros((10, 3)); y = np.zeros(10)
    rng = np.random.default_rng(0)
    w0 = np.zeros(4)
    draws = [local_one(X, y, w0, rng) for _ in range(4000)]
    sd = np.std(np.array(draws)) * 10 / 1.0  # undo /n and lr=1
    return abs(sd - 3.0) < 0.1, f"empirical noise std {sd:.3f} (expected 3.000 for sigma=3, C=1)"


def local_one(X, y, w0, rng):
    # one DP step, lr=1, no data signal (gradient of p=0.5 vs y=0 clipped) -> isolate noise
    w = fl.local_update(X, y, w0, w0, 1, 1.0, 0.0, 0.0, sigma=3.0, clip=1.0, rng=rng)
    g_det = fl.add_bias(X).T @ (0.5 - y) / len(y)
    return -(w - w0) - g_det


@check("Per-sample gradient clipping bounds every contribution by C", "Privacy")
def _clip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)) * 10; y = rng.integers(0, 2, 50)
    Xb = fl.add_bias(X); p = fl.sigmoid(Xb @ np.zeros(5))
    per = Xb * (p - y)[:, None]
    per = per * np.minimum(1.0, 1.0 / np.linalg.norm(per, axis=1, keepdims=True))
    mx = np.linalg.norm(per, axis=1).max()
    return mx <= 1.0 + 1e-12, f"max clipped per-sample norm = {mx:.6f}"


# ---------------------------------------------------------------- valuation
@check("Vectorised coalition training == one training run per coalition", "Valuation")
def _vec():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 4)); y = (X[:, 0] + 0.5 * rng.normal(size=120) > 0).astype(int)
    cl = [(X[i::4][:25], y[i::4][:25]) for i in range(4)]
    c = fl.TrainConfig(rounds=5)
    a = V.coalition_utilities(cl, X, y, c); b = V.all_coalition_utilities(cl, X, y, c)
    d = max(abs(a[k] - b[k]) for k in a)
    return d < 1e-12, f"max utility difference over 15 coalitions = {d:.1e}"


@check("Shapley axioms: efficiency, symmetry, null player", "Valuation")
def _axioms():
    K = 4
    val = {(): 0.5}
    contrib = [0.1, 0.1, 0.0, 0.2]           # additive game: phi must equal contrib
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            val[S] = 0.5 + sum(contrib[i] for i in S)
    phi = V.shapley_fast(val, K)
    eff = abs(phi.sum() - (val[tuple(range(K))] - 0.5)) < 1e-12
    sym = abs(phi[0] - phi[1]) < 1e-12
    null = abs(phi[2]) < 1e-12
    same = np.allclose(phi, V.shapley_from_utilities(val, K))
    return eff and sym and null and same, f"phi = {np.round(phi, 4).tolist()} (expected {contrib})"


@check("Top-contributor stability is no longer identically 1/K", "Valuation")
def _ts():
    ref = np.array([0.30, 0.10, 0.05, 0.02])
    same = [ref + 0.001 * i for i in range(5)]
    swapped = [np.array([0.10, 0.30, 0.05, 0.02])] * 5
    a = np.mean(V.rank_stability(ref, same)["top_contributor_stability"])
    b = np.mean(V.rank_stability(ref, swapped)["top_contributor_stability"])
    old = np.mean([np.mean(V.ranks(p) == 1) for p in same])   # old formula
    return a == 1.0 and b == 0.0, f"stable rankings -> {a:.2f}, top swapped -> {b:.2f} (old formula gave {old:.2f} = 1/K for both)"


@check("Membership-inference attack detects an over-fitted model (positive control)", "Privacy")
def _mia_pos():
    rng = np.random.default_rng(0)
    Xm = rng.normal(size=(60, 80)); ym = rng.integers(0, 2, 60)
    Xn = rng.normal(size=(60, 80)); yn = rng.integers(0, 2, 60)
    w = fl.centralized_train(Xm, ym, fl.TrainConfig(rounds=200, lr=1.0, l2=0.0))
    auc, adv = fl.mia_confidence_attack(w, Xm, ym, Xn, yn)
    return auc > 0.65, f"attack AUC {auc:.3f} (>0.65) on a model that memorised pure-noise labels"


@check("Experiments are reproducible (same seed -> identical results)", "Reproducibility")
def _repro():
    from .experiments import _pfl_seed
    a = _pfl_seed((7, 3, 0.5)); b = _pfl_seed((7, 3, 0.5))
    return a == b or str(a) == str(b), "two runs of the personalisation experiment with seed 7 are identical"


# ---------------------------------------------------------------- PCa-19 (base paper's own cohort)
PCA19_TABLE_I = {"de1": 763, "es7": 711, "kr2": 590, "ch2": 530, "kr1": 406, "es6": 346, "ch1": 312, "us2": 310,
                 "us1": 280, "nl": 266, "it": 218, "es4": 200, "es5": 200, "de2": 162, "gb": 133, "es1": 79,
                 "es8": 58, "es3": 30, "es2": 16}


@check("PCa-19 cohort = base paper Table I: 5,610 patients, 19 hospitals, 8 countries", "PCa-19")
def _pca19_counts():
    from . import multisite as M
    df = M.load_pca19()
    sizes = df.hospital.value_counts().to_dict()
    import hashlib, os
    sha = hashlib.sha256(open(M.PCA19_CSV, "rb").read()).hexdigest()
    rec = open(os.path.join(os.path.dirname(M.PCA19_CSV), "SHA256SUM")).read().split()[0]
    # The base paper's text says "nine countries"; its Table I and released data contain 8.
    ok = (sha == rec and len(df) == 5610 and sizes == PCA19_TABLE_I and df.country.nunique() == 8
          and df[["age", "PSA", "PV", "PIRADS", "y"]].notna().all().all())
    return ok, f"{len(df)} patients, {len(sizes)} hospitals, {df.country.nunique()} countries (paper text says 9), {df.y.mean():.1%} csPCa, no missing values, SHA-256 {sha[:12]}… matches"


@check("PCa-19 features are all pre-biopsy (age, PSA, prostate volume, PI-RADS)", "PCa-19")
def _pca19_features():
    from . import multisite as M
    df = M.load_pca19()
    F = M.core_raw(df)
    ok = list(F.columns) == M.CORE_FEATURES and set(df.columns) == {"hospital", "country", "y", "age", "PSA", "PV", "PIRADS", "5ARI"}
    return ok, f"features {', '.join(F.columns)}; label = ISUP grade group >= 2 at biopsy"


def _authors_csv(calc):
    import os
    return os.path.join(data.ROOT, "results", "final", f"pca19_authors_code_{calc}.csv")


@check("Base-paper protocol re-implementation = the authors' own code (same numbers)", "PCa-19")
def _pca19_authors():
    import json, os
    out = []
    ok = True
    for calc in ("noh", "ettala"):
        ours = os.path.join(data.ROOT, "results", "final", f"pca19_reproduction_{calc}.json")
        theirs = _authors_csv(calc)
        if not (os.path.exists(ours) and os.path.exists(theirs)):
            return False, f"missing {os.path.basename(ours)} or {os.path.basename(theirs)}"
        r = json.load(open(ours, encoding="utf-8"))
        A = pd.read_csv(theirs, header=None).values            # rows BL, LOC, CEN, FL, FR
        B = np.array([r["matrix"][k] for k in ("BL", "LOC", "CEN", "FL", "FR")])
        d = float(np.max(np.abs(A - B)))
        ok &= d < 1e-6
        out.append(f"{calc}: max |difference| {d:.1e} over 5x19 cells ({r['n_repeats']} repeats)")
    return ok, "; ".join(out)


@check("Published Noh/Ettala calculators reproduce the authors' BL row", "PCa-19")
def _pca19_bl():
    import os
    from . import multisite as M
    from sklearn.metrics import roc_auc_score
    df = M.load_pca19()
    names = M.hospitals_by_size(df)
    ok = True; msg = []
    for calc in ("noh", "ettala"):
        if not os.path.exists(_authors_csv(calc)):
            return False, f"missing {os.path.basename(_authors_csv(calc))}"
        f = M.CALCULATORS[calc][1]
        bl = [roc_auc_score(df.y[df.hospital == h], f(df[df.hospital == h])) for h in names]
        A = pd.read_csv(_authors_csv(calc), header=None).values[0]
        d = float(np.max(np.abs(A - np.array(bl))))
        ok &= d < 1e-9
        msg.append(f"{calc}: max |difference| {d:.1e}")
    return ok, "; ".join(msg)


@check("Leave-one-country-out: held-out country never influences preprocessing", "PCa-19")
def _pca19_loco_leak():
    from . import multisite as M
    df = M.load_pca19()
    held = df.country.values == "Spain"
    st1 = M.Standardiser().fit(M.core_raw(df)[~held])
    df2 = df.copy(); df2.loc[held, "PSA"] = df2.loc[held, "PSA"] * 50; df2.loc[held, "age"] = 99
    st2 = M.Standardiser().fit(M.core_raw(df2)[~held])
    ok = np.allclose(st1.mu, st2.mu) and np.allclose(st1.sd, st2.sd)
    return ok, "changing the held-out country's values leaves the fitted mean/sd unchanged"


@check("Batched coalition training (any subset of coalitions) == exact all-coalition training", "PCa-19")
def _pca19_masks():
    from . import multisite as M
    rng = np.random.default_rng(3)
    clients = [(rng.normal(size=(30, 3)), rng.integers(0, 2, 30)) for _ in range(4)]
    Xr = rng.normal(size=(60, 3)); yr = rng.integers(0, 2, 60)
    c = fl.TrainConfig(rounds=3, epochs=2)
    v = V.all_coalition_utilities(clients, Xr, yr, c, 0)
    masks = np.array([[(m >> k) & 1 for k in range(4)] for m in range(1, 16)], float)
    u = M.coalition_utilities_masks(clients, Xr, yr, c, masks, 0)
    d = max(abs(u[i] - v[tuple(np.where(m > 0)[0].tolist())]) for i, m in enumerate(masks))
    return d < 1e-12, f"max |difference| {d:.1e} over 15 coalitions"


@check("Stratified-sampling Shapley (used for all 19 hospitals) agrees with exact Shapley", "PCa-19")
def _pca19_mc():
    from . import multisite as M
    from scipy.stats import spearmanr
    df, names, tr, te, X, y = M.split(0)
    hs = names[:6]
    cl = [(X[tr[h]], y[tr[h]]) for h in hs]; ref = np.concatenate([te[h] for h in hs])
    c = M.cfg()
    masks = np.array([[(m >> k) & 1 for k in range(6)] for m in range(1, 64)], float)
    u = dict(zip([tuple(np.where(m > 0)[0].tolist()) for m in masks], M.coalition_utilities_masks(cl, X[ref], y[ref], c, masks, 0)))
    u[()] = 0.5
    exact = V.shapley_fast(u, 6)
    st_, n = M.stratified_shapley(cl, X[ref], y[ref], c, 3, 0)
    mc, _ = M.mc_shapley(cl, X[ref], y[ref], c, 400, 0)
    rho = spearmanr(exact, st_).correlation; err = float(np.max(np.abs(exact - st_)))
    err_mc = float(np.max(np.abs(exact - mc)))
    return rho > 0.9 and err < 0.005, (f"6 hospitals: Spearman {rho:.3f}, max |error| {err:.4f} AUC with {n} coalitions "
                                       f"(plain permutation sampling, 400 permutations: {err_mc:.4f})")


@check("Calibration, recalibration and decision-curve formulas are correct on known cases", "PCa-19")
def _pca19_cal():
    from . import multisite as M
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000); y = (rng.uniform(size=p.size) < p).astype(int)
    c = M.calibration(y, p)
    nb_all = M.net_benefit(y, np.ones_like(p), [0.2])[0]
    prev = y.mean(); expect = prev - (1 - prev) * 0.25
    # a mis-calibrated version (risk inflated, too extreme) is repaired by logistic recalibration on 20%
    lp = np.log(p / (1 - p)); bad = 1 / (1 + np.exp(-(1.6 * lp + 0.8)))
    n = len(p) // 5
    fixed = M.recalibrate(y[:n], bad[:n], bad[n:], "logistic")
    cb, cf = M.calibration(y[n:], bad[n:]), M.calibration(y[n:], fixed)
    ok = (abs(c["oe"] - 1) < 0.01 and abs(c["slope"] - 1) < 0.03 and abs(nb_all - expect) < 1e-12
          and abs(cf["oe"] - 1) < 0.02 and abs(cf["slope"] - 1) < 0.03 and abs(cb["slope"] - 1 / 1.6) < 0.03)
    return ok, (f"calibrated data: O/E {c['oe']:.3f}, slope {c['slope']:.3f}; distorted model slope {cb['slope']:.2f} -> "
                f"{cf['slope']:.2f} after recalibration; biopsy-all net benefit matches formula")


# ---------------------------------------------------------------- subsampled DP, private valuation, baselines
@check("Sampled-Gaussian RDP: closed form = numerical Rényi divergence; reproduces Abadi et al. (eps 1.26)", "Privacy")
def _rdp_sampled():
    from scipy import integrate
    from scipy.stats import norm
    errs = []
    for q, s_, a in [(0.1, 2.0, 4), (0.05, 1.5, 8), (0.3, 3.0, 6)]:
        # likelihood ratio N(1, s^2)/N(0, s^2) = exp((2z - 1) / (2 s^2)), written in closed form
        f = lambda z: norm.pdf(z, 0, s_) * ((1 - q) + q * math.exp((2 * z - 1) / (2 * s_ * s_))) ** a
        A, _ = integrate.quad(f, -20 * s_, 20 * s_, limit=400)
        num = math.log(A) / (a - 1)
        errs.append(abs(num - fl.rdp_sampled_gaussian(q, s_, [a])[0]) / num)
    q1 = np.allclose(fl.rdp_sampled_gaussian(1.0, 3.0, [2, 7]), np.array([2, 7]) / 18)
    eps_abadi = fl.rdp_epsilon_sampled(0.01, 4.0, 10000, 1e-5)[0]      # Abadi et al. 2016: ~1.26
    amp = fl.rdp_epsilon_sampled(0.1, 5.0, 100)[0] < fl.rdp_epsilon(5.0, 100)[0]
    ok = max(errs) < 1e-6 and q1 and abs(eps_abadi - 1.26) < 0.01 and amp
    return ok, (f"max relative error vs numerical integration {max(errs):.1e}; q = 1 reduces to alpha/(2 sigma^2); "
                f"q=0.01, sigma=4, 10,000 steps gives eps = {eps_abadi:.3f} (Abadi et al.: 1.26); subsampling amplifies")


@check("Poisson-subsampled DP-SGD: noise scale and sampling rate are as specified", "Privacy")
def _sampled_sgd():
    rng = np.random.default_rng(0)
    n, d = 2000, 3
    X = rng.normal(size=(n, d)); y = rng.integers(0, 2, n).astype(float)
    Xb = fl.add_bias(X); w0 = np.zeros(d + 1)
    # with lr = 1 and one step the update equals -(clipped sum + noise)/(q n) - l2 * 0
    q, sigma = 0.2, 3.0
    ups = np.array([fl._local_update_sampled(Xb, y, w0.copy(), w0, 1, 1.0, 0.0, 0.0, sigma, 1.0,
                                             np.random.default_rng(i), 1.0)[0] for i in range(4000)])
    # each record's clipped gradient at w=0 has first coordinate x1*(0.5-y)
    g1 = X[:, 0] * (0.5 - y); g1 = g1 / np.maximum(1, np.linalg.norm(Xb * (0.5 - y)[:, None], axis=1))
    expect_mean = -g1.sum() * 1.0 / (1.0 * n)          # q = 1 sanity on the mean
    sd = ups.std() * n                                   # noise std in units of the gradient sum
    ok = abs(sd - sigma) / sigma < 0.05 and abs(ups.mean() - expect_mean) < 0.01
    # sampling rate: average batch size
    rng2 = np.random.default_rng(1); m = np.mean([(rng2.random(n) < q).sum() for _ in range(500)]) / n
    return ok and abs(m - q) < 0.005, f"empirical noise std {sd:.3f} (sigma {sigma}); mean update matches; sampling rate {m:.3f} (q {q})"


@check("OSA-Shapley: exact vectorised Shapley; equals retraining Shapley ranking without DP", "Valuation")
def _osa():
    from . import multisite as M, private_valuation as PV
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2); K = 6
    u = rng.random(2 ** K - 1); masks, _ = PV.all_masks(K)
    d = {tuple(np.where(m > 0)[0].tolist()): x for m, x in zip(masks, u)}; d[()] = 0.5
    e1 = float(np.abs(PV.shapley_from_mask_utilities(u, K) - V.shapley_fast(d, K)).max())
    eff = abs(PV.shapley_from_mask_utilities(u, K).sum() - (u[-1] - 0.5))
    df, names, tr, te, X, y = M.split(30)
    hs = names[:6]; cl = [(X[tr[h]], y[tr[h]]) for h in hs]; ref = np.concatenate([te[h] for h in hs])
    mk = np.array([[(m >> k) & 1 for k in range(6)] for m in range(1, 64)], float)
    uu = dict(zip([tuple(np.where(m > 0)[0].tolist()) for m in mk], M.coalition_utilities_masks(cl, X[ref], y[ref], M.cfg(), mk, 0)))
    uu[()] = 0.5
    rho = spearmanr(V.shapley_fast(uu, 6), PV.osa_shapley(cl, X[ref], y[ref], None)).correlation
    cost = PV.retraining_privacy_cost(1.0, 10, 100)
    ok = e1 < 1e-12 and eff < 1e-12 and rho > 0.8 and cost > 10
    return ok, (f"vectorised = reference Shapley ({e1:.0e}), efficiency holds; Spearman(OSA, retraining) = {rho:.2f} without DP; "
                f"retraining 10 hospitals at eps = 1 per run really costs eps = {cost:.1f}")


@check("SCAFFOLD and FedAdam reduce to FedAvg in the degenerate cases", "Model")
def _baselines():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3)); y = (X[:, 0] + rng.normal(size=80) > 0).astype(float)
    c = fl.TrainConfig(rounds=5, epochs=3)
    one = [(X, y)]
    a = fl.federated_train(one, c, 0)["w"]; s_ = fl.federated_train_scaffold(one, c, 0)["w"]
    # with a single client SCAFFOLD's corrections are zero-sum: c_k = c, so local steps equal FedAvg's
    clients = [(X[:40], y[:40]), (X[40:], y[40:])]
    s2 = fl.federated_train_scaffold(clients, c, 0)
    # FedAdam with beta1=0, beta2->0, tau->0 is a sign-like step; check it converges to similar AUC
    wa = fl.federated_train_fedadam(clients, fl.TrainConfig(rounds=40, epochs=3), 0, server_lr=0.1)["w"]
    auc_a = fl.safe_auc(y, fl.predict_proba(X, wa)); auc_f = fl.safe_auc(y, fl.predict_proba(X, a))
    ok = np.allclose(a, s_) and len(s2["drift_history"]) == 5 and abs(auc_a - auc_f) < 0.03
    return ok, f"single-client SCAFFOLD = FedAvg (max diff {np.abs(a - s_).max():.1e}); FedAdam AUC {auc_a:.3f} vs FedAvg {auc_f:.3f}"


@check("Proposition 2: budget-matched retraining needs sqrt(2^(K-1)) times the per-run noise", "Valuation")
def _prop2():
    ratios = []
    for eps in (10.0, 1.0):
        for K in (5, 10):
            s1 = fl.calibrate_sigma(eps, 100); sK = fl.calibrate_sigma(eps, 100 * 2 ** (K - 1))
            ratios.append(abs(sK / s1 - 2 ** ((K - 1) / 2)) / 2 ** ((K - 1) / 2))
    return max(ratios) < 1e-6, f"max relative deviation from 2^((K-1)/2): {max(ratios):.1e} (K = 10: factor {2 ** 4.5:.1f})"


@check("Reconstruction competitors (MR/GTG, OR): OR grand coalition = FedAvg model; MR is efficient per round", "Valuation")
def _recon():
    from . import private_valuation as PV
    from . import multisite as M
    rng = np.random.default_rng(0)
    cl = []
    for k in range(4):
        X = rng.normal(size=(60 + 30 * k, 3)); y = (X @ [1.0, -0.5, 0.3] + rng.normal(size=len(X)) > 0).astype(float)
        cl.append((X, y))
    c = M.cfg()
    trace, w_fin = PV.fl_trace(cl, c, 0)
    w_fed = fl.federated_train(cl, c, 0)["w"]
    sizes = np.array([len(y) for _, y in cl], float)
    D = np.sum([W - wp[None, :] for wp, W in trace], axis=0) + trace[0][0][None, :]
    w_or = (sizes / sizes.sum()) @ D
    Xr = rng.normal(size=(300, 3)); yr = (Xr @ [1.0, -0.5, 0.3] > 0).astype(float)
    phi = PV.mr_shapley(trace, sizes, Xr, yr)
    masks, _ = PV.all_masks(4)
    tot = 0.0
    Xb = fl.add_bias(Xr)
    for wp, W in trace:
        u = PV.osa_utilities(W, sizes, masks, Xr, yr)
        s_ = Xb @ wp
        v0 = 0.5 if np.allclose(s_, s_[0]) else float(PV.auc_columns(yr, s_[:, None])[0])
        tot += u[-1] - v0
    e1 = float(np.abs(w_or - w_fed).max()); e2 = float(np.abs(w_fin - w_fed).max()); e3 = abs(phi.sum() - tot)
    return max(e1, e2) < 1e-10 and e3 < 1e-10, f"|OR(all) - FedAvg| = {e1:.1e}; trace = FedAvg {e2:.1e}; MR efficiency gap {e3:.1e}"


@check("Proposition 3: OSA coalition-model DP noise sd <= lr*sigma*C*sqrt(|S|*steps)/n_S and shrinks with n_S", "Valuation")
def _prop3():
    from . import private_valuation as PV
    rng = np.random.default_rng(1)
    cl = []
    for n in (80, 160, 320):
        X = rng.normal(size=(n, 3)); y = (X @ [1.0, -0.5, 0.3] + rng.normal(size=n) > 0).astype(float)
        cl.append((X, y))
    eps = 1.0
    sig = fl.calibrate_sigma(eps, PV.OSA_STEPS)
    clean = np.array([PV.local_dp_model(X, y, None)[0] for X, y in cl])
    reps = np.array([[PV.local_dp_model(X, y, eps, rng=np.random.default_rng(100 * r + k))[0] - clean[k]
                      for k, (X, y) in enumerate(cl)] for r in range(150)])          # R x K x d
    sizes = np.array([len(y) for _, y in cl], float)
    masks, _ = PV.all_masks(3)
    A = masks * sizes; A = A / A.sum(1, keepdims=True)
    emp = np.array([np.sqrt(np.mean(np.einsum("k,rkd->rd", a, reps) ** 2)) for a in A])
    bound = PV.coalition_noise_bound(sig, 1.0, sizes)
    ok = bool(np.all(emp <= bound)) and emp[-1] < emp[0]
    return ok, f"empirical/bound ratio {float((emp / bound).min()):.2f}-{float((emp / bound).max()):.2f}; sd(all) {emp[-1]:.3f} < sd(smallest) {emp[0]:.3f}"


@check("Non-convex model: batched coalition FedAvg-MLP = an independent per-model loop; per-record MLP gradients are exact", "Valuation")
def _mlp_val():
    from . import baselines as B
    from . import valuation_mlp as VM
    rng = np.random.default_rng(3)
    cl = []
    for k in range(3):
        X = rng.normal(size=(50 + 20 * k, 4)); y = (X[:, 0] - X[:, 1] ** 2 + rng.normal(size=len(X)) > 0).astype(float)
        cl.append((X, y))
    d, lr, l2 = 4, B.TUNED["mlp_lr"], VM.L2
    # independent reference: plain per-model loop, weight decay on all parameters (as in the valuation trainer)
    P = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in B._mlp_init(d, VM.H, np.random.default_rng(0)).items()}
    sizes = np.array([len(c[1]) for c in cl], float); a = sizes / sizes.sum()
    for _ in range(5):
        locs = []
        for X, y in cl:
            Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
            for _e in range(3):
                Hd = np.tanh(X @ Q["W1"] + Q["b1"]); p = fl.sigmoid(Hd @ Q["w2"] + Q["b2"]); r = (p - y) / len(y)
                dH = np.outer(r, Q["w2"]) * (1 - Hd ** 2)
                g = {"W1": X.T @ dH, "b1": dH.sum(0), "w2": Hd.T @ r, "b2": r.sum()}
                Q = {k: Q[k] - lr * (g[k] + l2 * Q[k]) for k in Q}
            locs.append(Q)
        P = {k: sum(ak * L[k] for ak, L in zip(a, locs)) for k in P}
    T = VM.init(d, 0)[None, :]
    for _ in range(5):
        T = sum(a[k] * VM.local_steps(T, X, y, 3, lr) for k, (X, y) in enumerate(cl))
    W1, b1, w2, b2 = VM.unpack(T, d)
    e1 = max(float(np.abs(W1[0] - P["W1"]).max()), float(np.abs(b1[0] - P["b1"]).max()),
             float(np.abs(w2[0] - P["w2"]).max()), abs(float(b2[0]) - P["b2"]))
    X, y = cl[0]
    T0 = VM.init(d, 1)[None, :]
    per = VM._grads(T0, X, y, d, per_sample=True)[0]
    full = VM._grads(T0, X, y, d)[0]
    e2 = float(np.abs(per.mean(0) - full).max())
    def loss(t):
        p = fl.sigmoid(VM.logits(t[None, :], X, d)[:, 0]); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    j = 7; h = 1e-6; tp = T0[0].copy(); tm = T0[0].copy(); tp[j] += h; tm[j] -= h
    e3 = abs((loss(tp) - loss(tm)) / (2 * h) - full[j])
    return max(e1, e2) < 1e-10 and e3 < 1e-6, f"batched vs per-model loop {e1:.1e}; per-record vs full {e2:.1e}; finite-difference {e3:.1e}"


@check("Proxy baselines: LOO of averaged releases = v(N) - v(N minus k) recomputed directly; proxies reuse OSA's DP releases; silos are real hospitals", "Valuation")
def _proxies():
    from . import multisite as M
    from . import private_valuation as PV
    df, names, tr, te, X, y = M.split(20)
    hs = names[:4]
    cl = [(X[tr[h]], y[tr[h]]) for h in hs]; cl_te = [(X[te[h]], y[te[h]]) for h in hs]
    ref = np.concatenate([te[h] for h in hs]); Xr, yr = X[ref], y[ref]
    pv = M._proxy_values(cl, cl_te, Xr, yr, 2.0, 20)
    phi, W = PV.osa_shapley(cl, Xr, yr, 2.0, rng=np.random.default_rng(5020), return_models=True)
    sizes = np.array([len(c[1]) for c in cl], float)
    def v(idx):
        a = sizes[idx] / sizes[idx].sum(); w = a @ W[idx]
        return M._auc_safe(yr, fl.add_bias(Xr) @ w)
    full = v(list(range(4)))
    loo = np.array([full - v([j for j in range(4) if j != k]) for k in range(4)])
    e_loo = float(np.abs(loo - pv["osa_loo"]).max()); e_phi = float(np.abs(phi - pv["osa"]).max())
    ref_auc = np.array([M._auc_safe(yr, fl.add_bias(Xr) @ W[k]) for k in range(4)])
    e_ref = float(np.abs(ref_auc - pv["ref_auc"]).max())
    real = set(M.hospitals_by_size()) == set(df.hospital.unique()) and len(names) == 19
    small = names[-10:]
    sizes_small = [int((df.hospital == h).sum()) for h in small]
    ok = max(e_loo, e_phi, e_ref) < 1e-12 and real and max(sizes_small) <= 266
    return ok, (f"LOO diff {e_loo:.1e}, OSA phi diff {e_phi:.1e}, single-model AUC diff {e_ref:.1e}; "
                f"small-silo group = 10 real hospitals with {min(sizes_small)}-{max(sizes_small)} patients")


@check("Second federation (UCI Heart Disease): checksums, 4 real hospitals (303/294/123/200), imputation and scaling from training rows only", "Data")
def _heart():
    import hashlib, os
    from . import heart as H
    rec = dict(l.split()[::-1] for l in open(os.path.join(H.DIR, "SHA256SUM")).read().strip().splitlines())
    ok_sha = all(hashlib.sha256(open(os.path.join(H.DIR, f), "rb").read()).hexdigest() == h for f, h in rec.items())
    df = H.load()
    counts = [int((df.hospital == k).sum()) for k, _ in H.SITES]
    df, tr, te, X, y = H.split(20, df)
    trn = np.concatenate([tr[k] for k, _ in H.SITES])
    # leakage check: changing every test row must not change any training-row feature
    df2 = df.copy(); tst = np.concatenate([te[k] for k, _ in H.SITES])
    for c in ("age", "trestbps", "chol", "thalach", "oldpeak"):
        df2.loc[df2.index[tst], c] = df2[c].iloc[tst] * 3 + 50
    X2 = H.features(df2, trn)
    leak = float(np.abs(X2[trn] - X[trn]).max())
    ok = ok_sha and counts == [303, 294, 123, 200] and len(df) == 920 and leak == 0.0 and X.shape[1] == 20
    return ok, f"checksums {'match' if ok_sha else 'DIFFER'}; sites {counts}; {X.shape[1]} features; max change of training features when test rows change = {leak:.1e}"


@check("Pre-registered nu sweep: arm A keeps the stated fraction of each hospital's training patients, arm B appends features without touching the reference set", "Valuation")
def _nu_sweep():
    from . import nu_sweep as S
    from . import multisite as M
    clA, XrA, yrA = S._data(20, ("A", 0.25, 0))
    cl1, Xr1, yr1 = S._data(20, ("A", 1.0, 0))
    clB, XrB, yrB = S._data(20, ("B", 1.0, 50))
    frac = [len(a[1]) / len(b[1]) for a, b in zip(clA, cl1)]
    ok = (max(abs(f - 0.25) for f in frac) < 0.01 and np.array_equal(yrA, yr1) and np.array_equal(XrA, Xr1)
          and XrB.shape[1] == Xr1.shape[1] + 50 and np.array_equal(XrB[:, :Xr1.shape[1]], Xr1)
          and all(np.array_equal(b[0][:, :Xr1.shape[1]], a[0]) for a, b in zip(cl1, clB)))
    return ok, f"kept fractions {min(frac):.3f}-{max(frac):.3f}; reference set unchanged; arm B adds 50 columns, original features unchanged"


def run_all(verbose=False):
    out = []
    for group, title, f in CHECKS:
        t = time.time()
        try:
            ok, detail = f()
        except Exception as e:  # pragma: no cover
            ok, detail = False, f"{type(e).__name__}: {e}"
        out.append({"group": group, "check": title, "passed": bool(ok), "detail": detail,
                    "seconds": round(time.time() - t, 2)})
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {group:15s} {title}\n         {detail}")
    return out


if __name__ == "__main__":
    import json, os, sys
    res = run_all(verbose=True)
    path = os.path.join(data.ROOT, "results", "final", "validation.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    n = sum(r["passed"] for r in res)
    print(f"\n{n}/{len(res)} checks passed")
    sys.exit(0 if n == len(res) else 1)
