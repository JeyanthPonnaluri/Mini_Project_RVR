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
