"""
All experiments shown in the Streamlit app. Every function returns a JSON-
serialisable dict; `run_all.py` stores them in results/final/.

Part 1 - Base-paper replication (Hospital participation in FL, clinical data,
         REAL hospitals = TCGA tissue-source sites).
Part 2 - Our framework (DP-FedProx + Shapley + personalisation) on the paper's
         matched clinical-protein cohort (347 patients), Dirichlet label skew.
"""
from __future__ import annotations

import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from . import data, fl
from . import valuation as V
from .stats import paired, summary

SEEDS = list(range(20))
EPSILONS = [10.0, 5.0, 2.0, 1.0, 0.5]
BASE_CFG = dict(rounds=20, epochs=5, lr=0.5, l2=1e-3, delta=1e-5, clip=1.0)
MU = 0.5
MIN_SITE = 20


def cfg(**kw):
    return fl.TrainConfig(**{**BASE_CFG, **kw})


def _nanmean(x):
    x = [v for v in x if v is not None and not math.isnan(v)]
    return float(np.mean(x)) if x else float("nan")


# =============================================================================
# PART 1 - BASE PAPER REPLICATION ON REAL SITES
# =============================================================================
def site_setup(seed, feature_set="full"):
    """Hospitals = tissue-source sites with >= MIN_SITE patients. Each site is split
    80/20 (stratified when possible). Sites with fewer patients form an external
    pool that never trains (used to evaluate free-riding)."""
    df = data.load_clinical_cohort()
    counts = df["site"].value_counts()
    sites = counts[counts >= MIN_SITE].index.tolist()
    rng = np.random.RandomState(seed)
    tr_idx, te_idx = {}, {}
    for s in sites:
        idx = np.where(df["site"].values == s)[0]
        ys = df["y"].values[idx]
        strat = ys if np.bincount(ys, minlength=2).min() >= 2 else None
        a, b = train_test_split(idx, test_size=0.2, random_state=rng.randint(1 << 30), stratify=strat)
        tr_idx[s], te_idx[s] = a, b
    ext = np.where(~df["site"].isin(sites).values)[0]
    all_tr = np.concatenate([tr_idx[s] for s in sites])
    # Shared preprocessing statistics (equivalent to securely aggregated means/variances)
    tp = data.TabularPreprocessor(feature_set).fit(df.iloc[all_tr])
    X = tp.transform(df)
    y = df["y"].values
    return df, sites, tr_idx, te_idx, ext, X, y


def _site_seed(seed):
    df, sites, tr, te, ext, X, y = site_setup(seed)
    c = cfg()
    clients = [(X[tr[s]], y[tr[s]]) for s in sites]
    out = {"seed": seed, "per_site": {}}
    w_fl = fl.federated_train(clients, c, seed)["w"]
    all_tr = np.concatenate([tr[s] for s in sites])
    w_cen = fl.centralized_train(X[all_tr], y[all_tr], c, seed)
    pooled = {k: ([], []) for k in ["LOC", "FL", "FR", "CEN", "BL"]}
    gleason = df["gleason_sum"].fillna(df["gleason_sum"].median()).values
    for i, s in enumerate(sites):
        w_loc = fl.local_only_train(X[tr[s]], y[tr[s]], c, seed)
        w_fr = fl.federated_train([cl for j, cl in enumerate(clients) if j != i], c, seed)["w"]
        Xt, yt = X[te[s]], y[te[s]]
        preds = {"LOC": fl.predict_proba(Xt, w_loc), "FL": fl.predict_proba(Xt, w_fl),
                 "FR": fl.predict_proba(Xt, w_fr), "CEN": fl.predict_proba(Xt, w_cen),
                 "BL": gleason[te[s]]}
        out["per_site"][s] = {k: fl.safe_auc(yt, p) for k, p in preds.items()}
        for k, p in preds.items():
            pooled[k][0].append(yt); pooled[k][1].append(p)
    out["pooled"] = {k: fl.safe_auc(np.concatenate(v[0]), np.concatenate(v[1])) for k, v in pooled.items()}
    # macro AUC = mean over sites whose test split has both classes (same sites for every strategy)
    ok = [s for s in sites if not math.isnan(out["per_site"][s]["FL"])]
    out["macro"] = {k: float(np.mean([out["per_site"][s][k] for s in ok])) for k in pooled}
    out["macro_sites"] = len(ok)
    # learning curve / free-riding (random participation order)
    rng = np.random.default_rng(1000 + seed)
    order = rng.permutation(len(sites))
    ext_X, ext_y = X[ext], y[ext]
    lc = []
    for K in range(1, len(sites) + 1):
        part = order[:K]; non = order[K:]
        w = fl.federated_train([clients[j] for j in part], c, seed)["w"]
        py = np.concatenate([y[te[sites[j]]] for j in part])
        pp = fl.predict_proba(np.vstack([X[te[sites[j]]] for j in part]), w)
        fy = np.concatenate([y[te[sites[j]]] for j in non] + [ext_y])
        fp = fl.predict_proba(np.vstack([X[te[sites[j]]] for j in non] + [ext_X]), w)
        lc.append({"K": K, "participants_auc": fl.safe_auc(py, pp), "free_rider_auc": fl.safe_auc(fy, fp),
                   "n_train": int(sum(len(clients[j][1]) for j in part))})
    out["learning_curve"] = lc
    # contribution: LOO (base paper) vs exact Shapley (ours); utility = pooled site test AUC
    ref_idx = np.concatenate([te[s] for s in sites])
    v = V.all_coalition_utilities(clients, X[ref_idx], y[ref_idx], c, seed)
    out["shapley"] = dict(zip(sites, V.shapley_fast(v, len(sites)).tolist()))
    out["loo"] = dict(zip(sites, V.loo_from_utilities(v, len(sites)).tolist()))
    return out


def base_paper_replication(seeds=SEEDS, processes=2):
    df, sites, tr, te, ext, X, y = site_setup(0)
    info = []
    for s in sites:
        idx = np.where(df["site"].values == s)[0]
        info.append({"site": s, "n": int(len(idx)), "n_train": int(len(tr[s])), "n_test": int(len(te[s])),
                     "prevalence": float(df["y"].values[idx].mean())})
    with Pool(processes) as pool:
        runs = pool.map(_site_seed, seeds)
    strategies = ["LOC", "FL", "FR", "CEN", "BL"]
    per_site = []
    for s in info:
        row = dict(s)
        for k in strategies:
            vals = [r["per_site"][s["site"]][k] for r in runs]
            sm = summary(vals)
            row[k] = sm["mean"]; row[k + "_n_defined"] = sm["n"]
        d = paired([r["per_site"][s["site"]]["FL"] for r in runs], [r["per_site"][s["site"]]["LOC"] for r in runs])
        row["FL_minus_LOC"] = d["mean"]; row["FL_minus_LOC_ci"] = [d["ci_low"], d["ci_high"]]
        f = paired([r["per_site"][s["site"]]["FR"] for r in runs], [r["per_site"][s["site"]]["FL"] for r in runs])
        row["FR_minus_FL"] = f["mean"]
        row["shapley"] = summary([r["shapley"][s["site"]] for r in runs])
        row["loo"] = summary([r["loo"][s["site"]] for r in runs])
        per_site.append(row)
    pooled = {k: summary([r["pooled"][k] for r in runs]) for k in strategies}
    macro = {k: summary([r["macro"][k] for r in runs]) for k in strategies}
    M = lambda a, b: paired([r["macro"][a] for r in runs], [r["macro"][b] for r in runs])
    macro_diff = {"FL-LOC": M("FL", "LOC"), "CEN-FL": M("CEN", "FL"), "FL-BL": M("FL", "BL"),
                  "FR-FL": M("FR", "FL"), "FR-LOC": M("FR", "LOC")}
    lc = []
    for K in range(1, len(sites) + 1):
        p = [r["learning_curve"][K - 1] for r in runs]
        lc.append({"K": K, "participants": summary([q["participants_auc"] for q in p]),
                   "free_riders": summary([q["free_rider_auc"] for q in p]),
                   "gap": paired([q["participants_auc"] for q in p], [q["free_rider_auc"] for q in p])})
    from scipy.stats import spearmanr
    rho_ls = [spearmanr([r["loo"][s] for s in sites], [r["shapley"][s] for s in sites]).correlation for r in runs]
    rho_size = spearmanr([s["n"] for s in info], [row["shapley"]["mean"] for row in per_site]).correlation
    return {"sites": info, "n_external": int(len(ext)), "per_site": per_site, "pooled": pooled, "macro": macro,
            "macro_diff": macro_diff, "macro_sites": summary([r["macro_sites"] for r in runs]), "learning_curve": lc,
            "loo_vs_shapley_spearman": summary(rho_ls), "shapley_vs_size_spearman": float(rho_size),
            "config": {**BASE_CFG, "seeds": len(seeds), "min_site_size": MIN_SITE, "feature_set": "full"}}


# =============================================================================
# PART 2 - OUR FRAMEWORK ON THE MATCHED COHORT
# =============================================================================
def matched_split(seed, feature_set="full", protein=False):
    clin, P = data.load_matched_cohort()
    idx = np.arange(len(clin))
    tr, te = train_test_split(idx, test_size=0.2, stratify=clin["y"], random_state=seed)
    fs = feature_set
    Ptr = P.iloc[tr] if protein else None
    Pte = P.iloc[te] if protein else None
    Xtr, Xte, names = data.build_features(clin.iloc[tr], clin.iloc[te], fs, Ptr, Pte)
    return Xtr, clin["y"].values[tr], Xte, clin["y"].values[te], names


def dirichlet_clients(Xtr, ytr, Xte, yte, K, alpha, seed):
    a, b = fl.dirichlet_partition(ytr, yte, K, alpha, seed)
    return [(Xtr[a == k], ytr[a == k]) for k in range(K)], [(Xte[b == k], yte[b == k]) for k in range(K)]


# ---- A0. centralized baselines & feature-set ablation (repeated CV) ---------
def centralized_baselines(n_splits=5, n_repeats=5):
    import warnings
    warnings.filterwarnings("ignore")  # sklearn 1.8 'penalty' deprecation / MLP convergence notices
    clin, P = data.load_matched_cohort()
    y = clin["y"].values
    sets = {"Clinical (full)": ("full", False), "Clinical (pre-op only)": ("preop", False),
            "Protein only (PCA)": ("none", True), "Clinical (full) + protein": ("full", True),
            "Pre-op + protein": ("preop", True)}
    models = {
        "NumPy LR (federated model, centralised)": None,
        "LR (L2)": lambda: LogisticRegression(C=1.0, max_iter=5000),
        "LR (L1)": lambda: LogisticRegression(C=1.0, penalty="l1", solver="liblinear"),
        "Random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=1),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=2000, random_state=0),
    }
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    folds = list(cv.split(clin, y))
    res = {s: {m: [] for m in models} for s in sets}
    dims = {}
    for tr, te in folds:
        for sname, (fs, prot) in sets.items():
            Xtr, Xte, names = data.build_features(clin.iloc[tr], clin.iloc[te], fs,
                                                  P.iloc[tr] if prot else None, P.iloc[te] if prot else None)
            dims[sname] = len(names)
            for mname, mk in models.items():
                if mk is None:
                    w = fl.centralized_train(Xtr, y[tr], cfg())
                    p = fl.predict_proba(Xte, w)
                else:
                    p = mk().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
                res[sname][mname].append(fl.safe_auc(y[te], p))
    table = [{"features": s, "dim_last_fold": dims[s], "model": m, **summary(v)} for s in sets for m, v in res[s].items()]
    key = "NumPy LR (federated model, centralised)"
    comps = {
        "protein_added_to_full": paired(res["Clinical (full) + protein"][key], res["Clinical (full)"][key]),
        "full_vs_preop": paired(res["Clinical (full)"][key], res["Clinical (pre-op only)"][key]),
        "protein_added_to_preop": paired(res["Pre-op + protein"][key], res["Clinical (pre-op only)"][key]),
    }
    return {"table": table, "comparisons": comps, "n_folds": len(folds),
            "note": "Fold-level paired differences are correlated across folds; CIs are indicative."}


# ---- A. client drift under heterogeneity ------------------------------------
def _drift_seed(args):
    seed, alpha = args
    Xtr, ytr, Xte, yte, _ = matched_split(seed)
    cl, _ = dirichlet_clients(Xtr, ytr, Xte, yte, 3, alpha, seed)
    out = {}
    for name, mu in [("FedAvg", 0.0), ("FedProx", MU)]:
        r = fl.federated_train(cl, cfg(mu=mu), seed, eval_set=(Xte, yte))
        out[name] = {"drift": float(np.mean(r["drift_history"])), "auc": r["auc_history"][-1],
                     "drift_curve": r["drift_history"], "auc_curve": r["auc_history"]}
    mus = {}
    if alpha == 0.5:
        for mu in [0.0, 0.01, 0.1, 0.5, 1.0]:
            r = fl.federated_train(cl, cfg(mu=mu), seed, eval_set=(Xte, yte))
            mus[str(mu)] = {"drift": float(np.mean(r["drift_history"])), "auc": r["auc_history"][-1]}
    out["mu_sweep"] = mus
    out["label_skew"] = [float(c[1].mean()) for c in cl]
    out["sizes"] = [int(len(c[1])) for c in cl]
    return out


def drift_experiment(alphas=(100.0, 10.0, 1.0, 0.5, 0.1), seeds=SEEDS, processes=2):
    jobs = [(s, a) for a in alphas for s in seeds]
    with Pool(processes) as pool:
        runs = pool.map(_drift_seed, jobs)
    rows = []
    for a in alphas:
        rr = [r for (s, al), r in zip(jobs, runs) if al == a]
        da = [r["FedAvg"]["drift"] for r in rr]; dp = [r["FedProx"]["drift"] for r in rr]
        rel = [(x - z) / x * 100 for x, z in zip(da, dp)]
        rows.append({"alpha": a, "fedavg_drift": summary(da), "fedprox_drift": summary(dp),
                     "drift_reduction_pct": summary(rel),
                     "fedavg_auc": summary([r["FedAvg"]["auc"] for r in rr]),
                     "fedprox_auc": summary([r["FedProx"]["auc"] for r in rr]),
                     "auc_diff": paired([r["FedProx"]["auc"] for r in rr], [r["FedAvg"]["auc"] for r in rr]),
                     "mean_label_spread": summary([max(r["label_skew"]) - min(r["label_skew"]) for r in rr]),
                     "fedavg_drift_curve": np.mean([r["FedAvg"]["drift_curve"] for r in rr], 0).tolist(),
                     "fedprox_drift_curve": np.mean([r["FedProx"]["drift_curve"] for r in rr], 0).tolist(),
                     "fedavg_auc_curve": np.nanmean([r["FedAvg"]["auc_curve"] for r in rr], 0).tolist(),
                     "fedprox_auc_curve": np.nanmean([r["FedProx"]["auc_curve"] for r in rr], 0).tolist()})
    rr = [r for (s, al), r in zip(jobs, runs) if al == 0.5]
    mu_rows = [{"mu": float(m), "drift": summary([r["mu_sweep"][m]["drift"] for r in rr]),
                "auc": summary([r["mu_sweep"][m]["auc"] for r in rr])} for m in rr[0]["mu_sweep"]]
    return {"rows": rows, "mu_sweep_alpha_0.5": mu_rows,
            "definition": "drift_t = mean_k ||w_k^t - w^{t-1}||_2, averaged over rounds (both algorithms)"}


# ---- B/C. privacy-utility and privacy x heterogeneity grid ------------------
def _grid_seed(args):
    seed, alpha = args
    Xtr, ytr, Xte, yte, _ = matched_split(seed)
    cl, _ = dirichlet_clients(Xtr, ytr, Xte, yte, 3, alpha, seed)
    out = {}
    for eps in [None] + EPSILONS:
        for name, kw in [("FedAvg", {}), ("FedProx", {"mu": MU}),
                         ("FedAvg_halfLR", {"lr": BASE_CFG["lr"] / 2})]:   # step-size control
            w = fl.federated_train(cl, cfg(epsilon=eps, **kw), seed)["w"]
            out[f"{name}|{eps}"] = fl.safe_auc(yte, fl.predict_proba(Xte, w))
    return out


def privacy_grid(alphas=(10.0, 1.0, 0.5, 0.1), seeds=SEEDS, processes=2):
    jobs = [(s, a) for a in alphas for s in seeds]
    with Pool(processes) as pool:
        runs = pool.map(_grid_seed, jobs)
    steps = BASE_CFG["rounds"] * BASE_CFG["epochs"]
    sig = {str(e): fl.calibrate_sigma(e, steps, BASE_CFG["delta"]) for e in EPSILONS}
    eps_check = {e: fl.rdp_epsilon(s, steps, BASE_CFG["delta"])[0] for e, s in sig.items()}
    cells = []
    for a in alphas:
        rr = [r for (s, al), r in zip(jobs, runs) if al == a]
        for eps in [None] + EPSILONS:
            fa = [r[f"FedAvg|{eps}"] for r in rr]; fp = [r[f"FedProx|{eps}"] for r in rr]
            fh = [r[f"FedAvg_halfLR|{eps}"] for r in rr]
            na = [r["FedAvg|None"] for r in rr]
            cells.append({"alpha": a, "epsilon": eps, "fedavg": summary(fa), "fedprox": summary(fp),
                          "fedavg_half_lr": summary(fh), "prox_minus_halflr_avg": paired(fp, fh),
                          "prox_minus_avg": paired(fp, fa),
                          "fedavg_minus_nodp": paired(fa, na) if eps is not None else None})
    return {"cells": cells, "sigma": sig, "epsilon_recomputed": eps_check, "steps": steps,
            "delta": BASE_CFG["delta"], "mu": MU, "clients": 3}


# ---- D. personalisation ------------------------------------------------------
def _pfl_seed(args):
    seed, K, alpha = args
    Xtr, ytr, Xte, yte, _ = matched_split(seed)
    cl, ct = dirichlet_clients(Xtr, ytr, Xte, yte, K, alpha, seed)
    c = cfg()
    w_avg = fl.federated_train(cl, c, seed)["w"]
    w_prox = fl.federated_train(cl, cfg(mu=MU), seed)["w"]
    res = {k: {"y": [], "p": []} for k in ["Local", "FedAvg", "FedProx", "PFL"]}
    per_client = []
    for (X, y), (Xt, yt) in zip(cl, ct):
        w_loc = fl.local_only_train(X, y, c, seed)
        w_p = fl.personalise(X, y, w_prox)
        pc = {}
        for k, w in [("Local", w_loc), ("FedAvg", w_avg), ("FedProx", w_prox), ("PFL", w_p)]:
            p = fl.predict_proba(Xt, w)
            res[k]["y"].append(yt); res[k]["p"].append(p)
            pc[k] = fl.safe_auc(yt, p)
        pc["n_train"] = int(len(y)); pc["n_test"] = int(len(yt)); pc["prevalence"] = float(y.mean()) if len(y) else None
        per_client.append(pc)
    ok = [c for c in per_client if not math.isnan(c["Local"])]
    macro = {k: (float(np.mean([c[k] for c in ok])) if ok else float("nan")) for k in res}
    return {"macro": macro, "n_defined": len(ok), "per_client": per_client}


def personalization(Ks=(3, 5, 10), alpha=0.5, seeds=SEEDS, processes=2):
    out = {}
    for K in Ks:
        jobs = [(s, K, alpha) for s in seeds]
        with Pool(processes) as pool:
            runs = pool.map(_pfl_seed, jobs)
        methods = ["Local", "FedAvg", "FedProx", "PFL"]
        macro = {m: summary([r["macro"][m] for r in runs]) for m in methods}
        defined = summary([r["n_defined"] for r in runs])
        worst = {m: summary([min([c[m] for c in r["per_client"] if not math.isnan(c[m])], default=float("nan"))
                             for r in runs]) for m in methods}
        out[str(K)] = {"macro": macro, "worst_client": worst,
                       "clients_with_defined_auc": defined,
                       "PFL_minus_Local": paired([r["macro"]["PFL"] for r in runs], [r["macro"]["Local"] for r in runs]),
                       "PFL_minus_FedProx": paired([r["macro"]["PFL"] for r in runs], [r["macro"]["FedProx"] for r in runs]),
                       "FedProx_minus_Local": paired([r["macro"]["FedProx"] for r in runs], [r["macro"]["Local"] for r in runs]),
                       "example_seed0": runs[0]["per_client"]}
    return {"alpha": alpha, "by_K": out,
            "note": "Per-client AUC is undefined when a client's local test split has one class. 'macro' = mean AUC "
                    "over the clients whose test split has both classes (identical clients for every method). "
                    "Pooling predictions of different local models is NOT used: it rewards models for "
                    "learning their own client's base rate and inflates AUC."}


# ---- E/F. Shapley vs LOO, Shapley stability under DP, client scaling ---------
def _shap_seed(args):
    seed, K, alpha, R = args
    Xtr, ytr, Xte, yte, _ = matched_split(seed)
    cl, _ = dirichlet_clients(Xtr, ytr, Xte, yte, K, alpha, seed)
    v0 = V.all_coalition_utilities(cl, Xte, yte, cfg(), seed)
    phi0 = V.shapley_fast(v0, K)
    loo0 = V.loo_from_utilities(v0, K)
    out = {"phi_nodp": phi0.tolist(), "loo_nodp": loo0.tolist(), "sizes": [int(len(c[1])) for c in cl],
           "v_full": v0[tuple(range(K))], "dp": {}}
    for eps in EPSILONS:
        c = cfg(epsilon=eps); sig = c.sigma()
        phis = [V.shapley_fast(V.all_coalition_utilities(cl, Xte, yte, c, seed * 1000 + r + 1, sigma=sig), K)
                for r in range(R)]
        st = V.rank_stability(phi0, phis)
        out["dp"][str(eps)] = {**st, "phi": [p.tolist() for p in phis]}
    return out


def shapley_study(Ks=(3, 5, 10), alpha=0.5, seeds=SEEDS, R_by_K=None, processes=2):
    from scipy.stats import spearmanr, wilcoxon
    R_by_K = R_by_K or {3: 5, 5: 5, 10: 2}
    out = {}
    for K in Ks:
        jobs = [(s, K, alpha, R_by_K[K]) for s in seeds]
        with Pool(processes) as pool:
            runs = pool.map(_shap_seed, jobs)
        rho_loo = [spearmanr(r["phi_nodp"], r["loo_nodp"]).correlation for r in runs]
        rho_size = [spearmanr(r["phi_nodp"], r["sizes"]).correlation for r in runs]
        per_eps = []
        for eps in EPSILONS:
            sp = [np.mean(r["dp"][str(eps)]["spearman"]) for r in runs]
            rr = [np.mean(r["dp"][str(eps)]["rank_reversal"]) for r in runs]
            ts = [np.mean(r["dp"][str(eps)]["top_contributor_stability"]) for r in runs]
            d = np.array(sp) - 0.90
            p = float(wilcoxon(d, alternative="less").pvalue) if np.any(d != 0) else None
            per_eps.append({"epsilon": eps, "spearman": summary(sp), "rank_reversal": summary(rr),
                            "top_contributor_stability": summary(ts), "wilcoxon_p_rho_below_0.90": p})
        out[str(K)] = {"per_epsilon": per_eps, "spearman_shapley_vs_loo": summary(rho_loo),
                       "spearman_shapley_vs_size": summary(rho_size),
                       "efficiency_check_max_abs": float(max(abs(sum(r["phi_nodp"]) - (r["v_full"] - 0.5)) for r in runs)),
                       "realizations_per_seed": R_by_K[K], "seeds": len(seeds),
                       "example_seed0": {"phi_nodp": runs[0]["phi_nodp"], "loo_nodp": runs[0]["loo_nodp"],
                                         "sizes": runs[0]["sizes"],
                                         "phi_dp_first": {e: runs[0]["dp"][e]["phi"][0] for e in runs[0]["dp"]}}}
    return {"alpha": alpha, "by_K": out,
            "definitions": {"rank_reversal": "fraction of clients whose DP rank differs from the no-DP rank",
                            "top_contributor_stability": "P(no-DP top contributor is still ranked 1 under DP) "
                                                         "[fixed; the old metric was identically 1/K]",
                            "random_TS_baseline": "1/K"}}


# ---- G. membership inference -------------------------------------------------
def _mia_seed(seed):
    Xtr, ytr, Xte, yte, _ = matched_split(seed)
    cl, _ = dirichlet_clients(Xtr, ytr, Xte, yte, 3, 10.0, seed)
    out = {}
    for eps in [None] + EPSILONS:
        w = fl.federated_train(cl, cfg(epsilon=eps), seed)["w"]
        auc, adv = fl.mia_confidence_attack(w, Xtr, ytr, Xte, yte, seed)
        out[str(eps)] = {"auc": auc, "adv": adv, "test_auc": fl.safe_auc(yte, fl.predict_proba(Xte, w)),
                         "train_auc": fl.safe_auc(ytr, fl.predict_proba(Xtr, w))}
    # deliberately over-fitted reference model (proves the attack works when there is leakage)
    rng = np.random.default_rng(seed)
    noise_tr = rng.normal(size=(len(ytr), 200)); noise_te = rng.normal(size=(len(yte), 200))
    Xa, Xb = np.hstack([Xtr, noise_tr]), np.hstack([Xte, noise_te])
    w = fl.centralized_train(Xa, ytr, cfg(rounds=200, l2=0.0, lr=1.0))
    auc, adv = fl.mia_confidence_attack(w, Xa, ytr, Xb, yte, seed)
    out["overfit_reference"] = {"auc": auc, "adv": adv}
    return out


def mia_experiment(seeds=SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_mia_seed, seeds)
    keys = ["None"] + [str(e) for e in EPSILONS]
    rows = [{"epsilon": k, "attack_auc": summary([r[k]["auc"] for r in runs]),
             "advantage": summary([r[k]["adv"] for r in runs]),
             "test_auc": summary([r[k]["test_auc"] for r in runs]),
             "generalisation_gap": summary([r[k]["train_auc"] - r[k]["test_auc"] for r in runs])} for k in keys]
    return {"rows": rows, "overfit_reference": {"attack_auc": summary([r["overfit_reference"]["auc"] for r in runs]),
                                                "advantage": summary([r["overfit_reference"]["adv"] for r in runs])},
            "note": "Balanced members (train) vs non-members (test). Advantage = max over thresholds of TPR-FPR "
                    "(optimistic by construction, ~0.1-0.2 even for a random score with 70+70 samples)."}


# ---- H. synthetic distribution-shift stress test ----------------------------
def _shift(X, y, kind, sev, n_num, rng):
    X = X.copy(); y = y.copy()
    if kind == "covariate":
        X[:, :n_num] = X[:, :n_num] * rng.uniform(1 - 0.25 * sev, 1 + 0.25 * sev, n_num) \
            + rng.normal(0, 0.5 * sev, size=(len(X), n_num))
    else:
        nf = int(len(y) * min(0.25 * sev, 0.5))
        if nf:
            i = rng.choice(len(y), nf, replace=False); y[i] = 1 - y[i]
    return X, y


def shift_experiment(seeds=SEEDS, severities=(0.0, 0.5, 1.0, 1.5, 2.0)):
    n_num = len(data.FEATURE_SETS["full"][0])
    res = {k: {s: [] for s in severities} for k in ["covariate", "concept"]}
    for seed in seeds:
        Xtr, ytr, Xte, yte, _ = matched_split(seed)
        cl, _ = dirichlet_clients(Xtr, ytr, Xte, yte, 3, 0.5, seed)
        w = fl.federated_train(cl, cfg(mu=MU), seed)["w"]
        rng = np.random.default_rng(seed)
        for kind in res:
            for s in severities:
                Xs, ys = _shift(Xte, yte, kind, s, n_num, rng)
                res[kind][s].append(fl.safe_auc(ys, fl.predict_proba(Xs, w)))
    return {"severities": list(severities),
            "covariate": [summary(res["covariate"][s]) for s in severities],
            "concept": [summary(res["concept"][s]) for s in severities],
            "note": "Synthetic perturbation of the held-out split (numeric features scaled + Gaussian noise; "
                    "concept shift flips 25%*severity of labels). Not an external clinical cohort."}


# ---- I. ablation table -------------------------------------------------------
def _abl_seed(seed):
    out = {}
    Xtr, ytr, Xte, yte, _ = matched_split(seed)
    cl, ct = dirichlet_clients(Xtr, ytr, Xte, yte, 3, 0.5, seed)
    out["Centralised (pooled data)"] = fl.safe_auc(yte, fl.predict_proba(Xte, fl.centralized_train(Xtr, ytr, cfg())))
    for name, kw in [("FedAvg", {}), ("FedProx", {"mu": MU}), ("FedProx + DP (eps=5)", {"mu": MU, "epsilon": 5.0}),
                     ("FedProx + DP (eps=1)", {"mu": MU, "epsilon": 1.0})]:
        out[name] = fl.safe_auc(yte, fl.predict_proba(Xte, fl.federated_train(cl, cfg(**kw), seed)["w"]))
    for label, fs, prot in [("FedProx, pre-op features only", "preop", False),
                            ("FedProx, clinical (full) + protein PCs", "full", True)]:
        a, b, c_, d, _ = matched_split(seed, fs, prot)
        cl2, _ = dirichlet_clients(a, b, c_, d, 3, 0.5, seed)
        out[label] = fl.safe_auc(d, fl.predict_proba(c_, fl.federated_train(cl2, cfg(mu=MU), seed)["w"]))
    return out


def ablation(seeds=SEEDS, processes=2):
    with Pool(processes) as pool:
        runs = pool.map(_abl_seed, seeds)
    keys = list(runs[0].keys())
    rows = [{"configuration": k, **summary([r[k] for r in runs]),
             "vs_FedProx": paired([r[k] for r in runs], [r["FedProx"] for r in runs]) if k != "FedProx" else None}
            for k in keys]
    return {"rows": rows, "alpha": 0.5, "clients": 3}


# ---- cohort facts --------------------------------------------------------------
def cohort_facts():
    raw = pd.read_csv(data.CLINICAL_TSV, sep="\t")
    clin = data.load_clinical_cohort()
    m, P = data.load_matched_cohort()
    tr, te = train_test_split(np.arange(len(m)), test_size=0.2, stratify=m["y"], random_state=42)
    surv = data.survival_events(m["sample"])
    ev = surv.set_index("sample")["OS"]
    pp = data.ProteinPreprocessor().fit(P)
    tp = data.TabularPreprocessor("full").fit(m)
    return {
        "raw_records": int(len(raw)), "raw_patients": int(raw["submitter_id"].nunique()),
        "raw_sample_types": raw["sample"].str[13:15].value_counts().to_dict(),
        "clinical_cohort": int(len(clin)), "clinical_cohort_pos": int(clin["y"].sum()),
        "clinical_sites": int(clin["site"].nunique()),
        "protein_samples": int(P.shape[0]) if False else int(pd.read_csv(data.PROTEIN_TSV, sep="\t").shape[1] - 1),
        "protein_targets_raw": int(P.shape[1]), "protein_targets_kept": pp.n_proteins,
        "protein_pcs_95pct_full_cohort": pp.n_components,
        "matched_cohort": int(len(m)), "matched_pos": int(m["y"].sum()),
        "matched_train": int(len(tr)), "matched_test": int(len(te)),
        "survival_events_matched": int(ev.sum()),
        "survival_events_train_seed42": int(ev.loc[m["sample"].iloc[tr]].sum()),
        "survival_events_test_seed42": int(ev.loc[m["sample"].iloc[te]].sum()),
        "survival_events_all_572": int(pd.read_csv(data.SURVIVAL_TSV, sep="\t")["OS"].sum()),
        "t_stage_counts_matched": m["t_stage"].value_counts().to_dict(),
        "features_full": tp.feature_names_,
        "features_preop": data.TabularPreprocessor("preop").fit(m).feature_names_,
        "excluded_columns": data.EXCLUDED_REASONS,
    }
