"""
Client contribution valuation: exact Shapley values (all 2^K coalitions) and
Leave-One-Out (LOO, the base paper's method). Utility v(S) = AUC on the server's
reference (global test) set of the FedAvg model trained on coalition S; v(empty) = 0.5.

Fix relative to scratch/client_scaling_experiments.py
-----------------------------------------------------
"Top-contributor stability" was computed as mean_k P(rank_k^DP == 1), which is
identically 1/K for every configuration (exactly one client is ranked first in
each realization). It is now defined as
    TS = P( argmax_k phi_k^DP == argmax_k phi_k^noDP )
i.e. the probability that the no-DP top contributor is still ranked first under DP.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import spearmanr

from .fl import TrainConfig, federated_train, predict_proba, safe_auc


def coalition_utilities(clients, X_ref, y_ref, cfg: TrainConfig, seed=0, sigma=None):
    K = len(clients)
    if sigma is None and cfg.epsilon is not None:
        sigma = cfg.sigma()
    v = {(): 0.5}
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            w = federated_train([clients[i] for i in S], cfg, seed=seed, sigma=sigma)["w"]
            a = safe_auc(y_ref, predict_proba(X_ref, w))
            v[S] = 0.5 if math.isnan(a) else a
    return v


def shapley_from_utilities(v, K):
    phi = np.zeros(K)
    fact = math.factorial
    for k in range(K):
        others = [i for i in range(K) if i != k]
        for r in range(K):
            wgt = fact(r) * fact(K - r - 1) / fact(K)
            for S in itertools.combinations(others, r):
                phi[k] += wgt * (v[tuple(sorted(S + (k,)))] - v[S])
    return phi


def loo_from_utilities(v, K):
    full = tuple(range(K))
    return np.array([v[full] - v[tuple(i for i in full if i != k)] for k in range(K)])


def ranks(values):
    """1 = largest contribution."""
    order = np.argsort(-np.asarray(values), kind="stable")
    r = np.empty(len(values), int)
    r[order] = np.arange(1, len(values) + 1)
    return r


def rank_stability(phi_ref, phi_list):
    """Spearman rho, rank-reversal probability and fixed top-contributor stability
    of DP realizations `phi_list` against the no-DP reference `phi_ref`."""
    r0 = ranks(phi_ref)
    top0 = int(np.argmax(phi_ref))
    rhos, rr, ts = [], [], []
    for phi in phi_list:
        rho = spearmanr(phi_ref, phi).correlation
        rhos.append(0.0 if np.isnan(rho) else float(rho))
        rr.append(float(np.mean(ranks(phi) != r0)))
        ts.append(float(int(np.argmax(phi)) == top0))
    return {"spearman": rhos, "rank_reversal": rr, "top_contributor_stability": ts}


# ----------------------------------------------------------------------------
# Vectorised exact coalition training (all coalitions trained simultaneously).
# Mathematically identical to calling federated_train() once per coalition with
# independent DP noise per coalition-run; verified against the loop in tests.
# ----------------------------------------------------------------------------
def all_coalition_utilities(clients, X_ref, y_ref, cfg: TrainConfig, seed=0, sigma=None):
    from .fl import add_bias, sigmoid
    K = len(clients)
    if sigma is None and cfg.epsilon is not None:
        sigma = cfg.sigma()
    rng = np.random.default_rng(seed)
    masks = np.array([[(m >> k) & 1 for k in range(K)] for m in range(1, 2 ** K)], float)  # (M,K)
    M = len(masks)
    sizes = np.array([len(c[1]) for c in clients], float)
    wts = masks * sizes
    wts = wts / wts.sum(1, keepdims=True)                                                     # (M,K)
    d = clients[0][0].shape[1] + 1
    W = np.zeros((M, d))
    Xbs = [add_bias(X) for X, _ in clients]
    for _ in range(cfg.rounds):
        newW = np.zeros_like(W)
        for k, (Xb, (_, y)) in enumerate(zip(Xbs, clients)):
            idx = np.where(masks[:, k] > 0)[0]
            Wk = W[idx].copy(); anchor = W[idx]
            n = len(y)
            for _e in range(cfg.epochs):
                P = sigmoid(Xb @ Wk.T)                    # (n, m)
                R = P - y[:, None]
                if sigma is None:
                    G = (R.T @ Xb) / n                    # (m, d)
                else:
                    per = R.T[:, :, None] * Xb[None, :, :]   # (m, n, d)
                    nr = np.maximum(np.linalg.norm(per, axis=2, keepdims=True), 1e-12)
                    per = per * np.minimum(1.0, cfg.clip / nr)
                    G = (per.sum(1) + rng.normal(0.0, sigma * cfg.clip, size=Wk.shape)) / n
                G = G + cfg.l2 * Wk
                if cfg.mu > 0:
                    G = G + cfg.mu * (Wk - anchor)
                Wk = Wk - cfg.lr * G
            newW[idx] += wts[idx, k][:, None] * Wk
        W = newW
    Pref = sigmoid(add_bias(X_ref) @ W.T)
    v = {(): 0.5}
    for i, m in enumerate(masks):
        a = safe_auc(y_ref, Pref[:, i])
        v[tuple(np.where(m > 0)[0].tolist())] = 0.5 if math.isnan(a) else a
    return v


def shapley_fast(v, K):
    """Exact Shapley from a utility dict, O(2^K * K)."""
    fact = [math.factorial(i) for i in range(K + 1)]
    phi = np.zeros(K)
    for S, val in v.items():
        s = len(S)
        if s == 0:
            continue
        inS = set(S)
        for k in range(K):
            if k in inS:
                T = tuple(i for i in S if i != k)
                phi[k] += fact[s - 1] * fact[K - s] / fact[K] * (val - v[T])
    return phi
