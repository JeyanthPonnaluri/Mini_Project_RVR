"""
Privacy-honest contribution valuation for cross-silo federated learning.

Problem
-------
Exact Shapley values need the utility v(S) of a model trained by every coalition S of hospitals.
The usual approach under differential privacy RETRAINS each coalition with DP-SGD. Every hospital
then takes part in 2^(K-1) private training runs; by composition its total privacy loss is far
larger than the per-run epsilon that is usually reported. If the total budget is held at epsilon,
each run needs much more noise and the valuation becomes useless (shown in the experiments).

Method: OSA-Shapley (one-shot-averaging Shapley)
------------------------------------------------
1. Each hospital k releases ONE model w_k trained locally with DP gradient descent at budget
   epsilon_v (per-record clipping, Gaussian noise, Renyi-DP accountant). Iterate (tail) averaging
   over the second half of the local steps damps the DP noise.
2. The server forms the model of every coalition S as the size-weighted average
       w_S = sum_{k in S} (n_k / n_S) w_k
   - pure post-processing of the K released models, so ALL 2^K coalition models together cost
   epsilon_v per hospital, independent of K. For a convex model trained to convergence the
   one-shot average approximates the federated (FedAvg) solution of the coalition.
3. v(S) = AUC of w_S on the server's reference set; exact Shapley values by vectorised bitmask
   summation over all 2^K coalitions (K = 19 takes about a minute).

All coalitions share the same released models, so the DP noise is common to every coalition and
largely cancels in marginal contributions (cf. the correlated-noise argument of Zhou et al.,
"Data value estimation on private gradients", and the gradient/model-reconstruction valuation of
Song et al. 2019 and GTG-Shapley, Liu et al. 2022).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

from . import fl

OSA_STEPS = 30        # local DP-GD steps (chosen on tuning seeds 0-3; evaluation uses other seeds)
OSA_LR = 1.0
OSA_L2 = 1e-3


def local_dp_model(X, y, epsilon=None, steps=OSA_STEPS, lr=OSA_LR, l2=OSA_L2, clip=1.0, delta=1e-5, rng=None):
    """One locally trained (DP) logistic-regression model, tail-averaged over the last half of the steps.
    Returns (weights, sigma). epsilon=None trains without DP."""
    sigma = None if epsilon is None else fl.calibrate_sigma(epsilon, steps, delta)
    Xb = fl.add_bias(X)
    n = len(y)
    w = np.zeros(Xb.shape[1])
    acc = np.zeros_like(w)
    start = steps - max(1, steps // 2)
    for t in range(steps):
        p = fl.sigmoid(Xb @ w)
        if sigma is None:
            g = Xb.T @ (p - y) / n
        else:
            per = Xb * (p - y)[:, None]
            nr = np.maximum(np.linalg.norm(per, axis=1, keepdims=True), 1e-12)
            per = per * np.minimum(1.0, clip / nr)
            g = (per.sum(0) + rng.normal(0.0, sigma * clip, size=w.shape)) / n
        w = w - lr * (g + l2 * w)
        if t >= start:
            acc += w
    return acc / (steps - start), sigma


def all_masks(K):
    m = np.arange(1, 2 ** K, dtype=np.int64)
    return ((m[:, None] >> np.arange(K)) & 1).astype(np.float64), m


def auc_columns(y, S):
    """AUC of every column of the score matrix S (n x m) against labels y, ties = average ranks."""
    y = np.asarray(y)
    npos = int(y.sum()); nneg = len(y) - npos
    R = rankdata(S, axis=0, method="average")
    return (R[y == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def osa_utilities(W_local, sizes, masks, X_ref, y_ref, chunk=20000):
    """Utility of every coalition (rows of masks) = AUC of the size-weighted average model."""
    sizes = np.asarray(sizes, float)
    Xb = fl.add_bias(X_ref)
    out = np.empty(len(masks))
    for i in range(0, len(masks), chunk):
        A = masks[i:i + chunk] * sizes
        A = A / A.sum(1, keepdims=True)
        out[i:i + chunk] = auc_columns(y_ref, Xb @ (A @ W_local).T)
    return out


def shapley_from_mask_utilities(u, K, empty=0.5):
    """Exact Shapley values from utilities indexed by bitmask (u[m-1] = v(S_m), m = 1..2^K-1)."""
    v = np.empty(2 ** K)
    v[0] = empty
    v[1:] = u
    ints = np.arange(2 ** K, dtype=np.int64)
    size = np.zeros(2 ** K, dtype=np.int64)
    for k in range(K):
        size += (ints >> k) & 1
    fact = [math.factorial(i) for i in range(K + 1)]
    wt = np.array([fact[s] * fact[K - s - 1] / fact[K] for s in range(K)])
    phi = np.zeros(K)
    for k in range(K):
        without = ints[((ints >> k) & 1) == 0]
        phi[k] = np.sum(wt[size[without]] * (v[without | (1 << k)] - v[without]))
    return phi


def osa_shapley(clients, X_ref, y_ref, epsilon=None, rng=None, steps=OSA_STEPS, return_models=False):
    """OSA-Shapley over all 2^K coalitions. clients = [(X_k, y_k)]."""
    K = len(clients)
    rng = np.random.default_rng(0) if rng is None else rng
    W = np.array([local_dp_model(X, y, epsilon, steps=steps, rng=rng)[0] for X, y in clients])
    sizes = [len(c[1]) for c in clients]
    masks, _ = all_masks(K)
    u = osa_utilities(W, sizes, masks, X_ref, y_ref)
    phi = shapley_from_mask_utilities(u, K)
    return (phi, W) if return_models else phi


def retraining_privacy_cost(epsilon, K, steps, delta=1e-5):
    """Total epsilon actually spent by each hospital when all coalitions are retrained with a
    per-run budget `epsilon` (it joins 2^(K-1) runs of `steps` Gaussian steps each)."""
    sigma = fl.calibrate_sigma(epsilon, steps, delta)
    return fl.rdp_epsilon(sigma, steps * 2 ** (K - 1), delta)[0]


# ---------------------------------------------------------------------------------------------
# Post-processing competitors: reconstruction from one DP-FedAvg run (Song et al. 2019; GTG-Shapley,
# Liu et al. 2022). In DP-FedAvg every hospital releases its noisy local model in every round and the
# accountant already covers all T*E steps, so any function of these releases also costs epsilon once.
# ---------------------------------------------------------------------------------------------
def fl_trace(clients, cfg, seed=0):
    """Run (DP-)FedAvg and record, per round, the global model before the round and the K local models.
    Returns list of (w_prev, W_local[K, d]) and the final global model."""
    rng = np.random.default_rng(seed)
    d = clients[0][0].shape[1] + 1
    sigma = cfg.sigma() if cfg.epsilon is not None else None
    sizes = np.array([len(c[1]) for c in clients], float)
    a = sizes / sizes.sum()
    w = np.zeros(d)
    trace = []
    for _ in range(cfg.rounds):
        W = np.array([fl.local_update(X, y, w, w, cfg.epochs, cfg.lr, cfg.l2, cfg.mu, sigma, cfg.clip, rng)
                      for X, y in clients])
        trace.append((w.copy(), W))
        w = a @ W
    return trace, w


def mr_shapley(trace, sizes, X_ref, y_ref):
    """Exact multi-round reconstruction Shapley (MR; GTG-Shapley is its truncated Monte-Carlo version).
    Round t: v_t(S) = AUC of the size-weighted average of the local models of S, v_t(empty) = AUC of the
    global model before the round; phi = sum over rounds of the per-round exact Shapley values."""
    K = len(sizes)
    masks, _ = all_masks(K)
    phi = np.zeros(K)
    Xb = fl.add_bias(X_ref)
    for w_prev, W in trace:
        u = osa_utilities(W, sizes, masks, X_ref, y_ref)
        s = Xb @ w_prev
        v0 = 0.5 if np.allclose(s, s[0]) else float(auc_columns(y_ref, s[:, None])[0])
        phi += shapley_from_mask_utilities(u, K, empty=v0)
    return phi


def or_shapley(trace, sizes, X_ref, y_ref):
    """One-round reconstruction (OR, Song et al. 2019): w_S = w_0 + sum_t sum_{k in S} (n_k/n_S) (w_k^t - w^{t-1});
    v(S) = AUC(w_S), v(empty) = 0.5."""
    K = len(sizes)
    masks, _ = all_masks(K)
    D = np.sum([W - w_prev[None, :] for w_prev, W in trace], axis=0)       # K x d: summed updates
    w0 = trace[0][0]
    u = osa_utilities(D + w0[None, :], sizes, masks, X_ref, y_ref)
    return shapley_from_mask_utilities(u, K)


def coalition_noise_bound(sigma, clip, sizes, steps=OSA_STEPS, lr=OSA_LR):
    """Proposition 3 helper: per-coordinate standard deviation of the DP perturbation of every coalition model
    for OSA, ignoring the contraction of the loss (upper bound for one tail-averaged DP-GD release):
    sd_k <= lr*sigma*clip/n_k * sqrt(steps) and sd(w_S) = sqrt(sum_k (n_k/n_S)^2 sd_k^2) = lr*sigma*clip*sqrt(|S| steps)/n_S."""
    sizes = np.asarray(sizes, float)
    masks, _ = all_masks(len(sizes))
    nS = masks @ sizes
    return lr * sigma * clip * np.sqrt(masks.sum(1) * steps) / nS
