"""
Privacy-honest valuation with a NON-CONVEX model: a one-hidden-layer MLP (tanh, 16 units).

Answers the question whether OSA-Shapley still works when the model is not convex, where averaging
independently trained models is no longer guaranteed to approximate the federated solution.

Coalition proxies compared (all evaluated against exact non-private retraining Shapley with the
federated MLP, i.e. FedAvg-MLP trained on every one of the 2^K - 1 coalitions):
  * retrain (per-run / budget-matched): DP FedAvg-MLP retrained on every coalition
  * OSA-param   : each hospital releases ONE DP-trained MLP from a SHARED initialisation;
                  coalition model = size-weighted parameter average (post-processing)
  * OSA-ensemble: same releases; coalition prediction = size-weighted average of the members'
                  log-odds (post-processing; no assumption about parameter alignment)
  * MR          : exact multi-round reconstruction from one DP FedAvg-MLP run (GTG upper bound)

Parameters are kept as flat vectors so that many models (all coalitions) train in one batch. Weight decay
(1e-3) is applied to all parameters, including biases (the stronger-baselines MLP decays only weights).
"""
from __future__ import annotations

import math

import numpy as np

from . import fl
from .baselines import TUNED
from .private_valuation import all_masks, auc_columns, shapley_from_mask_utilities

H = TUNED["mlp_hidden"]
L2 = TUNED["mlp_l2"]


# ------------------------------------------------------------------ flat parameterisation
def n_params(d, h=H):
    return d * h + h + h + 1


def unpack(T, d, h=H):
    """T: (M, P) -> W1 (M,d,h), b1 (M,h), w2 (M,h), b2 (M,)"""
    i = 0
    W1 = T[:, i:i + d * h].reshape(-1, d, h); i += d * h
    b1 = T[:, i:i + h]; i += h
    w2 = T[:, i:i + h]; i += h
    return W1, b1, w2, T[:, i]


def init(d, seed, h=H):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1 / np.sqrt(d), d * h), np.zeros(h), rng.normal(0, 1 / np.sqrt(h), h), [0.0]])


def logits(T, X, d, h=H):
    """T (M,P), X (n,d) -> (n, M) log-odds"""
    W1, b1, w2, b2 = unpack(T, d, h)
    Hd = np.tanh(np.einsum("nd,mdh->mnh", X, W1) + b1[:, None, :])
    return (np.einsum("mnh,mh->mn", Hd, w2) + b2[:, None]).T


def _grads(T, X, y, d, h=H, per_sample=False):
    """Gradients of the mean BCE for M models. Returns (M,P) or, per_sample, (M,n,P) of per-record grads."""
    W1, b1, w2, b2 = unpack(T, d, h)
    Z = np.einsum("nd,mdh->mnh", X, W1) + b1[:, None, :]
    Hd = np.tanh(Z)
    p = fl.sigmoid(np.einsum("mnh,mh->mn", Hd, w2) + b2[:, None])
    r = p - y[None, :]                                      # (M,n)
    dZ = r[:, :, None] * w2[:, None, :] * (1 - Hd ** 2)     # (M,n,h)
    if per_sample:
        gW1 = np.einsum("nd,mnh->mndh", X, dZ).reshape(T.shape[0], len(y), d * h)
        return np.concatenate([gW1, dZ, r[:, :, None] * Hd, r[:, :, None]], axis=2)
    n = len(y)
    gW1 = np.einsum("nd,mnh->mdh", X, dZ).reshape(T.shape[0], d * h) / n
    return np.concatenate([gW1, dZ.sum(1) / n, np.einsum("mn,mnh->mh", r, Hd) / n, r.sum(1, keepdims=True) / n], axis=1)


def local_steps(T, X, y, steps, lr, sigma=None, clip=1.0, rng=None, tail=False):
    """`steps` full-batch (DP-)GD steps for M models in parallel (L2 is post-processing).
    tail=True returns the average of the iterates of the second half of training."""
    d = X.shape[1]
    n = len(y)
    T = T.copy()
    acc = np.zeros_like(T); start = steps - max(1, steps // 2)
    for t in range(steps):
        if sigma is None:
            G = _grads(T, X, y, d)
        else:
            per = _grads(T, X, y, d, per_sample=True)
            nr = np.maximum(np.linalg.norm(per, axis=2, keepdims=True), 1e-12)
            per = per * np.minimum(1.0, clip / nr)
            G = (per.sum(1) + rng.normal(0.0, sigma * clip, size=T.shape)) / n
        T = T - lr * (G + L2 * T)
        if tail and t >= start:
            acc += T
    return acc / (steps - start) if tail else T


# ------------------------------------------------------------------ coalition training (truth / retraining)
def coalition_utilities(clients, X_ref, y_ref, masks, seed, rounds=None, epochs=None, lr=None, sigma=None,
                        chunk=None):
    """FedAvg-MLP trained on every coalition (rows of masks) from the same initialisation; returns AUCs.
    With sigma, every coalition run draws independent DP noise."""
    rounds = rounds or TUNED["mlp_rounds"]; epochs = epochs or TUNED["mlp_epochs"]; lr = lr or TUNED["mlp_lr"]
    d = clients[0][0].shape[1]
    sizes = np.array([len(c[1]) for c in clients], float)
    rng = np.random.default_rng(seed)
    M = len(masks)
    T = np.tile(init(d, seed), (M, 1))
    A = masks * sizes; A = A / A.sum(1, keepdims=True)
    for _ in range(rounds):
        new = np.zeros_like(T)
        for k, (X, y) in enumerate(clients):
            idx = np.where(masks[:, k] > 0)[0]
            if len(idx) == 0:
                continue
            step = chunk or len(idx)
            for s in range(0, len(idx), step):
                ii = idx[s:s + step]
                new[ii] += A[ii, k][:, None] * local_steps(T[ii], X, y, epochs, lr, sigma, rng=rng)
        T = new
    return auc_columns(y_ref, logits(T, X_ref, d))


def exact_shapley_retrain(clients, X_ref, y_ref, seed, sigma=None, chunk=None):
    K = len(clients)
    masks, _ = all_masks(K)
    u = coalition_utilities(clients, X_ref, y_ref, masks, seed, sigma=sigma, chunk=chunk)
    return shapley_from_mask_utilities(u, K)


# ------------------------------------------------------------------ OSA-Shapley for the MLP
# chosen on tuning seeds 0-3 (mean Spearman at eps = 2 and 1; grid 30-400 steps x lr 0.5-2.0):
# ensemble 200 steps, lr 2.0 -> 0.69; parameter averaging 200 steps, lr 0.5 -> 0.67.
OSA_MLP_CFG = {"ensemble": (200, 2.0), "param": (200, 0.5)}
OSA_MLP_STEPS, OSA_MLP_LR = OSA_MLP_CFG["ensemble"]


def osa_releases(clients, epsilon, seed, steps=None, lr=None, delta=1e-5):
    """One DP-trained MLP per hospital, all from the same (public) initialisation."""
    steps = steps or OSA_MLP_STEPS; lr = lr or OSA_MLP_LR
    d = clients[0][0].shape[1]
    sigma = None if epsilon is None else fl.calibrate_sigma(epsilon, steps, delta)
    rng = np.random.default_rng(5000 + seed)
    T0 = init(d, seed)[None, :]
    return np.vstack([local_steps(T0, X, y, steps, lr, sigma, rng=rng, tail=True) for X, y in clients])


def osa_shapley_mlp(clients, X_ref, y_ref, epsilon, seed, mode="param", steps=None, lr=None, chunk=20000):
    if steps is None and lr is None:
        steps, lr = OSA_MLP_CFG[mode]
    K = len(clients)
    d = clients[0][0].shape[1]
    R = osa_releases(clients, epsilon, seed, steps, lr)
    sizes = np.array([len(c[1]) for c in clients], float)
    masks, _ = all_masks(K)
    u = np.empty(len(masks))
    if mode == "ensemble":
        Lk = logits(R, X_ref, d)                            # (n, K)
    for i in range(0, len(masks), chunk):
        A = masks[i:i + chunk] * sizes; A = A / A.sum(1, keepdims=True)
        S = (A @ R) if mode == "param" else None
        u[i:i + chunk] = auc_columns(y_ref, logits(S, X_ref, d) if mode == "param" else Lk @ A.T)
    return shapley_from_mask_utilities(u, K)


# ------------------------------------------------------------------ MR reconstruction from one DP FedAvg-MLP run
def mr_shapley_mlp(clients, X_ref, y_ref, epsilon, seed, delta=1e-5):
    rounds, epochs, lr = TUNED["mlp_rounds"], TUNED["mlp_epochs"], TUNED["mlp_lr"]
    K = len(clients)
    d = clients[0][0].shape[1]
    sigma = None if epsilon is None else fl.calibrate_sigma(epsilon, rounds * epochs, delta)
    rng = np.random.default_rng(3000 + seed)
    sizes = np.array([len(c[1]) for c in clients], float)
    a = sizes / sizes.sum()
    masks, _ = all_masks(K)
    A = masks * sizes; A = A / A.sum(1, keepdims=True)
    w = init(d, seed)
    phi = np.zeros(K)
    for _ in range(rounds):
        W = np.vstack([local_steps(w[None, :], X, y, epochs, lr, sigma, rng=rng) for X, y in clients])
        u = auc_columns(y_ref, logits(A @ W, X_ref, d))
        s = logits(w[None, :], X_ref, d)[:, 0]
        v0 = 0.5 if np.allclose(s, s[0]) else float(auc_columns(y_ref, s[:, None])[0])
        phi += shapley_from_mask_utilities(u, K, empty=v0)
        w = a @ W
    return phi


def retrain_sigmas(epsilon, K, delta=1e-5):
    """Per-run sigma and the budget-matched sigma for DP FedAvg-MLP retraining of all coalitions."""
    N = TUNED["mlp_rounds"] * TUNED["mlp_epochs"]
    return fl.calibrate_sigma(epsilon, N, delta), fl.calibrate_sigma(epsilon, N * 2 ** (K - 1), delta)
