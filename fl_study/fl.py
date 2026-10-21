"""
NumPy federated logistic regression: FedAvg, FedProx, DP-SGD (full batch) with
an RDP accountant, local-only, centralized and personalised (fine-tuned) models.

Fixes relative to the original src/federated.py / src/fedprox_experiments.py
---------------------------------------------------------------------------
* The model has an intercept (the original had none).
* Client drift is measured the way the paper defines it,
      drift_t = mean_k || w_k^{t} - w^{t-1} ||_2
  (local model after local training minus the global model it started from),
  and it is recorded for BOTH FedAvg and FedProx. The original recorded 0.0 for
  FedAvg and, for FedProx, the change of the *global* model between rounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict_proba(X, w):
    return sigmoid(add_bias(X) @ w)


def log_loss(X, y, w):
    p = np.clip(predict_proba(X, w), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def safe_auc(y, p):
    """AUC, or NaN when y contains a single class (AUC undefined)."""
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ----------------------------------------------------------------------------
# Rényi-DP accountant for the (full-batch, q = 1) Gaussian mechanism
# ----------------------------------------------------------------------------
RDP_ORDERS = np.concatenate([np.linspace(1.1, 10.9, 99), np.arange(11, 64), np.array([64, 80, 96, 128, 160, 192, 256, 384, 512, 1024, 2048, 4096])])


def rdp_epsilon(sigma: float, steps: int, delta: float = 1e-5, orders=RDP_ORDERS):
    """(epsilon, best order) for `steps` compositions of a Gaussian mechanism with
    noise multiplier `sigma` (L2 sensitivity 1), using RDP(alpha) = alpha / (2 sigma^2)
    and the standard conversion eps = RDP_total + log(1/delta)/(alpha - 1)  (Mironov 2017)."""
    if sigma <= 0:
        return float("inf"), None
    orders = np.asarray(orders, float)
    eps = steps * orders / (2 * sigma ** 2) + math.log(1 / delta) / (orders - 1)
    i = int(np.argmin(eps))
    return float(eps[i]), float(orders[i])


def calibrate_sigma(target_eps: float, steps: int, delta: float = 1e-5):
    """Smallest noise multiplier whose composed epsilon is <= target (bisection)."""
    lo, hi = 1e-3, 1e5
    for _ in range(200):
        mid = (lo + hi) / 2
        if rdp_epsilon(mid, steps, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------
# Rényi-DP accountant for the POISSON-SUBSAMPLED Gaussian mechanism (q < 1)
# ----------------------------------------------------------------------------
RDP_INT_ORDERS = np.concatenate([np.arange(2, 257), np.array([320, 384, 512, 768, 1024])])


def rdp_sampled_gaussian(q: float, sigma: float, orders=RDP_INT_ORDERS):
    """RDP of one step of the sampled Gaussian mechanism (each record included independently with
    probability q, L2 sensitivity 1, noise multiplier sigma), at integer orders alpha:
        RDP(alpha) = 1/(alpha-1) * log sum_{k=0}^{alpha} C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2-k)/(2 sigma^2))
    (Mironov, Talwar & Zhang 2019, integer-order closed form; Zhu & Wang 2019). q = 1 gives alpha/(2 sigma^2)."""
    orders = np.asarray(orders)
    if q >= 1.0:
        return orders / (2 * sigma ** 2)
    out = np.empty(len(orders))
    lq, l1q = math.log(q), math.log1p(-q)
    for i, a in enumerate(orders):
        a = int(a)
        k = np.arange(a + 1)
        logc = np.array([math.lgamma(a + 1) - math.lgamma(j + 1) - math.lgamma(a - j + 1) for j in k])
        terms = logc + (a - k) * l1q + k * lq + (k * k - k) / (2 * sigma ** 2)
        m = terms.max()
        out[i] = (m + math.log(np.exp(terms - m).sum())) / (a - 1)
    return out


def rdp_epsilon_sampled(q: float, sigma: float, steps: int, delta: float = 1e-5, orders=RDP_INT_ORDERS):
    """(epsilon, best order) after `steps` compositions of the sampled Gaussian mechanism."""
    if sigma <= 0:
        return float("inf"), None
    orders = np.asarray(orders, float)
    eps = steps * rdp_sampled_gaussian(q, sigma, orders) + math.log(1 / delta) / (orders - 1)
    i = int(np.argmin(eps))
    return float(eps[i]), float(orders[i])


def calibrate_sigma_sampled(target_eps: float, q: float, steps: int, delta: float = 1e-5):
    lo, hi = 1e-2, 1e4
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if rdp_epsilon_sampled(q, mid, steps, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------
# Local training
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    rounds: int = 20
    epochs: int = 5
    lr: float = 0.5
    l2: float = 1e-3
    mu: float = 0.0               # FedProx proximal coefficient (0 = FedAvg)
    epsilon: float | None = None  # None = no DP
    delta: float = 1e-5
    clip: float = 1.0
    q: float = 1.0                # Poisson sampling rate per local step (1 = full batch)

    @property
    def steps_per_epoch(self):
        return 1 if self.q >= 1.0 else int(round(1.0 / self.q))

    @property
    def steps(self):
        return self.rounds * self.epochs * self.steps_per_epoch

    def sigma(self):
        if self.epsilon is None:
            return None
        if self.q >= 1.0:
            return calibrate_sigma(self.epsilon, self.steps, self.delta)
        return calibrate_sigma_sampled(self.epsilon, self.q, self.steps, self.delta)


def local_update(X, y, w_start, w_anchor, epochs, lr, l2=0.0, mu=0.0, sigma=None, clip=1.0, rng=None, q=1.0):
    """Full-batch gradient descent on BCE (+ L2) (+ FedProx term).
    With sigma: per-sample gradient clipping to `clip` and Gaussian noise N(0, (sigma*clip)^2)
    added to the summed gradient (DP-SGD with sampling rate q = 1). The L2 and proximal terms
    do not touch data, so they do not affect the privacy guarantee."""
    Xb = add_bias(X)
    n = len(y)
    w = w_start.copy()
    if q < 1.0:
        return _local_update_sampled(Xb, y, w, w_anchor, epochs, lr, l2, mu, sigma, clip, rng, q)
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        if sigma is None:
            g = Xb.T @ (p - y) / n
        else:
            per = Xb * (p - y)[:, None]
            norms = np.maximum(np.linalg.norm(per, axis=1, keepdims=True), 1e-12)
            per = per * np.minimum(1.0, clip / norms)
            g = (per.sum(0) + rng.normal(0.0, sigma * clip, size=w.shape)) / n
        g = g + l2 * w
        if mu > 0:
            g = g + mu * (w - w_anchor)
        w = w - lr * g
    return w


def _local_update_sampled(Xb, y, w, w_anchor, epochs, lr, l2, mu, sigma, clip, rng, q):
    """Poisson-subsampled (DP-)SGD: every record joins each step independently with probability q;
    the clipped per-record gradients of the sampled records are summed, Gaussian noise
    N(0, (sigma*clip)^2) is added and the sum is divided by the EXPECTED batch size q*n (a fixed
    denominator, as required by the sampled-Gaussian analysis). 1/q steps make one epoch."""
    n = len(y)
    for _ in range(epochs * int(round(1.0 / q))):
        mask = rng.random(n) < q
        Xs, ys = Xb[mask], y[mask]
        if len(ys):
            per = Xs * (sigmoid(Xs @ w) - ys)[:, None]
            if sigma is not None:
                norms = np.maximum(np.linalg.norm(per, axis=1, keepdims=True), 1e-12)
                per = per * np.minimum(1.0, clip / norms)
            gsum = per.sum(0)
        else:
            gsum = np.zeros_like(w)
        if sigma is not None:
            gsum = gsum + rng.normal(0.0, sigma * clip, size=w.shape)
        g = gsum / (q * n) + l2 * w
        if mu > 0:
            g = g + mu * (w - w_anchor)
        w = w - lr * g
    return w


def federated_train(clients, cfg: TrainConfig, seed=0, eval_set=None, sigma=None):
    """clients: list of (X_k, y_k). Returns dict with final weights, per-round test AUC
    (if eval_set given) and per-round client drift  mean_k ||w_k - w_global_prev||."""
    rng = np.random.default_rng(seed)
    d = clients[0][0].shape[1] + 1
    w = np.zeros(d)
    if sigma is None and cfg.epsilon is not None:
        sigma = cfg.sigma()
    sizes = np.array([len(c[1]) for c in clients], float)
    weights = sizes / sizes.sum()
    hist_auc, hist_drift = [], []
    for _ in range(cfg.rounds):
        locals_ = [local_update(X, y, w, w, cfg.epochs, cfg.lr, cfg.l2, cfg.mu, sigma, cfg.clip, rng, cfg.q)
                   for X, y in clients]
        hist_drift.append(float(np.mean([np.linalg.norm(wk - w) for wk in locals_])))
        w = np.sum([a * wk for a, wk in zip(weights, locals_)], axis=0)
        if eval_set is not None:
            hist_auc.append(safe_auc(eval_set[1], predict_proba(eval_set[0], w)))
    return {"w": w, "auc_history": hist_auc, "drift_history": hist_drift, "sigma": sigma}


def centralized_train(X, y, cfg: TrainConfig, seed=0):
    """Pooled-data model trained with the same optimiser/steps (upper bound)."""
    return federated_train([(X, y)], TrainConfig(**{**cfg.__dict__, "mu": 0.0}), seed)["w"]


def local_only_train(X, y, cfg: TrainConfig, seed=0):
    return federated_train([(X, y)], TrainConfig(**{**cfg.__dict__, "mu": 0.0}), seed)["w"]


def personalise(X, y, w_global, epochs=10, lr=0.1, l2=1e-3, mu=0.1):
    """PFL: fine-tune the global model locally, anchored to it with a proximal term."""
    if len(y) == 0:
        return w_global
    return local_update(X, y, w_global, w_global, epochs, lr, l2, mu)


# ----------------------------------------------------------------------------
# Partitions
# ----------------------------------------------------------------------------
def dirichlet_proportions(n_clients, n_classes, alpha, rng):
    return rng.dirichlet([alpha] * n_clients, size=n_classes)  # (classes, clients)


def split_by_proportions(y, props, rng, min_per_client=2):
    """Assign indices of each class to clients according to props[class].
    Every client receives at least `min_per_client` samples overall (if possible)."""
    n_clients = props.shape[1]
    assign = np.empty(len(y), int)
    for ci, c in enumerate(np.unique(y)):
        idx = rng.permutation(np.where(y == c)[0])
        counts = np.floor(props[ci] * len(idx)).astype(int)
        counts[np.argmax(props[ci])] += len(idx) - counts.sum()
        start = 0
        for k in range(n_clients):
            assign[idx[start:start + counts[k]]] = k
            start += counts[k]
    # guarantee non-empty clients: move samples from the largest client
    for k in range(n_clients):
        while (assign == k).sum() < min_per_client:
            big = np.bincount(assign, minlength=n_clients).argmax()
            assign[rng.choice(np.where(assign == big)[0])] = k
    return assign


def dirichlet_partition(y_train, y_test, n_clients, alpha, seed):
    """Same per-class client proportions applied to train and test, so each client
    has a local test split with the same label skew as its training data."""
    rng = np.random.default_rng(seed)
    props = dirichlet_proportions(n_clients, 2, alpha, rng)
    return split_by_proportions(y_train, props, rng), split_by_proportions(y_test, props, rng, 0)


def mia_confidence_attack(w, X_mem, y_mem, X_non, y_non, seed=0):
    """Confidence-threshold membership inference (Yeom et al. 2018 style).
    Score = model confidence on the true label. Members/non-members are balanced by
    subsampling the larger group. Returns attacker AUC and best-threshold advantage."""
    rng = np.random.default_rng(seed)
    m = min(len(y_mem), len(y_non))
    im = rng.choice(len(y_mem), m, replace=False)
    inn = rng.choice(len(y_non), m, replace=False)
    pm = predict_proba(X_mem[im], w); pn = predict_proba(X_non[inn], w)
    sm = np.where(y_mem[im] == 1, pm, 1 - pm); sn = np.where(y_non[inn] == 1, pn, 1 - pn)
    labels = np.r_[np.ones(m), np.zeros(m)]
    scores = np.r_[sm, sn]
    auc = roc_auc_score(labels, scores)
    thr = np.unique(scores)
    adv = max(np.mean(sm >= t) - np.mean(sn >= t) for t in thr)
    return float(auc), float(adv)


# ----------------------------------------------------------------------------
# Stronger federated baselines
# ----------------------------------------------------------------------------
def federated_train_scaffold(clients, cfg: TrainConfig, seed=0, sigma=None):
    """SCAFFOLD (Karimireddy et al. 2020), full participation, option-II control-variate update
    c_k+ = c_k - c + (w_global - w_k) / (E * lr). With DP the local steps are DP-SGD steps; the
    control variates are computed from released quantities only (post-processing), so they add no
    privacy cost. The FedProx term is not used (mu ignored)."""
    rng = np.random.default_rng(seed)
    d = clients[0][0].shape[1] + 1
    w = np.zeros(d); c = np.zeros(d); ck = [np.zeros(d) for _ in clients]
    if sigma is None and cfg.epsilon is not None:
        sigma = cfg.sigma()
    sizes = np.array([len(cl[1]) for cl in clients], float); a = sizes / sizes.sum()
    drift = []
    for _ in range(cfg.rounds):
        new_w, new_c = [], []
        for k, (X, y) in enumerate(clients):
            Xb = add_bias(X); n = len(y); wk = w.copy()
            for _e in range(cfg.epochs):
                p = sigmoid(Xb @ wk)
                if sigma is None:
                    g = Xb.T @ (p - y) / n
                else:
                    per = Xb * (p - y)[:, None]
                    nr = np.maximum(np.linalg.norm(per, axis=1, keepdims=True), 1e-12)
                    per = per * np.minimum(1.0, cfg.clip / nr)
                    g = (per.sum(0) + rng.normal(0.0, sigma * cfg.clip, size=w.shape)) / n
                wk = wk - cfg.lr * (g + cfg.l2 * wk - ck[k] + c)
            new_w.append(wk)
            new_c.append(ck[k] - c + (w - wk) / (cfg.epochs * cfg.lr))
        drift.append(float(np.mean([np.linalg.norm(wk - w) for wk in new_w])))
        dc = np.sum([ak * (nc - oc) for ak, nc, oc in zip(a, new_c, ck)], axis=0)
        ck = new_c
        w = np.sum([ak * wk for ak, wk in zip(a, new_w)], axis=0)
        c = c + dc
    return {"w": w, "drift_history": drift, "sigma": sigma}


def federated_train_fedadam(clients, cfg: TrainConfig, seed=0, sigma=None, server_lr=0.1,
                            beta1=0.9, beta2=0.99, tau=1e-3):
    """FedAdam (Reddi et al. 2021): clients run the usual local (DP-)GD; the server treats the
    weighted mean update as a pseudo-gradient and applies Adam. Server-side post-processing, so
    the DP guarantee of the client updates is unchanged."""
    rng = np.random.default_rng(seed)
    d = clients[0][0].shape[1] + 1
    w = np.zeros(d); m = np.zeros(d); v = np.zeros(d)
    if sigma is None and cfg.epsilon is not None:
        sigma = cfg.sigma()
    sizes = np.array([len(cl[1]) for cl in clients], float); a = sizes / sizes.sum()
    drift = []
    for _ in range(cfg.rounds):
        locs = [local_update(X, y, w, w, cfg.epochs, cfg.lr, cfg.l2, cfg.mu, sigma, cfg.clip, rng, cfg.q)
                for X, y in clients]
        drift.append(float(np.mean([np.linalg.norm(wk - w) for wk in locs])))
        delta = np.sum([ak * (wk - w) for ak, wk in zip(a, locs)], axis=0)
        m = beta1 * m + (1 - beta1) * delta
        v = beta2 * v + (1 - beta2) * delta ** 2
        w = w + server_lr * m / (np.sqrt(v) + tau)
    return {"w": w, "drift_history": drift, "sigma": sigma}
