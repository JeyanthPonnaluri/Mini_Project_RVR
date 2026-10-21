"""
Stronger baselines for the PCa-19 and TCGA evaluations.

Federated : FedAvg, FedProx (mu = 0.5), SCAFFOLD, FedAdam (all logistic regression), and a
            federated one-hidden-layer MLP trained with FedAvg (non-linear model).
Central   : logistic regression, histogram gradient boosting (LightGBM-style), random forest, MLP.
FedNova (Wang et al. 2020) normalises by the number of local steps; every hospital here runs the
same number of local steps, so FedNova reduces exactly to FedAvg and is not run separately.
Hyper-parameters of the new baselines were chosen on tuning seeds 0-3 (see TUNED) and are
evaluated on seeds 20-39.
"""
from __future__ import annotations

import numpy as np

from . import fl

TUNED = {"fedadam_server_lr": 0.1, "mlp_hidden": 16, "mlp_rounds": 40, "mlp_epochs": 10, "mlp_lr": 1.0, "mlp_l2": 1e-3}
# tuning (mean per-hospital AUC, seeds 0-3): FedAdam server lr 0.03 -> 0.813, 0.1 -> 0.825, 0.3 -> 0.824;
# federated MLP lr 0.3 -> 0.829, 1.0 -> 0.836


# ---------------------------------------------------------------- federated MLP (numpy)
def _mlp_init(d, h, rng):
    return {"W1": rng.normal(0, 1 / np.sqrt(d), (d, h)), "b1": np.zeros(h),
            "w2": rng.normal(0, 1 / np.sqrt(h), h), "b2": 0.0}


def mlp_predict(P, X):
    H = np.tanh(X @ P["W1"] + P["b1"])
    return fl.sigmoid(H @ P["w2"] + P["b2"])


def _mlp_local(P, X, y, epochs, lr, l2):
    P = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    n = len(y)
    for _ in range(epochs):
        Z = X @ P["W1"] + P["b1"]; H = np.tanh(Z)
        p = fl.sigmoid(H @ P["w2"] + P["b2"])
        r = (p - y) / n
        gw2 = H.T @ r + l2 * P["w2"]; gb2 = r.sum()
        dH = np.outer(r, P["w2"]) * (1 - H ** 2)
        gW1 = X.T @ dH + l2 * P["W1"]; gb1 = dH.sum(0)
        P["W1"] -= lr * gW1; P["b1"] -= lr * gb1; P["w2"] -= lr * gw2; P["b2"] -= lr * gb2
    return P


def federated_mlp(clients, seed=0, hidden=None, rounds=None, epochs=None, lr=None, l2=None):
    t = TUNED
    hidden = hidden or t["mlp_hidden"]; rounds = rounds or t["mlp_rounds"]
    epochs = epochs or t["mlp_epochs"]; lr = lr or t["mlp_lr"]; l2 = t["mlp_l2"] if l2 is None else l2
    rng = np.random.default_rng(seed)
    P = _mlp_init(clients[0][0].shape[1], hidden, rng)
    sizes = np.array([len(c[1]) for c in clients], float); a = sizes / sizes.sum()
    for _ in range(rounds):
        locs = [_mlp_local(P, X, y, epochs, lr, l2) for X, y in clients]
        P = {k: sum(ak * L[k] for ak, L in zip(a, locs)) for k in P}
    return P


# ---------------------------------------------------------------- central references
def central_models(Xtr, ytr, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    return {
        "Central LR": LogisticRegression(max_iter=2000).fit(Xtr, ytr),
        "Central gradient boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                                    random_state=seed).fit(Xtr, ytr),
        "Central random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed,
                                                        n_jobs=1).fit(Xtr, ytr),
        "Central MLP": MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, max_iter=2000, random_state=seed).fit(Xtr, ytr),
    }


def federated_models(clients, cfg_, seed=0, sigma=None):
    """All federated LR variants + the federated MLP (no DP for the MLP). Returns name -> predict(X)."""
    import copy
    out = {}
    w = fl.federated_train(clients, cfg_, seed, sigma=sigma)["w"]; out["FedAvg"] = lambda X, w=w: fl.predict_proba(X, w)
    if sigma is None and cfg_.epsilon is None:
        c4 = copy.copy(cfg_); c4.rounds, c4.epochs = 40, 10      # converged reference (400 local steps)
        w = fl.federated_train(clients, c4, seed)["w"]; out["FedAvg (400 steps)"] = lambda X, w=w: fl.predict_proba(X, w)
    c2 = copy.copy(cfg_); c2.mu = 0.5
    w = fl.federated_train(clients, c2, seed, sigma=sigma)["w"]; out["FedProx"] = lambda X, w=w: fl.predict_proba(X, w)
    w = fl.federated_train_scaffold(clients, cfg_, seed, sigma=sigma)["w"]; out["SCAFFOLD"] = lambda X, w=w: fl.predict_proba(X, w)
    w = fl.federated_train_fedadam(clients, cfg_, seed, sigma=sigma, server_lr=TUNED["fedadam_server_lr"])["w"]
    out["FedAdam"] = lambda X, w=w: fl.predict_proba(X, w)
    if sigma is None and cfg_.epsilon is None:
        P = federated_mlp(clients, seed); out["Federated MLP"] = lambda X, P=P: mlp_predict(P, X)
    return out
