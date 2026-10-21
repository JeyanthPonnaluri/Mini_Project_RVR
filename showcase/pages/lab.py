import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr

from fl_study import charts as ch
from fl_study import fl
from fl_study import heart as H
from fl_study import multisite as M
from fl_study import private_valuation as PV
from showcase import core, figs

EPS_OPTS = ["none", "5.0", "2.0", "1.0", "0.5"]
SEEDS = list(range(20, 40))
HEART_NAMES = {"cleveland": "Cleveland", "hungarian": "Budapest", "switzerland": "Zurich/Basel", "va": "Long Beach VA"}


# ----------------------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner=False)
def pca_split(seed: int):
    df, names, tr, te, X, y = M.split(seed)
    return names, tr, te, X, y.astype(float)


@st.cache_data(show_spinner=False)
def heart_split(seed: int):
    df, tr, te, X, y = H.split(seed)
    return [k for k, _ in H.SITES], tr, te, X, y


@st.cache_data(show_spinner=False)
def exact_truth(fed: str, seed: int):
    """Exact non-private retraining Shapley values stored with the results (only for the default hospital sets)."""
    if fed == "PCa-19":
        fn = os.path.join(core.ROOT, "results", "cache_valuation_proxies", f"seed{seed}.json")
        if os.path.exists(fn):
            d = json.load(open(fn))["large10"]
            return d["hospitals"], d["truth"]
    else:
        fn = os.path.join(core.ROOT, "results", "cache_heart", f"seed{seed}.json")
        if os.path.exists(fn):
            return [k for k, _ in H.SITES], json.load(open(fn))["truth"]
    return None, None


@st.cache_data(show_spinner=False, max_entries=64)
def run_valuation(fed: str, hospitals: tuple, seed: int, eps_key: str, corrupt: str | None, with_mr: bool):
    names, tr, te, X, y = pca_split(seed) if fed == "PCa-19" else heart_split(seed)
    cl = [(X[tr[h]], y[tr[h]]) for h in hospitals]
    if corrupt is not None:
        i = hospitals.index(corrupt)
        ycor = np.random.default_rng(777 + seed).binomial(1, cl[i][1].mean(), size=len(cl[i][1])).astype(float)
        cl[i] = (cl[i][0], ycor)
    ref = np.concatenate([te[h] for h in hospitals]); Xr, yr = X[ref], y[ref]
    eps = None if eps_key == "none" else float(eps_key)
    phi, W = PV.osa_shapley(cl, Xr, yr, eps, rng=np.random.default_rng(5000 + seed), return_models=True)
    phi_np = PV.osa_shapley(cl, Xr, yr, None, rng=np.random.default_rng(5000 + seed)) if eps is not None else phi
    out = {"osa": phi.tolist(), "osa_nodp": phi_np.tolist(), "sizes": [int(len(c[1])) for c in cl]}
    if with_mr:
        trace, _ = PV.fl_trace(cl, M.cfg(epsilon=eps), 3000 + seed)
        out["mr"] = PV.mr_shapley(trace, out["sizes"], Xr, yr).tolist()
    sizes = np.array(out["sizes"], float); a = sizes / sizes.sum()
    out["fed_auc"] = float(fl.safe_auc(yr, fl.predict_proba(Xr, a @ W)))
    return out


@st.cache_data(show_spinner=False)
def retrain_true_eps(eps: float, K: int) -> float:
    return float(PV.retraining_privacy_cost(eps, K, M.cfg(epsilon=eps).steps))


def _rho(a, b):
    r = spearmanr(a, b).correlation
    return float("nan") if r is None else float(r)


# ----------------------------------------------------------------------------- tabs
def valuation_tab():
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.1, 1.6, 1.1])
        with c1:
            fed = st.segmented_control("Federation", ["PCa-19", "Heart disease"], default="PCa-19", key="lab_fed") \
                or "PCa-19"
            seed = st.select_slider("Data split", SEEDS, value=20, help="One of the 20 held-out splits used in the paper")
        names, tr, te, X, y = pca_split(seed) if fed == "PCa-19" else heart_split(seed)
        label = (lambda h: h) if fed == "PCa-19" else (lambda h: HEART_NAMES.get(h, h))
        with c2:
            default = names[:10] if fed == "PCa-19" else names
            hospitals = st.multiselect("Hospitals in the federation (2–12)", names, default=default, max_selections=12,
                                       format_func=label, key=f"lab_h_{fed}")
            eps_key = st.pills("Privacy budget", EPS_OPTS, default="2.0", format_func=lambda x: core.EPS_LABEL[x],
                               key="lab_eps") or "2.0"
        with c3:
            bad = st.toggle("Give one hospital random labels", value=False,
                            help="Tests whether a valuation spots a hospital whose data are useless")
            corrupt = None
            if bad and hospitals:
                corrupt = st.selectbox("Which hospital", hospitals, index=min(5, len(hospitals) - 1), format_func=label)
            with_mr = st.toggle("Also compute MR reconstruction", value=True, help="Adds a DP-FedAvg run (a few seconds)")
    if len(hospitals) < 2:
        st.info("Select at least two hospitals.")
        return

    order = tuple(h for h in names if h in hospitals)
    with st.spinner(f"Training {len(order)} private models and evaluating {2 ** len(order) - 1:,} coalitions…"):
        res = run_valuation(fed, order, seed, eps_key, corrupt, with_mr)
    K = len(order); eps = None if eps_key == "none" else float(eps_key)
    t_names, truth = exact_truth(fed, seed)
    has_truth = truth is not None and tuple(t_names) == order and corrupt is None

    k = st.columns(4)
    with k[0]:
        core.kpi(f"{2 ** K - 1:,}", "coalitions valued exactly", f"{K} hospitals")
    with k[1]:
        core.kpi("ε = ∞" if eps is None else f"ε = {eps:g}", "privacy spent by OSA-Shapley", "once, for every coalition")
    with k[2]:
        if eps is None:
            core.kpi("–", "retraining at the same per-run ε", "no DP selected")
        else:
            te_ = retrain_true_eps(eps, K)
            core.kpi(f"ε ≈ {te_:,.0f}" if te_ < 1e5 else "ε > 10⁵", "what retraining would really cost",
                     f"{2 ** (K - 1):,} runs per hospital")
    with k[3]:
        if has_truth:
            core.kpi(f"ρ = {_rho(res['osa'], truth):.2f}", "OSA vs exact non-private ranking",
                     f"MR: {_rho(res['mr'], truth):.2f}" if "mr" in res else "")
        else:
            core.kpi(f"ρ = {_rho(res['osa'], res['osa_nodp']):.2f}", "agreement with OSA without DP",
                     "exact reference stored only for the default set")

    osa = np.array(res["osa"]); share = lambda v: 100 * np.asarray(v, float) / np.sum(np.abs(v))
    cols_ = [("osa", share(osa), f"OSA-Shapley ({core.EPS_LABEL[eps_key]})")]
    if "mr" in res:
        cols_.append(("mr", share(res["mr"]), f"MR reconstruction ({core.EPS_LABEL[eps_key]})"))
    if has_truth:
        cols_.append(("truth", share(truth), "Exact Shapley, no DP (reference)"))
    cols_.append(("size", share(res["sizes"]), "Share of patients"))
    left, right = st.columns([1.5, 1])
    with left:
        core.show(figs.shares([label(h) + (" ✱" if h == corrupt else "") for h in order], cols_,
                              f"{fed}: each hospital's share of the federation's value"))
    with right:
        rank = pd.DataFrame({"Hospital": [label(h) for h in order], "Train patients": res["sizes"],
                             "OSA value": osa, "OSA rank": (-osa).argsort().argsort() + 1})
        if "mr" in res:
            rank["MR rank"] = (-np.array(res["mr"])).argsort().argsort() + 1
        st.dataframe(rank.sort_values("OSA rank"), hide_index=True, use_container_width=True,
                     column_config={"OSA value": st.column_config.NumberColumn(format="%.4f")})
        if corrupt is not None:
            i = order.index(corrupt)
            last_osa = int(np.argmin(osa)) == i
            msg = f"<b>{label(corrupt)} (random labels) is ranked {'last' if last_osa else 'not last'} by OSA-Shapley</b>"
            if "mr" in res:
                msg += f"; MR ranks it {'last' if int(np.argmin(res['mr'])) == i else 'not last'}"
            msg += f". Hospital size cannot notice: it still gives {label(corrupt)} {share(res['sizes'])[i]:.0f}%."
            core.finding(msg, "good" if last_osa else "warn")
        else:
            core.finding(f"Federation model AUC on the reference set: <b>{res['fed_auc']:.3f}</b>. Values are exact "
                         "Shapley values of the coalition AUCs: they sum to the federation's AUC minus 0.5.")
    st.caption("The defaults reproduce the paper's setting exactly (same split seeds and noise seeds). "
               "✱ = hospital given random training labels.")


@st.cache_data(show_spinner=False, max_entries=32)
def run_training(seed: int, algo: str, eps_key: str, rounds: int):
    names, tr, te, X, y = pca_split(seed)
    eps = None if eps_key == "none" else float(eps_key)
    cfg = M.cfg(epsilon=eps, rounds=rounds, mu=M.MU if algo == "FedProx" else 0.0)
    cl = [(X[tr[h]], y[tr[h]]) for h in names]
    ref = np.concatenate([te[h] for h in names])
    res = fl.federated_train(cl, cfg, seed, eval_set=(X[ref], y[ref]))
    rows = []
    for h in names:
        w_loc = fl.local_only_train(X[tr[h]], y[tr[h]], M.cfg(rounds=rounds), seed)
        rows.append({"Hospital": h, "Train patients": int(len(tr[h])),
                     "Local only": fl.safe_auc(y[te[h]], fl.predict_proba(X[te[h]], w_loc)),
                     "Federated": fl.safe_auc(y[te[h]], fl.predict_proba(X[te[h]], res["w"]))})
    return res["auc_history"], res["sigma"], rows


def training_tab():
    st.markdown("Train the federation of all 19 prostate-cancer hospitals right now, with or without differential "
                "privacy, and compare every hospital's federated model with what it could build alone.")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            algo = st.segmented_control("Algorithm", ["FedAvg", "FedProx"], default="FedAvg", key="tr_algo") or "FedAvg"
        with c2:
            eps_key = st.pills("Privacy budget", EPS_OPTS, default="none", format_func=lambda x: core.EPS_LABEL[x],
                               key="tr_eps") or "none"
        with c3:
            rounds = st.slider("Rounds", 5, 40, 20, step=5)
        with c4:
            seed = st.select_slider("Data split", SEEDS, value=20, key="tr_seed")
    with st.spinner("Training…"):
        hist, sigma, rows = run_training(seed, algo, eps_key, rounds)
    df = pd.DataFrame(rows); df["Gain from joining"] = df["Federated"] - df["Local only"]
    k = st.columns(4)
    with k[0]:
        core.kpi(f"{hist[-1]:.3f}", "federated AUC, all test patients")
    with k[1]:
        core.kpi(f"{df['Federated'].mean():.3f}", "mean per-hospital AUC", f"local only: {df['Local only'].mean():.3f}")
    with k[2]:
        core.kpi(f"{(df['Gain from joining'] > 0).sum()} of 19", "hospitals better off in the federation")
    with k[3]:
        core.kpi("none" if sigma is None else f"σ = {sigma:.1f}", "DP noise multiplier",
                 "" if sigma is None else f"ε = {float(eps_key):g}, δ = 10⁻⁵")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        core.show(ch.lines(list(range(1, len(hist) + 1)), [{"name": algo, "color": ch.C[algo], "mean": hist}],
                           "Federated model AUC per round", "round", "AUC"))
    with c2:
        st.dataframe(df.sort_values("Train patients", ascending=False), hide_index=True, use_container_width=True,
                     column_config={"Local only": st.column_config.NumberColumn(format="%.3f"),
                                    "Federated": st.column_config.NumberColumn(format="%.3f"),
                                    "Gain from joining": st.column_config.NumberColumn(format="%+.3f")})
    st.caption("Single split; the paper averages 20 splits with confidence intervals. AUC is undefined (empty) for "
               "a test set with one class only.")


@st.cache_resource(show_spinner=False)
def calculator_model():
    df = M.load_pca19()
    names, tr, te, X, y = pca_split(20)
    idx = np.concatenate([tr[h] for h in names])
    raw = M.core_raw(df)
    stdz = M.Standardiser().fit(raw.iloc[idx])
    cl = [(X[tr[h]], y[tr[h]]) for h in names]
    w = fl.federated_train(cl, M.cfg(), 20)["w"]
    return stdz, w


def calculator_tab():
    st.warning("Research demonstration only. This model is not validated for clinical use and must not guide any "
               "decision about a patient.", icon="⚠️")
    st.markdown("The federated risk model trained on the 19 hospitals, as a calculator: the probability of "
                "clinically significant prostate cancer before biopsy.")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        with st.container(border=True):
            age = st.slider("Age (years)", 40, 85, 65)
            psa = st.number_input("PSA (ng/mL)", 0.1, 200.0, 7.5, step=0.5)
            pv = st.number_input("Prostate volume (mL)", 5.0, 250.0, 45.0, step=5.0)
            pirads = st.segmented_control("MRI PI-RADS score", [1, 2, 3, 4, 5], default=4, key="calc_pirads") or 4
    stdz, w = calculator_model()
    one = pd.DataFrame({"age": [age], "PSA": [psa], "PV": [pv], "PIRADS": [pirads]})
    x = stdz.transform(M.core_raw(one))
    p = float(fl.predict_proba(x, w)[0])
    with c2:
        core.kpi(f"{p:.0%}", "estimated risk of clinically significant cancer",
                 f"PSA density {psa / pv:.2f} ng/mL²")
        st.progress(min(max(p, 0.0), 1.0))
        core.finding("Remember what the paper found: the model ranks patients well in new countries, but its absolute "
                     "risks are not calibrated across countries. A hospital should re-estimate the intercept on its "
                     "own patients before quoting risks.", "warn")
        coef = pd.DataFrame({"Feature": M.CORE_FEATURES + ["intercept"], "Weight (standardised)": w})
        with st.expander("Model weights"):
            st.dataframe(coef, hide_index=True, use_container_width=True,
                         column_config={"Weight (standardised)": st.column_config.NumberColumn(format="%+.3f")})


def render():
    core.header("Try it", "Valuation lab",
                "Run the methods of the paper yourself, on the real hospitals, in seconds. Every run uses the same "
                "code that produced the published results.")
    t1, t2, t3 = st.tabs(["Value the hospitals", "Train a federation", "Risk calculator (demo)"])
    with t1:
        valuation_tab()
    with t2:
        training_tab()
    with t3:
        calculator_tab()
    core.next_steps(("rule", "Which method should a federation use?", ":material/rule:"),
                    ("claims", "Every claim and its verdict", ":material/fact_check:"))
