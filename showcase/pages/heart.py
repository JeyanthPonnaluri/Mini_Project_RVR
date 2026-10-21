import numpy as np
import streamlit as st

from showcase import core, figs
from showcase.pages.methods import _series


def render():
    hv = core.R("pca19_heart_valuation")
    core.header("Second federation", "Heart disease: where OSA-Shapley loses",
                "We repeated the whole comparison on an independent federation, the four hospitals of the UCI Heart "
                "Disease collection, with OSA-Shapley's settings unchanged. Without privacy noise it works; under "
                "differential privacy it does not. We report this in full.")
    rho = hv["rho"]
    k = st.columns(4)
    with k[0]:
        core.kpi(f"{core.f2(core.m(rho['none']['osa']))} vs {core.f2(core.m(rho['none']['mr']))}",
                 "OSA vs MR without DP", "OSA − MR " + core.ci(hv["osa_minus"]["none"]["mr"]))
    with k[1]:
        core.kpi(f"{core.f2(core.m(rho['2.0']['osa']))} vs {core.f2(core.m(rho['2.0']['mr']))}", "OSA vs MR at ε = 2",
                 "OSA − MR " + core.ci(hv["osa_minus"]["2.0"]["mr"]))
    with k[2]:
        core.kpi(core.f2(core.m(rho["none"]["size"])), "hospital size vs true contribution",
                 "bigger hospitals contribute less here")
    with k[3]:
        core.kpi(f"{core.m(hv['corrupt']['2.0']['osa']):.0%}", "useless hospital ranked last by OSA (ε = 2)",
                 f"size: {core.m(hv['corrupt']['2.0']['size']):.0%}")

    st.write("")
    left, right = st.columns([1.6, 1])
    with left:
        ser = _series(rho, ["osa", "mr", "ref_auc", "retrain_per_run", "retrain_budget_matched", "size"], core.EPS_KEYS)
        core.show(figs.rho_lines([core.EPS_LABEL[e] for e in core.EPS_KEYS], ser,
                                 f"4 heart-disease hospitals ({hv['n']} patients, 20 held-out splits)",
                                 ylim=(-0.6, 1.02)))
    with right:
        core.finding("<b>Without DP the picture is the same as for prostate cancer:</b> OSA-Shapley recovers the exact "
                     "ranking best.", "good")
        core.finding(f"<b>Under DP it loses to reconstruction</b> (ε = 2: OSA − MR = {core.ci(hv['osa_minus']['2.0']['mr'])}), "
                     "and at ε = 1 no method recovers the ranking.", "bad")
        core.finding("<b>Why?</b> Each hospital trains about 200 patients against 21 parameters (ν ≈ 43): a single "
                     "private release carries too little signal. The pre-registered test later showed that ν alone "
                     "does not fully explain this case; it remains an open question.", "warn")
        st.caption("With four hospitals, Spearman ρ moves in steps of 0.2, so these estimates are coarse.")

    st.subheader("Who contributes? The true shares")
    sites = hv["sites"]
    truth = np.array(hv["truth_mean"], float); truth_share = 100 * truth / truth.sum()
    size_share = 100 * np.array([s["n"] for s in sites], float) / sum(s["n"] for s in sites)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        core.show(figs.shares([s["name"] for s in sites], [("truth", truth_share, "Exact Shapley share (no DP)"), ("size", size_share, "Share of patients")],
                              "Exact non-private Shapley share vs share of patients"))
    with c2:
        st.markdown(" ")
        core.finding("The dark bars are the <b>exact</b> non-private Shapley shares (mean over 20 splits), not "
                     "OSA's estimate. Contributions are nearly equal although hospitals differ more than two-fold in "
                     "size, which is why size ranks them the wrong way round.")
        st.caption(f"True ε of per-run retraining here: {hv['true_epsilon']['1.0']:.1f}, {hv['true_epsilon']['2.0']:.1f} "
                   f"and {hv['true_epsilon']['5.0']:.0f} at a stated 1, 2 and 5 (only 8 runs per hospital with K = 4). "
                   f"Federated model AUC {core.f2(core.m(hv['fed_auc']))}.")

    core.next_steps(("rule", "The decision rule, tested in advance", ":material/rule:"),
                    ("lab", "Run the heart federation in the lab", ":material/science:"))
