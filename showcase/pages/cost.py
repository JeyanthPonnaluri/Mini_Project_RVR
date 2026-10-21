import math

import numpy as np
import pandas as pd
import streamlit as st

from fl_study import fl
from showcase import core, figs

STEPS = 100      # DP-GD steps per private training run (Table II of the paper)
DELTA = 1e-5


@st.cache_data(show_spinner=False)
def true_eps(eps_run: float, runs: int, steps: int = STEPS) -> float:
    sig = fl.calibrate_sigma(eps_run, steps, DELTA)
    return float(fl.rdp_epsilon(sig, steps * runs, DELTA)[0])


def fmt_eps(x: float) -> str:
    if x >= 1e5:
        return f"≈ 10^{int(math.floor(math.log10(x)))}"
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def render():
    core.header("Contribution 1", "The hidden privacy cost of valuation",
                "Exact Shapley values need a model for every coalition of hospitals. The common way to get them under "
                "differential privacy retrains each coalition and reports the budget of a single run. But a hospital "
                "takes part in many runs, and privacy losses add up.")

    c = st.columns(3)
    with c[0]:
        core.card("Every coalition is a training run",
                  "With K hospitals there are 2^K − 1 coalitions. Each hospital belongs to 2^(K−1) of them: 512 runs "
                  "for 10 hospitals, 262,144 for 19.", "STEP 1")
    with c[1]:
        core.card("Privacy budgets compose",
                  "A patient whose record enters r private runs pays for all of them. The reported per-run ε says "
                  "nothing about the total (Proposition 2).", "STEP 2")
    with c[2]:
        core.card("Honesty destroys the ranking",
                  "Holding the true total at ε needs √r times more noise in every run: 23× for 10 hospitals. At that "
                  "noise level the ranking is lost.", "STEP 3")

    st.write("")
    st.subheader("Calculator: what does a valuation really cost?")
    st.caption(f"Each private run is {STEPS} full-batch DP gradient steps calibrated to the per-run ε (δ = 10⁻⁵); the "
               "composed ε uses the same Rényi-DP accountant as the paper.")
    with st.container(border=True):
        i1, i2, i3 = st.columns([1, 1, 1.3])
        with i1:
            K = st.slider("Hospitals in the federation (K)", 2, 50, 10)
        with i2:
            eps = st.select_slider("Reported per-run budget ε", options=[0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
        with i3:
            method = st.segmented_control("Valuation", ["Exact retraining", "Sampled permutations", "OSA-Shapley / MR"],
                                          default="Exact retraining", key="cost_method")
            method = method or "Exact retraining"
            M = 100
            if method == "Sampled permutations":
                M = st.slider("Permutations sampled (M)", 10, 1000, 100, step=10)
        if method == "Exact retraining":
            r = 2 ** (K - 1)
        elif method == "Sampled permutations":
            r = max(1, int(round(M * (K + 1) / 2)))
        else:
            r = 1
        te = true_eps(float(eps), r)
        o = st.columns(3)
        with o[0]:
            core.kpi(f"{r:,}", "private runs that contain each hospital's patients")
        with o[1]:
            core.kpi(fmt_eps(te), "true ε per hospital", f"reported: {eps:g}")
        with o[2]:
            core.kpi(f"{math.sqrt(r):,.1f}×", "noise needed in every run to keep the reported ε honest")
        if r == 1:
            core.finding("<b>Post-processing pays once.</b> OSA-Shapley and reconstruction from the training run reuse "
                         "releases whose cost is already counted, so all coalitions together cost the stated ε.", "good")
        else:
            core.finding(f"<b>The reported ε = {eps:g} is really ε ≈ {fmt_eps(te)}.</b> "
                         f"{'Sampling fewer coalitions helps only for large K, and never removes the cost.' if method == 'Sampled permutations' else 'This is the cost that retraining-based valuation leaves unreported.'}",
                         "bad")

    Ks = list(range(2, 51))
    exact = [true_eps(float(eps), 2 ** (k - 1)) for k in Ks]
    perm = [true_eps(float(eps), int(round(100 * (k + 1) / 2))) for k in Ks]
    core.show(figs.accounting(Ks, exact, perm, float(eps), f"True privacy cost per hospital at a reported per-run ε = {eps:g}"))

    st.subheader("What honest accounting does to the ranking")
    pv = core.R("pca19_private_valuation"); rows = pv["rows"]
    tbl = pd.DataFrame([{
        "Per-run ε": f"{r_['epsilon']:g}",
        "True ε of per-run retraining": round(r_["retrain_true_epsilon"]),
        "Retraining, per run (ρ)": core.m(r_["retrain_per_run"]["spearman"]),
        "Retraining, honest budget (ρ)": core.m(r_["retrain_budget_matched"]["spearman"]),
        "OSA-Shapley, same honest budget (ρ)": core.m(r_["osa"]["spearman"]),
    } for r_ in rows])
    st.dataframe(tbl, hide_index=True, use_container_width=True,
                 column_config={c_: st.column_config.ProgressColumn(c_, format="%.2f", min_value=-0.2, max_value=1.0)
                                for c_ in ("Retraining, per run (ρ)", "Retraining, honest budget (ρ)",
                                           "OSA-Shapley, same honest budget (ρ)")} |
                 {"True ε of per-run retraining": st.column_config.NumberColumn(format="%d")})
    r2 = {r_["epsilon"]: r_ for r_ in rows}[2.0]
    core.finding(f"<b>Ten largest prostate hospitals, 20 held-out splits.</b> Retraining looks accurate only because it "
                 f"spends far more privacy than it reports (ε ≈ {r2['retrain_true_epsilon']:.0f} at a stated 2). At the "
                 f"honest budget it ranks hospitals with ρ = {core.f2(core.m(r2['retrain_budget_matched']['spearman']))}; "
                 f"OSA-Shapley, at that same budget, reaches {core.f2(core.m(r2['osa']['spearman']))} "
                 f"(difference {core.ci(r2['osa_minus_budget_matched'])}).", "good")

    with st.expander("Table II of the paper: composed ε for typical federations"):
        acc = []
        for lab, runs in [("Exact, K = 4", 8), ("Exact, K = 10", 512), ("Exact, K = 19", 2 ** 18),
                          ("100 permutations, K = 10", 550), ("100 permutations, K = 19", 1000),
                          ("100 permutations, K = 50", 2550), ("OSA-Shapley, MR, OR (any K)", 1)]:
            acc.append({"Valuation": lab, "Runs per hospital": runs,
                        **{f"per-run ε = {e:g}": fmt_eps(true_eps(e, runs)) for e in (10.0, 5.0, 2.0, 1.0)},
                        "Noise factor √r": f"{np.sqrt(runs):,.0f}"})
        st.dataframe(pd.DataFrame(acc), hide_index=True, use_container_width=True)

    core.next_steps(("methods", "Valuation methods compared", ":material/leaderboard:"),
                    ("lab", "Try the valuation yourself", ":material/science:"))
