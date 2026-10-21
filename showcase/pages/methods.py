import pandas as pd
import streamlit as st

from showcase import core, figs

METHODS = [
    ("osa", "OSA-Shapley (ours)", "Each hospital releases one DP model; coalition models are weighted averages of "
     "the releases; exact Shapley over all coalitions.", "Costs ε once"),
    ("mr", "MR reconstruction", "Rebuilds every coalition from the local updates of one DP-FedAvg run "
     "(the exact version of GTG-Shapley).", "Costs ε once"),
    ("ref_auc", "Single-model AUC", "Score each hospital's own released DP model on the server's reference set.",
     "Costs ε once"),
    ("retrain_budget_matched", "Retraining, honest budget", "Retrain every coalition with DP, with noise raised so "
     "the true total stays at ε.", "Costs ε, but √r× noise"),
    ("size", "Hospital size", "Credit proportional to the number of patients; nothing is trained.", "Free"),
    ("statistics", "Summary statistics", "Credit by how similar a hospital's feature summaries are to the federation's.",
     "Free"),
]


def _series(rho, keys, eps_keys, band_key="osa", extra=None):
    out = []
    for k in keys:
        vals = [rho[e].get(k) if e in rho else None for e in eps_keys]
        if extra and k in extra:
            vals = [extra[k].get(e, v) for e, v in zip(eps_keys, vals)]
        mean = [None if v is None else v["mean"] for v in vals]
        if all(x is None for x in mean):
            continue
        out.append({"key": k, "mean": mean, "band": k == band_key, "dash": "--" if k in ("size", "statistics") else "-",
                    "low": [None if v is None else v["ci_low"] for v in vals],
                    "high": [None if v is None else v["ci_high"] for v in vals]})
    return out


def render():
    core.header("Contribution 2", "Valuation methods compared",
                "Once retraining is ruled out, a federation can only value hospitals by post-processing releases whose "
                "cost is already counted, or by cheap proxies that cost nothing. Which of them recovers the true "
                "ranking? The reference is the exact non-private Shapley ranking, obtained by retraining every coalition "
                "without noise.")
    core.qa("Which valuation should a federation use if it pays the privacy budget only once?",
            "For the 19 prostate-cancer hospitals, OSA-Shapley: ρ = 0.89 with the exact ranking at ε = 2, against 0.75 for reconstruction, 0.18 for honest retraining and 0.08 for hospital size.")

    with st.expander("The six contenders", expanded=False):
        cols = st.columns(3)
        for i, (k, name, desc, cost) in enumerate(METHODS):
            with cols[i % 3]:
                st.markdown(f"<div class='sc-card'><h4><span style='color:{core.COL[k]}'>●</span> {name}</h4>"
                            f"<p>{desc}</p><p style='margin-top:.4rem'>{core.badge(cost, 'ok' if 'once' in cost or cost == 'Free' else 'no')}"
                            "</p></div>", unsafe_allow_html=True)
                st.write("")

    pv = core.R("pca19_private_valuation"); vq = core.R("pca19_valuation_proxies"); vm = core.R("pca19_valuation_mlp")
    vp = core.R("pca19_valuation_plus")
    group = st.segmented_control("Setting", ["10 largest hospitals", "10 smallest hospitals", "Neural network (MLP)"],
                                 default="10 largest hospitals", key="methods_group") or "10 largest hospitals"

    if group == "10 largest hospitals":
        rho = vq["large10"]["rho"]
        bm = {"retrain_budget_matched": {str(r["epsilon"]): r["retrain_budget_matched"]["spearman"] for r in pv["rows"]}}
        keys = ["osa", "mr", "ref_auc", "retrain_budget_matched", "size", "statistics"]
        title = "10 largest PCa-19 hospitals (266–763 patients, ν = 114)"
    elif group == "10 smallest hospitals":
        rho = vq["small10"]["rho"]; bm = None
        keys = ["osa", "mr", "ref_auc", "retrain_budget_matched", "size", "statistics"]
        title = "10 smallest PCa-19 hospitals (16–266 patients, ν = 45)"
    else:
        rho = {e: {"osa": vm["rho"][e]["osa_ensemble"], "mr": vm["rho"][e]["mr"]} for e in vm["rho"]}
        bm = {"retrain_budget_matched": {e: vm["budget_matched"][e]["retrain_budget_matched"] for e in vm["budget_matched"]}}
        keys = ["osa", "mr", "retrain_budget_matched"]
        title = f"Federated MLP ({vm['hidden']} hidden units) on the 10 largest hospitals (ν = 26)"

    ek = core.EPS_KEYS + (["0.5"] if group == "Neural network (MLP)" else [])
    ser = _series(rho, keys, ek, extra=bm)
    left, right = st.columns([1.6, 1])
    with left:
        core.show(figs.rho_lines([core.EPS_LABEL[e] for e in ek], ser, title))
    with right:
        if group == "10 largest hospitals":
            core.finding(f"<b>OSA-Shapley leads at every budget</b> (ρ {core.f2(core.m(rho['2.0']['osa']))} at ε = 2; "
                         f"OSA − MR {core.ci(vq['large10']['osa_minus']['2.0']['mr'])}).", "good")
            core.finding(f"<b>The single-model score is close</b> once releases exist (OSA − single-model "
                         f"{core.ci(vq['large10']['osa_minus']['2.0']['ref_auc'])} at ε = 2), and OSA pulls ahead under "
                         f"stronger privacy ({core.ci(vq['large10']['osa_minus']['1.0']['ref_auc'])} at ε = 1).")
            core.finding(f"<b>Size and statistics are unreliable:</b> ρ {core.f2(core.m(rho['2.0']['size']))} and "
                         f"{core.f2(core.m(rho['2.0']['statistics']))}. What a hospital adds depends on its case mix "
                         "relative to the others, not on its size.", "warn")
        elif group == "10 smallest hospitals":
            core.finding(f"<b>Small hospitals: size is enough.</b> When contribution is driven mainly by data volume, "
                         f"size reaches ρ {core.f2(core.m(rho['2.0']['size']))} and ties with OSA-Shapley "
                         f"({core.f2(core.m(rho['2.0']['osa']))}) and MR ({core.f2(core.m(rho['2.0']['mr']))}) at ε = 2.", "warn")
            core.finding(f"Without DP, OSA-Shapley is still the most accurate (ρ {core.f2(core.m(rho['none']['osa']))}).")
        else:
            core.finding(f"<b>Non-convex model: reconstruction wins under DP.</b> At ε = 2, OSA − MR = "
                         f"{core.ci(vm['osa_ensemble_minus_mr']['2.0'])}. A neural network trained alone for many noisy "
                         "steps accumulates the noise; MR's short updates restart from the shared model every round.", "bad")
            core.finding(f"Without DP, OSA-Shapley is better (ρ {core.f2(core.m(vm['rho']['none']['osa_ensemble']))} vs "
                         f"{core.f2(core.m(vm['rho']['none']['mr']))}).")

    st.subheader("All scores at one privacy level")
    e = st.pills("Privacy budget", core.EPS_KEYS, default="2.0", format_func=lambda x: core.EPS_LABEL[x],
                 key="methods_eps") or "2.0"
    rows = []
    for k in keys + ["own_test_auc", "osa_loo", "positives"]:
        v = rho.get(e, {}).get(k)
        if bm and k in bm:
            v = bm[k].get(e)
        if v is not None:
            rows.append((k, v))
    rows.sort(key=lambda kv: -kv[1]["mean"])
    core.show(figs.hbars([core.NAME[k] for k, _ in rows], [v["mean"] for _, v in rows], [core.COL[k] for k, _ in rows],
                         f"Agreement with the exact ranking at {core.EPS_LABEL[e]} (95% CI)",
                         "Spearman ρ", lows=[v["ci_low"] for _, v in rows], highs=[v["ci_high"] for _, v in rows],
                         xlim=(-0.6, 1.15), ref=0))

    if group == "10 largest hospitals":
        st.subheader("Can it spot a hospital with useless data?")
        st.markdown(f"In every split we replaced the labels of one large hospital (**{vp['corrupt_hospital']}**) with "
                    "random ones. A good valuation should rank it last.")
        cr = vq["large10"]["corrupt"]; cp = vp["corrupt"]
        data = []
        for k in ["osa", "ref_auc", "mr", "retrain_budget_matched", "osa_loo", "own_test_auc", "size", "statistics"]:
            row = {"Method": core.NAME[k]}
            for ee in ("none", "5.0", "2.0", "1.0"):
                src = cr.get(ee, {}).get(k) or cp.get(ee, {}).get(k)
                val = None if src is None else (src["last"]["mean"] if "last" in src else src["mean"])
                row[core.EPS_LABEL[ee]] = val
            data.append(row)
        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.ProgressColumn(c, format="%.2f", min_value=0.0, max_value=1.0)
                                    for c in df.columns if c != "Method"})
        st.caption("Share of the 20 splits in which the hospital with random labels is ranked last.")
        k19 = {x["epsilon"]: x for x in pv["k19"]["per_epsilon"]}
        core.finding(f"<b>All 19 hospitals, exactly.</b> Because no coalition is trained, OSA-Shapley values all "
                     f"524,287 coalitions of the 19 hospitals in about a minute. Its private ranking agrees with the "
                     f"non-private one at ρ = {core.f2(core.m(k19[5.0]['spearman']))} (ε = 5) and "
                     f"{core.f2(core.m(k19[1.0]['spearman']))} (ε = 1).", "good")

    core.next_steps(("heart", "A second federation: heart disease", ":material/cardiology:"),
                    ("rule", "When to use which method", ":material/rule:"))
