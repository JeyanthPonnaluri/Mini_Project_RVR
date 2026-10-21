import streamlit as st

from showcase import core, figs


def render():
    pv = core.R("pca19_private_valuation"); vq = core.R("pca19_valuation_proxies")
    hv = core.R("pca19_heart_valuation"); sc = core.R("pca19_nu_sweep_scoring"); val = core.R("validation")
    co = core.R("pca19_cohort")
    rows = {r["epsilon"]: r for r in pv["rows"]}
    L = vq["large10"]["rho"]

    st.markdown(f"""
<div class='sc-hero'>
  <div class='sc-kicker'>Research showcase · IEEE J-BHI manuscript</div>
  <h1>{core.TITLE}</h1>
  <p>Hospitals that train a clinical model together need to know what each one contributes. Under differential
  privacy, the usual way of measuring this secretly multiplies the privacy cost. We count that cost, compare every
  valuation that pays it only once, and test in advance when each one should be used.</p>
  <div class='sc-auth'>{core.AUTHORS}<br>{core.AFFIL}</div>
</div>""", unsafe_allow_html=True)

    k = st.columns(5)
    with k[0]:
        core.kpi(f"ε 1 → {rows[1.0]['retrain_true_epsilon']:.0f}", "real privacy cost of valuing 10 hospitals by retraining",
                 "per-run budget 1, composed over 512 runs")
    with k[1]:
        core.kpi(f"ρ {core.f2(core.m(L['2.0']['osa']))}", "OSA-Shapley ranking accuracy at ε = 2",
                 f"honest retraining {core.f2(core.m(rows[2.0]['retrain_budget_matched']['spearman']))}, "
                 f"size {core.f2(core.m(L['2.0']['size']))}")
    with k[2]:
        n_h = co["n_hospitals"] + len(hv["sites"])
        core.kpi(f"{n_h} hospitals", "two real federations, used as released",
                 f"{co['n']:,} + {hv['n']} patients · 10 countries")
    with k[3]:
        held = sum(bool(sc[h]) for h in ("H1", "H2", "H3", "H4", "H5", "H6"))
        core.kpi(f"{held} of 6", "pre-registered predictions held", "direction right, threshold revised")
    with k[4]:
        core.kpi(f"{sum(v['passed'] for v in val)}/{len(val)}", "automated validation checks pass",
                 "every number regenerated from code")

    st.write("")
    st.subheader("The research in three steps")
    c = st.columns(3)
    with c[0]:
        core.card("The hidden cost",
                  "Shapley values need a model for every coalition of hospitals. Retraining each coalition with "
                  "differential privacy puts every patient into hundreds of private runs, and privacy costs add up.",
                  "1 · PROBLEM")
    with c[1]:
        core.card("Pay once, value exactly",
                  "OSA-Shapley: each hospital releases one private model; every coalition model is a weighted average "
                  "of those releases. All 2^K coalitions are post-processing, so the budget is spent once.",
                  "2 · METHOD")
    with c[2]:
        core.card("Evidence, including where we lose",
                  "Two real federations, cheap proxies such as hospital size, reconstruction from the training run, "
                  "and a decision rule written down and hashed before it was tested.",
                  "3 · EVIDENCE")

    st.write("")
    st.subheader("Headline result")
    left, right = st.columns([1.55, 1])
    with left:
        xl = [core.EPS_LABEL[e] for e in core.EPS_KEYS]
        bm = [None] + [core.m(rows[float(e)]["retrain_budget_matched"]["spearman"]) for e in core.EPS_KEYS[1:]]
        ser = [{"key": "osa", "mean": [core.m(L[e]["osa"]) for e in core.EPS_KEYS], "band": True,
                "low": [L[e]["osa"]["ci_low"] for e in core.EPS_KEYS], "high": [L[e]["osa"]["ci_high"] for e in core.EPS_KEYS]},
               {"key": "mr", "mean": [core.m(L[e]["mr"]) for e in core.EPS_KEYS]},
               {"key": "ref_auc", "mean": [core.m(L[e]["ref_auc"]) for e in core.EPS_KEYS]},
               {"key": "retrain_budget_matched", "mean": bm},
               {"key": "size", "mean": [core.m(L[e]["size"]) for e in core.EPS_KEYS], "dash": "--"}]
        core.show(figs.rho_lines(xl, ser, "Ranking the 10 largest prostate-cancer hospitals (20 held-out splits)",
                                 ylim=(-0.25, 1.02)))
    with right:
        core.finding(f"<b>OSA-Shapley keeps the ranking at an honest budget:</b> ρ = {core.f2(core.m(L['2.0']['osa']))} "
                     f"at ε = 2, against {core.f2(core.m(L['2.0']['mr']))} for reconstruction from the training run and "
                     f"{core.f2(core.m(rows[2.0]['retrain_budget_matched']['spearman']))} for retraining at the same true "
                     "cost.", "good")
        core.finding(f"<b>Cheap proxies fail:</b> hospital size reaches ρ = {core.f2(core.m(L['2.0']['size']))} and "
                     "never spots a hospital whose data are useless.", "warn")
        core.finding(f"<b>No method wins everywhere.</b> With few patients per parameter (heart disease, a neural "
                     f"network) reconstruction is better; a pre-registered test placed the switch-over at "
                     f"ν ≈ {sc['crossover_eps2'][0]:.0f}.", "")

    st.write("")
    st.subheader("Choose your tour")
    t = st.columns(3)
    with t[0]:
        with st.container(border=True):
            st.markdown("**5-minute tour** · for examiners")
            st.caption("The problem, the method and the main results, in order.")
            core.link("cost", "1. The hidden privacy cost", ":material/payments:")
            core.link("methods", "2. Valuation methods compared", ":material/leaderboard:")
            core.link("rule", "3. The decision rule, tested in advance", ":material/rule:")
    with t[1]:
        with st.container(border=True):
            st.markdown("**The evidence** · for researchers")
            st.caption("Data, every claim with its verdict, and how to reproduce it.")
            core.link("federations", "The two federations", ":material/local_hospital:")
            core.link("claims", "Claims and verdicts", ":material/fact_check:")
            core.link("repro", "Reproducibility and checks", ":material/verified:")
    with t[2]:
        with st.container(border=True):
            st.markdown("**Try it live** · for everyone")
            st.caption("Run the valuation yourself, or ask the rule which method fits your federation.")
            core.link("lab", "Valuation lab", ":material/science:")
            core.link("rule", "Decision-rule advisor", ":material/tune:")
            core.link("paper", "Paper, downloads and citation", ":material/description:")
