import math

import pandas as pd
import streamlit as st

from showcase import core, figs

HYP = [
    ("H1", "OSA-Shapley beats MR wherever ν ≥ 100 (ε = 2)"),
    ("H2", "OSA-Shapley does not beat MR at ν ≤ 50 (ε = 2)"),
    ("H3", "OSA − MR increases with ν"),
    ("H4", "The switch-over lies between ν = 50 and 100"),
    ("H5", "Fewer patients and more parameters agree at matched ν"),
    ("H6", "Stronger privacy (ε = 1) moves the switch-over to higher ν"),
]


def advise(nu: float, convex: bool, eps: str, median_n: float):
    """Revised rule from the paper (Section VII-C). Returns (method, kind, reasons)."""
    if eps == "none":
        return "OSA-Shapley", "good", ["Without DP, OSA-Shapley was the most accurate valuation in every setting."]
    if not convex:
        return "MR reconstruction", "warn", [
            "For a non-convex model (e.g. a neural network) each private release accumulates noise over many local "
            "steps; reconstruction from the training run was better (MLP: ρ 0.82 vs 0.65 at ε = 2).",
            "Report the composed privacy cost of the training run you reconstruct from."]
    thr = 40.0 if eps == "1.0" else 30.0
    reasons = [f"Revised threshold for a convex model at {core.EPS_LABEL[eps]}: ν ≈ {thr:.0f} "
               "(pre-registered test; the original guess was 50–100)."]
    if median_n < 100:
        reasons.append("Hospitals are small: also check hospital size, which matched contribution for the ten smallest "
                       "PCa-19 hospitals.")
    if nu >= 1.25 * thr:
        return "OSA-Shapley", "good", reasons + [f"ν = {nu:.0f} is well above the threshold."]
    if nu >= 0.8 * thr:
        return "Compute both OSA-Shapley and MR", "", reasons + [
            f"ν = {nu:.0f} is close to the switch-over; both cost the stated ε once, so computing both is free in "
            "privacy terms. Note the heart-disease federation (ν = 43) where OSA lost, which ν does not explain."]
    return "MR reconstruction", "warn", reasons + [f"ν = {nu:.0f} is below the switch-over, where OSA-Shapley fell behind "
                                                   "(ν = 17: OSA − MR = −0.09 at ε = 2, −0.18 at ε = 1)."]


def render():
    sw = core.R("pca19_nu_sweep"); sc = core.R("pca19_nu_sweep_scoring")
    core.header("Contribution 3", "A decision rule, tested in advance",
                "No method wins everywhere, so a federation needs to know which one to use. We derived a rule from "
                "the first results, wrote down where it predicts the switch-over, fingerprinted that prediction, "
                "and only then ran the test. Two of six predictions failed, and we revised the rule.")
    core.qa("Can a federation tell in advance which valuation to use?",
            "Mostly. Patients per model parameter (ν) predicts the direction; 4 of 6 pre-registered predictions held, but the switch-over was at ν ≈ 21, not 50–100, so the rule was revised.")

    t1, t2, t3 = st.tabs(["Rule advisor", "The pre-registered test", "Integrity of the pre-registration"])

    with t1:
        st.markdown("**The index.** Proposition 3 says the noise in a private release scales as √p / n. "
                    "We therefore measure how much signal each release carries by")
        st.latex(r"\nu \;=\; \frac{\text{median training patients per hospital}}{\sqrt{\text{number of model parameters}}}")
        with st.container(border=True):
            i1, i2, i3, i4 = st.columns(4)
            with i1:
                n_med = st.number_input("Median training patients per hospital", 5, 100000, 300, step=10)
            with i2:
                p = st.number_input("Model parameters", 1, 1000000, 7, step=1,
                                    help="Logistic regression: features + 1 intercept")
            with i3:
                model = st.segmented_control("Model", ["Convex (logistic)", "Non-convex (neural net)"],
                                             default="Convex (logistic)", key="rule_model") or "Convex (logistic)"
            with i4:
                eps = st.pills("Privacy", ["none", "5.0", "2.0", "1.0"], default="2.0",
                               format_func=lambda x: core.EPS_LABEL[x], key="rule_eps") or "2.0"
            nu = n_med / math.sqrt(p)
            method, kind, reasons = advise(nu, model.startswith("Convex"), eps, n_med)
            o1, o2 = st.columns([1, 2])
            with o1:
                core.kpi(f"ν = {nu:.0f}", "your federation", f"{n_med:,} / √{p:,}")
            with o2:
                core.finding(f"<b>Recommended: {method}</b><br>" + "<br>".join("· " + r for r in reasons), kind)
        presets = pd.DataFrame([
            {"Setting": "10 largest PCa-19 hospitals, logistic", "ν": 114, "Winner under DP": "OSA-Shapley"},
            {"Setting": "10 smallest PCa-19 hospitals, logistic", "ν": 45, "Winner under DP": "tie (OSA, MR, size)"},
            {"Setting": "4 heart-disease hospitals, logistic", "ν": 43, "Winner under DP": "MR (unexplained exception)"},
            {"Setting": "10 largest PCa-19 hospitals, MLP", "ν": 26, "Winner under DP": "MR"},
        ])
        c1, c2 = st.columns([1.5, 1])
        with c1:
            core.show(figs.nu_sweep(sw["conditions"], highlight_nu=nu if 13 < nu < 160 else None))
        with c2:
            st.markdown("**The four settings the rule came from**")
            st.dataframe(presets, hide_index=True, use_container_width=True)
            st.caption("The chart shows the pre-registered sweep on the 10 largest PCa-19 hospitals; the dotted line "
                       "marks your setting.")

    with t2:
        st.markdown(
            "**Design (fixed before running).** Ten largest PCa-19 hospitals, logistic model, 20 held-out splits. "
            "**Arm A** keeps 100% → 15% of every hospital's training patients; **arm B** adds 21, 50 or 100 "
            "uninformative features. In each condition the reference is the exact non-private retraining Shapley "
            "ranking of that condition. Outcome: OSA − MR in Spearman ρ, with a paired 95% CI.")
        held = sum(bool(sc[h]) for h, _ in HYP)
        k = st.columns(3)
        with k[0]:
            core.kpi(f"{held} of 6", "predictions held")
        with k[1]:
            core.kpi(f"ν ≈ {sc['crossover_eps2'][0]:.0f}", "observed switch-over at ε = 2", "predicted: 50–100")
        with k[2]:
            core.kpi(f"ν ≈ {min(sc['crossover_eps1']):.0f}–{max(sc['crossover_eps1']):.0f}", "switch-over at ε = 1",
                     "higher, as predicted")
        st.write("")
        hrows = "".join(
            f"<tr><td style='padding:.45rem .6rem;font-weight:700'>{h}</td><td style='padding:.45rem .6rem'>{txt}</td>"
            f"<td style='padding:.45rem .6rem'>{core.badge('held', 'ok') if sc[h] else core.badge('failed', 'no')}</td></tr>"
            for h, txt in HYP)
        st.markdown("<table style='width:100%;border-collapse:collapse;border:1px solid #e7e5ef;border-radius:12px'>"
                    "<tr style='background:#f6f5fb'><th style='text-align:left;padding:.45rem .6rem'>#</th>"
                    "<th style='text-align:left;padding:.45rem .6rem'>Prediction written in advance</th>"
                    f"<th style='text-align:left;padding:.45rem .6rem'>Outcome</th></tr>{hrows}</table>",
                    unsafe_allow_html=True)
        st.write("")
        core.finding("<b>Direction right, threshold wrong.</b> OSA-Shapley's advantage does shrink with ν, both arms "
                     "agree, and stronger privacy moves the switch-over up. But OSA-Shapley stayed significantly ahead "
                     "down to ν ≈ 45 and fell behind only around ν ≈ 20. The original threshold was too cautious.", "warn")
        core.finding("<b>Revised rule.</b> No DP → OSA-Shapley. Under DP with a convex model → OSA-Shapley down to "
                     "ν ≈ 30 (ε = 2) or 40 (ε = 1). Non-convex model → reconstruction. Very small hospitals → also check "
                     "size. In doubt → compute both: they cost the same ε.", "good")
        rows = []
        for c in sorted(sw["conditions"], key=lambda c: -c["nu"]):
            rows.append({"Arm": "fewer patients" if c["arm"] == "A" else "more parameters",
                         "Patients kept": c["fraction"], "Extra features": c["extra_features"], "Parameters": c["p"],
                         "ν": round(c["nu"]), "OSA − MR, ε = 2": core.ci(c["2.0"]["osa_minus_mr"]),
                         "OSA − MR, ε = 1": core.ci(c["1.0"]["osa_minus_mr"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Patients kept": st.column_config.NumberColumn(format="%.2f")})

    with t3:
        live = core.prereg_hash()
        ok = live == core.PREREG_HASH
        st.markdown("The design and the six hypotheses were written to `results/preregistration_nu_sweep.md` and "
                    "fingerprinted with SHA-256 **before** the sweep was run (17 October 2026, 08:19 UTC). Any change to "
                    "the file, even one character, changes the fingerprint. The app recomputes it now:")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.code(f"recorded : {core.PREREG_HASH}\nrecomputed: {live}", language="text")
        with c2:
            if ok:
                core.finding("<b>Match.</b> The pre-registration file is unchanged since it was hashed.", "good")
            else:
                core.finding("<b>Mismatch.</b> The file differs from the hashed version.", "bad")
        core.finding("<b>Honest limitation:</b> this is an internal pre-registration (a hash kept with the code), not "
                     "an entry in a public registry such as OSF, and it was written and run on the same day.", "warn")
        with st.expander("Read the pre-registration document"):
            st.markdown(core.prereg_text())
        st.download_button("Download the pre-registration", core.prereg_text(), "preregistration_nu_sweep.md",
                           "text/markdown")

    core.next_steps(("lab", "Try the valuation yourself", ":material/science:"),
                    ("claims", "Every claim and its verdict", ":material/fact_check:"))
