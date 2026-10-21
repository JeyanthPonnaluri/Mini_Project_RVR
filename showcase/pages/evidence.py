"""Claims & verdicts, and reproducibility."""
import pandas as pd
import streamlit as st

from fl_study import claims as CL
from showcase import core

AREAS = ("PCa-19", "Heart", "Pre-registered")
MARK = {"supported": "✓ supported", "not significant": "○ not significant", "opposite": "✗ opposite",
        "not supported": "✗ not supported", "partly": "◐ partly", "explained by step size": "◐ step-size effect"}
AREA_SHORT = {"PCa-19 (19 hospitals)": "Prostate (PCa-19)", "Heart (4 hospitals)": "Heart disease",
              "Pre-registered rule test": "Pre-registered test"}


def claims_page():
    core.header("Evidence", "Every claim, with a computed verdict",
                "Each claim of the study is checked against the stored results every time this page loads. A claim is "
                "“supported” only if the 95% confidence interval of the paired difference over the 20 splits excludes "
                "zero in the claimed direction. Claims that failed are listed too.")
    rows = [r for r in CL.build(core.results()) if r["area"].startswith(AREAS)]
    df = pd.DataFrame([{"Area": AREA_SHORT.get(r["area"], r["area"]), "Claim": r["claim"].replace("eps", "ε"),
                        "Evidence (mean [95% CI])": r["evidence"].replace("eps", "ε").replace("rho", "ρ"),
                        "Verdict": MARK.get(r["verdict"], r["verdict"]), "_v": r["verdict"]} for r in rows])
    counts = df["_v"].value_counts()
    k = st.columns(4)
    with k[0]:
        core.kpi(str(len(df)), "claims tested")
    with k[1]:
        core.kpi(str(int(counts.get("supported", 0))), "supported")
    with k[2]:
        core.kpi(str(int(counts.get("not significant", 0))), "not significant")
    with k[3]:
        core.kpi(str(int(counts.get("opposite", 0) + counts.get("not supported", 0))), "opposite or not supported",
                 "reported, not hidden")
    st.write("")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        areas = st.pills("Area", list(dict.fromkeys(df["Area"])), selection_mode="multi",
                         default=list(dict.fromkeys(df["Area"])), key="cl_area")
    with c2:
        verdicts = st.pills("Verdict", list(dict.fromkeys(df["Verdict"])), selection_mode="multi",
                            default=list(dict.fromkeys(df["Verdict"])), key="cl_verdict")
    q = st.text_input("Search claims", placeholder="e.g. size, MLP, ν, calibration", label_visibility="collapsed")
    view = df[df["Area"].isin(areas or []) & df["Verdict"].isin(verdicts or [])]
    if q:
        view = view[view["Claim"].str.contains(q, case=False) | view["Evidence (mean [95% CI])"].str.contains(q, case=False)]
    st.dataframe(view.drop(columns="_v"), hide_index=True, use_container_width=True, height=min(640, 38 * len(view) + 40),
                 column_config={"Claim": st.column_config.TextColumn(width="large"),
                                "Evidence (mean [95% CI])": st.column_config.TextColumn(width="large")})
    st.caption(f"{len(view)} of {len(df)} claims shown. Pre-registered hypotheses are scored by the criteria written "
               "before the test was run.")
    core.next_steps(("repro", "Reproducibility and checks", ":material/verified:"),
                    ("paper", "Paper, downloads and citation", ":material/description:"))


def repro_page():
    val = core.R("validation")
    n = sum(v["passed"] for v in val)
    core.header("Evidence", "Reproducibility and automated checks",
                "Every number in the paper and in this app is regenerated from code. The checks below test the data, "
                "the privacy accounting, the valuation code and the absence of leakage.")
    k = st.columns(3)
    with k[0]:
        core.kpi(f"{n}/{len(val)}", "checks pass", "stored run")
    with k[1]:
        core.kpi(f"{len(set(v['group'] for v in val))}", "groups of checks")
    with k[2]:
        core.kpi("20", "held-out splits per result", "seeds 20–39, paired 95% CIs")
    st.write("")
    groups = list(dict.fromkeys(v["group"] for v in val))
    cols = st.columns(2)
    for i, g in enumerate(groups):
        items = [v for v in val if v["group"] == g]
        ok = sum(v["passed"] for v in items)
        with cols[i % 2]:
            with st.expander(f"{g} — {ok}/{len(items)} passed", expanded=False):
                for v in items:
                    st.markdown(f"{'✅' if v['passed'] else '❌'} **{v['check']}**  \n"
                                f"<span class='sc-small'>{v['detail']}</span>", unsafe_allow_html=True)
    st.caption("Several checks in the Data, Leakage and Model groups (the 500- and 347-patient cohorts, Dirichlet "
               "partitions) test the project's earlier TCGA pipeline, which is kept and validated but is not part of "
               "the paper. They test the same shared code (accountant, FedAvg, clipping).")

    st.subheader("Reproduce everything")
    st.code("pip install -r requirements.txt\n"
            "python run_pca19.py                      # recompute every experiment into results/final/\n"
            "python -m fl_study.validate              # the automated checks\n"
            "python Paper/generate_paper_assets.py    # every number, table and figure of the paper\n"
            "streamlit run app.py                     # this app", language="bash")
    st.subheader("Data sources")
    st.dataframe(pd.DataFrame([
        {"Data": "PCa-19 (19 prostate-cancer cohorts)", "Source": "github.com/phaseivai/Hospital-Participation-in-FL",
         "Licence / terms": "MIT", "Use": "as released (data.pkl converted to CSV)"},
        {"Data": "UCI Heart Disease (4 hospitals)", "Source": "UCI Machine Learning Repository, doi:10.24432/C52P4X",
         "Licence / terms": "CC BY 4.0", "Use": "as released; SHA-256 checked"},
    ]), hide_index=True, use_container_width=True)
    with st.expander("Re-run the checks now (takes several minutes)"):
        st.caption("Runs the full validation suite on this machine with the stored data.")
        if st.button("Run all checks", type="primary"):
            from fl_study import validate
            with st.spinner("Running checks…"):
                res = validate.run_all()
            m_ = sum(r["passed"] for r in res)
            (st.success if m_ == len(res) else st.error)(f"{m_} / {len(res)} checks passed")
    core.next_steps(("paper", "Paper, downloads and citation", ":material/description:"),
                    ("home", "Back to the overview", ":material/home:"))
