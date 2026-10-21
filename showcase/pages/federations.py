import pandas as pd
import streamlit as st

from fl_study import charts as ch
from showcase import core


def render():
    co = core.R("pca19_cohort"); hv = core.R("pca19_heart_valuation"); vq = core.R("pca19_valuation_proxies")
    core.header("The data", "Two real federations",
                "Every hospital in this study is a real institution, used exactly as its data were released. "
                "Nothing is simulated and no pooled data set is split into artificial hospitals.")

    k = st.columns(4)
    with k[0]:
        core.kpi(f"{co['n_hospitals']}", "prostate-cancer hospitals", f"{co['n_countries']} countries")
    with k[1]:
        core.kpi(f"{co['n']:,}", "men before prostate biopsy",
                 f"significant cancer {co['prevalence_range'][0]:.0%}–{co['prevalence_range'][1]:.0%} per hospital")
    with k[2]:
        core.kpi(f"{len(hv['sites'])}", "heart-disease hospitals", "USA, Hungary, Switzerland")
    with k[3]:
        core.kpi(f"{hv['n']}", "patients referred for angiography", f"{hv['n_features']} features after preprocessing")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["PCa-19: prostate cancer", "UCI Heart Disease", "How the data are used"])

    with tab1:
        st.markdown("The 19 cohorts collected by the base study (Kazlouski et al., IEEE EMBC 2025) to predict "
                    "**clinically significant prostate cancer** (ISUP grade group ≥ 2) before biopsy from age, PSA, "
                    "prostate volume and the MRI PI-RADS score.")
        h = pd.DataFrame(co["hospitals"])
        large = set(vq["large10"]["hospitals"]); small = set(vq["small10"]["hospitals"])
        h["group"] = ["10 largest + 10 smallest" if x in large and x in small else "10 largest" if x in large
                      else "10 smallest" for x in h["hospital"]]
        st.dataframe(
            h[["hospital", "country", "n", "prevalence", "age_median", "psa_median", "pv_median", "pirads_ge4", "group"]],
            hide_index=True, use_container_width=True,
            column_config={
                "hospital": st.column_config.TextColumn("Hospital"),
                "country": st.column_config.TextColumn("Country"),
                "n": st.column_config.ProgressColumn("Patients", format="%d", min_value=0, max_value=int(h["n"].max())),
                "prevalence": st.column_config.ProgressColumn("Significant cancer", format="%.2f", min_value=0.0,
                                                              max_value=1.0),
                "age_median": st.column_config.NumberColumn("Median age", format="%.0f"),
                "psa_median": st.column_config.NumberColumn("Median PSA (ng/mL)", format="%.1f"),
                "pv_median": st.column_config.NumberColumn("Median volume (mL)", format="%.0f"),
                "pirads_ge4": st.column_config.NumberColumn("PI-RADS ≥ 4", format="%.2f"),
                "group": st.column_config.TextColumn("Valuation group"),
            })
        c1, c2 = st.columns([1.3, 1])
        with c1:
            core.show(ch.scatter_labeled(h["n"].tolist(), h["prevalence"].tolist(), h["hospital"].tolist(),
                                         "Hospitals differ in size and in case mix", "patients",
                                         "share with significant cancer"))
        with c2:
            core.finding("<b>Real heterogeneity.</b> Hospitals range from 16 to 763 patients and their cancer rates "
                         "differ several-fold, which is exactly what makes fair credit hard.")
            st.caption(f"Source: {co['source']}. Converted to CSV without changing any value; our re-implementation "
                       "reproduces the authors' own code to within 1.2 × 10⁻¹⁴.")
            st.caption("The base paper's text mentions nine countries; its Table I and the released data contain eight.")

    with tab2:
        st.markdown("A second, independent federation: the four hospitals of the **UCI Heart Disease** collection "
                    "(Janosi et al., 1988; Detrano et al., 1989), predicting **coronary artery disease** at "
                    "angiography.")
        names = {"cleveland": "Cleveland Clinic Foundation", "hungarian": "Hungarian Institute of Cardiology, Budapest",
                 "switzerland": "University Hospitals Zurich and Basel", "va": "VA Medical Center, Long Beach"}
        hd = pd.DataFrame([{"Hospital": names.get(s["key"], s["name"]), "Country": s["name"].split("(")[-1].rstrip(")"),
                            "Patients": s["n"], "Disease rate": s["prevalence"]} for s in hv["sites"]])
        st.dataframe(hd, hide_index=True, use_container_width=True,
                     column_config={"Patients": st.column_config.ProgressColumn("Patients", format="%d", min_value=0,
                                                                                max_value=int(hd["Patients"].max())),
                                    "Disease rate": st.column_config.ProgressColumn("Disease rate", format="%.2f",
                                                                                    min_value=0.0, max_value=1.0)})
        with st.expander("Preprocessing (leakage-free)", expanded=False):
            st.markdown(
                "1. Drop `slope`, `ca` and `thal`, which are missing for most patients outside Cleveland.\n"
                "2. Treat a cholesterol of 0 as missing.\n"
                "3. Impute missing values with the **training-set** median and add a missing-value indicator.\n"
                "4. One-hot encode chest-pain type and resting ECG.\n"
                "5. Standardise with statistics from training rows only.\n\n"
                f"Result: {hv['n_features']} features. The four raw files are checked against their SHA-256 "
                "checksums by the validation suite.")
        core.finding(f"<b>Few patients per parameter.</b> With about 200 training patients per hospital and "
                     f"{hv['n_features'] + 1} parameters, each private release carries little signal "
                     f"(ν ≈ 43). This is where our method is tested hardest.", "warn")

    with tab3:
        st.markdown("Valuation accuracy needs an exact reference: the **non-private Shapley values obtained by "
                    "retraining every coalition**. That is computable for up to about ten hospitals "
                    "(1,023 coalitions), so we value these groups:")
        g = pd.DataFrame([
            {"Group": "10 largest PCa-19 hospitals", "Patients per hospital": "266–763",
             "Role": "main comparison; one hospital is given random labels to test bad-data detection"},
            {"Group": "10 smallest PCa-19 hospitals", "Patients per hospital": "16–266",
             "Role": "small silos, without simulating any"},
            {"Group": "All 19 PCa-19 hospitals", "Patients per hospital": "16–763",
             "Role": "feasibility: exact OSA-Shapley over 524,287 coalitions in about a minute"},
            {"Group": "4 heart-disease hospitals", "Patients per hospital": "123–303",
             "Role": "second federation, different disease and features"},
        ])
        st.dataframe(g, hide_index=True, use_container_width=True)
        st.caption("Each hospital's patients are split 80/20 into training and test sets, stratified by outcome; the "
                   "test patients of the valued hospitals form the server's reference set. Every result is averaged "
                   "over 20 held-out splits (seeds 20–39) with paired 95% confidence intervals.")

    core.next_steps(("training", "Federated training and privacy", ":material/model_training:"),
                    ("cost", "The hidden privacy cost", ":material/payments:"))
