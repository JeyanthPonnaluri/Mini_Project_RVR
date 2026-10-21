import os

import streamlit as st

from showcase import core

ABSTRACT = (
    "Hospitals that train a model together need to know what each contributes. Under differential privacy (DP), "
    "Shapley values are usually computed by retraining coalitions of hospitals, and the privacy cost of the valuation "
    "itself is rarely counted. A hospital whose patients enter r private runs pays for all of them: with ten hospitals "
    "a reported per-run ε = 1 costs ε ≈ 33 for exact Shapley values and ≈ 34 with 100 sampled permutations, and holding "
    "the true budget needs √r times the noise, which destroys the ranking. Valuation must therefore post-process "
    "releases whose cost is already counted. We compare such valuations (reconstruction from the DP training run, the "
    "basis of GTG-Shapley; a single-model score; and OSA-Shapley, which averages one DP-trained model per hospital) "
    "with cheap proxies such as hospital size, on two real federations: 19 prostate-cancer hospitals (5,610 men) and "
    "four heart-disease hospitals (920 patients). No method wins everywhere. OSA-Shapley is the most accurate without "
    "DP in every setting. Under DP it leads when hospitals hold many patients per model parameter (Spearman ρ = 0.89 "
    "with the exact ranking at ε = 2, against 0.75 for reconstruction and 0.08 for size); reconstruction is better for "
    "a neural network and in the heart-disease federation. Size and summary statistics are unreliable and never detect "
    "a hospital with random labels. We derive a decision rule indexed by patients per model parameter and test it in a "
    "pre-registered experiment: its direction holds, but OSA-Shapley stays competitive down to fewer patients per "
    "parameter than predicted, and we revise the rule.")

BIBTEX = """@article{jeyanth2026hiddencost,
  title   = {The Hidden Privacy Cost of Valuing Hospitals in Federated Learning:
             Accounting, a Decision Rule, and Evidence from Real Federations},
  author  = {Jeyanth, Ponnaluri and Jaideep, Pudota and Chandra Sekhara Rao, M. V. P.},
  journal = {Manuscript submitted to IEEE Journal of Biomedical and Health Informatics},
  year    = {2026}
}"""


def render():
    core.header("The paper", "Paper, downloads and citation")
    c1, c2 = st.columns([1.7, 1])
    with c1:
        st.markdown(f"#### {core.TITLE}")
        st.markdown(f"{core.AUTHORS}  \n<span class='sc-small'>{core.AFFIL}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='sc-badge info'>Manuscript submitted to IEEE J-BHI</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("**Abstract**")
        st.markdown(ABSTRACT)
    with c2:
        with st.container(border=True):
            st.markdown("**Downloads**")
            for fn, label in [("research_paper.pdf", "Main paper (10 pages)"),
                              ("supplementary.pdf", "Supplementary material (8 pages)")]:
                data = core.file_bytes(os.path.join(core.PAPER_DIR, fn))
                if data:
                    st.download_button(label, data, fn, "application/pdf", use_container_width=True)
            st.download_button("Pre-registration (markdown)", core.prereg_text(), "preregistration_nu_sweep.md",
                               "text/markdown", use_container_width=True)
            st.link_button("Code on GitHub", core.REPO_URL, use_container_width=True)
        with st.container(border=True):
            st.markdown("**Cite**")
            st.code(BIBTEX, language="bibtex")

    st.subheader("Contributions")
    k = st.columns(4)
    items = [
        ("The hidden cost, made explicit", "Proposition 2 gives the composed privacy cost of any retraining-based "
         "valuation and the √r-fold noise needed to keep it honest.", "cost"),
        ("OSA-Shapley", "One private release per hospital; exact Shapley over all coalitions at the stated budget "
         "(Proposition 1), with a noise bound (Proposition 3).", "methods"),
        ("A rule, tested in advance", "Patients per parameter predicts which valuation to use; 4 of 6 pre-registered "
         "predictions held and the threshold was revised.", "rule"),
        ("Honest negatives", "Proxies fail; OSA-Shapley loses under DP for a neural network and in the heart-disease "
         "federation; at ε = 1 nothing recovers the heart ranking.", "heart"),
    ]
    for col, (t, b, key) in zip(k, items):
        with col:
            core.card(t, b)
            core.link(key, "Explore", ":material/arrow_forward:")

    st.subheader("Limitations we state")
    st.markdown(
        "- **Retrospective, two federations.** With four heart-disease hospitals ρ moves in steps of 0.2, so those "
        "estimates are coarse.\n"
        "- **Model class.** One-shot averaging suits regularised logistic regression; for a federated neural network it "
        "loses to reconstruction under DP, and deeper networks are untested.\n"
        "- **Reference set.** Shapley values depend on the patients on which coalitions are scored; a reference set "
        "from outside the federation was not tested, and its patients are protected only if it is public or private itself.\n"
        "- **Privacy assumptions.** The guarantee is per patient and treats hospital sizes and shared preprocessing "
        "statistics as public; if the training run is also released, its budget composes with the valuation's.\n"
        "- **Pre-registration.** Internal (hashed), not in a public registry, and written and run on the same day.")

    st.subheader("Acknowledgements")
    st.markdown("We thank the authors of the base study (Kazlouski, Montoya Perez, Pahikkala and Airola, University of "
                "Turku) for releasing the PCa-19 collection and for their comments, and the creators of the UCI Heart "
                "Disease collection (Janosi, Steinbrunn, Pfisterer and Detrano).")
