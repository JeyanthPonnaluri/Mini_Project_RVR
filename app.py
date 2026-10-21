"""
Research showcase for
"The Hidden Privacy Cost of Valuing Hospitals in Federated Learning:
 Accounting, a Decision Rule, and Evidence from Real Federations"

Run:  streamlit run app.py          (Streamlit >= 1.40)
Data: validated results in results/final/ (regenerate with  python run_pca19.py)
The previous version of this app is kept as app_previous.py.
"""
import streamlit as st

st.set_page_config(page_title="Valuing Hospitals, Privately", page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

from showcase import core  # noqa: E402
from showcase.pages import cost, earlier, evidence, federations, heart, home, lab, methods, paper, rule, training  # noqa: E402

core.inject_css()

P = {
    "home": st.Page(home.render, title="Overview", icon=":material/home:", url_path="overview", default=True),
    "federations": st.Page(federations.render, title="The two federations", icon=":material/local_hospital:",
                           url_path="federations"),
    "training": st.Page(training.render, title="Federated training & privacy", icon=":material/model_training:",
                        url_path="training"),
    "cost": st.Page(cost.render, title="The hidden privacy cost", icon=":material/payments:", url_path="privacy-cost"),
    "methods": st.Page(methods.render, title="Valuation methods compared", icon=":material/leaderboard:",
                       url_path="methods"),
    "heart": st.Page(heart.render, title="Second federation: heart", icon=":material/cardiology:", url_path="heart"),
    "rule": st.Page(rule.render, title="Decision rule (pre-registered)", icon=":material/rule:", url_path="decision-rule"),
    "lab": st.Page(lab.render, title="Valuation lab", icon=":material/science:", url_path="lab"),
    "claims": st.Page(evidence.claims_page, title="Claims & verdicts", icon=":material/fact_check:", url_path="claims"),
    "repro": st.Page(evidence.repro_page, title="Reproducibility", icon=":material/verified:", url_path="reproducibility"),
    "paper": st.Page(paper.render, title="Paper & citation", icon=":material/description:", url_path="paper"),
    "earlier": st.Page(earlier.render, title="Earlier work (TCGA)", icon=":material/history:", url_path="earlier-work"),
}
core.register(P)

nav = st.navigation({
    "Start": [P["home"]],
    "Data and setting": [P["federations"], P["training"]],
    "Contributions": [P["cost"], P["methods"], P["heart"], P["rule"]],
    "Try it": [P["lab"]],
    "Evidence": [P["claims"], P["repro"], P["paper"]],
    "Archive": [P["earlier"]],
})
core.sidebar()
nav.run()
