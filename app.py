"""
Research showcase for
"The Hidden Privacy Cost of Valuing Hospitals in Federated Learning:
 Accounting, a Decision Rule, and Evidence from Real Federations"

Run:  streamlit run app.py          (Streamlit >= 1.40)
Data: validated results in results/final/ (regenerate with  python run_pca19.py)
The previous version of this app is kept as archive/app_previous.py.
"""
import streamlit as st

st.set_page_config(page_title="The Hidden Privacy Cost of Valuing Hospitals", page_icon=":material/biotech:",
                   layout="wide", initial_sidebar_state="expanded")

from showcase import core  # noqa: E402
from showcase.pages import cost, earlier, evidence, federations, heart, home, lab, methods, paper, rule, training  # noqa: E402

core.inject_css()

# (key, render function, title, icon, url) in reading order, grouped into the parts of the talk.
SECTIONS = [
    ("I · Introduction", [
        ("home", home.render, "Overview", ":material/home:", "overview"),
    ]),
    ("II · Data and setting", [
        ("federations", federations.render, "The two federations", ":material/local_hospital:", "federations"),
        ("training", training.render, "Federated training & privacy", ":material/model_training:", "training"),
    ]),
    ("III · Contributions", [
        ("cost", cost.render, "1 · The hidden privacy cost", ":material/payments:", "privacy-cost"),
        ("methods", methods.render, "2 · Valuation methods compared", ":material/leaderboard:", "methods"),
        ("heart", heart.render, "3 · Second federation: heart", ":material/cardiology:", "heart"),
        ("rule", rule.render, "4 · Decision rule, tested in advance", ":material/rule:", "decision-rule"),
    ]),
    ("IV · Interactive", [
        ("lab", lab.render, "Valuation lab", ":material/science:", "lab"),
    ]),
    ("V · Evidence", [
        ("claims", evidence.claims_page, "Claims & verdicts", ":material/fact_check:", "claims"),
        ("repro", evidence.repro_page, "Reproducibility", ":material/verified:", "reproducibility"),
        ("paper", paper.render, "Paper & citation", ":material/description:", "paper"),
    ]),
    ("Appendix", [
        ("earlier", earlier.render, "Earlier work (TCGA)", ":material/history:", "earlier-work"),
    ]),
]


def _page(key, fn):
    """Wrap a page: breadcrumb context before, previous/next pager after."""
    def run():
        core.set_current(key)
        fn()
        core.pager()
    run.__name__ = f"page_{key}"
    return run


P, nav_spec, order = {}, {}, []
for part, items in SECTIONS:
    nav_spec[part] = []
    for key, fn, title, icon, url in items:
        P[key] = st.Page(_page(key, fn), title=title, icon=icon, url_path=url, default=(key == "home"))
        nav_spec[part].append(P[key])
        order.append((key, part, title, icon))
core.register(P, order)

nav = st.navigation(nav_spec)
core.sidebar()
nav.run()
