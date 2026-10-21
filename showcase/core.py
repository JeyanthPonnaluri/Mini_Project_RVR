"""Shared state, data access, styling and small UI components for the research showcase."""
from __future__ import annotations

import hashlib
import json
import os

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "final")
PAPER_DIR = os.path.join(ROOT, "Paper")
PREREG = os.path.join(ROOT, "results", "preregistration_nu_sweep.md")
PREREG_HASH = "3d5dfcdfa98b5e7ceb028e0a0146e0caf614bbbc4daa7720a9ac041a964b68a1"
REPO_URL = "https://github.com/JeyanthPonnaluri/Mini_Project_RVR"

TITLE = ("The Hidden Privacy Cost of Valuing Hospitals in Federated Learning: "
         "Accounting, a Decision Rule, and Evidence from Real Federations")
SHORT_TITLE = "Valuing Hospitals, Privately"
AUTHORS = "Ponnaluri Jeyanth · Pudota Jaideep · Dr. M. V. P. Chandra Sekhara Rao"
AFFIL = "Department of CSE (Data Science), R.V.R. & J.C. College of Engineering, Guntur, India"

# One colour per method, used by every chart and badge in the app.
COL = {
    "osa": "#5b3fd1", "mr": "#14a37f", "or": "#7cc9b0", "ref_auc": "#2f7dd1", "osa_loo": "#d86aa0",
    "retrain_budget_matched": "#e8692d", "retrain_per_run": "#9b9994", "size": "#b9b6ae",
    "statistics": "#d9a21b", "own_test_auc": "#8c6d46", "positives": "#cfc9bd", "truth": "#2b2a38",
}
NAME = {
    "osa": "OSA-Shapley (ours)", "mr": "MR reconstruction", "or": "OR reconstruction",
    "ref_auc": "Single-model AUC", "osa_loo": "Leave-one-out (averaged)",
    "retrain_budget_matched": "Retraining, honest budget", "retrain_per_run": "Retraining, per run",
    "size": "Hospital size", "statistics": "Summary statistics", "own_test_auc": "Self-reported AUC",
    "positives": "Number of cases", "truth": "Exact Shapley (no DP)",
}
EPS_KEYS = ["none", "5.0", "2.0", "1.0"]
EPS_LABEL = {"none": "no DP", "10.0": "ε = 10", "5.0": "ε = 5", "2.0": "ε = 2", "1.0": "ε = 1", "0.5": "ε = 0.5"}

GLOSSARY = {
    "Federated learning (FL)": "Hospitals train one model together; patient records never leave the hospital, "
                               "only model updates are shared.",
    "Differential privacy (DP)": "A mathematical guarantee that a released model reveals almost nothing about any "
                                 "single patient. Smaller ε = stronger privacy, more noise.",
    "ε (epsilon)": "The privacy budget. Budgets add up: a patient used in r private runs pays for all r.",
    "Shapley value": "The fair share of a hospital in what the federation achieves, averaged over every order "
                     "in which hospitals could join.",
    "OSA-Shapley": "Our method: each hospital releases one private model; every coalition model is a weighted "
                   "average of those, so all 2^K coalitions cost ε once.",
    "MR reconstruction": "Rebuilds coalition models from the updates of one private training run (exact version "
                         "of GTG-Shapley).",
    "Spearman ρ": "Agreement of a ranking with the exact (non-private) ranking: 1 = identical order, 0 = unrelated.",
    "ν (nu)": "Median training patients per hospital divided by √(number of model parameters): how much signal "
              "each private release carries.",
    "Pre-registration": "Writing the predictions down, and fingerprinting them with a hash, before running the test.",
    "AUC": "How well a model ranks patients with the disease above those without (0.5 = chance, 1 = perfect).",
}


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def results() -> dict:
    out = {}
    for f in sorted(os.listdir(RES)):
        if f.endswith(".json"):
            with open(os.path.join(RES, f), encoding="utf-8") as fh:
                out[f[:-5]] = json.load(fh)
    return out


def R(key: str):
    """Result file without extension, e.g. R('pca19_heart_valuation'). Stops the page if missing."""
    res = results()
    if key not in res:
        st.error(f"Result file `{key}.json` is missing. Run `python run_pca19.py` to regenerate the results.")
        st.stop()
    return res[key]


def prereg_text() -> str:
    with open(PREREG, encoding="utf-8") as fh:
        return fh.read()


def prereg_hash() -> str:
    with open(PREREG, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def file_bytes(path: str):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    return None


# ----------------------------------------------------------------------------- formatting
def f2(x, d=2):
    return "–" if x is None else f"{x:.{d}f}".replace("-", "−")


def sgn(x, d=2):
    return "–" if x is None else f"{x:+.{d}f}".replace("-", "−")


def ci(s, d=2, signed=True):
    if s is None or s.get("ci_low") is None:
        return "–"
    fm = sgn if signed else f2
    return f"{fm(s['mean'], d)} [{fm(s['ci_low'], d)}, {fm(s['ci_high'], d)}]"


def m(s):
    return None if s is None else s.get("mean")


# ----------------------------------------------------------------------------- navigation registry
PAGES: dict = {}


def register(pages: dict):
    PAGES.clear()
    PAGES.update(pages)


def link(key: str, label: str, icon: str | None = None):
    if key in PAGES:
        st.page_link(PAGES[key], label=label, icon=icon)


def next_steps(*items):
    """Footer with links to the next pages: items = (key, label, icon)."""
    st.markdown("<div class='sc-sep'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sc-next'>Continue</div>", unsafe_allow_html=True)
    cols = st.columns(len(items))
    for c, (key, label, icon) in zip(cols, items):
        with c:
            with st.container(border=True):
                link(key, label, icon)


# ----------------------------------------------------------------------------- styling
CSS = """
<style>
:root { --sc-ink:#1d1b26; --sc-ink2:#55536a; --sc-muted:#8b899c; --sc-line:#e7e5ef; --sc-soft:#f6f5fb;
        --sc-osa:#5b3fd1; --sc-mr:#14a37f; --sc-warn:#e8692d; --sc-bad:#d64545; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.01em; color: var(--sc-ink); }
h1 { font-weight: 750; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: var(--sc-ink); }
div[data-testid="stMetricLabel"] p { font-size: 0.82rem; color: var(--sc-ink2); }
.sc-hero { background: linear-gradient(120deg, #2b1f6b 0%, #4a33b8 55%, #6d4fe0 100%); color: #fff;
           padding: 2.1rem 2.2rem 1.8rem; border-radius: 18px; margin-bottom: 1.4rem; }
.sc-hero h1 { color: #fff; font-size: 2.05rem; line-height: 1.18; margin: 0 0 .6rem 0; }
.sc-hero p { color: #e6e1ff; font-size: 1.02rem; margin: 0 0 .35rem 0; max-width: 60rem; }
.sc-hero .sc-auth { color: #cfc6ff; font-size: .86rem; margin-top: .9rem; }
.sc-kicker { text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700;
             color: var(--sc-osa); margin-bottom: .2rem; }
.sc-hero .sc-kicker { color: #c9bdff; }
.sc-lead { font-size: 1.06rem; color: var(--sc-ink2); max-width: 62rem; margin-bottom: 1rem; }
.sc-card { border: 1px solid var(--sc-line); border-radius: 14px; padding: 1rem 1.1rem; background: #fff; height: 100%; }
.sc-card h4 { margin: 0 0 .35rem 0; font-size: 1rem; color: var(--sc-ink); }
.sc-card p { margin: 0; color: var(--sc-ink2); font-size: .9rem; }
.sc-step { font-weight: 800; color: var(--sc-osa); font-size: .8rem; letter-spacing: .06em; }
.sc-kpi { border: 1px solid var(--sc-line); border-radius: 14px; padding: .9rem 1rem; background: #fff; }
.sc-kpi .v { font-size: 1.75rem; font-weight: 750; color: var(--sc-ink); line-height: 1.1; }
.sc-kpi .l { font-size: .82rem; color: var(--sc-ink2); margin-top: .25rem; }
.sc-kpi .s { font-size: .74rem; color: var(--sc-muted); margin-top: .2rem; }
.sc-find { border-left: 4px solid var(--sc-osa); background: var(--sc-soft); padding: .8rem 1rem;
           border-radius: 0 12px 12px 0; margin: .6rem 0 1rem; color: var(--sc-ink); font-size: .95rem; }
.sc-find.good { border-color: var(--sc-mr); background: #effaf6; }
.sc-find.warn { border-color: var(--sc-warn); background: #fff5ef; }
.sc-find.bad { border-color: var(--sc-bad); background: #fdf1f1; }
.sc-find b { color: var(--sc-ink); }
.sc-badge { display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .74rem; font-weight: 650; }
.sc-badge.ok { background: #e3f6ef; color: #0d7a5c; } .sc-badge.no { background: #fde8e8; color: #b03030; }
.sc-badge.ns { background: #efeef4; color: #55536a; } .sc-badge.info { background: #ece8ff; color: #4a33b8; }
.sc-sep { height: 1px; background: var(--sc-line); margin: 2rem 0 1rem; }
.sc-next { font-size: .78rem; font-weight: 700; color: var(--sc-muted); text-transform: uppercase;
           letter-spacing: .08em; margin-bottom: .4rem; }
.sc-side-title { font-weight: 800; font-size: 1.02rem; color: var(--sc-ink); line-height: 1.25; }
.sc-side-sub { font-size: .78rem; color: var(--sc-muted); margin-top: .2rem; }
.sc-legend span { display: inline-flex; align-items: center; margin-right: 1rem; font-size: .82rem; color: var(--sc-ink2); }
.sc-legend i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; margin-right: .35rem; }
.sc-small { font-size: .82rem; color: var(--sc-muted); }
footer { visibility: hidden; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- components
def header(kicker: str, title: str, lead: str | None = None):
    st.markdown(f"<div class='sc-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    if lead:
        st.markdown(f"<div class='sc-lead'>{lead}</div>", unsafe_allow_html=True)


def finding(text: str, kind: str = ""):
    st.markdown(f"<div class='sc-find {kind}'>{text}</div>", unsafe_allow_html=True)


def kpi(value: str, label: str, sub: str = ""):
    st.markdown(f"<div class='sc-kpi'><div class='v'>{value}</div><div class='l'>{label}</div>"
                f"{f'<div class=s>{sub}</div>' if sub else ''}</div>", unsafe_allow_html=True)


def card(title: str, body: str, step: str | None = None):
    st.markdown(f"<div class='sc-card'>{f'<div class=sc-step>{step}</div>' if step else ''}<h4>{title}</h4>"
                f"<p>{body}</p></div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    return f"<span class='sc-badge {kind}'>{text}</span>"


VERDICT_KIND = {"supported": "ok", "not significant": "ns", "opposite": "no", "not supported": "no",
                "partly": "ns", "explained by step size": "ns"}


def legend(keys):
    items = "".join(f"<span><i style='background:{COL[k]}'></i>{NAME[k]}</span>" for k in keys)
    st.markdown(f"<div class='sc-legend'>{items}</div>", unsafe_allow_html=True)


def show(fig):
    import matplotlib.pyplot as plt
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def sidebar():
    with st.sidebar:
        st.markdown(f"<div class='sc-side-title'>{SHORT_TITLE}</div>"
                    "<div class='sc-side-sub'>Contribution valuation under differential privacy, "
                    "on two real hospital federations</div>", unsafe_allow_html=True)
        st.write("")
        with st.popover("Glossary", use_container_width=True):
            for k, v in GLOSSARY.items():
                st.markdown(f"**{k}** — {v}")
        st.caption("All data are public and used as released: 19 prostate-cancer hospitals (PCa-19) and the "
                   "4 hospitals of the UCI Heart Disease collection. Every number is read from the validated "
                   "result files.")
        st.caption(f"[Code on GitHub]({REPO_URL})")
