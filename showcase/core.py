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
# ----------------------------------------------------------------------------- navigation registry
PAGES: dict = {}
ORDER: list = []          # [(key, part, title, icon)] in reading order
_CUR = {"key": None}


def register(pages: dict, order: list | None = None):
    PAGES.clear(); PAGES.update(pages)
    ORDER.clear(); ORDER.extend(order or [])


def set_current(key: str):
    _CUR["key"] = key
    st.session_state["_sc_fig"] = 0


def _pos(key):
    keys = [o[0] for o in ORDER]
    return keys.index(key) if key in keys else None


def link(key: str, label: str, icon: str | None = None):
    if key in PAGES:
        st.page_link(PAGES[key], label=label, icon=icon)


def next_steps(*items):
    """Compact 'see also' row of related pages: items = (key, label, icon)."""
    items = [it for it in items if it[0] in PAGES and it[0] != _CUR["key"]]
    if not items:
        return
    st.markdown("<div class='sc-sep'></div><div class='sc-next'>See also</div>", unsafe_allow_html=True)
    cols = st.columns(len(items))
    for c, (key, label, icon) in zip(cols, items):
        with c:
            link(key, label, icon)


def pager():
    """Previous / next page in reading order, shown at the foot of every page."""
    i = _pos(_CUR["key"])
    if i is None:
        return
    st.markdown("<div class='sc-sep'></div>", unsafe_allow_html=True)
    left, mid, right = st.columns([1, 0.9, 1])
    if i > 0:
        k, part, title, icon = ORDER[i - 1]
        with left:
            st.markdown("<div class='sc-pg'>← Previous</div>", unsafe_allow_html=True)
            st.page_link(PAGES[k], label=title, icon=icon)
    with mid:
        st.markdown(f"<div class='sc-pgc'>{ORDER[i][1]}<br><span>page {i + 1} of {len(ORDER)}</span></div>",
                    unsafe_allow_html=True)
    if i < len(ORDER) - 1:
        k, part, title, icon = ORDER[i + 1]
        with right:
            st.markdown("<div class='sc-pg r'>Next →</div>", unsafe_allow_html=True)
            st.page_link(PAGES[k], label=title, icon=icon)


# ----------------------------------------------------------------------------- styling
FONTS = ("https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700"
         "&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap")

CSS = """
<style>
@import url('__FONTS__');
:root { --ink:#101828; --ink2:#475467; --muted:#8a94a6; --line:#e4e7ec; --soft:#f6f7f9; --paper:#ffffff;
        --acc:#4c3bcf; --acc2:#14a37f; --warn:#d9622b; --bad:#c43c3c;
        --serif:'Source Serif 4', 'Georgia', serif; --sans:'Inter', 'Segoe UI', system-ui, sans-serif;
        --mono:'JetBrains Mono', 'Consolas', monospace; }
.stApp { background: var(--paper); }
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp input, .stApp textarea,
div[data-testid="stMarkdownContainer"], div[data-testid="stCaptionContainer"] { font-family: var(--sans); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--serif) !important; color: var(--ink);
        letter-spacing: -0.005em; }
.stApp h1 { font-weight: 700; font-size: 2.15rem; line-height: 1.15; }
.stApp h2 { font-weight: 650; font-size: 1.5rem; border-bottom: 1px solid var(--line); padding-bottom: .35rem;
        margin-top: 1.2rem; }
.stApp h3 { font-weight: 600; font-size: 1.22rem; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1220px; }
section[data-testid="stSidebar"] { background: #f7f8fa; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] { margin: .3rem 0; }
div[data-testid="stSidebarNav"] span { font-family: var(--sans); font-size: .9rem; }
header[data-testid="stHeader"] { background: rgba(255,255,255,.85); backdrop-filter: blur(6px); }
div[data-testid="stMetricValue"] { font-family: var(--serif); font-size: 1.6rem; font-weight: 700; color: var(--ink); }
div[data-testid="stMetricLabel"] p { font-size: .82rem; color: var(--ink2); }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: var(--sans); font-weight: 600; font-size: .9rem; padding: .45rem .9rem; }
div[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 10px; }
div[data-testid="stExpander"] summary p { font-weight: 600; color: var(--ink); }
div[data-testid="stPageLink"] a { border: 1px solid var(--line); border-radius: 10px; padding: .35rem .6rem;
        background: #fff; }
div[data-testid="stPageLink"] a:hover { border-color: var(--acc); background: #f5f3ff; }
code { font-family: var(--mono) !important; }

/* breadcrumb + page header */
.sc-crumb { display:flex; align-items:center; gap:.6rem; font-size:.74rem; font-weight:600; letter-spacing:.07em;
        text-transform:uppercase; color: var(--muted); margin-bottom:.25rem; }
.sc-crumb b { color: var(--acc); font-weight: 700; }
.sc-crumb .bar { flex: 0 0 140px; height: 4px; background: var(--line); border-radius: 4px; overflow: hidden; }
.sc-crumb .bar i { display:block; height:100%; background: var(--acc); border-radius: 4px; }
.sc-lead { font-family: var(--serif); font-size: 1.12rem; line-height: 1.55; color: var(--ink2); max-width: 64rem;
        margin: .1rem 0 1.1rem; }

/* paper title page (overview) */
.sc-paper { border: 1px solid var(--line); border-top: 5px solid var(--acc); border-radius: 6px 6px 14px 14px;
        padding: 1.9rem 2.2rem 1.6rem; background: linear-gradient(180deg, #fbfaff 0%, #ffffff 60%); margin-bottom: 1.3rem; }
.sc-venue { font-size: .74rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--acc); }
.sc-paper h1 { font-size: 2.25rem !important; margin: .45rem 0 .7rem !important; max-width: 60rem; }
.sc-authors { font-family: var(--serif); font-size: 1.05rem; color: var(--ink); }
.sc-affil { font-size: .86rem; color: var(--ink2); margin-top: .1rem; }
.sc-abstract { font-family: var(--serif); font-size: 1.04rem; line-height: 1.6; color: #2b3445; margin-top: 1rem;
        padding-top: .9rem; border-top: 1px solid var(--line); max-width: 66rem; }
.sc-abstract b { font-family: var(--sans); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
        color: var(--ink); margin-right: .4rem; }
.sc-tags { margin-top: .9rem; }
.sc-tag { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.15rem .6rem; margin:0 .3rem .3rem 0;
        font-size:.75rem; color: var(--ink2); background:#fff; }

/* components */
.sc-kicker { text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700; color: var(--acc); }
.sc-card { border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; background: #fff; height: 100%; }
.sc-card h4 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
.sc-card p { margin: 0; color: var(--ink2); font-size: .9rem; line-height: 1.5; }
.sc-step { font-weight: 700; color: var(--acc); font-size: .72rem; letter-spacing: .08em; }
.sc-kpi { border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1rem; background: #fff; height: 100%;
        border-top: 3px solid var(--acc); }
.sc-kpi .v { font-family: var(--serif); font-size: 1.7rem; font-weight: 700; color: var(--ink); line-height: 1.1; }
.sc-kpi .l { font-size: .82rem; color: var(--ink2); margin-top: .3rem; line-height: 1.35; }
.sc-kpi .s { font-size: .72rem; color: var(--muted); margin-top: .25rem; }
.sc-find { border-left: 3px solid var(--acc); background: #f7f6fe; padding: .75rem 1rem; border-radius: 0 10px 10px 0;
        margin: .55rem 0 .9rem; color: var(--ink); font-size: .94rem; line-height: 1.5; }
.sc-find.good { border-color: var(--acc2); background: #effaf5; }
.sc-find.warn { border-color: var(--warn); background: #fff6f0; }
.sc-find.bad { border-color: var(--bad); background: #fdf2f2; }
.sc-find b { color: var(--ink); }
.sc-qa { display:grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin: .3rem 0 1rem; }
.sc-qa div { border:1px solid var(--line); border-radius:10px; padding:.75rem 1rem; background:#fff; font-size:.93rem; line-height:1.5; color: var(--ink); }
.sc-qa div b { display:block; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color: var(--muted); margin-bottom:.2rem; }
.sc-qa div.a { border-color:#cfc8f5; background:#faf9ff; }
.sc-figcap { font-family: var(--serif); font-size: .93rem; color: var(--ink2); margin: -.2rem 0 1rem; line-height: 1.45; }
.sc-figcap b { font-family: var(--sans); font-size: .78rem; font-weight: 700; color: var(--ink); letter-spacing: .02em;
        margin-right: .35rem; }
.sc-badge { display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .74rem; font-weight: 650; }
.sc-badge.ok { background: #e3f6ef; color: #0d7a5c; } .sc-badge.no { background: #fde8e8; color: #b03030; }
.sc-badge.ns { background: #eef0f4; color: #475467; } .sc-badge.info { background: #ece8ff; color: #3c2db5; }
.sc-sep { height: 1px; background: var(--line); margin: 2rem 0 .9rem; }
.sc-next { font-size: .72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .08em;
        margin-bottom: .4rem; }
.sc-pg { font-size: .72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .08em;
        margin-bottom: .25rem; }
.sc-pg.r { text-align: right; }
.sc-pgc { text-align: center; font-size: .78rem; color: var(--ink2); padding-top: 1.1rem; }
.sc-pgc span { color: var(--muted); font-size: .72rem; }
.sc-side-title { font-family: var(--serif); font-weight: 700; font-size: 1.12rem; color: var(--ink); line-height: 1.25; }
.sc-side-sub { font-size: .78rem; color: var(--ink2); margin-top: .3rem; line-height: 1.4; }
.sc-legend span { display: inline-flex; align-items: center; margin-right: 1rem; font-size: .82rem; color: var(--ink2); }
.sc-legend i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; margin-right: .35rem; }
.sc-small { font-size: .82rem; color: var(--muted); }
footer { visibility: hidden; }
</style>
""".replace("__FONTS__", FONTS)

PRESENT_CSS = """
<style>
.stApp p, .stApp li, div[data-testid="stMarkdownContainer"] { font-size: 1.12rem !important; }
.stApp h1 { font-size: 2.6rem !important; } .stApp h2 { font-size: 1.85rem !important; }
.sc-lead { font-size: 1.3rem !important; } .sc-find { font-size: 1.12rem !important; }
.sc-kpi .v { font-size: 2.1rem !important; } .sc-kpi .l { font-size: .98rem !important; }
.block-container { max-width: 1400px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- components
def header(kicker: str, title: str, lead: str | None = None):
    """Page header: breadcrumb with reading progress, serif title, lead paragraph."""
    i = _pos(_CUR["key"])
    if i is not None:
        part = ORDER[i][1]
        pct = 100 * (i + 1) / len(ORDER)
        st.markdown(f"<div class='sc-crumb'><b>{part}</b><span>·</span><span>{kicker}</span>"
                    f"<span class='bar'><i style='width:{pct:.0f}%'></i></span><span>{i + 1} / {len(ORDER)}</span></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='sc-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    if lead:
        st.markdown(f"<div class='sc-lead'>{lead}</div>", unsafe_allow_html=True)


def qa(question: str, answer: str):
    """'Question / Answer' summary box at the top of a page."""
    st.markdown(f"<div class='sc-qa'><div><b>Question</b>{question}</div><div class='a'><b>Answer</b>{answer}</div></div>",
                unsafe_allow_html=True)


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


def _publication_style(fig):
    """Journal-figure look for the app only (the paper's own figures are generated separately)."""
    fig.patch.set_facecolor("#ffffff")
    fig.set_dpi(170)
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        ax.set_facecolor("#ffffff")
        for s in ("left", "bottom"):
            if ax.spines[s].get_visible():
                ax.spines[s].set_color("#98a2b3"); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(length=3, width=0.7, direction="out", color="#98a2b3", labelcolor="#344054")
        ax.xaxis.label.set_color("#344054"); ax.yaxis.label.set_color("#344054")


def show(fig, caption: str | None = None):
    """Render a figure as in a paper: the chart title becomes a numbered caption below it."""
    import matplotlib.pyplot as plt
    title = caption
    for ax in fig.axes:
        t = ax.get_title(loc="left") or ax.get_title()
        if t and title is None:
            title = t
        if t:
            ax.set_title("", loc="left"); ax.set_title("")
    _publication_style(fig)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.session_state["_sc_fig"] = st.session_state.get("_sc_fig", 0) + 1
    i = _pos(_CUR["key"])
    num = f"{(i or 0) + 1}.{st.session_state['_sc_fig']}" if i is not None else str(st.session_state["_sc_fig"])
    if title:
        st.markdown(f"<div class='sc-figcap'><b>Figure {num}.</b>{title}</div>", unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown(f"<div class='sc-side-title'>{SHORT_TITLE}</div>"
                    "<div class='sc-side-sub'>The hidden privacy cost of contribution valuation in federated learning, "
                    "studied on two real hospital federations.</div>", unsafe_allow_html=True)
        st.write("")
        if st.toggle("Presentation mode", key="sc_present", help="Larger text for projectors and lecture halls."):
            st.markdown(PRESENT_CSS, unsafe_allow_html=True)
        with st.popover("Glossary", use_container_width=True):
            for k, v in GLOSSARY.items():
                st.markdown(f"**{k}** — {v}")
        st.caption("Data are public and used as released: 19 prostate-cancer hospitals (PCa-19, 5,610 men) and the 4 "
                   "hospitals of the UCI Heart Disease collection (920 patients). Every number is read from the "
                   "validated result files.")
        st.caption(f"[Code on GitHub]({REPO_URL})")
