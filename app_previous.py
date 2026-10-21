"""
Federated Prostate-Cancer Staging - experiment showcase
=======================================================
Run:   streamlit run app.py
Data:  pre-computed, validated results in results/final/ (regenerate with  python run_all.py)

Story told by the app
  1. Overview          - problem, cohort, what was audited & fixed
  2. Base paper        - Kazlouski et al. (hospital participation in FL) replicated on
                         REAL TCGA hospitals: LOC vs FL vs FR vs CEN vs rule baseline
  3. Our model         - DP-FedProx + Shapley valuation + personalisation (DP-FPS)
  4. Final results     - base vs ours, every claim with a computed verdict
  5. Validation        - automated correctness checks (re-runnable live)
  6. Live demo         - train a federated model with your own settings
(The previous app is kept as app_legacy.py.)
"""
import json
import math
import os

import numpy as np
import pandas as pd
import streamlit as st

from fl_study import charts as ch
from fl_study import claims, data, fl

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "final")

st.set_page_config(page_title="Federated PCa Staging - Experiments", page_icon="🩺", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.small {font-size:0.85rem;color:#6b6a66}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def load_results():
    out = {}
    if os.path.isdir(RES):
        for f in os.listdir(RES):
            if f.endswith(".json"):
                with open(os.path.join(RES, f), encoding="utf-8") as fh:
                    out[f[:-5]] = json.load(fh)
    return out


R = load_results()


def need(*keys):
    missing = [k for k in keys if k not in R]
    if missing:
        st.error(f"Missing result files: {', '.join(missing)}. Run `python run_all.py` first.")
        st.stop()


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def ci_txt(s, d=3, signed=False):
    if s is None or s.get("mean") is None:
        return "—"
    f = f"{{:{'+' if signed else ''}.{d}f}}"
    if s.get("ci_low") is None:
        return f.format(s["mean"])
    return f"{f.format(s['mean'])} [{f.format(s['ci_low'])}, {f.format(s['ci_high'])}]"


def fmt_cell(v, f):
    """Format a number for a table cell; missing / undefined values become an em dash."""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f.format(v)
    except (TypeError, ValueError):
        return "—"


def _e(s):
    if s is None or s.get("ci_low") is None:
        return (0, 0)
    return (s["mean"] - s["ci_low"], s["ci_high"] - s["mean"])


def show(fig):
    import matplotlib.pyplot as plt
    st.pyplot(fig)
    plt.close(fig)


def verdict_badge(v):
    icon = {"supported": "✅", "not significant": "➖", "opposite": "❌", "not supported": "❌",
            "partly": "◐", "explained by step size": "◐", "insufficient data": "❔"}.get(v, "❔")
    return f"{icon} {v}"


def finding(text, kind="info"):
    getattr(st, kind)(text)


def cell(cells, a, e):
    return [x for x in cells if x["alpha"] == a and x["epsilon"] == e][0]


# ----------------------------------------------------------------------------- sidebar
PAGES = ["🏠 Overview", "📄 Base paper (replication)", "🌍 19 hospitals (PCa-19)", "🧪 Our model (DP-FPS)",
         "🏁 Final results", "✅ Validation", "🎛️ Live demo"]
with st.sidebar:
    st.markdown("### Federated PCa staging")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    seeds = R.get("meta", {}).get("base_paper", {}).get("seeds")
    st.caption(f"Results: `results/final/` · {seeds or '?'} seeds per experiment · "
               "regenerate with `python run_all.py`")
    st.caption("All AUCs are on held-out data. Brackets = 95% CI over seeds.")


# ============================================================================= 1. OVERVIEW
def page_overview():
    need("cohort")
    c = R["cohort"]
    st.title("Privacy-preserving federated learning for prostate-cancer staging")
    st.markdown(
        "**Tasks.** (1) **Primary:** predict clinically significant prostate cancer *before biopsy* from age, PSA, "
        "prostate volume and MRI PI-RADS on the base paper's own **19 real hospitals** (PCa-19, see the 🌍 page). "
        "(2) **Secondary:** predict whether a tumour is *advanced* (pathologic **T3/T4**) or *organ-confined* "
        "(**T1/T2**) from TCGA-PRAD data. In both, the data are spread across hospitals that cannot share records.")
    a, b, d, e = st.columns(4)
    a.metric("TCGA-PRAD records", c["raw_records"], f"{c['raw_patients']} patients", delta_color="off")
    b.metric("Clinical cohort (1 tumour / patient)", c["clinical_cohort"],
             f"{c['clinical_cohort_pos']} T3/T4 · {c['clinical_sites']} sites", delta_color="off")
    d.metric("Matched clinical + protein cohort", c["matched_cohort"],
             f"{c['matched_pos']} T3/T4 · split {c['matched_train']}/{c['matched_test']}", delta_color="off")
    e.metric("Deaths in matched cohort", c["survival_events_matched"],
             "too few for survival modelling", delta_color="off")

    st.subheader("The three stages of this project")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("#### 1 · Base paper")
        st.markdown("Kazlouski *et al.*, *Hospital Participation in Federated Learning: Evaluating "
                    "Sustainability and Clinical Utility*. Compares **local** training, **federated** training "
                    "(FedAvg), **free-riding** and a **baseline** model across 19 real hospitals. "
                    "We reproduce it **exactly on its own 19-hospital data (PCa-19, 5,610 patients)** and replicate the design "
                    "on the **9 TCGA hospitals with ≥ 20 patients**.")
    with s2:
        st.markdown("#### 2 · Our model (DP-FPS)")
        st.markdown("Adds what the base paper lacks: **differential privacy** (DP-SGD with a Rényi-DP "
                    "accountant), **FedProx** against client drift, **Shapley** contribution valuation, "
                    "**personalised** fine-tuning, and a **membership-inference** audit.")
    with s3:
        st.markdown("#### 3 · Final results")
        st.markdown("Every claim is tested over 20 seeds with paired 95% confidence intervals. The verdicts "
                    "on the *Final results* page are computed from the result files, not hand-written.")

    st.subheader("What the audit found and how it was fixed")
    fixes = pd.DataFrame([
        ["Identifier columns used as features", "~1,650 one-hot columns built from per-patient IDs, UUIDs and "
         "timestamps entered the model", "Explicit feature whitelist; IDs, timestamps and follow-up/outcome "
         "columns excluded (checked automatically)"],
        ["Proteomics claimed to help", "Reports said clinical + protein 'significantly outperforms' clinical",
         "Re-tested with repeated CV: adding protein PCs lowers AUC (Our model → Features)"],
        ["Label-adjacent features", "Gleason grade and pathologic N come from the same surgical report as the label",
         "Kept in the main model, disclosed, and a pre-operative-only ablation is reported"],
        ["Top-contributor stability metric", "Averaged P(rank = 1) over clients, so it was always exactly 1/K",
         "Now P(no-DP top contributor is still first under DP)"],
        ["Client-drift metric", "FedAvg drift silently recorded as 0.0; FedProx measured global-model change",
         "Both use the paper's definition mean‖w_k − w_global‖"],
        ["Survival event count", "Paper: '12 deaths (9 train, 0 test)', which does not add up",
         f"{c['survival_events_matched']} deaths in the 347-patient cohort (12 is over all 572 records)"],
        ["Pooled AUC of local models", "Concatenating predictions of different local models rewards learning "
         "each site's base rate", "Site/client comparisons use per-site (macro) AUC"],
        ["No intercept in the NumPy model", "Logistic regression without a bias term", "Intercept added"],
    ], columns=["Issue", "Before", "Fix"])
    st.dataframe(fixes, hide_index=True)

    with st.expander("Features used by the model"):
        st.markdown("**Full set (main model):** " + ", ".join(f"`{x}`" for x in c["features_full"]))
        st.markdown("**Pre-operative set (ablation):** " + ", ".join(f"`{x}`" for x in c["features_preop"]))
        st.markdown("**Never used as features:**")
        for reason, cols in c["excluded_columns"].items():
            st.markdown(f"- *{reason}*: " + ", ".join(f"`{x}`" for x in cols))
    with st.expander("Cohort construction"):
        st.markdown(
            f"- {c['raw_records']} records → keep primary-tumour samples (`-01`), valid pathologic T stage, one "
            f"sample per patient → **{c['clinical_cohort']} patients** (used for the base-paper replication).\n"
            f"- Protein (RPPA) data exist for {c['protein_samples']} samples → matched cohort **{c['matched_cohort']}** "
            f"(used for our model, so the multimodal comparison is on identical patients).\n"
            f"- Proteins: {c['protein_targets_raw']} targets → {c['protein_targets_kept']} with ≤ 30% missing → PCA to 95% "
            f"variance, fitted on the training split only (~115 components; {c['protein_pcs_95pct_full_cohort']} if "
            f"it were fitted on all rows).\n"
            f"- T stage in matched cohort: " + ", ".join(f"{k}: {v}" for k, v in sorted(c["t_stage_counts_matched"].items())))


# ============================================================================= 2. BASE PAPER
def page_base():
    need("base_paper")
    bp = R["base_paper"]
    st.title("Base paper — replicated on real TCGA hospitals")
    with st.expander("What the base paper did", expanded=True):
        st.markdown(
            "**Kazlouski, Montoya Perez, Pahikkala, Airola** — *Hospital Participation in Federated Learning: "
            "Evaluating Sustainability and Clinical Utility* (University of Turku).\n\n"
            "- 19 real hospital datasets (5,610 patients), predicting clinically significant prostate cancer before "
            "biopsy with logistic-regression risk calculators (Ettala, Noh).\n"
            "- Each hospital can get a model four ways: **LOC** (train on own data), **FL** (join FedAvg), "
            "**FR** (free-ride: use the federated model without contributing) or **BL** (external pre-trained "
            "model); **CEN** (pooled data) is the upper reference.\n"
            "- Findings: FL generalises better but large hospitals gain little over LOC; free-riders reach "
            "participant-level performance once ≈10 hospitals train the model; a small consortium of strong "
            "hospitals could serve the others.\n"
            "- It does **not** add differential privacy, handle client drift, or value contributions.")
    st.markdown(
        f"**Our replication.** Same strategies, same model family (logistic regression, FedAvg), on the "
        f"**{len(bp['sites'])} TCGA tissue-source sites with ≥ {bp['config']['min_site_size']} patients** "
        f"(the {bp['n_external']} patients from smaller sites never train and act as external free-riders). Each "
        f"site is split 80/20, repeated over {bp['config']['seeds']} seeds. PSA is not available in TCGA, so BL is "
        "a training-free rule: the Gleason sum.")

    sites = pd.DataFrame(bp["per_site"])
    tab1, tab2, tab3, tab4 = st.tabs(["Strategy comparison", "Per hospital", "How many hospitals?",
                                      "Who contributes?"])
    with tab1:
        keys = ["LOC", "FL", "FR", "CEN", "BL"]
        names = ["Local only", "Federated", "Free-rider", "Centralised", "Rule (Gleason)"]
        show(ch.bar_with_ci(names, [bp["macro"][k] for k in keys], [ch.C[k] for k in keys],
                            "Mean per-hospital test AUC (95% CI over seeds)", ylim=(0.5, 1.0)))
        d = bp["macro_diff"]
        c1, c2, c3 = st.columns(3)
        c1.metric("FL − LOC", ci_txt(d["FL-LOC"], signed=True))
        c2.metric("FR − FL (free-rider penalty)", ci_txt(d["FR-FL"], signed=True))
        c3.metric("CEN − FL", ci_txt(d["CEN-FL"], signed=True))
        st.caption(f"Macro AUC averages the hospitals whose test split contains both classes "
                   f"({fmt(bp['macro_sites']['mean'], 1)} per seed on average). PROCURE Biobank (100% T3/T4) "
                   "never has a defined AUC.")
        lo, fl_ = bp["macro"]["LOC"]["mean"], bp["macro"]["FL"]["mean"]
        if d["FL-LOC"]["ci_high"] is not None and d["FL-LOC"]["ci_high"] < 0:
            finding(f"**Local models beat the federated model on their own hospital** ({fmt(lo)} vs {fmt(fl_)}). "
                    "TCGA sites differ in case mix and grading practice, so a site's own model fits its patients "
                    "better. This is consistent with the base paper's finding that joining FL brings large "
                    "hospitals little local gain.", "warning")
        elif d["FL-LOC"]["ci_low"] is not None and d["FL-LOC"]["ci_low"] > 0:
            finding("Federated training improves on local training per hospital.", "success")
        else:
            finding(f"**No significant difference between federated ({fmt(fl_)}) and local ({fmt(lo)}) models per "
                    "hospital.** As in the base paper, joining FL brings little local gain on average.", "info")
    with tab2:
        tbl = sites[["site", "n", "prevalence", "LOC", "FL", "FR", "CEN", "BL", "LOC_n_defined", "FL_minus_LOC"]].copy()
        tbl.columns = ["Hospital", "Patients", "T3/T4 rate", "LOC", "FL", "FR", "CEN", "Rule", "seeds with AUC", "FL − LOC"]
        for col, f in [("T3/T4 rate", "{:.2f}"), ("LOC", "{:.3f}"), ("FL", "{:.3f}"), ("FR", "{:.3f}"),
                       ("CEN", "{:.3f}"), ("Rule", "{:.3f}"), ("FL − LOC", "{:+.3f}")]:
            tbl[col] = [fmt_cell(v, f) for v in tbl[col]]   # undefined AUC -> "—"
        st.dataframe(tbl, hide_index=True)
        ok = sites.dropna(subset=["FL_minus_LOC"])
        show(ch.scatter_labeled(ok["n"].tolist(), ok["FL_minus_LOC"].tolist(),
                                [ch.short(s) for s in ok["site"]],
                                "Gain from joining FL vs hospital size", "patients at hospital",
                                "FL − LOC (AUC)", ref_y=0))
        st.caption("Small test splits (5–19 patients per site) make single-site AUCs noisy; read the averages "
                   "over seeds, not single values.")
    with tab3:
        lc = bp["learning_curve"]
        K = [r["K"] for r in lc]
        ser = [{"name": name, "color": ch.C[col], "mean": [r[key]["mean"] for r in lc],
                "low": [r[key]["ci_low"] for r in lc], "high": [r[key]["ci_high"] for r in lc]}
               for key, name, col in [("participants", "Participants", "FL"), ("free_riders", "Free-riders", "FR")]]
        show(ch.lines(K, ser, "Federated model AUC as more hospitals join", "hospitals training the model",
                      "pooled test AUC", clip=(0, 1)))
        st.dataframe(pd.DataFrame({"hospitals": K,
                                   "participants AUC": [ci_txt(r["participants"]) for r in lc],
                                   "free-riders AUC": [ci_txt(r["free_riders"]) for r in lc],
                                   "gap": [ci_txt(r["gap"], signed=True) for r in lc]}),
                     hide_index=True)
        st.caption("Free-riders = hospitals not (yet) participating plus the external small-site pool. Hospitals "
                   "join in a random order per seed, as in the base paper.")
    with tab4:
        st.markdown("The base paper does not value contributions. We add the two standard methods: **leave-one-out "
                    "(LOO)** — AUC lost when a hospital is removed — and the **exact Shapley value** over all "
                    f"2^{len(bp['sites'])} coalitions.")
        sh = [r["shapley"]["mean"] for r in bp["per_site"]]
        lo_ = [r["loo"]["mean"] for r in bp["per_site"]]
        short = [ch.short(r["site"]) for r in bp["per_site"]]
        show(ch.grouped_bars(short, ["Shapley", "LOO"], [sh, lo_], [ch.C["FedAvg"], ch.C["FedProx"]],
                             "Contribution of each hospital (mean over seeds)", "Δ AUC", ref=0))
        c1, c2 = st.columns(2)
        c1.metric("Spearman(LOO, Shapley) per seed", ci_txt(bp["loo_vs_shapley_spearman"], 2))
        c2.metric("Spearman(Shapley, hospital size)", fmt(bp["shapley_vs_size_spearman"], 2))
        st.caption("LOO values are tiny and often negative: removing one hospital barely changes a 9-hospital "
                   "model, so LOO cannot separate contributors. Shapley averages over all coalition sizes and "
                   "satisfies efficiency (values sum to the total gain over a random model).")


# ============================================================================= 2b. PCa-19 (base paper's own data)
NOH_C, ETT_C = "#4a3aa7", "#8a8985"


def page_pca19():
    need("pca19_cohort", "pca19_strategies", "pca19_loco", "pca19_privacy", "pca19_personalisation", "pca19_mia")
    co, s = R["pca19_cohort"], R["pca19_strategies"]
    st.title("19 real hospitals — the base paper's own data")
    st.markdown(
        f"**PCa-19** is the exact patient-level collection used by the base paper: **{co['n']:,} men from "
        f"{co['n_hospitals']} hospitals in {co['n_countries']} countries**, released in the authors' public repository "
        "(MIT licence). The task is the base paper's: predict **clinically significant prostate cancer "
        "(ISUP grade group ≥ 2) before biopsy**, from age, **PSA**, prostate volume and the MRI **PI-RADS** score. "
        "All four are known before biopsy, so this is a real pre-biopsy risk calculator.")
    a, b, c, d = st.columns(4)
    a.metric("Patients", f"{co['n']:,}", f"{co['n_pos']:,} csPCa ({co['n_pos'] / co['n']:.0%})", delta_color="off")
    b.metric("Hospitals", co["n_hospitals"], f"{co['n_countries']} countries", delta_color="off")
    c.metric("csPCa rate per hospital", f"{co['prevalence_range'][0]:.0%}–{co['prevalence_range'][1]:.0%}",
             "real heterogeneity", delta_color="off")
    d.metric("Features", len(co["features"]), "age, log PSA, log volume, PI-RADS", delta_color="off")
    st.caption("The base paper's text says nine countries; its Table I and released data contain eight "
               "(Spain, South Korea, Germany, China, USA, Netherlands, Italy, UK).")

    tabs = st.tabs(["Cohort", "Base paper reproduced", "Strategies (our protocol)", "External validation",
                    "Privacy & drift", "Personalisation", "Contribution under DP", "Privacy audit",
                    "Stronger baselines", "Subsampled DP", "Heart federation", "Decision rule (pre-registered)"])

    with tabs[0]:
        h = pd.DataFrame(co["hospitals"])
        tbl = h.copy()
        tbl.columns = ["Hospital", "Country", "Patients", "csPCa rate", "Median age", "Median PSA", "Median volume",
                       "PI-RADS ≥ 4"]
        for col, f in [("csPCa rate", "{:.0%}"), ("Median age", "{:.0f}"), ("Median PSA", "{:.1f}"),
                       ("Median volume", "{:.0f}"), ("PI-RADS ≥ 4", "{:.0%}")]:
            tbl[col] = [fmt_cell(v, f) for v in tbl[col]]
        st.dataframe(tbl, hide_index=True)
        show(ch.scatter_labeled(h["n"].tolist(), h["prevalence"].tolist(), h["hospital"].tolist(),
                                "Hospitals differ in size and case mix", "patients", "csPCa rate"))
        st.caption("es2 (16 patients) has extreme PSA values (median 277 ng/mL); log-transforming PSA keeps it "
                   "from dominating the model.")

    with tabs[1]:
        st.markdown(
            "First we re-ran the **authors' exact protocol**: scikit-learn logistic regression on the raw Ettala "
            "and Noh variables, a stratified 80/20 split per hospital, **one-shot FedAvg** (average of fully trained "
            "local models), free-riding = all other hospitals train and the excluded one is tested on 100% of its "
            "patients, Monte-Carlo random states drawn exactly as in their scripts.")
        rows = []
        for calc in ("noh", "ettala"):
            k = f"pca19_reproduction_{calc}"
            if k not in R:
                st.info(f"`{k}.json` not computed yet.")
                continue
            r = R[k]
            for strat, lab in [("BL", "Published calculator (BL)"), ("LOC", "Local (LOC)"), ("CEN", "Centralised (CEN)"),
                               ("FL", "Federated (FL)"), ("FR", "Free-rider (FR)")]:
                rows.append({"Calculator": calc.capitalize(), "Strategy": lab, "AV (mean over hospitals)": r["AV"][strat],
                             "AVW (patient-weighted)": r["AVW"][strat]})
        if rows:
            t = pd.DataFrame(rows)
            for col in ("AV (mean over hospitals)", "AVW (patient-weighted)"):
                t[col] = [fmt_cell(v, "{:.3f}") for v in t[col]]
            st.dataframe(t, hide_index=True)
            r = R.get("pca19_reproduction_noh")
            if r:
                st.caption(f"{r['n_repeats']} Monte-Carlo repeats. AV and AVW are the two summary columns of the "
                           "base paper's Figure 3.")
        v = [x for x in R.get("validation", []) if x["group"] == "PCa-19" and "authors' own code" in x["check"]]
        if v and v[0]["passed"]:
            finding(f"**Agreement with the authors' own code:** {v[0]['detail']}", "success")
        elif v:
            finding(f"**Agreement check with the authors' code failed:** {v[0]['detail']}", "error")
        st.markdown("Same conclusions as the paper: the published **Noh** calculator is almost as good as centralised "
                    "training, **Ettala**'s spline penalises extreme PSA values, and FL ≈ FR ≈ CEN.")

    with tabs[2]:
        st.markdown("Our protocol: standardised features (statistics from training rows only), iterative FedAvg/FedProx "
                    f"(T = 20 rounds, E = 5 local epochs), {R['pca19_meta'].get('strategies', {}).get('seeds', 20)} seeds, "
                    "paired 95% CIs. **Macro** = mean over the 19 hospitals; **weighted** = by patients.")
        keys = ["LOC", "FL", "FedProx", "CEN", "FR_test", "BL_noh", "BL_ettala"]
        names = ["Local", "FedAvg", "FedProx", "Central.", "Free-rider", "Noh", "Ettala"]
        cols = [ch.C["LOC"], ch.C["FL"], ch.C["FedProx"], ch.C["CEN"], ch.C["FR"], NOH_C, ETT_C]
        g1, g2 = st.columns(2)
        with g1:
            show(ch.bar_with_ci(names, [s["macro"][k] for k in keys], cols, "Mean per-hospital test AUC", ylim=(0.5, 1.0)))
        with g2:
            show(ch.bar_with_ci(names, [s["weighted"][k] for k in keys], cols, "Patient-weighted test AUC", ylim=(0.5, 1.0)))
        d = s["diff"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FL − LOC (macro)", ci_txt(d["FL-LOC"], signed=True))
        c2.metric("Free-rider − FL", ci_txt(d["FR_test-FL"], signed=True))
        c3.metric("CEN − FL", ci_txt(d["CEN-FL"], signed=True))
        c4.metric("FL − Noh (published)", ci_txt(d["FL-BL_noh"], signed=True))
        finding(f"**The base paper's conclusion holds on its own data under our protocol:** joining the federation gives "
                f"no significant average gain ({ci_txt(d['FL-LOC'], signed=True)}), and a free-rider gets almost the same "
                f"model ({ci_txt(d['FR_test-FL'], signed=True)}). Per hospital, {s['n_gain_sig']} hospitals gain "
                f"significantly and {s['n_loss_sig']} lose (Spearman of gain vs size {fmt(s['gain_vs_size_spearman'], 2)}).")
        finding(f"**Honest negative:** the published Noh calculator (trained in Korea, no federation) is not beaten on "
                f"hospital-averaged AUC ({ci_txt(d['FL-BL_noh'], signed=True)}); on the patient-weighted AUC FL is ahead "
                f"({fmt(s['weighted']['FL']['mean'])} vs {fmt(s['weighted']['BL_noh']['mean'])}).", "warning")
        ph = pd.DataFrame(s["hospitals"])
        tbl = pd.DataFrame({"Hospital": ph["hospital"], "Country": ph["country"], "Patients": ph["n"],
                            "csPCa": [fmt_cell(v, "{:.0%}") for v in ph["prevalence"]],
                            "LOC": [fmt_cell(v, "{:.3f}") for v in ph["LOC"]], "FL": [fmt_cell(v, "{:.3f}") for v in ph["FL"]],
                            "FR (100%)": [fmt_cell(v, "{:.3f}") for v in ph["FR_all"]],
                            "Noh": [fmt_cell(v, "{:.3f}") for v in ph["BL_noh"]],
                            "FL − LOC [95% CI]": [ci_txt(x, signed=True) for x in ph["FL_minus_LOC"]],
                            "FR calibration O/E": [fmt_cell(x["oe"], "{:.2f}") for x in ph["FR_cal"]]})
        st.dataframe(tbl, hide_index=True)
        lc = s["learning_curve"]
        K = [x["K"] for x in lc]
        show(ch.lines(K, [{"name": "Participants", "color": ch.C["FL"], "mean": [x["FL"]["mean"] for x in lc],
                           "low": [x["FL"]["ci_low"] for x in lc], "high": [x["FL"]["ci_high"] for x in lc]},
                          {"name": "Free-riders", "color": ch.C["FR"], "mean": [x["FR"]["mean"] for x in lc],
                           "low": [x["FR"]["ci_low"] for x in lc], "high": [x["FR"]["ci_high"] for x in lc]}],
                      "AUC as more hospitals join (random order, 5 orders × seeds)", "hospitals training the model",
                      "mean per-hospital AUC", clip=(0, 1)))

    with tabs[3]:
        lo = R["pca19_loco"]
        st.markdown("**Leave-one-country-out**: train on 7 countries, test on **all** patients of the 8th. This is "
                    "external validation the base paper did not do. We report discrimination (AUC), calibration "
                    "(observed/expected ratio, calibration slope; ideal = 1) and clinical usefulness (net benefit).")
        rows = []
        for r in lo["countries"]:
            row = {"Country": r["country"], "Patients": r["n"], "csPCa": fmt_cell(r["prevalence"], "{:.0%}")}
            for m, lab in [("FedAvg", "FedAvg"), ("FedProx+DP(eps=5)", "FedProx+DP ε=5"), ("Noh (published)", "Noh")]:
                row[f"{lab} AUC"] = fmt_cell(r[m]["auc"], "{:.3f}")
                row[f"{lab} O/E"] = fmt_cell(r[m]["oe"], "{:.2f}")
                row[f"{lab} slope"] = fmt_cell(r[m]["slope"], "{:.2f}")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        bc = lo.get("diff_by_country", {})
        if bc:
            c1, c2, c3 = st.columns(3)
            c1.metric("FedAvg − Noh, AUC (8 countries)", ci_txt(bc["FedAvg - Noh (published)"], signed=True))
            c2.metric("DP ε=5 − no DP, AUC", ci_txt(bc["FedProx+DP(eps=5) - FedProx"], signed=True))
            c3.metric("FedAvg − centralised, AUC", ci_txt(bc["FedAvg - Centralised"], signed=True))
            st.caption("Each of the 8 countries is one independent external test (n = 8); seed-level intervals only "
                       "reflect which training patients were drawn and are therefore much narrower.")
        sl = lo.get("calibration_slope_by_model", {})
        if sl:
            finding(f"**Discrimination transports, calibration does not.** All models reach similar external AUC, but "
                    f"observed/expected ratios range widely between countries. DP noise makes predictions too extreme: "
                    f"mean calibration slope {fmt(sl['FedProx']['mean'], 2)} without DP vs "
                    f"{fmt(sl['FedProx+DP(eps=5)']['mean'], 2)} at ε = 5 and {fmt(sl['FedProx+DP(eps=1)']['mean'], 2)} at ε = 1.",
                    "warning")
        rc = lo.get("recalibration")
        if rc:
            st.markdown("**Fix: local recalibration.** A new hospital uses 20% of its own patients to re-estimate the "
                        "model's intercept (baseline risk) or intercept + slope; the other 80% evaluate it. AUC is unchanged by "
                        "construction.")
            rr = []
            for m in ("FedAvg", "FedProx+DP(eps=5)", "FedProx+DP(eps=1)", "Noh (published)"):
                for meth, lab in [("raw", "as transported"), ("intercept", "intercept re-estimated"),
                                  ("logistic", "intercept + slope re-estimated")]:
                    x = rc[m][meth]
                    rr.append({"Model": m, "Recalibration": lab, "|log O/E| (0 = ideal)": ci_txt(x["abs_log_oe"]),
                               "Calibration slope (1 = ideal)": ci_txt(x["slope"], 2), "Brier": ci_txt(x["brier"])})
            st.dataframe(pd.DataFrame(rr), hide_index=True)
        th = lo["thresholds"]
        ser = [{"name": n, "color": col, "mean": lo["dca"][n]} for n, col in
               [("FedAvg", ch.C["FL"]), ("FedProx+DP(eps=5)", ch.C["FedProx"]), ("Noh (published)", NOH_C),
                ("Biopsy all", ch.C["reference"])]]
        ser[-1]["dash"] = "--"
        show(ch.lines(th, ser, "Decision curve: net benefit on held-out countries", "risk threshold for biopsy",
                      "net benefit", ref=0, ref_label="biopsy none"))
        st.caption("Net benefit = true positives − false positives × threshold odds, per patient. A model is useful "
                   "where its curve is above both 'biopsy all' and 'biopsy none'.")

    with tabs[4]:
        pv = R["pca19_privacy"]
        cells = pv["cells"]
        eps = [x["epsilon"] for x in cells if x["epsilon"] is not None]
        nod = [x for x in cells if x["epsilon"] is None][0]
        dp = [x for x in cells if x["epsilon"] is not None]
        ser = [{"name": n, "color": col, "dash": dash, "mean": [x[k]["macro"]["mean"] for x in dp],
                "low": [x[k]["macro"]["ci_low"] for x in dp], "high": [x[k]["macro"]["ci_high"] for x in dp]}
               for k, n, col, dash in [("FedAvg", "FedAvg", ch.C["FedAvg"], "-"), ("FedProx", "FedProx", ch.C["FedProx"], "-"),
                                       ("FedAvg_half_lr", "FedAvg (lr/2)", ch.C["FedAvg"], ":")]]
        show(ch.lines(eps, ser, "Mean per-hospital AUC vs privacy budget ε (19 real hospitals)", "ε (smaller = more private)",
                      "AUC", xlog=True, ref=nod["FedAvg"]["macro"]["mean"], ref_label="no DP"))
        rows = [{"ε": "no DP" if x["epsilon"] is None else f"{x['epsilon']:g}",
                 "σ": "—" if x["epsilon"] is None else f"{pv['sigma'][str(x['epsilon'])]:.1f}",
                 "FedAvg": ci_txt(x["FedAvg"]["macro"]), "FedProx": ci_txt(x["FedProx"]["macro"]),
                 "FedAvg (lr/2)": ci_txt(x["FedAvg_half_lr"]["macro"]),
                 "FedAvg − no DP": "—" if x["epsilon"] is None else ci_txt(x["FedAvg-noDP"], signed=True),
                 "FedProx − FedAvg(lr/2)": ci_txt(x["FedProx-FedAvg_half_lr"], signed=True),
                 "drift FedAvg": fmt(x["FedAvg"]["drift"]["mean"]), "drift FedProx": fmt(x["FedProx"]["drift"]["mean"])}
                for x in cells]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        worst = [x for x in dp if x["epsilon"] == min(eps)][0]
        finding(f"**With 5,610 patients, privacy is almost free.** Even at ε = {min(eps):g} the mean per-hospital AUC changes by "
                f"{ci_txt(worst['FedAvg-noDP'], signed=True)}. On the 347-patient TCGA cohort ε = 1 cost 0.06 AUC — the "
                "cost of DP depends on how many patients each hospital has.", "success")
        finding(f"**FedProx reduces client drift by {fmt(pv['drift_reduction_pct']['mean'], 1)}%**, but at every ε it equals "
                "FedAvg with half the learning rate — the step-size explanation found on TCGA holds on real hospitals.")

    with tabs[5]:
        pf = R["pca19_personalisation"]["groups"]
        gs = ["small (< 100)", "medium (100-399)", "large (>= 400)", "all"]
        show(ch.grouped_bars([g.split(" (")[0] if g != "all" else "all" for g in gs], ["Local", "FedProx", "PFL"],
                             [[pf[g][m]["mean"] for g in gs] for m in ("Local", "FedProx", "PFL")],
                             [ch.C["Local"], ch.C["FedProx"], ch.C["PFL"]], "Mean per-hospital AUC by hospital size",
                             "AUC", ylim=(0.6, 0.95)))
        st.dataframe(pd.DataFrame([{"Hospitals": g, "n hospitals": len(pf[g]["hospitals"]),
                                    "PFL − Local": ci_txt(pf[g]["PFL-Local"], signed=True),
                                    "PFL − FedProx": ci_txt(pf[g]["PFL-FedProx"], signed=True),
                                    "FedProx − Local": ci_txt(pf[g]["FedProx-Local"], signed=True)} for g in gs]),
                     hide_index=True)
        finding(f"**Personalisation helps exactly where it should — small hospitals:** PFL − FedProx = "
                f"{ci_txt(pf['small (< 100)']['PFL-FedProx'], signed=True)} for hospitals with < 100 patients.", "success")

    with tabs[6]:
        if "pca19_shapley" not in R:
            st.info("`pca19_shapley.json` not computed yet (run `python run_pca19.py --only shapley`).")
        else:
            sh = R["pca19_shapley"]
            ex, mc = sh["exact10"], sh["mc19"]
            st.markdown(f"**Exact Shapley** over the {sh['K10']} largest hospitals (all {2 ** sh['K10'] - 1} coalitions) and "
                        f"**stratified-sampling Shapley** over all 19 ({mc['coalitions_per_seed']} coalitions per seed; "
                        "validated against exact Shapley). Rankings with DP are compared with the no-DP ranking.")
            rows = []
            for lab, block in (("10 hospitals (exact)", ex), ("19 hospitals (sampled)", mc)):
                for x in block["per_epsilon"]:
                    rows.append({"Hospitals": lab, "ε": f"{x['epsilon']:g}", "Spearman ρ": ci_txt(x["spearman"], 2),
                                 "Rank reversal": ci_txt(x["rank_reversal"], 2),
                                 "Top contributor kept": ci_txt(x["top_contributor_stability"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            hs = ex["hospitals"]
            show(ch.grouped_bars(hs, ["Shapley", "Leave-one-out"], [[ex["phi_nodp"][h]["mean"] for h in hs],
                                                                   [ex["loo_nodp"][h]["mean"] for h in hs]],
                                 [ch.C["FedAvg"], ch.C["FedProx"]], "Contribution of the 10 largest hospitals (no DP)",
                                 "Δ AUC", ref=0))
            c1, c2 = st.columns(2)
            c1.metric("Spearman(Shapley, LOO)", ci_txt(ex["spearman_shapley_vs_loo"], 2))
            c2.metric("Spearman(Shapley, size), 19 hospitals", fmt(mc["spearman_shapley_vs_size"], 2))

        pvr = R.get("pca19_private_valuation")
        if pvr:
            st.markdown("---")
            st.markdown("#### Privacy-honest valuation: OSA-Shapley (our method)")
            st.markdown(
                "Retraining every coalition under DP (the table above) releases 2^(K−1) private models that involve each "
                "hospital, so its **real** privacy cost is far above the per-run ε. **OSA-Shapley** spends the budget once: each "
                "hospital releases one DP-trained local model, every coalition model is the size-weighted average of its "
                "members' models (post-processing, no extra privacy), and exact Shapley values follow from all coalitions.")
            rows = []
            for x in pvr["rows"]:
                rows.append({"ε": f"{x['epsilon']:g}", "real ε of per-run retraining": f"{x['retrain_true_epsilon']:.0f}",
                             "ρ per-run retraining": ci_txt(x["retrain_per_run"]["spearman"], 2),
                             "ρ budget-matched retraining": ci_txt(x["retrain_budget_matched"]["spearman"], 2),
                             "ρ OSA-Shapley": ci_txt(x["osa"]["spearman"], 2),
                             "top kept (OSA)": fmt(x["osa"]["top_contributor_stability"]["mean"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("ρ = Spearman correlation with the exact non-private Shapley ranking of the 10 largest hospitals "
                       f"({len(pvr['seeds'])} held-out seeds). Budget-matched = retraining with noise calibrated so the total "
                       "cost equals ε.")
            r2 = {x["epsilon"]: x for x in pvr["rows"]}
            finding(f"At the same real privacy cost, OSA-Shapley keeps the ranking (ρ = {fmt(r2[2.0]['osa']['spearman']['mean'], 2)} "
                    f"at ε = 2) while budget-matched retraining loses it (ρ = {fmt(r2[2.0]['retrain_budget_matched']['spearman']['mean'], 2)}). "
                    f"Per-run retraining looks stable only because it spends ε ≈ {r2[2.0]['retrain_true_epsilon']:.0f}.", "success")
            k19 = pvr["k19"]
            st.markdown(f"**All 19 hospitals, exact OSA-Shapley (524,287 coalitions):** Spearman with the no-DP OSA ranking "
                        + ", ".join(f"{ci_txt(x['spearman'], 2)} at ε = {x['epsilon']:g}" for x in k19["per_epsilon"]) + ".")
            hs = k19["hospitals"]
            show(ch.bar_with_ci(hs, [k19["phi_nodp"][h] for h in hs], [ch.C["FedAvg"]] * len(hs),
                                "OSA-Shapley contribution of every hospital (no DP, Δ AUC)", ylabel="Shapley value", fmt="{:.3f}"))

        vp = R.get("pca19_valuation_plus")
        if vp:
            st.markdown("---")
            st.markdown("#### OSA-Shapley vs other valuations that also cost ε once")
            st.markdown(
                "DP-FedAvg already releases every hospital's noisy local model in every round, so coalition models can also be "
                "**reconstructed** from one private training run at no extra cost. **MR** (multi-round reconstruction) is the exact "
                "version of GTG-Shapley; **OR** (one-round reconstruction) accumulates all updates. These are the fair competitors.")
            keys = ["none", "5.0", "2.0", "1.0", "0.5"]
            rows = []
            for k in keys:
                a, t = vp["methods"][k], vp["tcga"][k]
                rows.append({"ε": "no DP" if k == "none" else f"{float(k):g}",
                             "PCa-19 OSA": fmt(a["osa"]["mean"], 2), "PCa-19 MR": fmt(a["mr"]["mean"], 2),
                             "PCa-19 OR": fmt(a["or"]["mean"], 2), "PCa-19 OSA − MR": ci_txt(vp["osa_minus_mr"][k], 2),
                             "TCGA OSA": fmt(t["osa"]["mean"], 2), "TCGA MR": fmt(t["mr"]["mean"], 2),
                             "TCGA OSA − MR": ci_txt(vp["tcga_osa_minus_mr"][k], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Spearman ρ with the exact non-private retraining Shapley ranking, ten hospitals. PCa-19: hospitals of "
                       "200–763 patients; TCGA: ten simulated hospitals of about 28 patients.")
            finding(f"On real hospitals OSA-Shapley beats reconstruction at the same cost (ε = 2: OSA − MR = "
                    f"{ci_txt(vp['osa_minus_mr']['2.0'], 2)}). On tiny TCGA silos reconstruction is at least as good, as the noise "
                    "bound (Proposition 3: noise ∝ 1/n_k) predicts.", "success")
            c = vp["corrupt"]
            st.markdown(f"**A hospital with useless data.** The training labels of {vp['corrupt_hospital']} were replaced by random labels. "
                        "Share of splits in which each valuation ranks it last:")
            names = {"osa": "OSA (ε)", "mr": "MR (ε)", "or": "OR (ε)", "retrain_budget_matched": "Retraining, budget-matched (ε)",
                     "retrain_per_run": "Retraining, per run (real ε 85 / 33)"}
            st.dataframe(pd.DataFrame([{"method": nm, "ε = 2": f"{c['2.0'][k]['last']['mean']:.0%}",
                                        "ε = 1": f"{c['1.0'][k]['last']['mean']:.0%}"} for k, nm in names.items()]), hide_index=True)
            sn = vp["sensitivity"]
            st.markdown("**Local steps** 10 / 30 / 60 at ε = 2: "
                        + " / ".join(fmt(sn[f'steps={s_},eps=2.0']['mean'], 2) for s_ in (10, 30, 60)) + ".")

        vm = R.get("pca19_valuation_mlp")
        if vm:
            st.markdown("---")
            st.markdown("#### Non-convex model: valuing hospitals for a federated MLP")
            rows = []
            for k in ["none", "5.0", "2.0", "1.0", "0.5"]:
                r = vm["rho"][k]; bm = vm["budget_matched"].get(k)
                rows.append({"ε": "no DP" if k == "none" else f"{float(k):g}",
                             "OSA (ensemble)": fmt(r["osa_ensemble"]["mean"], 2), "OSA (parameter avg.)": fmt(r["osa_param"]["mean"], 2),
                             "MR reconstruction": fmt(r["mr"]["mean"], 2), "OSA − MR": ci_txt(vm["osa_ensemble_minus_mr"][k], 2),
                             "Budget-matched retraining": fmt(bm["retrain_budget_matched"]["mean"], 2) if bm else "–"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Spearman ρ with exact non-private retraining Shapley of the federated MLP (16 tanh units), ten largest "
                       f"hospitals, {len(vm['seeds'])} held-out splits; budget-matched retraining on "
                       f"{vm['budget_matched']['2.0']['n_seeds']} splits.")
            finding("With a non-convex model, OSA-Shapley is excellent without DP but loses to reconstruction under DP "
                    f"(ε = 2: OSA − MR = {ci_txt(vm['osa_ensemble_minus_mr']['2.0'], 2)}). Rule: OSA for convex risk models in "
                    "large hospitals; reconstruction for non-convex models and tiny silos.", "warning")

    with tabs[7]:
        m = R["pca19_mia"]
        rows = m["rows"]
        show(ch.bar_with_ci(["no DP" if x["epsilon"] is None else f"ε={x['epsilon']:g}" for x in rows] + ["over-fitted\ncontrol"],
                            [x["attack_auc"] for x in rows] + [m["control"]],
                            [ch.C["FedProx"]] * len(rows) + [ch.C["reference"]],
                            "Membership-inference attack AUC (0.5 = chance)", ylim=(0.4, 1.0), ref=0.5))
        finding(f"The attack is at chance for every model ({fmt(rows[0]['attack_auc']['mean'])} without DP) while it "
                f"detects the over-fitted control ({fmt(m['control']['mean'])}). A 7-parameter model trained on about 4,500 "
                "patients does not memorise individuals; DP still matters as a guarantee against all attacks.")


    with tabs[8]:
        bl = R.get("pca19_baselines")
        if not bl:
            st.info("`pca19_baselines.json` not computed yet (run `python run_pca19.py --only baselines`).")
        else:
            st.markdown("SCAFFOLD (control variates), FedAdam (adaptive server optimiser), a federated one-hidden-layer MLP, "
                        "a converged FedAvg (400 local steps) and centralised non-linear references. FedNova equals FedAvg here "
                        "because every hospital runs the same number of local steps.")
            rows = []
            for mname in bl["models"]:
                w_ = bl["within"][mname]; lo_ = bl["loco"][mname]
                dw = bl["within_diff_vs_FedAvg"].get(mname); dl = bl["loco_diff_vs_FedAvg_by_country"].get(mname)
                dp = bl["within_dp_eps1"].get(mname)
                rows.append({"Model": mname, "Within-hospital AUC": ci_txt(w_["macro"]),
                             "− FedAvg": "—" if dw is None else ci_txt(dw, signed=True),
                             "Unseen-country AUC": ci_txt(lo_["auc"]),
                             "− FedAvg (8 countries)": "—" if dl is None else ci_txt(dl, signed=True),
                             "Calibration slope": fmt(lo_["slope"]["mean"], 2),
                             "AUC with DP ε=1": "—" if dp is None else ci_txt(dp)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            d = bl["drift"]
            st.caption(f"Client drift (no DP): FedAvg {fmt(d['FedAvg']['mean'])}, FedProx {fmt(d['FedProx']['mean'])}, "
                       f"SCAFFOLD {fmt(d['SCAFFOLD']['mean'])}. Hyper-parameters of the new baselines were tuned on seeds 0–3 and "
                       f"evaluated on seeds 20–39: {bl['tuned']}.")
        bt = R.get("pca19_baselines_tcga")
        if bt:
            st.markdown("**TCGA-PRAD, 10 Dirichlet hospitals** (global test AUC): " + "; ".join(
                f"α = {a}: " + ", ".join(f"{m_} {fmt(bt[a][m_]['mean'])}" for m_ in bt[a]) for a in ("1.0", "0.1")))

    with tabs[9]:
        ss = R.get("pca19_subsampling")
        if not ss:
            st.info("`pca19_subsampling.json` not computed yet (run `python run_pca19.py --only subsampling`).")
        else:
            st.markdown("Poisson subsampling: each patient joins each local step with probability q; privacy is accounted with "
                        "the Rényi-DP bound of the sampled Gaussian mechanism (validated against numerical integration and the "
                        "ε = 1.26 example of Abadi et al.). Learning rates for q < 1 were tuned on seeds 0–3.")
            rows = []
            for x in ss["rows"]:
                rows.append({"Sampling": x["config"], "ε": "no DP" if x["epsilon"] is None else f"{x['epsilon']:g}",
                             "σ": "—" if x["sigma"] is None else f"{x['sigma']:.1f}", "local steps": x["steps"],
                             "TCGA AUC": ci_txt(x["tcga_auc"]),
                             "vs full batch": ci_txt(x["tcga_auc_vs_full"], signed=True) if "tcga_auc_vs_full" in x else "—",
                             "PCa-19 AUC": ci_txt(x["pca19_macro"]), "PCa-19 slope": fmt(x["pca19_slope"]["mean"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)


    with tabs[10]:
        hv = R.get("pca19_heart_valuation")
        if not hv:
            st.info("`pca19_heart_valuation.json` not computed yet (run `python run_pca19.py --only heart_valuation`).")
        else:
            st.markdown("A second real federation: the four hospitals of the UCI Heart Disease collection "
                        f"({hv['n']} patients, coronary artery disease, {hv['n_features']} features), "
                        f"{len(hv['seeds'])} held-out splits. Reference: exact non-private retraining Shapley values.")
            st.dataframe(pd.DataFrame([{"Hospital": s_["name"], "Patients": s_["n"], "Disease rate": f"{s_['prevalence']:.0%}"}
                                       for s_ in hv["sites"]]), hide_index=True)
            names = {"size": "Hospital size", "statistics": "Summary statistics", "own_test_auc": "Self-reported AUC",
                     "ref_auc": "Single-model AUC", "osa_loo": "Leave-one-out (averaged)", "mr": "MR reconstruction",
                     "or": "OR reconstruction", "retrain_per_run": "Retraining, per run",
                     "retrain_budget_matched": "Retraining, budget-matched", "osa": "OSA-Shapley (ours)"}
            rows = []
            for key, lab in names.items():
                row = {"Score": lab}
                for e in ["none", "5.0", "2.0", "1.0"]:
                    v = hv["rho"][e].get(key)
                    row["no DP" if e == "none" else f"ε = {float(e):g}"] = "–" if v is None else fmt(v["mean"], 2)
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Spearman ρ with the exact ranking; with four hospitals ρ moves in steps of 0.2.")
            finding("Without DP OSA-Shapley recovers the ranking (OSA − MR = "
                    f"{ci_txt(hv['osa_minus']['none']['mr'], 2)}); under DP it loses to reconstruction "
                    f"(ε = 2: {ci_txt(hv['osa_minus']['2.0']['mr'], 2)}) and at ε = 1 nothing works. Hospital size is "
                    f"anti-correlated with contribution (ρ = {fmt(hv['rho']['none']['size']['mean'], 2)}).", "warning")

    with tabs[11]:
        ns = R.get("pca19_nu_sweep"); sc = R.get("pca19_nu_sweep_scoring")
        if not ns or not sc:
            st.info("`pca19_nu_sweep.json` not computed yet (run `python run_pca19.py --only nu_sweep`).")
        else:
            st.markdown("**Rule tested:** OSA-Shapley beats MR reconstruction under DP when ν = median training patients per "
                        "hospital / √(parameters) is about 100 or more, and stops beating it at about 50 or less. The "
                        "predictions were written and hashed (`results/preregistration_nu_sweep.md`) before the sweep ran.")
            rows = []
            for c_ in sorted(ns["conditions"], key=lambda c_: -c_["nu"]):
                rows.append({"Arm": "fewer patients" if c_["arm"] == "A" else "more parameters",
                             "Patients kept": f"{c_['fraction']:.0%}", "Extra features": c_["extra_features"],
                             "Parameters": c_["p"], "ν": f"{c_['nu']:.0f}",
                             "OSA − MR (ε = 2)": ci_txt(c_["2.0"]["osa_minus_mr"], 2),
                             "OSA − MR (ε = 1)": ci_txt(c_["1.0"]["osa_minus_mr"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            hyp = {"H1": "OSA wins at ν ≥ 100", "H2": "OSA does not win at ν ≤ 50", "H3": "advantage grows with ν",
                   "H4": "switch-over between ν 50 and 100", "H5": "both arms agree", "H6": "ε = 1 moves the switch-over up"}
            st.dataframe(pd.DataFrame([{"Hypothesis": f"{k}: {v}", "Result": "held" if sc[k] else "failed"}
                                       for k, v in hyp.items()]), hide_index=True)
            finding(f"4 of 6 predictions held. The direction is right, but the switch-over is at ν ≈ "
                    f"{sc['crossover_eps2'][0]:.0f} (ε = 2), not 50–100. Revised rule: for a convex model under DP use "
                    "OSA-Shapley down to ν ≈ 30 (ε = 2) or 40 (ε = 1); for non-convex models use reconstruction.", "info")


# ============================================================================= 3. OUR MODEL
def page_ours():
    need("centralized", "drift", "privacy_grid", "personalization", "shapley", "mia", "shift", "ablation")
    st.title("Our model — DP-FedProx with Shapley valuation (DP-FPS)")
    st.markdown(
        "Cohort: **347 patients** with clinical + protein data (277 train / 70 test per seed). Hospitals are "
        "simulated with **Dirichlet label skew** (small α = very different T3/T4 rates per hospital). Model: "
        "logistic regression trained with full-batch gradient descent, 20 rounds × 5 local epochs, learning rate "
        "0.5, FedProx μ = 0.5. Every number is a mean over 20 seeds.")
    with st.expander("Method in one screen"):
        st.latex(r"\text{Local step:}\ w \leftarrow w - \eta\Big(\tfrac{1}{n_k}\big[\textstyle\sum_i "
                 r"\mathrm{clip}_C(\nabla\ell_i(w)) + \mathcal{N}(0,\sigma^2C^2I)\big] + \lambda w + \mu (w-w^{t})\Big)")
        st.latex(r"\text{RDP accountant:}\ \varepsilon = \min_{\alpha>1}\ \frac{T\alpha}{2\sigma^2} + "
                 r"\frac{\log(1/\delta)}{\alpha-1},\quad T = 20\times5 = 100,\ \delta=10^{-5}")
        st.latex(r"\text{Shapley:}\ \phi_k = \sum_{S\subseteq N\setminus\{k\}} \frac{|S|!\,(K-|S|-1)!}{K!}"
                 r"\big[v(S\cup\{k\}) - v(S)\big],\quad v(S) = \text{test AUC of FedAvg trained on } S")
        st.markdown("Privacy unit: one patient (one sample per patient). Each gradient is clipped to C = 1 and "
                    "every patient is used once per local epoch (sampling rate q = 1); hospitals hold disjoint "
                    "patients, so the guarantee composes over T = 100 steps. μ = 0 gives FedAvg.")
    tabs = st.tabs(["Features & baselines", "Heterogeneity & drift", "Privacy vs utility", "Personalisation",
                    "Contribution valuation", "Privacy attack", "Stress test", "Ablation"])

    # --- features
    with tabs[0]:
        c = R["centralized"]
        df = pd.DataFrame(c["table"])
        order = ["Clinical (full)", "Clinical (pre-op only)", "Protein only (PCA)", "Clinical (full) + protein",
                 "Pre-op + protein"]
        piv = df.pivot(index="model", columns="features", values="mean")[order]
        st.markdown(f"Centralised models, **repeated 5×5 stratified cross-validation** ({c['n_folds']} folds) on the "
                    "347-patient cohort. Every preprocessing step (imputation, scaling, protein filtering, PCA) is "
                    "fitted inside each training fold.")
        show(ch.heatmap(piv.values, list(piv.index), [o.replace(" + ", "\n+ ").replace(" (", "\n(") for o in order],
                        "Mean cross-validated AUC", fmt="{:.3f}", vlim=(0.5, 0.85)))
        cmp = c["comparisons"]
        a1, a2, a3 = st.columns(3)
        a1.metric("Protein added to full clinical", ci_txt(cmp["protein_added_to_full"], signed=True))
        a2.metric("Full vs pre-op clinical", ci_txt(cmp["full_vs_preop"], signed=True))
        a3.metric("Protein added to pre-op", ci_txt(cmp["protein_added_to_preop"], signed=True))
        st.caption("Differences are for the NumPy logistic model used in federation (fold-paired).")
        finding("**Proteomics does not help.** With ~115 protein components for 277 training patients the "
                "logistic model over-fits, and adding them lowers AUC, so the main model uses clinical features. "
                "Gleason grade and pathologic N carry most of the signal. They are post-surgical and come from the "
                "same pathology report as the label, so the pre-operative AUC (~0.65) is the realistic number "
                "for a tool used before surgery.", "warning")
        with st.expander("Table view"):
            st.dataframe(df.assign(**{"95% CI": [f"[{fmt(a)}, {fmt(b)}]" for a, b in zip(df.ci_low, df.ci_high)]})
                         [["features", "model", "mean", "sd", "95% CI", "dim_last_fold"]],
                         hide_index=True)
            st.caption(c["note"])

    # --- drift
    with tabs[1]:
        dr = R["drift"]
        rows = dr["rows"]
        lab = [f"α={r['alpha']:g}" for r in rows]
        show(ch.grouped_bars(lab, ["FedAvg", "FedProx"], [[r["fedavg_drift"]["mean"] for r in rows],
                                                          [r["fedprox_drift"]["mean"] for r in rows]],
                             [ch.C["FedAvg"], ch.C["FedProx"]],
                             "Client drift mean‖w_k − w_global‖ (small α = more heterogeneous)", "drift (L2)",
                             errs=[[_e(r["fedavg_drift"]) for r in rows], [_e(r["fedprox_drift"]) for r in rows]]))
        st.dataframe(pd.DataFrame({
            "α": lab,
            "T3/T4-rate spread across hospitals": [fmt(r["mean_label_spread"]["mean"], 2) for r in rows],
            "FedAvg drift": [ci_txt(r["fedavg_drift"], 4) for r in rows],
            "FedProx drift": [ci_txt(r["fedprox_drift"], 4) for r in rows],
            "drift reduction %": [ci_txt(r["drift_reduction_pct"], 1) for r in rows],
            "FedAvg AUC": [fmt(r["fedavg_auc"]["mean"]) for r in rows],
            "FedProx AUC": [fmt(r["fedprox_auc"]["mean"]) for r in rows],
            "AUC difference": [ci_txt(r["auc_diff"], 4, True) for r in rows]}), hide_index=True)
        mu = dr["mu_sweep_alpha_0.5"]
        show(ch.lines([m["mu"] for m in mu], [{"name": "drift", "color": ch.C["FedProx"],
                                              "mean": [m["drift"]["mean"] for m in mu],
                                              "low": [m["drift"]["ci_low"] for m in mu],
                                              "high": [m["drift"]["ci_high"] for m in mu]}],
                      "Effect of the proximal coefficient μ on drift (α = 0.5)", "μ", "drift (L2)"))
        red = [r["drift_reduction_pct"]["mean"] for r in rows]
        auc_sig = [r for r in rows if r["auc_diff"]["ci_low"] is not None
                   and (r["auc_diff"]["ci_low"] > 0 or r["auc_diff"]["ci_high"] < 0)]
        sig_alphas = ", ".join(f"{r['alpha']:g}" for r in auc_sig)
        finding(f"**FedProx reliably reduces client drift** by {min(red):.0f}–{max(red):.0f}% at every heterogeneity "
                "level, and drift grows as α shrinks. " +
                ("**Without DP this does not change accuracy** (no α has a significant AUC difference): the logistic "
                 "model is convex and FedAvg already converges to a good solution." if not auc_sig else
                 f"The AUC difference is significant for α = {sig_alphas}."),
                "success")

    # --- privacy
    with tabs[2]:
        g = R["privacy_grid"]
        cells = g["cells"]
        ep_all = [None, 10.0, 5.0, 2.0, 1.0, 0.5]
        xs = [100, 10, 5, 2, 1, 0.5]
        a10 = [cell(cells, 10.0, e) for e in ep_all]
        ser = [{"name": name, "color": color, "dash": dash, "mean": [x[key]["mean"] for x in a10],
                "low": [x[key]["ci_low"] for x in a10], "high": [x[key]["ci_high"] for x in a10]}
               for key, name, color, dash in [("fedavg", "FedAvg", ch.C["FedAvg"], "-"),
                                              ("fedprox", "FedProx", ch.C["FedProx"], "-"),
                                              ("fedavg_half_lr", "FedAvg (lr/2)", ch.C["FedAvg"], "--")]]
        show(ch.lines(xs, ser, "Test AUC vs privacy budget ε (near-IID, α = 10)",
                      "ε (log scale; 'none' = no DP)", "test AUC", xlog=True,
                      xticks=([100, 10, 5, 2, 1, 0.5], ["none", "10", "5", "2", "1", "0.5"]), clip=(0, 1), direct=False))
        st.markdown("Calibrated noise multiplier (T = 100 steps, δ = 1e-5, C = 1): " + ", ".join(
            f"ε = {e}: σ = {float(s):.1f}" for e, s in g["sigma"].items()))
        al = sorted({x["alpha"] for x in cells}, reverse=True)
        ep = [10.0, 5.0, 2.0, 1.0, 0.5]

        def sig(d):
            return "*" if d["ci_low"] is not None and (d["ci_low"] > 0 or d["ci_high"] < 0) else ""
        M = [[cell(cells, a, e)["prox_minus_avg"]["mean"] for e in ep] for a in al]
        S = [[sig(cell(cells, a, e)["prox_minus_avg"]) for e in ep] for a in al]
        vmax = float(np.nanmax(np.abs(M)))
        show(ch.heatmap(M, [f"α={a:g}" for a in al], [f"ε={e:g}" for e in ep],
                        "FedProx − FedAvg test AUC under DP (* = 95% CI excludes 0)", center=0.0, stars=S, vlim=vmax))
        M2 = [[cell(cells, a, e)["prox_minus_halflr_avg"]["mean"] for e in ep] for a in al]
        S2 = [[sig(cell(cells, a, e)["prox_minus_halflr_avg"]) for e in ep] for a in al]
        show(ch.heatmap(M2, [f"α={a:g}" for a in al], [f"ε={e:g}" for e in ep],
                        "Control: FedProx − FedAvg with half the learning rate", center=0.0, stars=S2, vlim=vmax))
        dpc = [x for x in cells if x["epsilon"] is not None]
        n_sig = sum(x["prox_minus_avg"]["ci_low"] > 0 for x in dpc)
        n_ctl = sum(x["prox_minus_halflr_avg"]["ci_low"] > 0 for x in dpc)
        loss1 = cell(cells, 10.0, 1.0)["fedavg_minus_nodp"]["mean"]
        txt = (f"**Privacy costs utility**: at ε = 1 FedAvg loses {abs(loss1):.3f} AUC versus no DP, and the cost grows "
               f"quickly below ε ≈ 2. FedProx beats FedAvg in **{n_sig}/{len(dpc)}** DP cells (95% CI > 0). ")
        if n_ctl <= len(dpc) // 4:
            txt += (f"**But the step-size control explains it**: against a FedAvg with half the learning rate, FedProx "
                    f"wins in only {n_ctl}/{len(dpc)} cells. Under DP the proximal term mainly acts as a smaller "
                    "effective step that damps the injected noise; it is not a heterogeneity effect.")
        else:
            txt += (f"The advantage survives a step-size control in {n_ctl}/{len(dpc)} cells, so it is not only "
                    "a smaller effective learning rate.")
        finding(txt, "info")
        with st.expander("Full grid (table)"):
            st.dataframe(pd.DataFrame([{"α": x["alpha"], "ε": "no DP" if x["epsilon"] is None else f"{x['epsilon']:g}",
                                        "FedAvg": ci_txt(x["fedavg"]), "FedProx": ci_txt(x["fedprox"]),
                                        "FedAvg lr/2": ci_txt(x["fedavg_half_lr"]),
                                        "FedProx − FedAvg": ci_txt(x["prox_minus_avg"], 4, True),
                                        "Wilcoxon p": fmt(x["prox_minus_avg"].get("wilcoxon_p"), 4),
                                        "FedAvg − no DP": ci_txt(x["fedavg_minus_nodp"], 3, True)
                                        if x["fedavg_minus_nodp"] else "—"}
                                       for x in cells]), hide_index=True)

    # --- personalisation
    with tabs[3]:
        p = R["personalization"]
        Ks = sorted(p["by_K"], key=int)
        meth = ["Local", "FedAvg", "FedProx", "PFL"]
        show(ch.grouped_bars([f"{k} hospitals" for k in Ks], ["Local only", "FedAvg", "FedProx", "Personalised (PFL)"],
                             [[p["by_K"][k]["macro"][m]["mean"] for k in Ks] for m in meth],
                             [ch.C[m] for m in meth], "Mean per-hospital test AUC (α = 0.5)", "AUC",
                             errs=[[_e(p["by_K"][k]["macro"][m]) for k in Ks] for m in meth], ylim=(0.5, 0.95)))
        st.dataframe(pd.DataFrame([{"hospitals": k,
                                    "hospitals with defined AUC (mean)": fmt(p["by_K"][k]["clients_with_defined_auc"]["mean"], 1),
                                    "PFL − Local": ci_txt(p["by_K"][k]["PFL_minus_Local"], 3, True),
                                    "PFL − FedProx": ci_txt(p["by_K"][k]["PFL_minus_FedProx"], 3, True),
                                    "FedProx − Local": ci_txt(p["by_K"][k]["FedProx_minus_Local"], 3, True)}
                                   for k in Ks]), hide_index=True)
        g3 = p["by_K"][Ks[0]]
        sig_l = [k for k in Ks if p["by_K"][k]["PFL_minus_Local"]["ci_low"] is not None
                 and p["by_K"][k]["PFL_minus_Local"]["ci_low"] > 0]
        sig_p = [k for k in Ks if p["by_K"][k]["PFL_minus_FedProx"]["ci_low"] is not None
                 and p["by_K"][k]["PFL_minus_FedProx"]["ci_low"] > 0]
        finding(f"**Federated models beat local-only training** for small hospitals (e.g. {Ks[0]} hospitals: FedProx "
                f"{fmt(g3['macro']['FedProx']['mean'])} vs local {fmt(g3['macro']['Local']['mean'])}). Personalised "
                f"fine-tuning beats local-only significantly for K = {', '.join(sig_l) or 'none'}, but it beats the "
                f"global FedProx model for K = {', '.join(sig_p) or 'none'}: an 18-feature logistic model has "
                "little left to personalise.", "info")
        st.caption(p["note"])

    # --- valuation
    with tabs[4]:
        sh = R["shapley"]
        Ks = sorted(sh["by_K"], key=int)
        ep = [10.0, 5.0, 2.0, 1.0, 0.5]
        xs = list(range(len(ep)))
        cols = ["#2a78d6", "#eb6834", "#1baf7a"]
        for metric, title, ylab in [("spearman", "Rank agreement with no-DP Shapley (Spearman ρ)", "ρ"),
                                    ("top_contributor_stability",
                                     "P(no-DP top contributor still ranked first)", "probability")]:
            ser = [{"name": f"{k} hospitals", "color": cols[i % 3],
                    "mean": [x[metric]["mean"] for x in sh["by_K"][k]["per_epsilon"]],
                    "low": [x[metric]["ci_low"] for x in sh["by_K"][k]["per_epsilon"]],
                    "high": [x[metric]["ci_high"] for x in sh["by_K"][k]["per_epsilon"]]} for i, k in enumerate(Ks)]
            show(ch.lines(xs, ser, title, "privacy budget ε (stronger privacy →)", ylab,
                          xticks=(xs, [f"{e:g}" for e in ep]), ylim=(-0.1, 1.05), clip=(-1, 1)))
        st.dataframe(pd.DataFrame([{"hospitals": k, "ε": f"{x['epsilon']:g}", "Spearman ρ": ci_txt(x["spearman"], 2),
                                    "rank reversal": ci_txt(x["rank_reversal"], 2),
                                    "top-contributor stability": ci_txt(x["top_contributor_stability"], 2),
                                    "random baseline": f"{1 / int(k):.2f}",
                                    "Wilcoxon p (ρ < 0.90)": fmt(x["wilcoxon_p_rho_below_0.90"], 4)}
                                   for k in Ks for x in sh["by_K"][k]["per_epsilon"]]),
                     hide_index=True)
        cc = st.columns(len(Ks))
        for col, k in zip(cc, Ks):
            col.metric(f"Spearman(Shapley, LOO), {k} hospitals", ci_txt(sh["by_K"][k]["spearman_shapley_vs_loo"], 2))
        finding("**DP noise scrambles contribution rankings.** Agreement with the no-DP ranking falls steadily as ε "
                "shrinks and is far below 0.9 even at ε = 10. With more hospitals the overall rank correlation "
                "degrades about as much, but exact ranks and the identity of the top contributor become less stable. "
                "A payment scheme based on Shapley values from DP-trained models would often reward the wrong "
                "hospital. (Top-contributor stability uses the corrected definition; the old one was always 1/K.)",
                "warning")
        st.caption(f"Exact Shapley over all coalitions; 20 partitions × "
                   f"{', '.join(str(sh['by_K'][k]['realizations_per_seed']) for k in Ks)} DP-noise realisations "
                   f"for {', '.join(Ks)} hospitals; efficiency axiom checked (max error "
                   f"{max(sh['by_K'][k]['efficiency_check_max_abs'] for k in Ks):.1e}).")

    # --- MIA
    with tabs[5]:
        m = R["mia"]
        rows = m["rows"]
        names = ["no DP" if r["epsilon"] == "None" else f"ε={float(r['epsilon']):g}" for r in rows]
        show(ch.bar_with_ci(names + ["over-fitted\ncontrol"], [r["attack_auc"] for r in rows] +
                            [m["overfit_reference"]["attack_auc"]],
                            [ch.C["FedAvg"]] * len(rows) + [ch.C["reference"]],
                            "Membership-inference attack AUC (0.5 = attacker is guessing)", ylabel="attack AUC",
                            ylim=(0.3, 0.85), ref=0.5, ref_label="chance"))
        st.dataframe(pd.DataFrame([{"privacy": n, "attack AUC": ci_txt(r["attack_auc"]),
                                    "train − test AUC gap": ci_txt(r["generalisation_gap"], 3, True),
                                    "model test AUC": fmt(r["test_auc"]["mean"])}
                                   for n, r in zip(names, rows)]), hide_index=True)
        finding("**The attack is weak against this model even without DP** (attack AUC close to 0.5), because an "
                "18-feature logistic regression barely memorises. DP's formal guarantee still holds, but its "
                "*empirical* benefit cannot be shown with this attack. The over-fitted control shows the attack "
                "does work when a model memorises. (The old pipeline's higher no-DP attack AUC came from the "
                "identifier features, which let the model memorise patients.)", "info")
        st.caption(m["note"])

    # --- shift
    with tabs[6]:
        s = R["shift"]
        sev = s["severities"]
        show(ch.lines(sev, [{"name": "covariate shift P(X)", "color": ch.C["FedAvg"],
                             "mean": [x["mean"] for x in s["covariate"]], "low": [x["ci_low"] for x in s["covariate"]],
                             "high": [x["ci_high"] for x in s["covariate"]]},
                            {"name": "concept shift P(Y|X)", "color": ch.C["FedProx"],
                             "mean": [x["mean"] for x in s["concept"]], "low": [x["ci_low"] for x in s["concept"]],
                             "high": [x["ci_high"] for x in s["concept"]]}],
                      "FedProx global model under synthetic shift of the test split", "severity", "test AUC",
                      ref=0.5, ref_label="chance", clip=(0, 1)))
        st.caption(s["note"])

    # --- ablation
    with tabs[7]:
        a = R["ablation"]
        st.dataframe(pd.DataFrame([{"configuration": r["configuration"], "test AUC": ci_txt(r),
                                    "vs FedProx": ci_txt(r["vs_FedProx"], 3, True) if r["vs_FedProx"] else "reference"}
                                   for r in a["rows"]]), hide_index=True)
        st.caption("Global test split, 3 hospitals, α = 0.5, 20 seeds. Local-only and personalised models are "
                   "compared per hospital in the Personalisation tab.")


# ============================================================================= 4. FINAL RESULTS
def page_final():
    need("base_paper", "ablation", "privacy_grid")
    st.title("Final results")
    bp = R["base_paper"]
    ab = {r["configuration"]: r for r in R["ablation"]["rows"]}
    st.subheader("Base paper approach vs our framework")
    comp = pd.DataFrame([
        ["Data", "Clinical risk factors, 19 real hospitals", "The same 19 hospitals (PCa-19, primary) + TCGA-PRAD "
         "(secondary: real sites and Dirichlet-skewed hospitals)"],
        ["External validation", "None (within-hospital splits)", "Leave-one-country-out, calibration, decision curves, "
         "local recalibration"],
        ["Aggregation", "FedAvg", "FedAvg and FedProx (μ = 0.5)"],
        ["Privacy", "None (plain weight sharing)", "DP-SGD, per-patient clipping, Rényi-DP accountant (ε 0.5–10, δ = 1e-5)"],
        ["Heterogeneity", "Natural (hospital differences)", "Natural sites + controlled Dirichlet label skew"],
        ["Contribution valuation", "None", "Exact Shapley vs leave-one-out; stability under DP"],
        ["Personalisation", "None", "Proximal fine-tuning of the global model"],
        ["Privacy audit", "None", "Membership-inference attack with a positive control"],
        ["Statistics", "Single runs / heatmaps", "20 seeds, paired 95% CIs, Wilcoxon tests"],
    ], columns=["", "Base paper (Kazlouski et al.)", "Ours (DP-FPS)"])
    st.dataframe(comp, hide_index=True)

    st.subheader("Headline numbers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base-paper FL, per-hospital AUC (TCGA sites)", fmt(bp["macro"]["FL"]["mean"]))
    c2.metric("Our FedProx, test AUC (no DP)", fmt(ab["FedProx"]["mean"]))
    c3.metric("Our FedProx + DP ε = 5", fmt(ab["FedProx + DP (eps=5)"]["mean"]),
              f"{ab['FedProx + DP (eps=5)']['vs_FedProx']['mean']:+.3f} vs no DP")
    c4.metric("Our FedProx + DP ε = 1", fmt(ab["FedProx + DP (eps=1)"]["mean"]),
              f"{ab['FedProx + DP (eps=1)']['vs_FedProx']['mean']:+.3f} vs no DP")
    st.caption("The base-paper and our numbers come from different cohorts and test designs (493 patients, "
               "per-site tests vs 347 patients, global test), so compare each against its own baseline.")

    st.subheader("Every claim, with a computed verdict")
    rows = claims.build(R)
    st.dataframe(pd.DataFrame([{"area": r["area"], "claim": r["claim"], "evidence (mean [95% CI])": r["evidence"],
                                "verdict": verdict_badge(r["verdict"])} for r in rows]),
                 hide_index=True)
    st.caption("Rule: 'supported' only if the 95% CI of the paired difference over seeds excludes 0 in the claimed "
               "direction. Verdicts are computed from results/final/*.json each time the page loads.")

    st.subheader("Limitations")
    st.markdown(
        "- PCa-19 (5,610 patients) was harmonised by the base paper's authors from published studies; two hospitals have "
        "< 50 patients. Prospective validation is still needed.\n"
        "- TCGA cohorts are small (347 / 493 patients); per-hospital test splits of 5–19 patients make single-site AUCs noisy.\n"
        "- On TCGA the strongest features (Gleason grade, pathologic N) are post-surgical (pre-operative model ≈ 0.65 AUC); "
        "PCa-19 removes this limitation with pre-biopsy PSA and MRI.\n"
        "- Dirichlet hospitals are simulated; the real-site replication uses TCGA tissue-source sites, which are "
        "biobanks rather than independent clinical deployments.\n"
        "- The DP guarantee is per patient (one sample per patient), assumes full-batch updates (q = 1) and "
        "treats hospital sizes and shared preprocessing statistics as public.\n"
        "- Survival modelling is not possible: 9 deaths among 347 patients.\n"
        "- The membership-inference audit uses one (confidence-threshold) attack.")


# ============================================================================= 5. VALIDATION
def page_validation():
    st.title("Validation")
    st.markdown("Automated checks of the data, leakage, model, privacy and valuation code. They test the same "
                "code that produced the results.")
    if st.button("▶ Re-run all checks now", type="primary"):
        from fl_study import validate
        with st.spinner("Running checks…"):
            st.session_state["val"] = validate.run_all()
    res = st.session_state.get("val") or R.get("validation")
    if not res:
        st.info("No stored validation results. Press the button to run the checks.")
    else:
        n = sum(r["passed"] for r in res)
        (st.success if n == len(res) else st.error)(f"{n} / {len(res)} checks passed")
        for grp in dict.fromkeys(r["group"] for r in res):
            st.markdown(f"**{grp}**")
            for r in [x for x in res if x["group"] == grp]:
                st.markdown(f"{'✅' if r['passed'] else '❌'} {r['check']}  \n<span class='small'>{r['detail']}</span>",
                            unsafe_allow_html=True)
    st.divider()
    st.subheader("Reproducibility")
    meta = R.get("meta", {})
    if meta:
        st.dataframe(pd.DataFrame([{"experiment": k, **v} for k, v in meta.items() if k != "environment"]),
                     hide_index=True)
        st.caption(f"Environment used for the stored results: {meta.get('environment', {})}")
    st.code("python run_all.py              # regenerate every result (~25 min on 2 cores)\n"
            "python -m fl_study.validate    # run the checks from the command line", language="bash")


# ============================================================================= 6. LIVE DEMO
@st.cache_data(show_spinner=False)
def demo_data(seed, feature_set):
    from sklearn.model_selection import train_test_split
    clin, _ = data.load_matched_cohort()
    tr, te = train_test_split(np.arange(len(clin)), test_size=0.2, stratify=clin["y"], random_state=seed)
    Xtr, Xte, names = data.build_features(clin.iloc[tr], clin.iloc[te], feature_set)
    return Xtr, clin["y"].values[tr], Xte, clin["y"].values[te], names


def page_demo():
    st.title("Live demo — train a federated model")
    st.markdown("Pick a setting and train on the 347-patient cohort right now (about a second).")
    c1, c2, c3 = st.columns(3)
    with c1:
        K = st.slider("Hospitals", 2, 10, 3)
        alpha = st.select_slider("Heterogeneity α (smaller = more skewed)", [0.1, 0.5, 1.0, 10.0, 100.0], value=0.5)
        feature_set = st.radio("Features", ["full", "preop"], horizontal=True,
                               format_func=lambda x: "full clinical" if x == "full" else "pre-operative only")
    with c2:
        algo = st.radio("Algorithm", ["FedAvg", "FedProx"], horizontal=True)
        mu = st.slider("FedProx μ", 0.0, 1.0, 0.5, 0.05, disabled=algo == "FedAvg")
        rounds = st.slider("Rounds", 5, 50, 20)
    with c3:
        use_dp = st.checkbox("Differential privacy", value=False)
        eps = st.select_slider("ε (privacy budget)", [0.5, 1.0, 2.0, 5.0, 10.0], value=5.0, disabled=not use_dp)
        seed = int(st.number_input("Seed", 0, 999, 0))
    cfg = fl.TrainConfig(rounds=rounds, epochs=5, lr=0.5, l2=1e-3, mu=mu if algo == "FedProx" else 0.0,
                         epsilon=eps if use_dp else None)
    Xtr, ytr, Xte, yte, names = demo_data(seed, feature_set)
    a, b = fl.dirichlet_partition(ytr, yte, K, alpha, seed)
    clients = [(Xtr[a == k], ytr[a == k]) for k in range(K)]
    tests = [(Xte[b == k], yte[b == k]) for k in range(K)]
    with st.spinner("Training…"):
        res = fl.federated_train(clients, cfg, seed, eval_set=(Xte, yte))
        w_cen = fl.centralized_train(Xtr, ytr, fl.TrainConfig(rounds=rounds))
    auc = res["auc_history"][-1]
    cen = fl.safe_auc(yte, fl.predict_proba(Xte, w_cen))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{algo} test AUC", fmt(auc), f"{auc - cen:+.3f} vs centralised")
    m2.metric("Centralised (pooled) AUC", fmt(cen))
    m3.metric("Final client drift", fmt(res["drift_history"][-1], 4))
    if use_dp:
        e_chk, order = fl.rdp_epsilon(res["sigma"], cfg.steps, cfg.delta)
        m4.metric("Noise multiplier σ", f"{res['sigma']:.1f}", f"ε = {e_chk:.2f} (best α = {order:g})",
                  delta_color="off")
    else:
        m4.metric("Privacy", "none", "ε = ∞", delta_color="off")
    r = list(range(1, rounds + 1))
    g1, g2 = st.columns(2)
    with g1:
        show(ch.lines(r, [{"name": algo, "color": ch.C[algo], "mean": res["auc_history"]}],
                      "Test AUC per round", "round", "AUC", ref=cen, ref_label="centralised"))
    with g2:
        show(ch.lines(r, [{"name": algo, "color": ch.C[algo], "mean": res["drift_history"]}],
                      "Client drift per round", "round", "mean‖w_k − w_global‖"))
    rows = []
    for k, ((X, y), (Xt, yt)) in enumerate(zip(clients, tests)):
        w_loc = fl.local_only_train(X, y, fl.TrainConfig(rounds=rounds), seed)
        w_p = fl.personalise(X, y, res["w"])
        rows.append({"hospital": k + 1, "train patients": len(y), "T3/T4 rate": float(y.mean()) if len(y) else None,
                     "test patients": len(yt), "local-only AUC": fl.safe_auc(yt, fl.predict_proba(Xt, w_loc)),
                     f"{algo} AUC": fl.safe_auc(yt, fl.predict_proba(Xt, res["w"])),
                     "personalised AUC": fl.safe_auc(yt, fl.predict_proba(Xt, w_p))})
    dft = pd.DataFrame(rows)
    for col, f in [("T3/T4 rate", "{:.2f}"), ("local-only AUC", "{:.3f}"), (f"{algo} AUC", "{:.3f}"),
                   ("personalised AUC", "{:.3f}")]:
        dft[col] = [fmt_cell(v, f) for v in dft[col]]
    st.dataframe(dft, hide_index=True)
    st.caption("'—' = the hospital's test split has only one class, so AUC is undefined.")
    with st.expander("Model coefficients"):
        st.dataframe(pd.DataFrame({"feature": names + ["(intercept)"], "weight": res["w"]})
                     .sort_values("weight", key=abs, ascending=False), hide_index=True)


{PAGES[0]: page_overview, PAGES[1]: page_base, PAGES[2]: page_pca19, PAGES[3]: page_ours, PAGES[4]: page_final,
 PAGES[5]: page_validation, PAGES[6]: page_demo}[page]()
