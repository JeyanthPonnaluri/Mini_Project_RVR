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
PAGES = ["🏠 Overview", "📄 Base paper (replication)", "🧪 Our model (DP-FPS)", "🏁 Final results",
         "✅ Validation", "🎛️ Live demo"]
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
        "**Task.** Predict whether a prostate tumour is *advanced* (pathologic **T3/T4**) or *organ-confined* "
        "(**T1/T2**) from TCGA-PRAD patient data, when the data are spread across hospitals that cannot "
        "share records.")
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
                    "We replicate the design on the **9 TCGA hospitals with ≥ 20 patients**.")
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
        ["Data", "Clinical risk factors, 19 real hospitals", "TCGA-PRAD clinical (+ protein ablation); real sites "
         "and Dirichlet-skewed hospitals"],
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
        "- Small cohorts (347 / 493 patients); per-hospital test splits of 5–19 patients make single-site AUCs noisy.\n"
        "- The strongest features (Gleason grade, pathologic N) are post-surgical; a pre-operative model reaches "
        "only about 0.65 AUC.\n"
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


{PAGES[0]: page_overview, PAGES[1]: page_base, PAGES[2]: page_ours, PAGES[3]: page_final,
 PAGES[4]: page_validation, PAGES[5]: page_demo}[page]()
