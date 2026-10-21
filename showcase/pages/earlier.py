"""Earlier work: the project's first phase on TCGA-PRAD (not part of the paper).

Kept for transparency. The code below is the original app's TCGA pages, unchanged except for headings."""
import math

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st

from fl_study import charts as ch
from fl_study import claims, data, fl  # noqa: F401
from showcase import core

R = core.results()

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
    core.show(fig)


def verdict_badge(v):
    icon = {"supported": "✅", "not significant": "➖", "opposite": "❌", "not supported": "❌",
            "partly": "◐", "explained by step size": "◐", "insufficient data": "❔"}.get(v, "❔")
    return f"{icon} {v}"


def finding(text, kind="info"):
    getattr(st, kind)(text)


def cell(cells, a, e):
    return [x for x in cells if x["alpha"] == a and x["epsilon"] == e][0]




def page_base():
    need("base_paper")
    bp = R["base_paper"]
    st.subheader("The base paper's protocol on TCGA tissue-source sites")
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



def page_ours():
    need("centralized", "drift", "privacy_grid", "personalization", "shapley", "mia", "shift", "ablation")
    st.subheader("DP-FedProx with Shapley valuation on TCGA (Dirichlet-split hospitals)")
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


def render():
    core.header("Earlier work · not in the paper", "Where the project started: TCGA-PRAD",
                "Before moving to the base paper's own 19 hospitals, we studied federated learning on the TCGA "
                "prostate cohort (post-surgical staging, 347–493 patients). These results are kept for "
                "transparency but are <b>not part of the paper</b>: the cohort is small, and the hospitals were "
                "either tiny tissue-source sites or simulated Dirichlet splits.")
    core.finding("<b>Why we moved on.</b> Valuing hospitals only makes sense for real institutions. The paper "
                 "therefore uses two real federations as released (19 prostate-cancer hospitals and 4 heart-disease "
                 "hospitals) and never simulates or partitions hospitals.", "warn")
    part = st.segmented_control("Section", ["Base-paper protocol on TCGA", "Our first model (DP-FPS)"],
                                default="Base-paper protocol on TCGA", key="earlier_part")
    if part == "Our first model (DP-FPS)":
        page_ours()
    else:
        page_base()
    core.next_steps(("home", "Back to the overview", ":material/home:"),
                    ("paper", "Paper & citation", ":material/description:"))
