"""
Generate every number, table and figure used in research_paper.tex from
results/final/*.json, so the manuscript can never drift from the experiments.

    python Paper/generate_paper_assets.py

Writes
  Paper/generated/numbers.tex   - \\newcommand macros used in the prose
  Paper/generated/tab_*.tex     - tables (\\input in the manuscript)
  Paper/figures/figure*.pdf     - figures
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from fl_study import charts as ch  # noqa: E402
from fl_study import claims  # noqa: E402

RES = os.path.join(ROOT, "results", "final")
GEN = os.path.join(HERE, "generated")
FIG = os.path.join(HERE, "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

R = {f[:-5]: json.load(open(os.path.join(RES, f), encoding="utf-8")) for f in os.listdir(RES) if f.endswith(".json")}
MAC = {}


def m(name, value):
    assert name.isalpha(), name
    MAC[name] = value


def f3(x):
    return f"{x:.3f}"


def pm(s, d=3):
    return f"{s['mean']:.{d}f} \\pm {s['sd']:.{d}f}"


def ci(s, d=3, signed=True):
    sg = "+" if signed else ""
    return f"{s['mean']:{sg}.{d}f}\\ [{s['ci_low']:{sg}.{d}f}, {s['ci_high']:{sg}.{d}f}]"


def p_fmt(p):
    if p is None:
        return "--"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}".replace("e-0", "e-")


# ------------------------------------------------------------------ cohort
c = R["cohort"]
m("nRaw", c["raw_records"]); m("nRawPatients", c["raw_patients"])
m("nClinical", c["clinical_cohort"]); m("nClinicalPos", c["clinical_cohort_pos"])
m("nMatched", c["matched_cohort"]); m("nMatchedPos", c["matched_pos"])
m("nTrain", c["matched_train"]); m("nTest", c["matched_test"])
m("nProteinSamples", c["protein_samples"]); m("nProteinRaw", c["protein_targets_raw"])
m("nProteinKept", c["protein_targets_kept"]); m("nPCAll", c["protein_pcs_95pct_full_cohort"])
m("nEvents", c["survival_events_matched"]); m("nEventsTrain", c["survival_events_train_seed42"])
m("nEventsTest", c["survival_events_test_seed42"]); m("nEventsAll", c["survival_events_all_572"])
m("nFeatFull", len(c["features_full"])); m("nFeatPreop", len(c["features_preop"]))

# ------------------------------------------------------------------ base paper
bp = R["base_paper"]
m("nSites", len(bp["sites"])); m("nSiteTotal", sum(s["n"] for s in bp["sites"])); m("nExternal", bp["n_external"])
for k, name in [("LOC", "LOC"), ("FL", "FL"), ("FR", "FR"), ("CEN", "CEN"), ("BL", "BL")]:
    m(f"bp{name}", f3(bp["macro"][k]["mean"]))
for k, name in [("FL-LOC", "FLminusLOC"), ("FR-FL", "FRminusFL"), ("CEN-FL", "CENminusFL"), ("FL-BL", "FLminusBL")]:
    m(f"bp{name}", ci(bp["macro_diff"][k]))
lc = bp["learning_curve"]
m("bpLcPartOne", f3(lc[0]["participants"]["mean"])); m("bpLcFrOne", f3(lc[0]["free_riders"]["mean"]))
m("bpLcPartAll", f3(lc[-1]["participants"]["mean"])); m("bpLcFrAll", f3(lc[-1]["free_riders"]["mean"]))
m("bpLooShapRho", ci(bp["loo_vs_shapley_spearman"], 2, False))
m("bpShapSizeRho", f"{bp['shapley_vs_size_spearman']:.2f}")
m("bpMacroSites", f"{bp['macro_sites']['mean']:.1f}")

rows = []
for r in bp["per_site"]:
    def g(k):
        return "--" if r[k] is None else f3(r[k])
    rows.append(f"{ch.short(r['site'])} & {r['n']} & {r['prevalence']:.2f} & {g('LOC')} & {g('FL')} & {g('FR')} & "
                f"{g('CEN')} & {g('BL')} & {r['shapley']['mean']:+.3f} & {r['loo']['mean']:+.4f} \\\\")
mac = bp["macro"]
rows.append("\\midrule")
rows.append(f"Macro mean & {sum(s['n'] for s in bp['sites'])} & -- & {f3(mac['LOC']['mean'])} & {f3(mac['FL']['mean'])} & "
            f"{f3(mac['FR']['mean'])} & {f3(mac['CEN']['mean'])} & {f3(mac['BL']['mean'])} & & \\\\")
open(os.path.join(GEN, "tab_base_paper.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Base-paper replication on the " + str(len(bp["sites"])) +
    " TCGA tissue-source sites with $\\geq 20$ patients: mean per-site test AUC over " + str(bp["config"]["seeds"]) +
    " seeds (LOC = local only, FL = FedAvg over all sites, FR = free-rider, CEN = pooled, BL = Gleason-sum rule) "
    "and mean contribution (exact Shapley, leave-one-out). -- = AUC undefined (single-class test split).}\n"
    "\\centering\\small\n\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nSite & $n$ & T3/T4 & LOC & FL & FR & CEN & BL & "
    "Shapley & LOO \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:base_paper}\n"
    "\\end{table*}\n")

# ------------------------------------------------------------------ centralised baselines
cb = R["centralized"]
order = ["Clinical (full)", "Clinical (pre-op only)", "Protein only (PCA)", "Clinical (full) + protein", "Pre-op + protein"]
models = ["NumPy LR (federated model, centralised)", "LR (L2)", "LR (L1)", "Random forest", "MLP"]
tab = {(r["features"], r["model"]): r for r in cb["table"]}
rows = []
for mo in models:
    rows.append(mo.replace("NumPy LR (federated model, centralised)", "NumPy LR (ours)") + " & " +
                " & ".join(f"{tab[(fs, mo)]['mean']:.3f}" for fs in order) + " \\\\")
open(os.path.join(GEN, "tab_centralized.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Centralised baselines: mean AUC over repeated stratified cross-validation (" +
    str(cb["n_folds"]) + " folds, 5$\\times$5) on the " + str(c["matched_cohort"]) + "-patient cohort. All preprocessing "
    "is fitted inside each training fold.}\n\\centering\\small\n\\begin{tabular}{lccccc}\n\\toprule\nModel & Clinical "
    "(full) & Clinical (pre-op) & Protein only & Full + protein & Pre-op + protein \\\\\n\\midrule\n" +
    "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:centralized}\n\\end{table*}\n")
key = models[0]
m("cvFull", f3(tab[("Clinical (full)", key)]["mean"])); m("cvPreop", f3(tab[("Clinical (pre-op only)", key)]["mean"]))
m("cvProt", f3(tab[("Protein only (PCA)", key)]["mean"])); m("cvFullProt", f3(tab[("Clinical (full) + protein", key)]["mean"]))
m("cvSkFull", f3(tab[("Clinical (full)", "LR (L2)")]["mean"]))
m("cvRFFullProt", f3(tab[("Clinical (full) + protein", "Random forest")]["mean"]))
m("cvRFFull", f3(tab[("Clinical (full)", "Random forest")]["mean"]))
m("cvProtDiff", ci(cb["comparisons"]["protein_added_to_full"]))
m("cvPreopDiff", ci(cb["comparisons"]["full_vs_preop"]))
m("cvProtPreopDiff", ci(cb["comparisons"]["protein_added_to_preop"]))
m("nFolds", cb["n_folds"])

# ------------------------------------------------------------------ drift
dr = R["drift"]
rows = []
for r in dr["rows"]:
    rows.append(f"{r['alpha']:g} & {r['mean_label_spread']['mean']:.2f} & ${pm(r['fedavg_drift'], 3)}$ & "
                f"${pm(r['fedprox_drift'], 3)}$ & ${r['drift_reduction_pct']['mean']:.1f}$ "
                f"[{r['drift_reduction_pct']['ci_low']:.1f}, {r['drift_reduction_pct']['ci_high']:.1f}] & "
                f"${ci(r['auc_diff'], 4)}$ \\\\")
open(os.path.join(GEN, "tab_drift.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Client drift $\\bar d = \\mathrm{mean}_{t,k}\\|w_k^t - w^{t-1}\\|_2$ without DP "
    "(3 clients, 20 seeds, mean $\\pm$ sd). Spread = max$-$min T3/T4 rate across clients. Last column: paired "
    "FedProx$-$FedAvg test-AUC difference with 95\\% CI.}\n\\centering\\small\n\\begin{tabular}{cccccc}\n\\toprule\n"
    "$\\alpha$ & Spread & FedAvg drift & FedProx drift & Reduction \\% [95\\% CI] & $\\Delta$AUC [95\\% CI] \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:drift}\n\\end{table*}\n")
red = [r["drift_reduction_pct"]["mean"] for r in dr["rows"]]
m("driftRedMin", f"{min(red):.1f}"); m("driftRedMax", f"{max(red):.1f}")
d0 = {r["alpha"]: r for r in dr["rows"]}
m("driftAvgIID", f"{d0[100.0]['fedavg_drift']['mean']:.3f}"); m("driftAvgSkew", f"{d0[0.1]['fedavg_drift']['mean']:.3f}")
mus = dr["mu_sweep_alpha_0.5"]
m("driftMuZero", f"{mus[0]['drift']['mean']:.3f}"); m("driftMuOne", f"{mus[-1]['drift']['mean']:.3f}")

# ------------------------------------------------------------------ privacy grid
g = R["privacy_grid"]
cells = g["cells"]
cell = {(x["alpha"], x["epsilon"]): x for x in cells}
for e, s in g["sigma"].items():
    m("sigma" + {"10.0": "Ten", "5.0": "Five", "2.0": "Two", "1.0": "One", "0.5": "Half"}[e], f"{float(s):.2f}")
m("auNoDP", f3(cell[(10.0, None)]["fedavg"]["mean"]))
for e, nm in [(10.0, "Ten"), (5.0, "Five"), (2.0, "Two"), (1.0, "One"), (0.5, "Half")]:
    x = cell[(10.0, e)]
    m("auAvg" + nm, f3(x["fedavg"]["mean"])); m("auProx" + nm, f3(x["fedprox"]["mean"]))
    m("auLoss" + nm, ci(x["fedavg_minus_nodp"]))
dp_cells = [x for x in cells if x["epsilon"] is not None]
sup = sum(x["prox_minus_avg"]["ci_low"] > 0 for x in dp_cells)
sup_ctl = sum(x["prox_minus_halflr_avg"]["ci_low"] > 0 for x in dp_cells)
m("gridSup", sup); m("gridCells", len(dp_cells)); m("gridSupCtl", sup_ctl)
nodp = [x for x in cells if x["epsilon"] is None]
m("gridNoDPmax", f"{max(abs(x['prox_minus_avg']['mean']) for x in nodp):.4f}")
m("gridOneHalfMin", f"{min(x['prox_minus_avg']['mean'] for x in dp_cells if x['epsilon'] in (1.0, 0.5)):.3f}")
m("gridOneHalfMax", f"{max(x['prox_minus_avg']['mean'] for x in dp_cells if x['epsilon'] in (1.0, 0.5)):.3f}")
m("gridCtlOneHalfMax", f"{max(x['prox_minus_halflr_avg']['mean'] for x in dp_cells if x['epsilon'] in (1.0, 0.5)):.3f}")
alphas = sorted({x["alpha"] for x in cells}, reverse=True)
eps_list = [None, 10.0, 5.0, 2.0, 1.0, 0.5]
lines = []
for a in alphas:
    for lab, k in [("FedAvg", "fedavg"), ("FedProx", "fedprox"), ("FedAvg ($\\eta/2$)", "fedavg_half_lr")]:
        lines.append(("$\\alpha=" + f"{a:g}" + "$" if lab == "FedAvg" else "") + f" & {lab} & " +
                     " & ".join(f"{cell[(a, e)][k]['mean']:.3f}" for e in eps_list) + " \\\\")
    lines.append(" & $\\Delta$ Prox$-$Avg & " + " & ".join(
        "--" if e is None else (f"\\textbf{{{cell[(a, e)]['prox_minus_avg']['mean']:+.3f}}}"
                                if cell[(a, e)]['prox_minus_avg']['ci_low'] > 0 or cell[(a, e)]['prox_minus_avg']['ci_high'] < 0
                                else f"{cell[(a, e)]['prox_minus_avg']['mean']:+.3f}") for e in eps_list) + " \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
open(os.path.join(GEN, "tab_grid.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Global test AUC under joint label skew ($\\alpha$) and privacy ($\\varepsilon$), "
    "3 clients, mean over 20 seeds. FedAvg ($\\eta/2$) is a step-size control. $\\Delta$ = paired FedProx$-$FedAvg "
    "difference; bold = 95\\% CI excludes 0.}\n\\centering\\small\n\\begin{tabular}{llcccccc}\n\\toprule\n & Model & "
    "no DP & $\\varepsilon=10$ & $\\varepsilon=5$ & $\\varepsilon=2$ & $\\varepsilon=1$ & $\\varepsilon=0.5$ \\\\\n"
    "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:interaction}\n\\end{table*}\n")

# ------------------------------------------------------------------ personalisation
pf = R["personalization"]["by_K"]
rows = []
for k in sorted(pf, key=int):
    v = pf[k]
    rows.append(f"{k} & {v['clients_with_defined_auc']['mean']:.1f} & " +
                " & ".join(f"{v['macro'][mm]['mean']:.3f}" for mm in ["Local", "FedAvg", "FedProx", "PFL"]) +
                f" & ${ci(v['PFL_minus_Local'])}$ & ${ci(v['PFL_minus_FedProx'])}$ \\\\")
open(os.path.join(GEN, "tab_pfl.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Mean per-client test AUC ($\\alpha=0.5$, 20 seeds) over clients whose local "
    "test split contains both classes (\\#def. = mean number of such clients). PFL = proximal fine-tuning of the "
    "FedProx model. Paired differences with 95\\% CI.}\n\\centering\\small\n\\begin{tabular}{ccccccrr}\n\\toprule\n"
    "$K$ & \\#def. & Local & FedAvg & FedProx & PFL & PFL$-$Local & PFL$-$FedProx \\\\\n\\midrule\n" +
    "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:local_adaptation}\n\\end{table*}\n")
for k, nm in [("3", "Three"), ("5", "Five"), ("10", "Ten")]:
    if k in pf:
        m("pflLocal" + nm, f3(pf[k]["macro"]["Local"]["mean"])); m("pflProx" + nm, f3(pf[k]["macro"]["FedProx"]["mean"]))
        m("pflPFL" + nm, f3(pf[k]["macro"]["PFL"]["mean"])); m("pflGain" + nm, ci(pf[k]["PFL_minus_Local"]))
        m("pflVsProx" + nm, ci(pf[k]["PFL_minus_FedProx"]))

# ------------------------------------------------------------------ Shapley
sh = R["shapley"]["by_K"]
Ks = sorted(sh, key=int)
rows = []
for k in Ks:
    for x in sh[k]["per_epsilon"]:
        rows.append(f"{k} & {x['epsilon']:g} & {x['spearman']['mean']:.2f} [{x['spearman']['ci_low']:.2f}, "
                    f"{x['spearman']['ci_high']:.2f}] & {x['rank_reversal']['mean']:.2f} & "
                    f"{x['top_contributor_stability']['mean']:.2f} & {1 / int(k):.2f} & {p_fmt(x['wilcoxon_p_rho_below_0.90'])} \\\\")
    rows.append("\\midrule")
rows = rows[:-1]
open(os.path.join(GEN, "tab_shapley.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Stability of exact Shapley rankings under DP relative to the no-DP ranking "
    "(20 partitions; " + ", ".join(f"{sh[k]['realizations_per_seed']}" for k in Ks) + " noise realisations per "
    "partition for $K=" + ", ".join(Ks) + "$). TS = probability that the no-DP top contributor is still ranked first "
    "(chance = $1/K$). $p$: one-sided Wilcoxon test of $\\rho-0.90<0$.}\n\\centering\\small\n"
    "\\begin{tabular}{ccccccc}\n\\toprule\n$K$ & $\\varepsilon$ & Spearman $\\rho$ [95\\% CI] & Rank reversal & TS & "
    "Chance TS & $p$ \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:wilcoxon_results}\n\\end{table*}\n")
for k, nm in [("3", "Three"), ("5", "Five"), ("10", "Ten")]:
    if k in sh:
        e = {x["epsilon"]: x for x in sh[k]["per_epsilon"]}
        m("shRhoTen" + nm, f"{e[10.0]['spearman']['mean']:.2f}"); m("shRhoHalf" + nm, f"{e[0.5]['spearman']['mean']:.2f}")
        m("shTSTen" + nm, f"{e[10.0]['top_contributor_stability']['mean']:.2f}")
        m("shTSHalf" + nm, f"{e[0.5]['top_contributor_stability']['mean']:.2f}")
        m("shRRHalf" + nm, f"{e[0.5]['rank_reversal']['mean']:.2f}")
        m("shLoo" + nm, ci(sh[k]["spearman_shapley_vs_loo"], 2, False))
        m("shSize" + nm, ci(sh[k]["spearman_shapley_vs_size"], 2, False))
        m("shR" + nm, sh[k]["realizations_per_seed"])

# ------------------------------------------------------------------ MIA, shift, ablation
mi = R["mia"]
mr = {r["epsilon"]: r for r in mi["rows"]}
m("miaNoDP", ci(mr["None"]["attack_auc"], 3, False)); m("miaOne", ci(mr["1.0"]["attack_auc"], 3, False))
m("miaHalf", ci(mr["0.5"]["attack_auc"], 3, False)); m("miaCtl", ci(mi["overfit_reference"]["attack_auc"], 3, False))
m("miaGap", ci(mr["None"]["generalisation_gap"]))
sf = R["shift"]
m("shiftCovTwo", f3(sf["covariate"][-1]["mean"])); m("shiftConTwo", f3(sf["concept"][-1]["mean"]))
m("shiftBase", f3(sf["covariate"][0]["mean"]))
ab = {r["configuration"]: r for r in R["ablation"]["rows"]}
rows = []
for k, r in ab.items():
    d = r["vs_FedProx"]
    rows.append(f"{k} & ${pm(r)}$ & " + ("reference" if d is None else f"${ci(d)}$") + " \\\\")
open(os.path.join(GEN, "tab_ablation.tex"), "w", encoding="utf-8").write(
    "\\begin{table}[t]\n\\caption{Ablation on the global test split ($K=3$, $\\alpha=0.5$, 20 seeds, "
    "mean $\\pm$ sd; paired difference to FedProx with 95\\% CI).}\n\\centering\\scriptsize\n"
    "\\begin{tabular}{p{3.1cm}cc}\n\\toprule\nConfiguration & Test AUC & vs FedProx \\\\\n\\midrule\n" +
    "\n".join(rows).replace("eps=", "$\\varepsilon$=").replace("FedProx, ", "FedProx: ") +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:global_ablation}\n\\end{table}\n")
m("abCen", f3(ab["Centralised (pooled data)"]["mean"])); m("abAvg", f3(ab["FedAvg"]["mean"]))
m("abProx", f3(ab["FedProx"]["mean"]))
m("abDPFive", f3(ab["FedProx + DP (eps=5)"]["mean"])); m("abDPOne", f3(ab["FedProx + DP (eps=1)"]["mean"]))
m("abPreop", f3(ab["FedProx, pre-op features only"]["mean"]))
m("abProt", f3(ab["FedProx, clinical (full) + protein PCs"]["mean"]))

# ------------------------------------------------------------------ hypothesis table (computed verdicts)
rows = []
for r in claims.build(R):
    ev = (r["evidence"].replace("%", "\\%").replace("_", "\\_").replace("[", "{[}").replace("]", "{]}")
          .replace(" > ", " $>$ ").replace(" < ", " $<$ ")
          .replace("->", "$\\rightarrow$").replace("eps", "$\\varepsilon$").replace("alpha", "$\\alpha$"))
    cl = (r["claim"].replace("eps", "$\\varepsilon$").replace("%", "\\%").replace("<", "$<$")
          .replace("alpha", "$\\alpha$"))
    rows.append(f"{r['area']} & {cl} & {ev} & {r['verdict']} \\\\")
open(os.path.join(GEN, "tab_hypotheses.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Claim--evidence summary. Verdicts are computed from the result files: "
    "`supported' only if the 95\\% CI of the paired difference over seeds excludes 0 in the claimed direction.}\n"
    "\\centering\\scriptsize\n\\begin{tabular}{p{1.4cm}p{4.2cm}p{8.6cm}p{1.5cm}}\n\\toprule\nArea & Claim & Evidence "
    "(mean [95\\% CI]) & Verdict \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:hyp_summary}\n\\end{table*}\n")

with open(os.path.join(GEN, "numbers.tex"), "w", encoding="utf-8") as fh:
    fh.write("% AUTO-GENERATED by Paper/generate_paper_assets.py from results/final/*.json - do not edit by hand\n")
    for k, v in MAC.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(MAC)} macros, tables written to {GEN}")

# ------------------------------------------------------------------ figures
import matplotlib.pyplot as plt  # noqa: E402


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


xs = [100, 10, 5, 2, 1, 0.5]
a10 = [cell[(10.0, e)] for e in eps_list]
save(ch.lines(xs, [{"name": n, "color": col, "dash": dsh, "mean": [x[k]["mean"] for x in a10],
                    "low": [x[k]["ci_low"] for x in a10], "high": [x[k]["ci_high"] for x in a10]}
                   for k, n, col, dsh in [("fedavg", "FedAvg", ch.C["FedAvg"], "-"),
                                          ("fedprox", "FedProx", ch.C["FedProx"], "-")]],
              "Test AUC vs privacy budget ($\\alpha=10$, 20 seeds)", "$\\varepsilon$ ('none' = no DP)", "test AUC",
              xlog=True, xticks=(xs, ["none", "10", "5", "2", "1", "0.5"]), clip=(0, 1), direct=False),
     "figure3_privacy_utility.pdf")
ep5 = [10.0, 5.0, 2.0, 1.0, 0.5]
M = [[cell[(a, e)]["prox_minus_avg"]["mean"] for e in ep5] for a in alphas]
S = [["*" if (cell[(a, e)]["prox_minus_avg"]["ci_low"] > 0 or cell[(a, e)]["prox_minus_avg"]["ci_high"] < 0) else ""
      for e in ep5] for a in alphas]
save(ch.heatmap(M, [f"$\\alpha$={a:g}" for a in alphas], [f"$\\varepsilon$={e:g}" for e in ep5],
                "FedProx $-$ FedAvg test AUC under DP (* = 95% CI excludes 0)", center=0.0, stars=S),
     "figure4_heatmap.pdf")
rws = dr["rows"]
save(ch.grouped_bars([f"$\\alpha$={r['alpha']:g}" for r in rws], ["FedAvg", "FedProx"],
                     [[r["fedavg_drift"]["mean"] for r in rws], [r["fedprox_drift"]["mean"] for r in rws]],
                     [ch.C["FedAvg"], ch.C["FedProx"]], "Client drift (mean $\\pm$ 95% CI, 20 seeds)", "drift ($L_2$)",
                     errs=[[(r["fedavg_drift"]["mean"] - r["fedavg_drift"]["ci_low"],
                             r["fedavg_drift"]["ci_high"] - r["fedavg_drift"]["mean"]) for r in rws],
                           [(r["fedprox_drift"]["mean"] - r["fedprox_drift"]["ci_low"],
                             r["fedprox_drift"]["ci_high"] - r["fedprox_drift"]["mean"]) for r in rws]]),
     "figure5_drift.pdf")
ex = sh["3"]["example_seed0"]
labels = ["no DP"] + [f"$\\varepsilon$={float(e):g}" for e in ex["phi_dp_first"]]
vals = [ex["phi_nodp"]] + [ex["phi_dp_first"][e] for e in ex["phi_dp_first"]]
save(ch.grouped_bars(labels, [f"Hospital {i + 1}" for i in range(3)], [[v[i] for v in vals] for i in range(3)],
                     ["#2a78d6", "#eb6834", "#1baf7a"], "Exact Shapley values of 3 hospitals, one partition, one "
                     "noise draw per $\\varepsilon$", "$\\phi_k$ ($\\Delta$AUC)", ref=0), "figure6_shapley.pdf")
mrows = mi["rows"]
nm = ["no DP" if r["epsilon"] == "None" else f"$\\varepsilon$={float(r['epsilon']):g}" for r in mrows]
save(ch.bar_with_ci(nm + ["over-fitted\ncontrol"], [r["attack_auc"] for r in mrows] + [mi["overfit_reference"]["attack_auc"]],
                    [ch.C["FedAvg"]] * len(mrows) + [ch.C["reference"]], "Membership-inference attack AUC (20 seeds)",
                    ylabel="attack AUC", ylim=(0.3, 0.85), ref=0.5, ref_label="chance"), "figure7_mia.pdf")
sev = sf["severities"]
save(ch.lines(sev, [{"name": n, "color": col, "mean": [x["mean"] for x in sf[k]], "low": [x["ci_low"] for x in sf[k]],
                     "high": [x["ci_high"] for x in sf[k]]}
                    for k, n, col in [("covariate", "covariate shift", ch.C["FedAvg"]),
                                      ("concept", "concept shift", ch.C["FedProx"])]],
              "FedProx model under synthetic test-set shift", "severity", "test AUC", ref=0.5, ref_label="chance",
              clip=(0, 1)), "figure8_stress_test.pdf")
xs5 = list(range(5))
for metric, ttl, fname in [("spearman", "Spearman $\\rho$ with the no-DP Shapley ranking", "figure9_shapley_instability.pdf"),
                           ("top_contributor_stability", "P(no-DP top contributor still ranked first)",
                            "figure11_shapley_scaling_instability.pdf")]:
    save(ch.lines(xs5, [{"name": f"K={k}", "color": ["#2a78d6", "#eb6834", "#1baf7a"][i],
                         "mean": [x[metric]["mean"] for x in sh[k]["per_epsilon"]]} for i, k in enumerate(Ks)],
                  ttl, "$\\varepsilon$ (stronger privacy $\\rightarrow$)", metric.replace("_", " "),
                  xticks=(xs5, [f"{e:g}" for e in ep5]), ylim=(-0.05, 1.05)), fname)
Kp = sorted(pf, key=int)
save(ch.grouped_bars([f"K={k}" for k in Kp], ["Local", "FedAvg", "FedProx", "PFL"],
                     [[pf[k]["macro"][mm]["mean"] for k in Kp] for mm in ["Local", "FedAvg", "FedProx", "PFL"]],
                     [ch.C[mm] for mm in ["Local", "FedAvg", "FedProx", "PFL"]],
                     "Mean per-client test AUC vs number of clients ($\\alpha=0.5$)", "AUC", ylim=(0.5, 0.95)),
     "figure10_fedprox_scaling_benefit.pdf")
keys = ["LOC", "FL", "FR", "CEN", "BL"]
save(ch.bar_with_ci(["Local", "Federated", "Free-rider", "Centralised", "Rule"], [bp["macro"][k] for k in keys],
                    [ch.C[k] for k in keys], "Base-paper strategies on 9 TCGA sites (mean per-site AUC, 95% CI)",
                    ylim=(0.5, 1.0)), "figure12_base_paper.pdf")
save(ch.lines([r["K"] for r in lc], [{"name": n, "color": ch.C[cc], "mean": [r[k]["mean"] for r in lc],
                                      "low": [r[k]["ci_low"] for r in lc], "high": [r[k]["ci_high"] for r in lc]}
                                     for k, n, cc in [("participants", "participants", "FL"), ("free_riders", "free-riders", "FR")]],
              "Federated AUC as hospitals join (TCGA sites)", "hospitals training", "AUC", clip=(0, 1)),
     "figure13_learning_curve.pdf")
print("figures written to", FIG)
