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
    "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{p{2.9cm}cc}\n\\toprule\nConfiguration & Test AUC & vs FedProx \\\\\n\\midrule\n" +
    "\n".join(rows).replace("eps=", "$\\varepsilon$=").replace("FedProx, ", "FedProx: ") +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:global_ablation}\n\\end{table}\n")
m("abCen", f3(ab["Centralised (pooled data)"]["mean"])); m("abAvg", f3(ab["FedAvg"]["mean"]))
m("abProx", f3(ab["FedProx"]["mean"]))
m("abDPFive", f3(ab["FedProx + DP (eps=5)"]["mean"])); m("abDPOne", f3(ab["FedProx + DP (eps=1)"]["mean"]))
m("abPreop", f3(ab["FedProx, pre-op features only"]["mean"]))
m("abProt", f3(ab["FedProx, clinical (full) + protein PCs"]["mean"]))

# ------------------------------------------------------------------ hypothesis table (computed verdicts)
import re as _re2
rows = []
for r in claims.build(R):
    if not r["area"].startswith(("PCa-19", "Heart", "Pre-registered")):
        continue
    ev = (r["evidence"].replace("%", "\\%").replace("_", "\\_").replace("[", "{[}").replace("]", "{]}")
          .replace(" > ", " $>$ ").replace(" < ", " $<$ ")
          .replace("->", "$\\rightarrow$").replace("alpha", "$\\alpha$").replace("|", "$|$"))
    ev = _re2.sub(r"\beps\b", r"$\\varepsilon$", ev)
    cl = (_re2.sub(r"\beps\b", r"$\\varepsilon$", r["claim"]).replace("%", "\\%").replace("<=", "LEQ").replace("<", "$<$").replace("LEQ", "<=")
          .replace("alpha", "$\\alpha$"))
    import re as _re
    ev = _re.sub(r"\bnu\b", r"$\\nu$", ev.replace(">=", r"$\ge$").replace("<=", r"$\le$"))
    cl = _re.sub(r"\bnu\b", r"$\\nu$", cl.replace(">=", r"$\ge$").replace("<=", r"$\le$"))
    rows.append(f"{r['area']} & {cl} & {ev} & {r['verdict']} \\\\")
open(os.path.join(GEN, "tab_hypotheses.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Claim--evidence summary. Verdicts are computed from the result files: "
    "`supported' only if the 95\\% CI of the paired difference over seeds excludes 0 in the claimed direction; the pre-registered rows are scored by the criteria fixed in advance "
    "(Section~\\ref{sec:prereg}).}\n"
    "\\centering\\scriptsize\n\\begin{tabular}{p{1.4cm}p{4.2cm}p{8.6cm}p{1.5cm}}\n\\toprule\nArea & Claim & Evidence "
    "(mean [95\\% CI]) & Verdict \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:hyp_summary}\n\\end{table*}\n")

# ------------------------------------------------------------------ PCa-19 (base paper's own 19-hospital cohort)
def ci2(s, d=3, signed=True):
    return ci(s, d, signed)


P19 = {k[6:]: v for k, v in R.items() if k.startswith("pca19_")}
pc = P19["cohort"]
m("pNineN", f"{pc['n']:,}".replace(",", "{,}")); m("pNinePos", f"{pc['n_pos']:,}".replace(",", "{,}"))
m("pNineHosp", pc["n_hospitals"]); m("pNineCountries", pc["n_countries"])
m("pNinePrevLo", f"{100 * pc['prevalence_range'][0]:.0f}"); m("pNinePrevHi", f"{100 * pc['prevalence_range'][1]:.0f}")
m("pNinePrev", f"{100 * pc['n_pos'] / pc['n']:.1f}")
rows = [f"{h['hospital']} & {h['country']} & {h['n']} & {100 * h['prevalence']:.0f} & {h['age_median']:.0f} & "
        f"{h['psa_median']:.1f} & {h['pv_median']:.0f} & {100 * h['pirads_ge4']:.0f} \\\\" for h in pc["hospitals"]]
open(os.path.join(GEN, "tab_pca19_cohort.tex"), "w", encoding="utf-8").write(
    "\\begin{table}[t]\n\\caption{PCa-19: the base paper's 19-hospital pre-biopsy cohort (" + MAC["pNineN"] +
    " men). csPCa = ISUP grade group $\\geq 2$ at biopsy; medians of age (years), PSA (ng/mL) and prostate volume (mL); "
    "PI-RADS $\\geq 4$ in \\%.}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llrrrrrr}\n\\toprule\n"
    "Hosp. & Country & $n$ & csPCa\\% & Age & PSA & Vol. & PI-RADS$\\geq$4 \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:pca19_cohort}\n\\end{table}\n")

# authors' protocol reproduction
rp = {c_: P19.get(f"reproduction_{c_}") for c_ in ("noh", "ettala")}
if all(rp.values()):
    m("pNineReps", rp["noh"]["n_repeats"])
    rows = []
    for strat, lab in [("BL", "Published calculator (BL)"), ("LOC", "Local (LOC)"), ("CEN", "Centralised (CEN)"),
                       ("FL", "One-shot FedAvg (FL)"), ("FR", "Free-rider (FR)")]:
        rows.append(f"{lab} & {rp['noh']['AV'][strat]:.3f} & {rp['noh']['AVW'][strat]:.3f} & "
                    f"{rp['ettala']['AV'][strat]:.3f} & {rp['ettala']['AVW'][strat]:.3f} \\\\")
    open(os.path.join(GEN, "tab_pca19_repro.tex"), "w", encoding="utf-8").write(
        "\\begin{table}[t]\n\\caption{Reproduction of the base paper's protocol on PCa-19 (" + str(rp["noh"]["n_repeats"]) +
        " Monte-Carlo repeats; AV = mean over hospitals, AVW = patient-weighted). Our re-implementation gives the same "
        "numbers as the authors' code (Section~\\ref{sec:p19_repro}).}\n\\centering\\footnotesize\n\\begin{tabular}{lcccc}\n"
        "\\toprule\n& \\multicolumn{2}{c}{Noh variables} & \\multicolumn{2}{c}{Ettala variables} \\\\\nStrategy & AV & AVW & AV & AVW \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:pca19_repro}\n\\end{table}\n")
    for c_, nm in (("noh", "Noh"), ("ettala", "Ett")):
        for k_ in ("BL", "LOC", "CEN", "FL", "FR"):
            m(f"pRep{nm}{k_}", f3(rp[c_]["AV"][k_]))

# strategies under our protocol
s9 = P19["strategies"]
for k_, nm in [("LOC", "LOC"), ("FL", "FL"), ("FedProx", "Prox"), ("CEN", "CEN"), ("FR_test", "FR"), ("BL_noh", "Noh"), ("BL_ettala", "Ett")]:
    m(f"pS{nm}", f3(s9["macro"][k_]["mean"])); m(f"pW{nm}", f3(s9["weighted"][k_]["mean"]))
for k_, nm in [("FL-LOC", "FLminusLOC"), ("FR_test-FL", "FRminusFL"), ("CEN-FL", "CENminusFL"), ("FL-BL_noh", "FLminusNoh"),
               ("FL-BL_ettala", "FLminusEtt"), ("FedProx-FL", "ProxminusFL")]:
    m(f"pD{nm}", ci(s9["diff"][k_]))
m("pGainSig", s9["n_gain_sig"]); m("pLossSig", s9["n_loss_sig"]); m("pGainSizeRho", f"{s9['gain_vs_size_spearman']:.2f}")
lc9 = s9["learning_curve"]
m("pLcFrOne", f3(lc9[0]["FR"]["mean"])); m("pLcFrFive", f3(lc9[4]["FR"]["mean"])); m("pLcFrLast", f3(lc9[-1]["FR"]["mean"]))
fc = s9["fr_calibration_pooled"]
m("pFrOE", f"{fc['oe']['mean']:.2f}"); m("pFrSlope", f"{fc['slope']['mean']:.2f}")

# privacy & drift on real hospitals
pv9 = P19["privacy"]
cells9 = {("none" if x["epsilon"] is None else x["epsilon"]): x for x in pv9["cells"]}
m("pDriftRed", f"{pv9['drift_reduction_pct']['mean']:.1f}")
m("pDpLossHalf", ci(cells9[0.5]["FedAvg-noDP"])); m("pDpLossOne", ci(cells9[1.0]["FedAvg-noDP"]))
m("pDpLossFive", ci(cells9[5.0]["FedAvg-noDP"]))
m("pProxHalfMax", f"{max(abs(x['FedProx-FedAvg_half_lr']['mean']) for x in pv9['cells']):.3f}")
m("pSigmaHalf", f"{pv9['sigma']['0.5']:.0f}")
rows = []
for k_ in ["none", 10.0, 5.0, 2.0, 1.0, 0.5]:
    x = cells9[k_]
    lab = "no DP" if k_ == "none" else f"{k_:g}"
    sg = "--" if k_ == "none" else f"{pv9['sigma'][str(k_)]:.1f}"
    dl = "--" if k_ == "none" else ci(x["FedAvg-noDP"])
    rows.append(f"{lab} & {sg} & {f3(x['FedAvg']['macro']['mean'])} & {f3(x['FedProx']['macro']['mean'])} & "
                f"{f3(x['FedAvg_half_lr']['macro']['mean'])} & ${dl}$ & ${ci(x['FedProx-FedAvg_half_lr'])}$ & "
                f"{x['FedAvg']['drift']['mean']:.2f} / {x['FedProx']['drift']['mean']:.2f} \\\\")
open(os.path.join(GEN, "tab_pca19_privacy.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Privacy and drift on the 19 real hospitals (mean per-hospital test AUC, 20 seeds). "
    "The last two AUC columns are paired differences with 95\\% CI; drift = mean $\\|w_k-w^{t-1}\\|$.}\n\\centering\\small\n"
    "\\begin{tabular}{lccccccc}\n\\toprule\n$\\varepsilon$ & $\\sigma$ & FedAvg & FedProx & FedAvg ($\\eta/2$) & FedAvg $-$ no DP & "
    "FedProx $-$ FedAvg($\\eta/2$) & Drift Avg / Prox \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:pca19_privacy}\n\\end{table*}\n")

# personalisation
pf9 = P19["personalisation"]["groups"]
m("pPflSmall", ci(pf9["small (< 100)"]["PFL-FedProx"])); m("pPflAll", ci(pf9["all"]["PFL-FedProx"]))
m("pPflLarge", ci(pf9["large (>= 400)"]["PFL-FedProx"])); m("pPflLargeLoc", ci(pf9["large (>= 400)"]["FedProx-Local"]))
m("pNSmall", len(pf9["small (< 100)"]["hospitals"]))

# membership inference
mi9 = P19["mia"]
m("pMiaNoDP", f3(mi9["rows"][0]["attack_auc"]["mean"])); m("pMiaCtl", f3(mi9["control"]["mean"]))

# leave-one-country-out
lo9 = P19["loco"]
bc9 = lo9["diff_by_country"]
m("pLocoAvgNoh", ci(bc9["FedAvg - Noh (published)"])); m("pLocoDpFive", ci(bc9["FedProx+DP(eps=5) - FedProx"]))
m("pLocoAvgCen", ci(bc9["FedAvg - Centralised"]))
sl9 = lo9["calibration_slope_by_model"]
m("pSlopeNoDP", ci(sl9["FedProx"], 2, False)); m("pSlopeDpFive", ci(sl9["FedProx+DP(eps=5)"], 2, False))
m("pSlopeDpOne", ci(sl9["FedProx+DP(eps=1)"], 2, False)); m("pSlopeNoh", ci(sl9["Noh (published)"], 2, False))
rc9 = lo9["recalibration"]
m("pRecalOeRaw", f3(rc9["FedAvg"]["raw"]["abs_log_oe"]["mean"])); m("pRecalOeInt", f3(rc9["FedAvg"]["intercept"]["abs_log_oe"]["mean"]))
m("pRecalOeDiff", ci(rc9["FedAvg"]["intercept_vs_raw_abs_log_oe"]))
m("pRecalSlopeDpRaw", f"{rc9['FedProx+DP(eps=5)']['raw']['slope']['mean']:.2f}")
m("pRecalSlopeDpLog", f"{rc9['FedProx+DP(eps=5)']['logistic']['slope']['mean']:.2f}")
oes = [r["FedAvg"]["oe"] for r in lo9["countries"]]
m("pOeMin", f"{min(oes):.2f}"); m("pOeMax", f"{max(oes):.2f}")
rows = []
for r in lo9["countries"]:
    rows.append(f"{r['country']} & {r['n']} & {100 * r['prevalence']:.0f} & {f3(r['FedAvg']['auc'])} & {r['FedAvg']['oe']:.2f} & "
                f"{r['FedAvg']['slope']:.2f} & {f3(r['FedProx+DP(eps=5)']['auc'])} & {r['FedProx+DP(eps=5)']['slope']:.2f} & "
                f"{f3(r['Noh (published)']['auc'])} & {r['Noh (published)']['oe']:.2f} \\\\")
open(os.path.join(GEN, "tab_pca19_loco.tex"), "w", encoding="utf-8").write(
    "\\begin{table*}[t]\n\\caption{Leave-one-country-out external validation on PCa-19: train on the other seven countries, test "
    "on all patients of the held-out country (mean of 20 seeds). O/E = observed/expected events; slope = calibration "
    "slope (ideal 1).}\n\\centering\\small\n\\begin{tabular}{lrrccccccc}\n\\toprule\n& & & \\multicolumn{3}{c}{FedAvg} & "
    "\\multicolumn{2}{c}{FedProx + DP ($\\varepsilon=5$)} & \\multicolumn{2}{c}{Noh (published)} \\\\\nCountry & $n$ & csPCa\\% & "
    "AUC & O/E & slope & AUC & slope & AUC & O/E \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\label{tab:pca19_loco}\n\\end{table*}\n")
rows = []
for mod, lab in [("FedAvg", "FedAvg"), ("FedProx+DP(eps=5)", "DP $\\varepsilon=5$"),
                 ("FedProx+DP(eps=1)", "DP $\\varepsilon=1$"), ("Noh (published)", "Noh")]:
    for meth, ml in [("raw", "none"), ("intercept", "intercept"), ("logistic", "int.\\,+\\,slope")]:
        x = rc9[mod][meth]
        rows.append(f"{lab if meth == 'raw' else ''} & {ml} & {f3(x['abs_log_oe']['mean'])} & {x['slope']['mean']:.2f} & "
                    f"{f3(x['brier']['mean'])} \\\\")
    rows.append("\\midrule")
open(os.path.join(GEN, "tab_pca19_recal.tex"), "w", encoding="utf-8").write(
    "\\begin{table}[t]\n\\caption{Local recalibration on 20\\% of each held-out country, evaluated on the other 80\\% "
    "(mean over 8 countries and 20 seeds). $|\\log$ O/E$|$ = 0 and slope = 1 are ideal; DP rows are FedProx + DP.}\n\\centering\\scriptsize\n"
    "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llccc}\n\\toprule\nModel & Recalibration & $|\\log$ O/E$|$ & Slope & Brier \\\\\n\\midrule\n" +
    "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:pca19_recal}\n\\end{table}\n")

# Shapley under DP on real hospitals
if "shapley" in P19:
    sh9 = P19["shapley"]
    ex9, mc9 = sh9["exact10"], sh9["mc19"]
    pe = {x["epsilon"]: x for x in ex9["per_epsilon"]}
    m("pShRhoTen", f"{pe[10.0]['spearman']['mean']:.2f}"); m("pShRhoHalf", f"{pe[0.5]['spearman']['mean']:.2f}")
    m("pShTSTen", f"{pe[10.0]['top_contributor_stability']['mean']:.2f}"); m("pShTSHalf", f"{pe[0.5]['top_contributor_stability']['mean']:.2f}")
    m("pShRhoTwo", f"{pe[2.0]['spearman']['mean']:.2f}"); m("pShRhoOne", f"{pe[1.0]['spearman']['mean']:.2f}")
    phis = [v["mean"] for v in ex9["phi_nodp"].values()]
    m("pShPhiMin", f"{min(phis):.3f}"); m("pShPhiMax", f"{max(phis):.3f}")
    pm19 = {x["epsilon"]: x for x in mc9["per_epsilon"]}
    m("pShNineRhoFive", f"{pm19[5.0]['spearman']['mean']:.2f}"); m("pShNineRhoOne", f"{pm19[1.0]['spearman']['mean']:.2f}")
    m("pShNineTSFive", f"{pm19[5.0]['top_contributor_stability']['mean']:.2f}"); m("pShNineTSOne", f"{pm19[1.0]['top_contributor_stability']['mean']:.2f}")
    m("pShNineCoal", f"{mc9['coalitions_per_seed']:,}".replace(",", "{,}"))
    m("pShLooRho", ci(ex9["spearman_shapley_vs_loo"], 2, False)); m("pShSizeRho", f"{mc9['spearman_shapley_vs_size']:.2f}")
    rows = []
    for lab, blk in (("10 (exact)", ex9["per_epsilon"]), ("19 (stratified)", mc9["per_epsilon"])):
        for x in blk:
            rows.append(f"{lab} & {x['epsilon']:g} & ${ci(x['spearman'], 2, False)}$ & {x['rank_reversal']['mean']:.2f} & "
                        f"{x['top_contributor_stability']['mean']:.2f} \\\\")
    open(os.path.join(GEN, "tab_pca19_shapley.tex"), "w", encoding="utf-8").write(
        "\\begin{table}[t]\n\\caption{Stability of Shapley contribution rankings under DP on the real PCa-19 hospitals (20 seeds). "
        "$\\rho$ = Spearman correlation with the no-DP ranking; Rank rev.\\ = share of hospital pairs whose order flips relative to the no-DP ranking; TS = probability that the no-DP top contributor stays first.}\n"
        "\\centering\\scriptsize\n\\begin{tabular}{lcccc}\n\\toprule\nHospitals & $\\varepsilon$ & $\\rho$ [95\\% CI] & Rank rev. & TS \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:pca19_shapley}\n\\end{table}\n")

# ------------------------------------------------------------------ privacy-honest valuation (OSA-Shapley)
if "private_valuation" in P19:
    pv_ = P19["private_valuation"]
    rows = []
    for x in pv_["rows"]:
        e = x["epsilon"]
        rows.append(f"{e:g} & {x['retrain_true_epsilon']:.0f} & {x['retrain_per_run']['spearman']['mean']:.2f} & "
                    f"{x['retrain_budget_matched']['spearman']['mean']:.2f} & \\textbf{{{x['osa']['spearman']['mean']:.2f}}} & "
                    f"{x['retrain_per_run']['top_contributor_stability']['mean']:.2f} & {x['retrain_budget_matched']['top_contributor_stability']['mean']:.2f} & "
                    f"\\textbf{{{x['osa']['top_contributor_stability']['mean']:.2f}}} \\\\")
    open(os.path.join(GEN, "tab_osa.tex"), "w", encoding="utf-8").write(
        "\\begin{table*}[t]\n\\caption{Contribution valuation under DP on the ten largest PCa-19 hospitals (" + str(len(pv_["seeds"])) +
        " held-out seeds). Agreement with the exact non-private Shapley ranking (Spearman $\\rho$) and probability that the true "
        "top contributor is ranked first (TS). \\emph{Per-run}: every coalition retrained at the stated $\\varepsilon$, whose true "
        "cost for each hospital is the composed $\\varepsilon$ in column 2. \\emph{Budget-matched}: retraining with noise "
        "calibrated so that the total cost is the stated $\\varepsilon$. OSA-Shapley costs the stated $\\varepsilon$ once.}\n"
        "\\centering\\small\n\\begin{tabular}{lccccccc}\n\\toprule\n & True $\\varepsilon$ of & \\multicolumn{3}{c}{Spearman $\\rho$ with exact Shapley} & "
        "\\multicolumn{3}{c}{TS} \\\\\n$\\varepsilon$ & per-run retraining & per-run & budget-matched & OSA & per-run & budget-matched & OSA \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:osa}\n\\end{table*}\n")
    r = {x["epsilon"]: x for x in pv_["rows"]}
    m("osaRhoTen", f"{r[10.0]['osa']['spearman']['mean']:.2f}"); m("osaRhoTwo", f"{r[2.0]['osa']['spearman']['mean']:.2f}")
    m("osaRhoOne", f"{r[1.0]['osa']['spearman']['mean']:.2f}"); m("osaRhoHalf", f"{r[0.5]['osa']['spearman']['mean']:.2f}")
    m("bmRhoTwo", f"{r[2.0]['retrain_budget_matched']['spearman']['mean']:.2f}")
    m("bmRhoTen", f"{r[10.0]['retrain_budget_matched']['spearman']['mean']:.2f}"); m("bmRhoOne", f"{r[1.0]['retrain_budget_matched']['spearman']['mean']:.2f}")
    m("prRhoTen", f"{r[10.0]['retrain_per_run']['spearman']['mean']:.2f}"); m("prRhoOne", f"{r[1.0]['retrain_per_run']['spearman']['mean']:.2f}")
    m("trueEpsOne", f"{r[1.0]['retrain_true_epsilon']:.0f}"); m("trueEpsFive", f"{r[5.0]['retrain_true_epsilon']:.0f}")
    m("trueEpsTen", f"{r[10.0]['retrain_true_epsilon']:.0f}")
    m("osaVsBmTwo", ci(r[2.0]["osa_minus_budget_matched"], 2)); m("osaVsPrTwo", ci(r[2.0]["osa_minus_per_run"], 2))
    m("osaVsPrOne", ci(r[1.0]["osa_minus_per_run"], 2))
    m("osaNoDP", f"{pv_['osa_nodp']['spearman']['mean']:.2f}")
    m("osaSteps", pv_["osa_steps"])
    k19 = {x["epsilon"]: x for x in pv_["k19"]["per_epsilon"]}
    m("osaNineRhoFive", f"{k19[5.0]['spearman']['mean']:.2f}"); m("osaNineRhoOne", f"{k19[1.0]['spearman']['mean']:.2f}")
    m("osaNineTSFive", f"{k19[5.0]['top_contributor_stability']['mean']:.2f}"); m("osaNineTSOne", f"{k19[1.0]['top_contributor_stability']['mean']:.2f}")
    m("osaNineSize", f"{pv_['k19']['spearman_vs_size']:.2f}")
if "private_valuation_tcga" in P19:
    pt_ = {x["epsilon"]: x for x in P19["private_valuation_tcga"]["rows"]}
    m("tOsaTwo", f"{pt_[2.0]['osa']['spearman']['mean']:.2f}"); m("tPrTwo", f"{pt_[2.0]['retrain_per_run']['spearman']['mean']:.2f}")
    m("tBmTwo", f"{pt_[2.0]['retrain_budget_matched']['spearman']['mean']:.2f}")
    m("tOsaTen", f"{pt_[10.0]['osa']['spearman']['mean']:.2f}"); m("tPrTen", f"{pt_[10.0]['retrain_per_run']['spearman']['mean']:.2f}")
    m("tBmTen", f"{pt_[10.0]['retrain_budget_matched']['spearman']['mean']:.2f}")
    m("tOsaOne", f"{pt_[1.0]['osa']['spearman']['mean']:.2f}"); m("tPrOne", f"{pt_[1.0]['retrain_per_run']['spearman']['mean']:.2f}")
    m("tOsaNoDP", f"{P19['private_valuation_tcga']['osa_nodp']['spearman']['mean']:.2f}")

# ------------------------------------------------------------------ OSA-Shapley vs post-processing competitors, robustness
if "valuation_plus" in P19:
    vp = P19["valuation_plus"]
    keys = ["none", "5.0", "2.0", "1.0", "0.5"]
    lab = lambda k: "none" if k == "none" else f"{float(k):g}"
    _sm = P19.get("valuation_proxies", {}).get("small10")
    rows = []
    for k in keys:
        a = vp["methods"][k]
        if _sm is not None and k in _sm["rho"]:
            t = _sm["rho"][k]; dd = _sm["osa_minus"][k]["mr"]
            small = (f"\\textbf{{{t['osa']['mean']:.2f}}} & {t['mr']['mean']:.2f} & {t['or']['mean']:.2f} & ${ci(dd, 2)}$")
        else:
            small = "-- & -- & -- & --"
        rows.append(f"{lab(k)} & \\textbf{{{a['osa']['mean']:.2f}}} & {a['mr']['mean']:.2f} & {a['or']['mean']:.2f} & "
                    f"${ci(vp['osa_minus_mr'][k], 2)}$ & " + small + " \\\\")
    _smn = "" if _sm is None else f"{min(_sm['sizes'])}--{max(_sm['sizes'])}"
    open(os.path.join(GEN, "tab_osa_post.tex"), "w", encoding="utf-8").write(
        "\\begin{table*}[t]\n\\caption{Valuations that cost the privacy budget once (Spearman $\\rho$ with the exact non-private retraining "
        "Shapley ranking). OSA: one DP model per hospital. MR: exact multi-round reconstruction from one DP-FedAvg run "
        "(GTG-Shapley is its truncated Monte-Carlo approximation). OR: one-round reconstruction from the same run. All three are "
        "post-processing of releases that already cost $\\varepsilon$. OSA$-$MR: paired difference with 95\\% CI. Both groups are real "
        "PCa-19 hospitals, " + str(len(vp["seeds"])) + " held-out splits; no hospital is simulated or partitioned.}\n"
        "\\centering\\small\n\\begin{tabular}{lcccccccc}\n\\toprule\n & \\multicolumn{4}{c}{Ten largest hospitals (266--763 patients)} & "
        "\\multicolumn{4}{c}{Ten smallest hospitals (" + _smn + " patients)} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "$\\varepsilon$ & OSA & MR & OR & OSA$-$MR & OSA & MR & OR & OSA$-$MR \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\label{tab:osa_post}\n\\end{table*}\n")
    for k, nm in (("none", "NoDP"), ("2.0", "Two"), ("1.0", "One"), ("0.5", "Half"), ("5.0", "Five")):
        m(f"mrRho{nm}", f"{vp['methods'][k]['mr']['mean']:.2f}"); m(f"orRho{nm}", f"{vp['methods'][k]['or']['mean']:.2f}")
        m(f"osaVsMr{nm}", ci(vp["osa_minus_mr"][k], 2)); m(f"osaVsOr{nm}", ci(vp["osa_minus_or"][k], 2))
        m(f"tMrRho{nm}", f"{vp['tcga'][k]['mr']['mean']:.2f}"); m(f"tOsaVsMr{nm}", ci(vp["tcga_osa_minus_mr"][k], 2))
        m(f"tHoOsaVsMr{nm}", ci(vp["tcga_heldout_seeds"]["osa_minus_mr"][k], 2))
    cr = vp["corrupt"]
    m("corrHosp", vp["corrupt_hospital"])
    for k, nm in (("2.0", "Two"), ("1.0", "One")):
        for meth, tag in (("osa", "Osa"), ("mr", "Mr"), ("or", "Or"), ("retrain_per_run", "Pr"), ("retrain_budget_matched", "Bm")):
            m(f"corr{tag}{nm}", f"{100 * cr[k][meth]['last']['mean']:.0f}\\%")
    m("corrTruth", f"{100 * cr['truth']['last']['mean']:.0f}\\%")
    rm = vp["removal"]["2.0"]
    for meth, tag in (("osa", "Osa"), ("mr", "Mr"), ("or", "Or"), ("truth", "Truth"), ("random", "Rand"), ("all", "All")):
        m(f"rem{tag}", f"{rm[meth]['mean']:.3f}")
    sn = vp["sensitivity"]
    srows = []
    for steps in (10, 30, 60):
        srows.append(f"Local steps = {steps}" + ("" if steps != 30 else " (chosen)") +
                     f" & {sn[f'steps={steps},eps=2.0']['mean']:.2f} & {sn[f'steps={steps},eps=1.0']['mean']:.2f} \\\\")
    for frac in (0.25, 0.5):
        srows.append(f"Reference set {int(100 * frac)}\\% of hospital test data & {sn[f'ref={frac},eps=2.0']['mean']:.2f} & "
                     f"{sn[f'ref={frac},eps=1.0']['mean']:.2f} \\\\")
    crows = []
    for meth, nmx in (("osa", "OSA (cost $\\varepsilon$)"), ("mr", "MR (cost $\\varepsilon$)"), ("or", "OR (cost $\\varepsilon$)"),
                      ("retrain_budget_matched", "Retraining, budget-matched (cost $\\varepsilon$)"),
                      ("retrain_per_run", "Retraining, per run (true $\\varepsilon$ 85 and 33)")):
        crows.append(f"{nmx} & {100 * cr['2.0'][meth]['last']['mean']:.0f}\\% & {100 * cr['1.0'][meth]['last']['mean']:.0f}\\% \\\\")
    open(os.path.join(GEN, "tab_osa_robust.tex"), "w", encoding="utf-8").write(
        "\\begin{table}[t]\n\\caption{OSA-Shapley robustness on PCa-19 (ten hospitals, " + str(len(vp["seeds"])) + " held-out seeds). "
        "Top: Spearman $\\rho$ with the exact non-private ranking. Bottom: share of splits in which a hospital with random training "
        "labels (" + vp["corrupt_hospital"] + ") is ranked last (no-DP retraining Shapley: " + f"{100 * cr['truth']['last']['mean']:.0f}" + "\\%).}\n"
        "\\centering\\scriptsize\n\\begin{tabular}{lcc}\n\\toprule\nSetting & $\\varepsilon=2$ & $\\varepsilon=1$ \\\\\n\\midrule\n" +
        "\n".join(srows) + "\n\\midrule\n\\multicolumn{3}{l}{\\emph{Useless hospital ranked last}} \\\\\n" + "\n".join(crows) +
        "\n\\bottomrule\n\\end{tabular}\n\\label{tab:osa_robust}\n\\end{table}\n")

# ------------------------------------------------------------------ non-convex model (federated MLP)
if "valuation_mlp" in P19:
    vm = P19["valuation_mlp"]
    keys = ["none", "5.0", "2.0", "1.0", "0.5"]
    lab = lambda k: "none" if k == "none" else f"{float(k):g}"
    rows = []
    for k in keys:
        r = vm["rho"][k]
        bm = vm["budget_matched"].get(k)
        rows.append(f"{lab(k)} & {r['osa_ensemble']['mean']:.2f} & {r['osa_param']['mean']:.2f} & {r['mr']['mean']:.2f} & "
                    f"${ci(vm['osa_ensemble_minus_mr'][k], 2)}$ & " +
                    (f"{bm['retrain_budget_matched']['mean']:.2f}" if bm else "--") + " \\\\")
    open(os.path.join(GEN, "tab_osa_mlp.tex"), "w", encoding="utf-8").write(
        "\\begin{table}[t]\n\\caption{Valuation with a non-convex model: a one-hidden-layer MLP (16 tanh units, weight decay $10^{-3}$) trained by FedAvg. "
        "Spearman $\\rho$ with the exact non-private retraining Shapley ranking of the ten largest PCa-19 hospitals (" +
        str(len(vm["seeds"])) + " held-out splits; budget-matched retraining on " + str(vm["budget_matched"]["2.0"]["n_seeds"]) +
        " of them). OSA-ens.: coalition prediction = size-weighted average of the members' log-odds; OSA-par.: size-weighted "
        "parameter average from a shared initialisation; MR: exact multi-round reconstruction from one DP FedAvg-MLP run. All "
        "columns cost the stated $\\varepsilon$.}\n"
        "\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccc}\n\\toprule\n$\\varepsilon$ & OSA-ens. & OSA-par. & MR & "
        "OSA-ens.$-$MR & Budget-m. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:osa_mlp}\n\\end{table}\n")
    for k, nm in (("none", "NoDP"), ("2.0", "Two"), ("1.0", "One"), ("5.0", "Five"), ("0.5", "Half")):
        m(f"mlpOsa{nm}", f"{vm['rho'][k]['osa_ensemble']['mean']:.2f}")
        m(f"mlpPar{nm}", f"{vm['rho'][k]['osa_param']['mean']:.2f}")
        m(f"mlpMr{nm}", f"{vm['rho'][k]['mr']['mean']:.2f}")
        m(f"mlpOsaVsMr{nm}", ci(vm["osa_ensemble_minus_mr"][k], 2))
    for k, nm in (("2.0", "Two"), ("1.0", "One")):
        b = vm["budget_matched"][k]
        m(f"mlpBm{nm}", f"{b['retrain_budget_matched']['mean']:.2f}")
        m(f"mlpOsaVsBm{nm}", ci(b["osa_ensemble_minus_bm"], 2))
        m(f"mlpTrueEps{nm}", f"{vm['true_eps_per_run'][k]:.0f}")
    m("mlpBmSeeds", vm["budget_matched"]["2.0"]["n_seeds"])

# ------------------------------------------------------------------ stronger baselines
if "baselines" in P19:
    bl_ = P19["baselines"]
    rows = []
    for mname in bl_["models"]:
        w_ = bl_["within"][mname]; lo_ = bl_["loco"][mname]
        dw = bl_["within_diff_vs_FedAvg"].get(mname); dl = bl_["loco_diff_vs_FedAvg_by_country"].get(mname)
        dp = bl_["within_dp_eps1"].get(mname)
        rows.append(f"{mname} & {w_['macro']['mean']:.3f} & {'--' if dw is None else '$' + ci(dw) + '$'} & "
                    f"{lo_['auc']['mean']:.3f} & {'--' if dl is None else '$' + ci(dl) + '$'} & {lo_['slope']['mean']:.2f} & "
                    f"{'--' if dp is None else format(dp['mean'], '.3f')} \\\\")
        if mname == "Federated MLP":
            rows.append("\\midrule")
    open(os.path.join(GEN, "tab_baselines.tex"), "w", encoding="utf-8").write(
        "\\begin{table*}[t]\n\\caption{Stronger baselines on PCa-19 (" + str(len(bl_["seeds"])) + " held-out seeds). Within-hospital: "
        "mean per-hospital test AUC over 19 hospitals. Leave-one-country-out: mean AUC over the 8 held-out countries and mean "
        "calibration slope; differences vs FedAvg are paired over countries ($n=8$). Last column: federated variants trained with "
        "DP at $\\varepsilon=1$. FedNova equals FedAvg here (identical local steps at every hospital).}\n\\centering\\small\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n& \\multicolumn{2}{c}{Within hospital} & \\multicolumn{3}{c}{Leave-one-country-out} & DP \\\\\n"
        "Model & AUC & $-$FedAvg [95\\% CI] & AUC & $-$FedAvg [95\\% CI] & Slope & $\\varepsilon=1$ \\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:baselines}\n\\end{table*}\n")
    W_ = bl_["within"]
    m("blAvg", f3(W_["FedAvg"]["macro"]["mean"])); m("blAvgLong", f3(W_["FedAvg (400 steps)"]["macro"]["mean"]))
    m("blAdam", f3(W_["FedAdam"]["macro"]["mean"])); m("blScaf", f3(W_["SCAFFOLD"]["macro"]["mean"]))
    m("blMLP", f3(W_["Federated MLP"]["macro"]["mean"])); m("blGB", f3(W_["Central gradient boosting"]["macro"]["mean"]))
    m("blRF", f3(W_["Central random forest"]["macro"]["mean"])); m("blCMLP", f3(W_["Central MLP"]["macro"]["mean"]))
    m("blCLR", f3(W_["Central LR"]["macro"]["mean"]))
    D_ = bl_["within_diff_vs_FedAvg"]
    m("blDLong", ci(D_["FedAvg (400 steps)"])); m("blDAdam", ci(D_["FedAdam"])); m("blDMLP", ci(D_["Federated MLP"]))
    m("blDGB", ci(D_["Central gradient boosting"])); m("blDScaf", ci(D_["SCAFFOLD"]))
    L_ = bl_["loco_diff_vs_FedAvg_by_country"]
    m("blLMLP", ci(L_["Federated MLP"])); m("blLGB", ci(L_["Central gradient boosting"]))
    m("blScafDrift", f"{100 * (1 - bl_['drift']['SCAFFOLD']['mean'] / bl_['drift']['FedAvg']['mean']):.0f}")
    m("blProxDrift", f"{100 * (1 - bl_['drift']['FedProx']['mean'] / bl_['drift']['FedAvg']['mean']):.0f}")
    dpd = bl_["within_dp_eps1_diff_vs_FedAvg"]
    m("blDpScaf", ci(dpd["SCAFFOLD"])); m("blDpAdam", ci(dpd["FedAdam"])); m("blDpProx", ci(dpd["FedProx"]))
if "baselines_tcga" in P19:
    bt_ = P19["baselines_tcga"]
    for a_, nm in (("0.1", "Low"), ("1.0", "Mid")):
        m(f"btScaf{nm}", ci(bt_[a_ + "_diff_vs_FedAvg"]["SCAFFOLD"])); m(f"btAdam{nm}", ci(bt_[a_ + "_diff_vs_FedAvg"]["FedAdam"]))
        m(f"btMLP{nm}", ci(bt_[a_ + "_diff_vs_FedAvg"]["Federated MLP"])); m(f"btAvg{nm}", f3(bt_[a_]["FedAvg"]["mean"]))
    m("btGB", f3(bt_["central"]["Central gradient boosting"]["mean"])); m("btCLR", f3(bt_["central"]["Central LR"]["mean"]))
if "strategies_converged" in P19:
    sc_ = P19["strategies_converged"]
    m("cvgFLminusLOC", ci(sc_["diff"]["FL-LOC"])); m("cvgFLminusNoh", ci(sc_["diff"]["FL-BL_noh"]))
    m("cvgFRminusFL", ci(sc_["diff"]["FR_test-FL"])); m("cvgCENminusFL", ci(sc_["diff"]["CEN-FL"]))
    m("cvgFL", f3(sc_["macro"]["FL"]["mean"])); m("cvgWFL", f3(sc_["weighted"]["FL"]["mean"])); m("cvgWNoh", f3(sc_["weighted"]["BL_noh"]["mean"]))

# ------------------------------------------------------------------ subsampling
if "subsampling" in P19:
    ss_ = P19["subsampling"]
    rows = []
    for x in ss_["rows"]:
        e = "no DP" if x["epsilon"] is None else f"{x['epsilon']:g}"
        sg = "--" if x["sigma"] is None else f"{x['sigma']:.1f}"
        vs = "" if x["q"] >= 1 else f" (${x['pca19_macro_vs_full']['mean']:+.3f}$)"
        rows.append(f"{x['config']} & {e} & {sg} & {x['steps']} & {x['pca19_macro']['mean']:.3f}{vs} & {x['pca19_slope']['mean']:.2f} \\\\")
        if x["epsilon"] == 0.5:
            rows.append("\\midrule")
    open(os.path.join(GEN, "tab_subsampling.tex"), "w", encoding="utf-8").write(
        "\\begin{table}[t]\n\\caption{Poisson subsampling with sampled-Gaussian RDP accounting (FedProx, " + str(len(ss_["seeds"])) +
        " held-out seeds). $\\sigma$ = noise multiplier needed for the stated $\\varepsilon$ over the given number of local steps; "
        "PCa-19: mean per-hospital AUC (difference to full batch in brackets) and "
        "calibration slope.}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
        "Sampling & $\\varepsilon$ & $\\sigma$ & Steps & AUC & Slope \\\\\n\\midrule\n" + "\n".join(rows[:-1]) +
        "\n\\bottomrule\n\\end{tabular}\n\\label{tab:subsampling}\n\\end{table}\n")
    R_ = {(x["config"], x["epsilon"]): x for x in ss_["rows"]}
    m("ssSigFull", f"{R_[('full batch (q=1)', 1.0)]['sigma']:.1f}"); m("ssSigThree", f"{R_[('q=0.3', 1.0)]['sigma']:.1f}")
    m("ssSigOne", f"{R_[('q=0.1', 1.0)]['sigma']:.1f}")
    m("ssTThreeOne", ci(R_[("q=0.3", 1.0)]["tcga_auc_vs_full"])); m("ssTThreeHalf", ci(R_[("q=0.3", 0.5)]["tcga_auc_vs_full"]))
    m("ssTOneOne", ci(R_[("q=0.1", 1.0)]["tcga_auc_vs_full"])); m("ssPThreeOne", ci(R_[("q=0.3", 1.0)]["pca19_macro_vs_full"]))
    m("ssTFullOne", f3(R_[("full batch (q=1)", 1.0)]["tcga_auc"]["mean"])); m("ssTThreeOneAbs", f3(R_[("q=0.3", 1.0)]["tcga_auc"]["mean"]))

# ------------------------------------------------------------------ OSA-Shapley vs cheap proxies (practicality)
if "valuation_proxies" in P19:
    vq = P19["valuation_proxies"]; L, Sm = vq["large10"], vq["small10"]
    _pv = {x["epsilon"]: x for x in P19["private_valuation"]["rows"]} if "private_valuation" in P19 else {}
    NAMES = [("size", "Hospital size $n_k$", "size (public)"),
             ("positives", "Number of csPCa cases", "one count"),
             ("statistics", "Similarity of summary statistics", "feature means, prevalence"),
             ("own_test_auc", "Self-reported local AUC", "one AUC (unverifiable)"),
             ("ref_auc", "Single-model AUC on reference set", "one DP model ($\\varepsilon$)"),
             ("osa_loo", "Leave-one-out of averaged models", "one DP model ($\\varepsilon$)"),
             ("or", "OR reconstruction", "DP-FedAvg run ($\\varepsilon$)"),
             ("mr", "MR reconstruction (GTG bound)", "DP-FedAvg run ($\\varepsilon$)"),
             ("retrain_budget_matched", "Retraining, budget-matched", "$2^{K-1}$ runs ($\\varepsilon$)"),
             ("osa", "\\textbf{OSA-Shapley (ours)}", "one DP model ($\\varepsilon$)")]
    def _cell(blk, k, mth, large=False):
        if mth == "retrain_budget_matched" and large:
            if k == "none":
                return "--"
            return f"{_pv[float(k)]['retrain_budget_matched']['spearman']['mean']:.2f}" if float(k) in _pv else "--"
        if k not in blk["rho"] or mth not in blk["rho"][k]:
            return "--"
        v = blk["rho"][k][mth]["mean"]
        return f"\\textbf{{{v:.2f}}}" if mth == "osa" else f"{v:.2f}"
    def _det(mth):
        c = L.get("corrupt", {}).get("2.0", {})
        if mth in c:
            return f"{100 * c[mth]['mean']:.0f}\\%"
        if mth in ("mr", "or"):
            return f"{100 * P19['valuation_plus']['corrupt']['2.0'][mth]['last']['mean']:.0f}\\%"
        if mth == "retrain_budget_matched":
            return f"{100 * P19['valuation_plus']['corrupt']['2.0']['retrain_budget_matched']['last']['mean']:.0f}\\%"
        return "--"
    rows = []
    for mth, nm, cost in NAMES:
        rows.append(f"{nm} & {cost} & " + " & ".join(_cell(L, k, mth, True) for k in ("none", "2.0", "1.0")) +
                    f" & {_det(mth)} & " + " & ".join(_cell(Sm, k, mth) for k in ("none", "2.0", "1.0")) + " \\\\")
    open(os.path.join(GEN, "tab_proxies.tex"), "w", encoding="utf-8").write(
        "\\begin{table*}[t]\n\\caption{Is OSA-Shapley worth its cost? Agreement (Spearman $\\rho$) of cheap contribution proxies and of "
        "privacy-honest valuations with the exact non-private retraining Shapley ranking, on real PCa-19 hospitals (" + str(len(vq["seeds"])) +
        " held-out splits). \\emph{What is shared}: what each hospital must reveal. Proxies that use a model use the same DP releases as "
        "OSA-Shapley and therefore cost the same $\\varepsilon$; size, counts and statistics are shared without DP. \\emph{Useless}: share of "
        "splits in which a hospital whose training labels were replaced by random labels (" + str(L.get("corrupt_hospital", "")) +
        ") is ranked last at $\\varepsilon=2$; a tie for last place does not count.}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llccccccc}\n\\toprule\n & & \\multicolumn{4}{c}{Ten largest hospitals} & \\multicolumn{3}{c}{Ten smallest hospitals} \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\nContribution score & What is shared & no DP & $\\varepsilon=2$ & $\\varepsilon=1$ & Useless & "
        "no DP & $\\varepsilon=2$ & $\\varepsilon=1$ \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\label{tab:proxies}\n\\end{table*}\n")
    for tag, blk in (("L", L), ("S", Sm)):
        for k, nm in (("none", "NoDP"), ("5.0", "Five"), ("2.0", "Two"), ("1.0", "One")):
            if k not in blk["rho"]:
                continue
            for mth, mn in (("size", "Size"), ("positives", "Pos"), ("statistics", "Stat"), ("own_test_auc", "Own"),
                            ("ref_auc", "Ref"), ("osa_loo", "Loo"), ("osa", "Osa"), ("mr", "Mr"), ("or", "Or"),
                            ("retrain_budget_matched", "Bm")):
                if mth in blk["rho"][k]:
                    m(f"px{tag}{mn}{nm}", f"{blk['rho'][k][mth]['mean']:.2f}")
                if mth in blk["osa_minus"][k]:
                    m(f"pxd{tag}{mn}{nm}", ci(blk["osa_minus"][k][mth], 2))
    for k, nm in (("none", "NoDP"), ("2.0", "Two"), ("1.0", "One")):
        for mth, mn in (("size", "Size"), ("statistics", "Stat"), ("own_test_auc", "Own"), ("ref_auc", "Ref"), ("osa_loo", "Loo"), ("osa", "Osa")):
            m(f"pxC{mn}{nm}", f"{100 * L['corrupt'][k][mth]['mean']:.0f}\\%")
    m("pxSmallMin", min(Sm["sizes"])); m("pxSmallMax", max(Sm["sizes"]))
    m("pxSmallHosp", ", ".join(Sm["hospitals"]))

# ------------------------------------------------------------------ privacy accounting of retraining-based valuation
from fl_study import private_valuation as _PV, fl
_EP = [10.0, 5.0, 2.0, 1.0]
def _true(eps, runs, steps=100, delta=1e-5):
    sig = fl.calibrate_sigma(eps, steps, delta)
    return fl.rdp_epsilon(sig, steps * runs, delta)[0]
def _fmt(x):
    if x >= 1e5:
        e = int(np.floor(np.log10(x))); return f"$\\approx 10^{{{e}}}$"
    return f"{x:,.0f}".replace(",", "{,}") if x >= 10 else f"{x:.1f}"
acc_rows = []
for K in (4, 10, 19, 50):
    r = 2 ** (K - 1)
    acc_rows.append(f"Exact, $K={K}$ & {_fmt(float(r)) if r >= 1e5 else str(r).replace('512', '512')} & " + " & ".join(_fmt(_true(e, r)) for e in _EP) +
                    f" & {_fmt(2 ** ((K - 1) / 2)) if K < 40 else _fmt(2 ** ((K - 1) / 2))} \\\\")
for K in (10, 19, 50):
    r = int(100 * (K + 1) / 2)
    acc_rows.append(f"100 permutations, $K={K}$ & " + f"{r:,}".replace(",", "{,}") + " & " + " & ".join(_fmt(_true(e, r)) for e in _EP) +
                    f" & {np.sqrt(r):.0f} \\\\")
acc_rows.append("OSA-Shapley, MR, OR (any $K$) & 1 & 10 & 5 & 2 & 1 & 1 \\\\")
open(os.path.join(GEN, "tab_accounting.tex"), "w", encoding="utf-8").write(
    "\\begin{table}[t]\n\\caption{The privacy cost that retraining-based valuation actually spends (Proposition~2). $r$: number of DP training "
    "runs that contain a given hospital's patients (exact Shapley: $2^{K-1}$; permutation sampling without caching: on average $100(K+1)/2$). "
    "Each run is $N=100$ full-batch DP-GD steps calibrated to the per-run $\\varepsilon$; the table gives the composed $\\varepsilon$ per hospital "
    "($\\delta=10^{-5}$). Last column: factor by which every run's noise must grow to keep the stated budget ($\\sqrt r$). Post-processing "
    "valuations spend the stated budget once.}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lrccccr}\n\\toprule\n"
    " & & \\multicolumn{4}{c}{True $\\varepsilon$ for a per-run $\\varepsilon$ of} & Noise \\\\\n\\cmidrule(lr){3-6}\n"
    "Valuation & $r$ & 10 & 5 & 2 & 1 & factor \\\\\n\\midrule\n" + "\n".join(acc_rows) +
    "\n\\bottomrule\n\\end{tabular}}\n\\label{tab:accounting}\n\\end{table}\n")
m("accMcNineteenOne", f"{_true(1.0, 1000):.0f}"); m("accMcTenOne", f"{_true(1.0, 550):.0f}"); m("accMcFiftyOne", f"{_true(1.0, 2550):.0f}")
m("accFourOne", f"{_true(1.0, 8):.1f}"); m("accFourTwo", f"{_true(2.0, 8):.1f}")

# ------------------------------------------------------------------ second real federation: UCI Heart Disease (4 hospitals)
_HP = os.path.join(ROOT, "results", "final", "pca19_heart_valuation.json")
if os.path.exists(_HP):
    hv = json.load(open(_HP))
    HN = [("size", "Hospital size $n_k$"), ("statistics", "Similarity of summary statistics"),
          ("own_test_auc", "Self-reported local AUC"), ("ref_auc", "Single-model AUC on reference set"),
          ("osa_loo", "Leave-one-out of averaged models"), ("or", "OR reconstruction"), ("mr", "MR reconstruction (GTG bound)"),
          ("retrain_per_run", "Retraining, per run"), ("retrain_budget_matched", "Retraining, budget-matched"),
          ("osa", "\\textbf{OSA-Shapley (ours)}")]
    def _hc(k, mth):
        if k not in hv["rho"] or mth not in hv["rho"][k]:
            return "--"
        v = hv["rho"][k][mth]["mean"]
        return f"\\textbf{{{v:.2f}}}" if mth == "osa" else f"{v:.2f}"
    def _hd(mth):
        c = hv["corrupt"].get("2.0", {})
        return f"{100 * c[mth]['mean']:.0f}\\%" if mth in c else "--"
    rows = [f"{nm} & " + " & ".join(_hc(k, mth) for k in ("none", "5.0", "2.0", "1.0")) + f" & {_hd(mth)} \\\\" for mth, nm in HN]
    sites = ", ".join(f"{x['name'].split(' (')[0]} ({x['n']})" for x in hv["sites"])
    open(os.path.join(GEN, "tab_heart.tex"), "w", encoding="utf-8").write(
        "\\begin{table}[t]\n\\caption{Second real federation: the four hospitals of the UCI Heart Disease collection (" + sites + " patients), "
        "coronary artery disease, " + str(len(hv["seeds"])) + " held-out splits. Spearman $\\rho$ with the exact non-private retraining Shapley "
        "ranking; with four hospitals $\\rho$ moves in steps of 0.2. Per-run retraining really costs $\\varepsilon\\approx" +
        f"{hv['true_epsilon']['5.0']:.0f}$, {hv['true_epsilon']['2.0']:.1f} and {hv['true_epsilon']['1.0']:.1f}" +
        " at a stated 5, 2 and 1. \\emph{Useless}: share of splits in which " + hv["corrupt_hospital"].split(" (")[0] +
        ", given random training labels, is ranked last at $\\varepsilon=2$.}\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\nContribution score & no DP & $\\varepsilon=5$ & $\\varepsilon=2$ & $\\varepsilon=1$ & Useless \\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:heart}\n\\end{table}\n")
    for k, nm in (("none", "NoDP"), ("5.0", "Five"), ("2.0", "Two"), ("1.0", "One")):
        for mth, mn in (("size", "Size"), ("statistics", "Stat"), ("own_test_auc", "Own"), ("ref_auc", "Ref"), ("osa_loo", "Loo"),
                        ("osa", "Osa"), ("mr", "Mr"), ("or", "Or"), ("retrain_budget_matched", "Bm"), ("retrain_per_run", "Pr")):
            if mth in hv["rho"].get(k, {}):
                m(f"hr{mn}{nm}", f"{hv['rho'][k][mth]['mean']:.2f}")
            if mth in hv["osa_minus"].get(k, {}):
                m(f"hrd{mn}{nm}", ci(hv["osa_minus"][k][mth], 2))
    for k, nm in (("2.0", "Two"), ("1.0", "One")):
        for mth, mn in (("osa", "Osa"), ("ref_auc", "Ref"), ("size", "Size")):
            m(f"hrC{mn}{nm}", f"{100 * hv['corrupt'][k][mth]['mean']:.0f}\\%")
    m("hrN", hv["n"]); m("hrFeat", hv["n_features"]); m("hrAuc", f"{hv['fed_auc']['mean']:.2f}")
    m("hrCorrupt", hv["corrupt_hospital"].split(" (")[0])
    m("hrPhiMin", f"{min(hv['truth_mean']):.3f}"); m("hrPhiMax", f"{max(hv['truth_mean']):.3f}")
    m("hrTrueOne", f"{hv['true_epsilon']['1.0']:.1f}"); m("hrTrueTwo", f"{hv['true_epsilon']['2.0']:.1f}")
    # evidence behind the decision rule. nu = median training patients per hospital / sqrt(model parameters),
    # the quantity that sets the signal-to-noise of one DP release (Proposition 3)
    def _nu(sizes, p):
        return float(np.median(np.array(sizes, float) * 0.8) / np.sqrt(p))
    vq_ = P19["valuation_proxies"]; vm_ = P19["valuation_mlp"]
    p_lr = 7; p_mlp = 6 * 16 + 16 + 16 + 1; p_h = hv["n_features"] + 1
    sett = [("PCa-19, 10 largest", "logistic", 10, p_lr, _nu(vq_["large10"]["sizes"], p_lr),
             vq_["large10"]["rho"]["none"]["osa"]["mean"], vq_["large10"]["rho"]["2.0"]["osa"]["mean"], vq_["large10"]["rho"]["2.0"]["mr"]["mean"],
             vq_["large10"]["osa_minus"]["2.0"]["mr"], vq_["large10"]["rho"]["2.0"]["size"]["mean"], "OSA"),
            ("PCa-19, 10 smallest", "logistic", 10, p_lr, _nu(vq_["small10"]["sizes"], p_lr),
             vq_["small10"]["rho"]["none"]["osa"]["mean"], vq_["small10"]["rho"]["2.0"]["osa"]["mean"], vq_["small10"]["rho"]["2.0"]["mr"]["mean"],
             vq_["small10"]["osa_minus"]["2.0"]["mr"], vq_["small10"]["rho"]["2.0"]["size"]["mean"], "OSA, MR or size"),
            ("Heart Disease, 4", "logistic", 4, p_h, _nu([x["n"] for x in hv["sites"]], p_h),
             hv["rho"]["none"]["osa"]["mean"], hv["rho"]["2.0"]["osa"]["mean"], hv["rho"]["2.0"]["mr"]["mean"],
             hv["osa_minus"]["2.0"]["mr"], hv["rho"]["2.0"]["size"]["mean"], "MR"),
            ("PCa-19, 10 largest", "MLP", 10, p_mlp, _nu(vq_["large10"]["sizes"], p_mlp),
             vm_["rho"]["none"]["osa_ensemble"]["mean"], vm_["rho"]["2.0"]["osa_ensemble"]["mean"], vm_["rho"]["2.0"]["mr"]["mean"],
             vm_["osa_ensemble_minus_mr"]["2.0"], None, "MR")]
    rows = []
    for nm, mod, K, p_, nu, a0, a2, m2, d2, sz, best in sett:
        rows.append(f"{nm} & {mod} & {K} & {p_} & {nu:.0f} & {a0:.2f} & {a2:.2f} & {m2:.2f} & ${ci(d2, 2)}$ & " +
                    ("--" if sz is None else f"{sz:.2f}") + f" & {best} \\\\")
        m("nu" + {"PCa-19, 10 largest|logistic": "Large", "PCa-19, 10 smallest|logistic": "Small", "Heart Disease, 4|logistic": "Heart",
                  "PCa-19, 10 largest|MLP": "Mlp"}[nm + "|" + mod], f"{nu:.0f}")
    open(os.path.join(GEN, "tab_decision.tex"), "w", encoding="utf-8").write(
        "\\begin{table*}[t]\n\\caption{Evidence behind the decision rule, from two real federations. $p$: model parameters; "
        "$\\nu$: median training patients per hospital divided by $\\sqrt p$, which sets the signal-to-noise ratio of one DP release "
        "(Proposition~3). $\\rho$: Spearman correlation with the exact non-private retraining Shapley ranking (20 held-out splits). "
        "OSA$-$MR: paired difference with 95\\% CI at $\\varepsilon=2$. Size: agreement of hospital size with the same ranking.}\n"
        "\\centering\\small\n\\begin{tabular}{llrrrcccccl}\n\\toprule\n & & & & & OSA & OSA & MR & & & \\\\\n"
        "Federation & Model & $K$ & $p$ & $\\nu$ & no DP & $\\varepsilon=2$ & $\\varepsilon=2$ & OSA$-$MR & Size & Best at $\\varepsilon=2$ \\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\label{tab:decision}\n\\end{table*}\n")
# ------------------------------------------------------------------ pre-registered test of the decision rule (nu sweep)
_NP = os.path.join(ROOT, "results", "final", "pca19_nu_sweep.json")
if os.path.exists(_NP):
    ns = json.load(open(_NP)); sc = json.load(open(os.path.join(ROOT, "results", "final", "pca19_nu_sweep_scoring.json")))
    rows = []
    for c in ns["conditions"]:
        lab = (f"A: {100 * c['fraction']:.0f}\\% of patients" if c["arm"] == "A" else f"B: +{c['extra_features']} features")
        rows.append(f"{lab} & {c['p']} & {c['nu']:.0f} & {c['2.0']['osa']['mean']:.2f} & {c['2.0']['mr']['mean']:.2f} & "
                    f"${ci(c['2.0']['osa_minus_mr'], 2)}$ & {c['1.0']['osa']['mean']:.2f} & {c['1.0']['mr']['mean']:.2f} & "
                    f"${ci(c['1.0']['osa_minus_mr'], 2)}$ \\\\")
    open(os.path.join(GEN, "tab_nu_sweep.tex"), "w", encoding="utf-8").write(
        "\\begin{table*}[t]\n\\caption{Pre-registered test of the decision rule on the ten largest PCa-19 hospitals (" + str(len(ns["seeds"])) +
        " held-out splits). Arm A keeps a random fraction of every hospital's training patients; arm B adds uninformative features. "
        "In every condition the reference is the exact non-private retraining Shapley ranking of that condition. $\\rho$: Spearman correlation "
        "with the reference; OSA$-$MR: paired difference with 95\\% CI.}\n\\centering\\small\n\\begin{tabular}{lrrcccccc}\n\\toprule\n"
        " & & & \\multicolumn{3}{c}{$\\varepsilon=2$} & \\multicolumn{3}{c}{$\\varepsilon=1$} \\\\\n\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
        "Condition & $p$ & $\\nu$ & OSA & MR & OSA$-$MR & OSA & MR & OSA$-$MR \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\label{tab:nu_sweep}\n\\end{table*}\n")
    import matplotlib.pyplot as _pl
    _f, _a = _pl.subplots(figsize=(3.5, 2.5))
    for arm, mk, nmA in (("A", "o", "fewer patients"), ("B", "s", "more parameters")):
        for e, col in (("2.0", "#4A3AA7"), ("1.0", "#EB6834")):
            C_ = sorted([c for c in ns["conditions"] if c["arm"] == arm], key=lambda c: c["nu"])
            x = [c["nu"] * (0.97 if e == "2.0" else 1.03) for c in C_]; d = [c[e]["osa_minus_mr"] for c in C_]
            _a.errorbar(x, [v["mean"] for v in d], yerr=[[v["mean"] - v["ci_low"] for v in d], [v["ci_high"] - v["mean"] for v in d]],
                        fmt=mk + ("-" if arm == "A" else "--"), color=col, ms=3.5, lw=1.1, capsize=2,
                        label=f"$\\varepsilon$={float(e):g}, {nmA}")
    _a.axhline(0, color="#8a8985", lw=0.8)
    _a.axvspan(50, 100, color="#e8e6f6", zorder=0, lw=0)
    _a.text(70, _a.get_ylim()[1] * 0.92, "predicted\nswitch-over", ha="center", va="top", fontsize=6.5, color="#4A3AA7")
    from matplotlib.ticker import NullFormatter, FixedLocator
    _a.set_xscale("log"); _a.xaxis.set_major_locator(FixedLocator([20, 30, 50, 100])); _a.set_xticklabels(["20", "30", "50", "100"])
    _a.xaxis.set_minor_formatter(NullFormatter())
    _a.set_xlabel("$\\nu$ = median training patients / $\\sqrt{\\mathrm{parameters}}$", fontsize=7)
    _a.set_ylabel("OSA $-$ MR (Spearman $\\rho$)", fontsize=7); _a.tick_params(labelsize=6.5)
    _a.legend(fontsize=5.6, frameon=False, loc="lower right")
    _a.spines["top"].set_visible(False); _a.spines["right"].set_visible(False)
    _f.tight_layout(); _f.savefig(os.path.join(FIG, "figure20_nu_sweep.pdf")); _pl.close(_f)
    C0 = {(c["arm"], c["fraction"], c["extra_features"]): c for c in ns["conditions"]}
    m("nsCrossTwo", f"{sc['crossover_eps2'][0]:.0f}" if sc["crossover_eps2"] else "--")
    m("nsCrossOneMax", f"{max(sc['crossover_eps1']):.0f}" if sc["crossover_eps1"] else "--")
    m("nsCrossOneMin", f"{min(sc['crossover_eps1']):.0f}" if sc["crossover_eps1"] else "--")
    m("nsSpear", f"{sc['H3_spearman']:.2f}"); m("nsLowerCount", sc["H6_lower_count"])
    for key, nm in (((("A", 0.4, 0)), "AFortyFive"), ((("A", 0.25, 0)), "ATwentyEight"), ((("A", 0.15, 0)), "ASeventeen"),
                    ((("A", 0.6, 0)), "ASixtyEight"), ((("B", 1.0, 21)), "BTwentyOne"), ((("B", 1.0, 50)), "BFifty"), ((("B", 1.0, 100)), "BHundred")):
        c = C0[key]
        m(f"ns{nm}Nu", f"{c['nu']:.0f}"); m(f"ns{nm}Two", ci(c["2.0"]["osa_minus_mr"], 2)); m(f"ns{nm}One", ci(c["1.0"]["osa_minus_mr"], 2))

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
# ------------------------------------------------------------------ PCa-19 figures
keys9 = ["LOC", "FL", "FedProx", "CEN", "FR_test", "BL_noh", "BL_ettala"]
save(ch.bar_with_ci(["Local", "FedAvg", "FedProx", "Central.", "Free-rider", "Noh", "Ettala"], [s9["macro"][k] for k in keys9],
                    [ch.C["LOC"], ch.C["FL"], ch.C["FedProx"], ch.C["CEN"], ch.C["FR"], "#4a3aa7", "#8a8985"],
                    "PCa-19: mean per-hospital test AUC over 19 hospitals (95% CI, 20 seeds)", ylim=(0.5, 1.0)),
     "figure14_pca19_strategies.pdf")
dp9 = [cells9[e] for e in [10.0, 5.0, 2.0, 1.0, 0.5]]
save(ch.lines([10, 5, 2, 1, 0.5], [{"name": n, "color": col, "dash": dsh, "mean": [x[k]["macro"]["mean"] for x in dp9],
                                     "low": [x[k]["macro"]["ci_low"] for x in dp9], "high": [x[k]["macro"]["ci_high"] for x in dp9]}
                                    for k, n, col, dsh in [("FedAvg", "FedAvg", ch.C["FedAvg"], "-"),
                                                           ("FedProx", "FedProx", ch.C["FedProx"], "-")]],
              "PCa-19: AUC vs privacy budget (19 real hospitals)", "$\\varepsilon$", "mean per-hospital AUC", xlog=True,
              xticks=([10, 5, 2, 1, 0.5], ["10", "5", "2", "1", "0.5"]),
              ref=cells9["none"]["FedAvg"]["macro"]["mean"], ref_label="no DP", ylim=(0.78, 0.86), direct=False),
     "figure15_pca19_privacy.pdf")
th9 = lo9["thresholds"]
ser = [{"name": n, "color": col, "mean": lo9["dca"][n]} for n, col in
       [("FedAvg", ch.C["FL"]), ("FedProx+DP(eps=5)", ch.C["FedProx"]), ("Noh (published)", "#4a3aa7"), ("Biopsy all", ch.C["reference"])]]
ser[-1]["dash"] = "--"
save(ch.lines(th9, ser, "Decision curve on held-out countries (patient-weighted)", "risk threshold for biopsy", "net benefit",
              ref=0, ref_label="biopsy none", direct=False), "figure16_pca19_dca.pdf")
mods = ["FedAvg", "FedProx+DP(eps=5)", "FedProx+DP(eps=1)", "Noh (published)"]
save(ch.grouped_bars(["FedAvg", "DP $\\varepsilon$=5", "DP $\\varepsilon$=1", "Noh"], ["as transported", "intercept", "intercept + slope"],
                     [[rc9[mm][meth]["slope"]["mean"] for mm in mods] for meth in ("raw", "intercept", "logistic")],
                     ["#8a8985", "#2a78d6", "#1baf7a"], "Calibration slope on held-out countries (ideal = 1)", "slope",
                     ref=1.0, ylim=(0.5, 1.3)), "figure17_pca19_recalibration.pdf")
print("figures written to", FIG)

# ------------------------------------------------------------------ figure 2: system architecture (vector)
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def _box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=7.3, fontweight="bold", color="#1f1f1d")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.085 - i * 0.052, ln, ha="center", va="top", fontsize=6.1, color="#3d3d3a")


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#6b6a66", lw=0.9, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
G, B, O, P, Y = ("#eef7f1", "#1baf7a"), ("#eaf2fc", "#2a78d6"), ("#fdf0ea", "#eb6834"), ("#f3effa", "#4a3aa7"), ("#fdf6e3", "#c98a00")
_box(ax, 0.01, 0.56, 0.20, 0.40, "Cohorts", ["PCa-19: 19 hospitals,", "5,610 men, 8 countries", "UCI Heart: 4 hospitals,", "920 patients", "all real hospitals"], *G)
_box(ax, 0.25, 0.56, 0.20, 0.40, "Preprocessing", ["leakage-free whitelist", "statistics from training", "rows only (secure sums)", "held-out country never", "seen (LOCO)"], *G)
_box(ax, 0.49, 0.56, 0.22, 0.40, "Hospital k (local DP)", ["per-patient clipping, C=1", "Gaussian noise sigma*C", "full batch or Poisson q", "FedProx / SCAFFOLD", "control variates"], *B)
_box(ax, 0.75, 0.56, 0.24, 0.40, "Server", ["FedAvg / FedProx /", "SCAFFOLD / FedAdam", "aggregation", "Renyi-DP accountant", "(full-batch or sampled)"], *B)
_box(ax, 0.01, 0.02, 0.30, 0.44, "Model evaluation", ["within-hospital AUC (19 sites)", "leave-one-country-out AUC,", "calibration, decision curves", "local recalibration (20%)", "personalisation (PFL)", "membership-inference audit"], *O)
_box(ax, 0.35, 0.02, 0.34, 0.44, "OSA-Shapley valuation (ours)", ["each hospital releases ONE", "DP-trained local model (eps_v)", "coalition model = size-weighted", "average (post-processing)", "exact Shapley over all 2^K", "coalitions; cost eps_v once"], *P)
_box(ax, 0.73, 0.02, 0.26, 0.44, "Statistics", ["20 paired splits", "95% CIs, Wilcoxon", "countries as units (n=8)", "tuning splits disjoint", "41 automated checks"], *Y)
_arrow(ax, 0.21, 0.78, 0.25, 0.78); _arrow(ax, 0.45, 0.78, 0.49, 0.78)
ax.annotate("", xy=(0.75, 0.82), xytext=(0.71, 0.82), arrowprops=dict(arrowstyle="-|>", color="#6b6a66", lw=0.9))
ax.annotate("", xy=(0.71, 0.72), xytext=(0.75, 0.72), arrowprops=dict(arrowstyle="-|>", color="#6b6a66", lw=0.9))
ax.text(0.73, 0.835, "$w_k$", ha="center", va="bottom", fontsize=6.5, color="#6b6a66"); ax.text(0.73, 0.705, "$w$", ha="center", va="top", fontsize=6.5, color="#6b6a66")
_arrow(ax, 0.87, 0.56, 0.16, 0.46); _arrow(ax, 0.60, 0.56, 0.52, 0.46)
save(fig, "figure2_architecture.pdf")
print("architecture figure written")


# ------------------------------------------------------------------ main valuation figure (OSA vs competitors)
if "private_valuation" in P19 and "valuation_plus" in P19:
    import matplotlib.pyplot as _plt
    _pv = {x["epsilon"]: x for x in P19["private_valuation"]["rows"]}
    _vp = P19["valuation_plus"]
    _E = [5.0, 2.0, 1.0, 0.5]
    _x = list(range(len(_E)))
    _fig, _ax = _plt.subplots(figsize=(3.5, 2.6))
    _ser = [("OSA-Shapley (ours)", [_pv[e]["osa"]["spearman"] for e in _E], "#4A3AA7", "-"),
            ("MR reconstruction (GTG upper bound)", [_vp["methods"][str(e)]["mr"] for e in _E], "#1BAF7A", "-"),
            ("Retraining, budget-matched", [_pv[e]["retrain_budget_matched"]["spearman"] for e in _E], "#EB6834", "-"),
            ("Retraining, per run (true $\\varepsilon$ 14--334)", [_pv[e]["retrain_per_run"]["spearman"] for e in _E], "#8A8985", "--")]
    for _n, _v, _c, _ls in _ser:
        _mu = [d["mean"] for d in _v]
        _ax.errorbar(_x, _mu, yerr=[[d["mean"] - d["ci_low"] for d in _v], [d["ci_high"] - d["mean"] for d in _v]],
                     fmt="o" + _ls, color=_c, ms=3.5, lw=1.4, capsize=2, label=_n)
    _ax.set_xticks(_x); _ax.set_xticklabels([f"{e:g}" for e in _E])
    _ax.set_xlabel("privacy budget $\\varepsilon$", fontsize=8); _ax.set_ylabel("Spearman $\\rho$ with exact Shapley", fontsize=8)
    _ax.set_ylim(-0.05, 1.0); _ax.tick_params(labelsize=7); _ax.grid(axis="y", color="#dddddd", lw=0.5)
    for _sp in ("top", "right"):
        _ax.spines[_sp].set_visible(False)
    _ax.legend(fontsize=6.3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _fig.savefig(os.path.join(FIG, "figure19_osa_valuation.pdf"), bbox_inches="tight")
    _plt.close(_fig)


# ------------------------------------------------------------------ typography: true minus signs in every generated file
import re as _re3
_MINUS = _re3.compile(r"(?<=[\s\[\(\{=&])-(?=\d)|^-(?=\d)", _re3.M)
for _f in os.listdir(GEN):
    if _f.endswith(".tex"):
        _p = os.path.join(GEN, _f)
        _t = open(_p, encoding="utf-8").read()
        _t2 = _MINUS.sub(r"\\ensuremath{-}", _t)
        if _t2 != _t:
            open(_p, "w", encoding="utf-8").write(_t2)
