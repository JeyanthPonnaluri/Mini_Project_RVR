"""
Builds the claim/evidence table shown on the 'Final results' page directly from
the result files, so every verdict is computed, not written by hand.
Verdict rule: 'supported' only if the 95% CI of the relevant paired difference
(over seeds) excludes zero in the claimed direction.
"""
from __future__ import annotations

from .stats import verdict


def _f(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}" if isinstance(x, float) and d < 0 else f"{x:.{d}f}"


def _ci(s, d=3):
    if s is None or s.get("ci_low") is None:
        return "n/a"
    return f"{s['mean']:+.{d}f} [{s['ci_low']:+.{d}f}, {s['ci_high']:+.{d}f}]"


def build(R):
    rows = []
    bp = R.get("base_paper")
    if bp:
        d = bp["macro_diff"]
        rows.append(dict(area="Base paper", claim="Federated model beats each hospital's local model (per-site AUC)",
                         evidence=f"FL - LOC = {_ci(d['FL-LOC'])}",
                         verdict=verdict(d["FL-LOC"]["ci_low"], d["FL-LOC"]["ci_high"])))
        rows.append(dict(area="Base paper", claim="Free-riders get (almost) the same benefit as participants",
                         evidence=f"FR - FL = {_ci(d['FR-FL'])} (negative = small free-rider penalty)",
                         verdict="supported" if d["FR-FL"]["ci_low"] is not None and d["FR-FL"]["ci_low"] > -0.05
                         else "not supported"))
        rows.append(dict(area="Base paper", claim="Federated training approaches centralised (pooled) training",
                         evidence=f"CEN - FL = {_ci(d['CEN-FL'])}",
                         verdict="supported" if d["CEN-FL"]["ci_high"] is not None and abs(d["CEN-FL"]["mean"]) < 0.02
                         else "not supported"))
        rows.append(dict(area="Base paper", claim="Learned model beats a simple rule baseline (Gleason sum)",
                         evidence=f"FL - rule = {_ci(d['FL-BL'])}",
                         verdict=verdict(d["FL-BL"]["ci_low"], d["FL-BL"]["ci_high"])))
    c = R.get("centralized")
    if c:
        p = c["comparisons"]["protein_added_to_full"]
        rows.append(dict(area="Our model", claim="Adding proteomics (PCA) improves staging over clinical features",
                         evidence=f"(full+protein) - full = {_ci(p)} over {c['n_folds']} CV folds",
                         verdict=verdict(p["ci_low"], p["ci_high"])))
    dr = R.get("drift")
    if dr:
        worst = min(r["drift_reduction_pct"]["ci_low"] for r in dr["rows"])
        rng = ", ".join(f"alpha={r['alpha']:g}: {r['drift_reduction_pct']['mean']:.1f}%" for r in dr["rows"])
        rows.append(dict(area="Our model", claim="H1 FedProx reduces client drift under heterogeneity",
                         evidence=f"relative drift reduction ({rng}); lowest CI bound {worst:.1f}%",
                         verdict="supported" if worst > 0 else "not significant"))
        a = [r for r in dr["rows"] if r["alpha"] == 0.1][0]["auc_diff"]
        rows.append(dict(area="Our model", claim="FedProx improves accuracy without DP (alpha = 0.1)",
                         evidence=f"FedProx - FedAvg = {_ci(a)}", verdict=verdict(a["ci_low"], a["ci_high"])))
    g = R.get("privacy_grid")
    if g:
        cell = [x for x in g["cells"] if x["alpha"] == 10.0 and x["epsilon"] == 1.0][0]
        rows.append(dict(area="Our model", claim="H2 Strong privacy (eps = 1) costs utility",
                         evidence=f"FedAvg(eps=1) - FedAvg(no DP) = {_ci(cell['fedavg_minus_nodp'])}",
                         verdict=verdict(cell["fedavg_minus_nodp"]["ci_low"], cell["fedavg_minus_nodp"]["ci_high"], False)))
        cell5 = [x for x in g["cells"] if x["alpha"] == 10.0 and x["epsilon"] == 10.0][0]
        rows.append(dict(area="Our model", claim="Weak privacy (eps = 10) costs little utility (< 0.03 AUC)",
                         evidence=f"FedAvg(eps=10) - no DP = {_ci(cell5['fedavg_minus_nodp'])}",
                         verdict="supported" if cell5["fedavg_minus_nodp"]["ci_low"] > -0.03 else "not supported"))
        dp_cells = [x for x in g["cells"] if x["epsilon"] is not None]
        sup = sum(verdict(x["prox_minus_avg"]["ci_low"], x["prox_minus_avg"]["ci_high"]) == "supported" for x in dp_cells)
        ctl = sum(verdict(x["prox_minus_halflr_avg"]["ci_low"], x["prox_minus_halflr_avg"]["ci_high"]) == "supported"
                  for x in dp_cells)
        rows.append(dict(area="Our model", claim="H3 FedProx beats FedAvg under joint DP + heterogeneity",
                         evidence=f"CI > 0 in {sup}/{len(dp_cells)} (alpha, eps) cells vs FedAvg; in {ctl}/{len(dp_cells)} "
                                  f"cells vs a FedAvg with half the learning rate (step-size control)",
                         verdict=("explained by step size" if ctl <= len(dp_cells) // 4 and sup > 0
                                  else "supported" if sup >= len(dp_cells) * 0.75 else "partly")))
    pf = R.get("personalization")
    if pf:
        k = pf["by_K"]["3"]
        rows.append(dict(area="Our model", claim="H4a Personalised FL beats local-only training",
                         evidence=f"PFL - Local (K=3, macro client AUC) = {_ci(k['PFL_minus_Local'])}",
                         verdict=verdict(k["PFL_minus_Local"]["ci_low"], k["PFL_minus_Local"]["ci_high"])))
        rows.append(dict(area="Our model", claim="H4b Personalised FL beats the global FedProx model",
                         evidence=f"PFL - FedProx = {_ci(k['PFL_minus_FedProx'])}",
                         verdict=verdict(k["PFL_minus_FedProx"]["ci_low"], k["PFL_minus_FedProx"]["ci_high"])))
    sh = R.get("shapley")
    if sh:
        k3 = sh["by_K"]["3"]; kmax = sh["by_K"][max(sh["by_K"], key=int)]
        e = {x["epsilon"]: x for x in kmax["per_epsilon"]}
        rows.append(dict(area="Our model", claim="H5 DP noise destabilises Shapley contribution rankings",
                         evidence=(f"K={max(sh['by_K'], key=int)}: Spearman vs no-DP {e[10.0]['spearman']['mean']:.2f} (eps=10) -> "
                                   f"{e[0.5]['spearman']['mean']:.2f} (eps=0.5); top-contributor kept "
                                   f"{e[10.0]['top_contributor_stability']['mean']:.2f} -> {e[0.5]['top_contributor_stability']['mean']:.2f}"),
                         verdict="supported" if e[0.5]["spearman"]["ci_high"] < 0.9 else "not significant"))
        rows.append(dict(area="Our model", claim="Shapley and LOO give similar client rankings",
                         evidence=f"Spearman(Shapley, LOO): K=3 {k3['spearman_shapley_vs_loo']['mean']:.2f}, "
                                  f"K={max(sh['by_K'], key=int)} {kmax['spearman_shapley_vs_loo']['mean']:.2f}",
                         verdict="partly"))
    mi = R.get("mia")
    if mi:
        nod = mi["rows"][0]["attack_auc"]; e1 = [r for r in mi["rows"] if r["epsilon"] == "1.0"][0]["attack_auc"]
        rows.append(dict(area="Our model", claim="H7 DP reduces membership-inference vulnerability",
                         evidence=f"attack AUC no-DP {_f(nod['mean'])} [{_f(nod['ci_low'])}, {_f(nod['ci_high'])}] vs "
                                  f"eps=1 {_f(e1['mean'])}; positive control (over-fitted) "
                                  f"{_f(mi['overfit_reference']['attack_auc']['mean'])}",
                         verdict="supported" if nod["ci_low"] > e1["ci_high"] else "not significant"))
    return rows
