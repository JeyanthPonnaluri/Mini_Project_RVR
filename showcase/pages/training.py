import pandas as pd
import streamlit as st

from fl_study import charts as ch
from showcase import core, figs

NOH_C = "#4a3aa7"; ETT_C = "#8a8985"


def render():
    core.header("Supplementary study", "Federated training and privacy",
                "Before valuing hospitals, we checked the federation that is valued: the base paper reproduced on "
                "its own data, transport to unseen countries, the accuracy cost of privacy, stronger baselines and a "
                "privacy audit. These results are summarised in the paper and detailed in its supplement.")
    s = core.R("pca19_strategies"); d = s["diff"]
    k = st.columns(4)
    with k[0]:
        core.kpi("1.2 × 10⁻¹⁴", "largest difference from the authors' own code", "Noh and Ettala, 50 repeats")
    with k[1]:
        core.kpi(core.sgn(d["FL-LOC"]["mean"], 3), "AUC gain from joining (FL − local)", core.ci(d["FL-LOC"], 3) + " · n.s.")
    pr = core.R("pca19_privacy"); worst = [c for c in pr["cells"] if c["epsilon"] == 0.5][0]
    with k[2]:
        core.kpi(core.sgn(worst["FedAvg-noDP"]["mean"], 3), "AUC cost of strong privacy (ε = 0.5)",
                 "5,610 patients, 7 parameters")
    lo = core.R("pca19_loco"); rc = lo["recalibration"]["FedAvg"]
    with k[3]:
        core.kpi(f"{core.f2(rc['raw']['abs_log_oe']['mean'])} → {core.f2(rc['intercept']['abs_log_oe']['mean'])}",
                 "|log O/E| after local recalibration", "20% of a new country's patients")

    st.write("")
    tabs = st.tabs(["Strategies", "New countries", "Privacy vs accuracy", "Stronger baselines",
                    "Personalisation & audit"])

    with tabs[0]:
        keys = ["LOC", "FL", "FedProx", "CEN", "FR_test", "BL_noh", "BL_ettala"]
        names = ["Local", "FedAvg", "FedProx", "Pooled", "Free-rider", "Noh", "Ettala"]
        cols = [ch.C["LOC"], ch.C["FL"], ch.C["FedProx"], ch.C["CEN"], ch.C["FR"], NOH_C, ETT_C]
        c1, c2 = st.columns(2)
        with c1:
            core.show(ch.bar_with_ci(names, [s["macro"][x] for x in keys], cols, "Mean per-hospital test AUC (20 splits)",
                                     ylim=(0.6, 0.9)))
        with c2:
            lc = s["learning_curve"]
            core.show(ch.lines([x["K"] for x in lc],
                               [{"name": "Participants", "color": ch.C["FL"], "mean": [x["FL"]["mean"] for x in lc],
                                 "low": [x["FL"]["ci_low"] for x in lc], "high": [x["FL"]["ci_high"] for x in lc]},
                                {"name": "Free-riders", "color": ch.C["FR"], "mean": [x["FR"]["mean"] for x in lc],
                                 "low": [x["FR"]["ci_low"] for x in lc], "high": [x["FR"]["ci_high"] for x in lc]}],
                               "AUC as hospitals join the federation", "hospitals training the model",
                               "mean per-hospital AUC", clip=(0, 1)))
        core.finding(f"<b>The base paper's finding holds on its own data:</b> joining brings no significant average "
                     f"gain (FL − local {core.ci(d['FL-LOC'], 3)}), and a free-rider gets almost the same model "
                     f"(free-rider − FL {core.ci(d['FR_test-FL'], 3)}). A federation that is to last therefore needs "
                     "a fair way to credit contributions, which is the subject of this work.")
        core.finding(f"<b>Honest negative:</b> the published Noh calculator is not beaten on hospital-averaged AUC "
                     f"(FL − Noh {core.ci(d['FL-BL_noh'], 3)}); patient-weighted, FL is ahead "
                     f"({core.f2(s['weighted']['FL']['mean'], 3)} vs {core.f2(s['weighted']['BL_noh']['mean'], 3)}).",
                     "warn")
        with st.expander("Per-hospital results"):
            ph = pd.DataFrame(s["hospitals"])
            st.dataframe(pd.DataFrame({"Hospital": ph["hospital"], "Country": ph["country"], "Patients": ph["n"],
                                       "Local AUC": ph["LOC"], "FL AUC": ph["FL"], "Noh AUC": ph["BL_noh"],
                                       "FL − local [95% CI]": [core.ci(x, 3) for x in ph["FL_minus_LOC"]]}),
                         hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ("Local AUC", "FL AUC", "Noh AUC")})

    with tabs[1]:
        st.markdown("**Leave-one-country-out:** train on seven countries and test on every patient of the eighth.")
        cc = lo["countries"]
        short = {"United Kingdom": "UK", "South Korea": "Korea", "Netherlands": "NL"}
        c1, c2 = st.columns([1.3, 1])
        with c1:
            core.show(figs.country_auc([short.get(c["country"], c["country"]) for c in cc],
                                       [c["FedAvg"]["auc"] for c in cc], [c["FedAvg"]["oe"] for c in cc],
                                       "FedAvg in each held-out country (O/E = observed / expected cancers)"))
        with c2:
            bc = lo["diff_by_country"]
            core.finding(f"<b>Discrimination transports.</b> FedAvg − published Noh calculator = "
                         f"{core.ci(bc['FedAvg - Noh (published)'], 3)} with countries as units.", "good")
            sl = lo["calibration_slope_by_model"]
            core.finding(f"<b>Calibration does not.</b> O/E ranges from "
                         f"{min(c['FedAvg']['oe'] for c in cc):.2f} to {max(c['FedAvg']['oe'] for c in cc):.2f}; "
                         f"DP makes predictions more extreme (slope {core.f2(sl['FedProx']['mean'])} → "
                         f"{core.f2(sl['FedProx+DP(eps=5)']['mean'])} at ε = 5).", "warn")
            core.finding(f"<b>Fix:</b> re-estimating the intercept on 20% of a new country's patients, already common "
                         f"practice for risk calculators, cuts |log O/E| from {core.f2(rc['raw']['abs_log_oe']['mean'])} "
                         f"to {core.f2(rc['intercept']['abs_log_oe']['mean'])}.")
        th = lo["thresholds"]
        ser = [{"name": n, "color": c, "mean": lo["dca"][n]} for n, c in
               [("FedAvg", ch.C["FL"]), ("FedProx+DP(eps=5)", ch.C["FedProx"]), ("Noh (published)", NOH_C),
                ("Biopsy all", ch.C["reference"])]]
        ser[-1]["dash"] = "--"
        core.show(ch.lines(th, ser, "Decision curve on held-out countries: net benefit of biopsy decisions",
                           "risk threshold for biopsy", "net benefit", ref=0, ref_label="biopsy none", direct=False))

    with tabs[2]:
        cells = pr["cells"]; dp = [x for x in cells if x["epsilon"] is not None]
        nod = [x for x in cells if x["epsilon"] is None][0]
        ser = [{"name": n, "color": c, "dash": dsh, "mean": [x[key]["macro"]["mean"] for x in dp],
                "low": [x[key]["macro"]["ci_low"] for x in dp], "high": [x[key]["macro"]["ci_high"] for x in dp]}
               for key, n, c, dsh in [("FedAvg", "FedAvg", ch.C["FedAvg"], "-"), ("FedProx", "FedProx", ch.C["FedProx"], "-"),
                                      ("FedAvg_half_lr", "FedAvg, half step", ch.C["FedAvg"], ":")]]
        c1, c2 = st.columns([1.4, 1])
        with c1:
            core.show(ch.lines([x["epsilon"] for x in dp], ser, "Mean per-hospital AUC against the privacy budget",
                               "ε (smaller = more private)", "AUC", xlog=True, ref=nod["FedAvg"]["macro"]["mean"],
                               ref_label="no DP"))
        with c2:
            core.finding(f"<b>Privacy is almost free here:</b> even at ε = 0.5 (noise σ = {pr['sigma']['0.5']:.0f}) the "
                         f"AUC changes by {core.ci(worst['FedAvg-noDP'], 3)}.", "good")
            core.finding("<b>Do not generalise this.</b> It holds for a seven-parameter logistic model trained on "
                         "5,610 patients. For a neural network, or for the heart-disease federation, the same ε "
                         "perturbs the released models far more, which matters for valuation.", "warn")
            core.finding(f"FedProx cuts client drift by {core.f2(pr['drift_reduction_pct']['mean'], 1)}%, but its "
                         "effect equals FedAvg with half the step size.")

    with tabs[3]:
        b = core.R("pca19_baselines"); wd = b["within_diff_vs_FedAvg"]
        order = [x for x in b["models"] if x in wd]
        core.show(figs.hbars(order, [wd[x]["mean"] for x in order],
                             [ch.C["FedProx"] if x == "FedProx" else ch.C["FedAvg"] if not x.startswith("Central")
                              else ch.MUTED for x in order],
                             "Within-hospital AUC difference from FedAvg (20 splits, 95% CI)", "AUC − FedAvg",
                             lows=[wd[x]["ci_low"] for x in order], highs=[wd[x]["ci_high"] for x in order],
                             fmt="{:+.3f}", ref=0))
        core.finding("<b>Stronger optimisers and non-linear models change little</b> (all within 0.015 AUC): with four "
                     "strong predictors a logistic model is near the ceiling, and the federation's value lies in access "
                     "to data, not in a fancier optimiser.")

    with tabs[4]:
        pf = core.R("pca19_personalisation")["groups"]; mia = core.R("pca19_mia")
        c1, c2 = st.columns(2)
        gs = ["small (< 100)", "medium (100-399)", "large (>= 400)"]
        with c1:
            core.show(ch.grouped_bars(["small (<100)", "medium", "large (≥400)"], ["Local", "FedProx", "Personalised"],
                                      [[pf[g][x]["mean"] for g in gs] for x in ("Local", "FedProx", "PFL")],
                                      [ch.C["Local"], ch.C["FedProx"], ch.C["PFL"]], "Per-hospital AUC by hospital size",
                                      "AUC", ylim=(0.6, 0.95)))
            core.finding(f"<b>Personalisation helps small hospitals:</b> personalised − FedProx = "
                         f"{core.ci(pf['small (< 100)']['PFL-FedProx'], 3)} for hospitals with fewer than 100 patients.",
                         "good")
        with c2:
            rows = mia["rows"]
            core.show(ch.bar_with_ci(["no DP" if r["epsilon"] is None else f"ε={r['epsilon']:g}" for r in rows] +
                                     ["over-fitted\ncontrol"], [r["attack_auc"] for r in rows] + [mia["control"]],
                                     [ch.C["FedAvg"]] * len(rows) + [ch.C["reference"]],
                                     "Membership-inference attack AUC (0.5 = guessing)", ylabel="attack AUC",
                                     ylim=(0.3, 1.0), ref=0.5, ref_label="chance"))
            core.finding("<b>The attack is at chance</b> for every model, while it detects an over-fitted control. A "
                         "small logistic model does not memorise patients; DP still provides the formal guarantee.")

    core.next_steps(("cost", "The hidden privacy cost", ":material/payments:"),
                    ("methods", "Valuation methods compared", ":material/leaderboard:"))
