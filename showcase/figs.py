"""Charts for the showcase. Matplotlib (static, rendered and checked offline), one visual language:
light surface, thin marks, recessive grid, one colour per method, direct labels instead of legends."""
from __future__ import annotations

import numpy as np

from fl_study import charts as ch
from . import core

INK, INK2, MUTED, GRID, SURFACE = ch.INK, ch.INK2, ch.MUTED, ch.GRID, ch.SURFACE


def _fig(w=7.2, h=3.7):
    return ch._fig(w, h)


def _title(ax, title, xlabel=None, ylabel=None):
    ch._labels(ax, title, xlabel, ylabel)


def rho_lines(xlabels, series, title, ylabel="Spearman ρ with exact ranking", ylim=(-0.4, 1.02), note=None):
    """series: list of dicts {key, mean[], low[], high[], dash, band, label?}. Legend below the plot."""
    fig, ax = _fig(7.4, 4.2)
    x = np.arange(len(xlabels))
    for s in series:
        mvals = np.array([np.nan if v is None else v for v in s["mean"]], float)
        col = core.COL[s["key"]]
        lw = 2.8 if s["key"] == "osa" else 1.8
        ax.plot(x, mvals, color=col, lw=lw, ls=s.get("dash", "-"), marker="o", ms=5 if s["key"] == "osa" else 3.5,
                zorder=4 if s["key"] == "osa" else 3, label=s.get("label", core.NAME[s["key"]]))
        if s.get("band") and s.get("low") is not None:
            lo = np.array([np.nan if v is None else v for v in s["low"]], float)
            hi = np.array([np.nan if v is None else v for v in s["high"]], float)
            ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0, zorder=2)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(x, xlabels, fontsize=8.5, color=INK2)
    ax.set_xlim(-0.2, len(xlabels) - 0.8)
    if ylim:
        ax.set_ylim(*ylim)
    _title(ax, title, "privacy budget (stronger privacy →)", ylabel)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=7.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    if note:
        ax.text(0, -0.2, note, transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    fig.tight_layout()
    return fig


def hbars(labels, values, colors, title, xlabel, lows=None, highs=None, xlim=None, fmt="{:.2f}", ref=None,
          ref_label=None, bold_first=False):
    n = len(labels)
    fig, ax = _fig(7.2, 0.42 * n + 1.2)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.7)
    y = np.arange(n)[::-1]
    vals = np.array([np.nan if v is None else v for v in values], float)
    ax.barh(y, vals, color=colors, height=0.62, zorder=2)
    if lows is not None:
        lo = vals - np.array(lows, float); hi = np.array(highs, float) - vals
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=3)
    for yi, v, hiv in zip(y, vals, (highs if highs is not None else vals)):
        if not np.isnan(v):
            xpos = max(v, hiv if hiv is not None else v)
            ax.text(xpos + 0.015 * (xlim[1] - xlim[0] if xlim else 1), yi, fmt.format(v).replace("-", "−"),
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=8.5, color=INK)
    if bold_first:
        ax.get_yticklabels()[0].set_fontweight("bold")
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls="--", zorder=1)
        if ref_label:
            ax.text(ref, n - 0.45, " " + ref_label, fontsize=7.2, color=MUTED, va="bottom")
    if xlim:
        ax.set_xlim(*xlim)
    _title(ax, title, xlabel)
    fig.tight_layout()
    return fig


def accounting(Ks, exact, perm, stated, title):
    fig, ax = _fig(7.2, 3.6)
    ax.plot(Ks, exact, color=core.COL["retrain_budget_matched"], lw=2.2, marker="o", ms=3)
    ax.plot(Ks, perm, color=core.COL["retrain_per_run"], lw=2.2, ls="--", marker="o", ms=3)
    ax.axhline(stated, color=core.COL["osa"], lw=2.2)
    ax.set_yscale("log")
    ax.annotate("exact Shapley by retraining", (Ks[min(len(Ks) - 1, 8)], exact[min(len(Ks) - 1, 8)]),
                xytext=(6, 6), textcoords="offset points", fontsize=7.8, color=core.COL["retrain_budget_matched"],
                fontweight="bold")
    ax.annotate("100 sampled permutations", (Ks[-1], perm[-1]), xytext=(-6, 8), textcoords="offset points",
                fontsize=7.8, color=INK2, ha="right", fontweight="bold")
    ax.text(Ks[0], stated * 2.2, "OSA-Shapley / MR: stated ε, once", fontsize=7.8, color=core.COL["osa"],
            fontweight="bold")
    _title(ax, title, "number of hospitals K", "true ε per hospital (log scale)")
    fig.tight_layout()
    return fig


def nu_sweep(conds, highlight_nu=None, title="OSA − MR against patients per parameter"):
    fig, ax = _fig(7.4, 4.3)
    ax.axvspan(50, 100, color=core.COL["osa"], alpha=0.08, lw=0)
    ax.text(70.7, -0.25, "predicted\nswitch-over", ha="center", va="top", fontsize=7.5, color=core.COL["osa"])
    ax.axhline(0, color=MUTED, lw=0.9)
    styles = {("A", "2.0"): (core.COL["osa"], "-", "o", "ε = 2, fewer patients"),
              ("A", "1.0"): (core.COL["retrain_budget_matched"], "-", "o", "ε = 1, fewer patients"),
              ("B", "2.0"): (core.COL["osa"], "--", "s", "ε = 2, more parameters"),
              ("B", "1.0"): (core.COL["retrain_budget_matched"], "--", "s", "ε = 1, more parameters")}
    for (arm, e), (col, ls, mk, lab) in styles.items():
        cs = sorted([c for c in conds if c["arm"] == arm], key=lambda c: c["nu"])
        nu = np.array([c["nu"] for c in cs]) * (1.025 if e == "1.0" else 0.975)
        d = [c[e]["osa_minus_mr"] for c in cs]
        mv = np.array([x["mean"] for x in d]); lo = mv - np.array([x["ci_low"] for x in d])
        hi = np.array([x["ci_high"] for x in d]) - mv
        ax.errorbar(nu, mv, yerr=[lo, hi], color=col, ls=ls, marker=mk, ms=4.5, lw=1.6, elinewidth=0.9,
                    capsize=2.5, label=lab, zorder=3)
    if highlight_nu:
        ax.axvline(highlight_nu, color=INK, lw=1.2, ls=":")
        ax.text(highlight_nu, 0.335, f" your setting: ν = {highlight_nu:.0f}", fontsize=7.5, color=INK, va="top",
                ha="right" if highlight_nu > 60 else "left")
    ax.set_xscale("log")
    ax.set_xticks([15, 20, 30, 50, 100, 150], ["15", "20", "30", "50", "100", "150"])
    ax.minorticks_off()
    ax.set_xlim(13, 160); ax.set_ylim(-0.36, 0.36)
    _title(ax, title, "ν = median training patients per hospital / √parameters (log scale)",
           "OSA − MR (Spearman ρ, 95% CI)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=7.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    return fig


def shares(hospitals, columns, title):
    """Grouped horizontal bars of each hospital's share (%) under several scores. columns: list of (key, values)."""
    n = len(hospitals); k = len(columns)
    fig, ax = _fig(7.2, 0.5 * n + 1.3)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.7)
    y = np.arange(n)[::-1].astype(float)
    h = 0.8 / k
    for i, col in enumerate(columns):
        key, vals = col[0], col[1]
        label = col[2] if len(col) > 2 else core.NAME[key]
        v = np.array(vals, float)
        ax.barh(y + 0.4 - h * (i + 0.5), v, height=h * 0.92, color=core.COL[key], label=label, zorder=2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, hospitals, fontsize=8.5, color=INK)
    _title(ax, title, "share of the federation's value (%)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.06 - 0.4 / n))
    fig.tight_layout()
    return fig


def country_auc(countries, aucs, oes, title):
    fig, ax = _fig(7.2, 3.4)
    x = np.arange(len(countries))
    ax.bar(x, aucs, color=ch.C["FedAvg"], width=0.6, zorder=2)
    for xi, a, o in zip(x, aucs, oes):
        ax.text(xi, a + 0.004, f"{a:.2f}", ha="center", va="bottom", fontsize=7.8, color=INK)
        ax.text(xi, 0.705, f"O/E\n{o:.2f}", ha="center", va="bottom", fontsize=6.8,
                color="white" if abs(np.log(o)) < 0.15 else "#ffe0cc")
    ax.set_xticks(x, countries, fontsize=8, color=INK2)
    ax.set_ylim(0.7, 0.95)
    _title(ax, title, None, "AUC in the held-out country")
    fig.tight_layout()
    return fig
