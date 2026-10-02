"""
Matplotlib figure builders for the Streamlit app (static, so they can be rendered
and checked offline). One light chart surface, thin marks, recessive grid,
fixed entity -> colour mapping across every chart, legends + direct labels.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"

# Colour follows the entity, in the validated categorical order.
C = {
    "FedAvg": "#2a78d6", "FL": "#2a78d6",
    "FedProx": "#eb6834",
    "Local": "#1baf7a", "LOC": "#1baf7a",
    "Centralised": "#eda100", "CEN": "#eda100",
    "Free-rider": "#e87ba4", "FR": "#e87ba4",
    "PFL": "#008300",
    "Rule (Gleason)": "#4a3aa7", "BL": "#4a3aa7",
    "FedAvg (lr/2)": "#2a78d6",
    "reference": "#8a8985",
}
LABEL = {"LOC": "Local only (LOC)", "FL": "Federated (FL)", "FR": "Free-rider (FR)",
         "CEN": "Centralised (CEN)", "BL": "Rule baseline (Gleason sum)"}
SITE_SHORT = {
    "University of Pittsburgh": "Pittsburgh", "International Genomics Consortium": "IGC",
    "MD Anderson Cancer Center": "MD Anderson", "Roswell Park": "Roswell Park", "Indivumed": "Indivumed",
    "University of California San Francisco": "UCSF", "PROCURE Biobank": "PROCURE",
    "University Medical Center Hamburg-Eppendorf": "Hamburg", "ABS - Lahey Clinic": "Lahey Clinic",
}


def short(site):
    return SITE_SHORT.get(site, site if len(site) < 14 else site[:12] + "…")


SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b3b2", "#f0efec", "#9ec5f4", "#2a78d6"])


def _fig(w=7.0, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    return fig, ax


def _labels(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2, **kw)
    return leg


def _m(s):
    return np.nan if s is None or s.get("mean") is None else s["mean"]


def _ci(s):
    if s is None or s.get("ci_low") is None:
        return (0, 0)
    return (s["mean"] - s["ci_low"], s["ci_high"] - s["mean"])


def bar_with_ci(names, summaries, colors, title, ylabel="AUC", ylim=None, ref=None, ref_label=None, fmt="{:.3f}"):
    fig, ax = _fig(7, 3.4)
    x = np.arange(len(names))
    means = [_m(s) for s in summaries]
    errs = np.array([_ci(s) for s in summaries]).T
    ax.bar(x, means, width=0.55, color=colors, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(x, means, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    span = (ylim[1] - ylim[0]) if ylim else 1.0
    for xi, m, e in zip(x, means, errs[1]):
        if not math.isnan(m):
            ax.text(xi, m + e + 0.02 * span, fmt.format(m), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, names, fontsize=8, color=INK2)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls="--", zorder=1)
        if ref_label:
            ax.text(-0.45, ref, ref_label, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    if ylim:
        ax.set_ylim(*ylim)
    _labels(ax, title, None, ylabel)
    fig.tight_layout()
    return fig


def lines(x, series, title, xlabel, ylabel, ylim=None, ref=None, ref_label=None, xlog=False, xticks=None,
          clip=None, direct=True):
    """series: list of dicts {name, mean[], low[], high[], color, dash}"""
    fig, ax = _fig(7, 3.6)
    for s in series:
        m = np.array(s["mean"], float)
        ax.plot(x, m, color=s["color"], lw=2, ls=s.get("dash", "-"), marker="o", ms=4, label=s["name"], zorder=3)
        if s.get("low") is not None and len(series) <= 2:
            lo = np.array([np.nan if v is None else v for v in s["low"]], float)
            hi = np.array([np.nan if v is None else v for v in s["high"]], float)
            if clip:
                lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
            ax.fill_between(x, lo, hi, color=s["color"], alpha=0.12, lw=0, zorder=2)
        if direct and len(series) >= 2:
            ax.annotate(s["name"], (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls="--")
        if ref_label:
            ax.text(x[0], ref, " " + ref_label, va="bottom", fontsize=7.5, color=MUTED)
    if xlog:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks[0], xticks[1])
    if ylim:
        ax.set_ylim(*ylim)
    ax.margins(x=0.12)
    _labels(ax, title, xlabel, ylabel)
    if len(series) >= 2:
        _legend(ax, loc="best")
    fig.tight_layout()
    return fig


def heatmap(matrix, row_labels, col_labels, title, cmap="seq", center=None, fmt="{:+.3f}", stars=None,
            xlabel=None, ylabel=None, vlim=None):
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    M = np.array(matrix, float)
    if center is not None:
        v = vlim or np.nanmax(np.abs(M - center))
        im = ax.imshow(M, cmap=DIV, vmin=center - v, vmax=center + v, aspect="auto")
    else:
        im = ax.imshow(M, cmap=SEQ, aspect="auto", vmin=vlim[0] if vlim else None, vmax=vlim[1] if vlim else None)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if math.isnan(M[i, j]):
                continue
            rgba = im.cmap(im.norm(M[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            t = fmt.format(M[i, j]) + (stars[i][j] if stars else "")
            ax.text(j, i, t, ha="center", va="center", fontsize=8, color="white" if lum < 0.5 else INK)
    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=8, color=INK2)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, M.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    _labels(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def scatter_labeled(x, y, labels, title, xlabel, ylabel, color="#2a78d6", ref_y=None, ref_line=False):
    fig, ax = _fig(7, 3.6)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.scatter(x, y, s=46, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    seen = {}
    for xi, yi, l in zip(x, y, labels):
        key = (round(xi, 1), round(yi, 2)); k = seen.get(key, 0); seen[key] = k + 1
        ax.annotate(l, (xi, yi), xytext=(5, 4 - 10 * k), textcoords="offset points", fontsize=7.5, color=INK2)
    if ref_y is not None:
        ax.axhline(ref_y, color=MUTED, lw=1, ls="--")
    if ref_line:
        lo = min(min(x), min(y)); hi = max(max(x), max(y))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    _labels(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def grouped_bars(groups, series_names, values, colors, title, ylabel, errs=None, ylim=None, ref=None, fmt="{:.3f}"):
    """values[s][g]"""
    fig, ax = _fig(7.2, 3.6)
    G, S = len(groups), len(series_names)
    width = 0.8 / S
    x = np.arange(G)
    for i, (n, c) in enumerate(zip(series_names, colors)):
        xs = x - 0.4 + width * (i + 0.5)
        v = np.array(values[i], float)
        ax.bar(xs, v, width=width, color=c, edgecolor=SURFACE, linewidth=1.5, label=n, zorder=2)
        if errs is not None:
            ax.errorbar(xs, v, yerr=np.array(errs[i]).T, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
    ax.set_xticks(x, groups, fontsize=8, color=INK2, rotation=25 if G > 6 else 0, ha="right" if G > 6 else "center")
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls="--")
    if ylim:
        ax.set_ylim(*ylim)
    _labels(ax, title, None, ylabel)
    _legend(ax, ncol=min(S, 4), loc="upper center", bbox_to_anchor=(0.5, -0.22 if G > 6 else -0.12))
    fig.tight_layout()
    return fig
