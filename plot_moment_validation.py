#!/usr/bin/env python3
"""Fig. 4: independent single-layer solver versus archived FEM spectra."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
VEC = ROOT / "figures_vector"
SRC = ROOT / "source_data"
VEC.mkdir(exist_ok=True)
SRC.mkdir(exist_ok=True)

PALETTE = {
    "blue_main": "#0F4D92",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
})

GRP = {
    "base": (PALETTE["red_strong"], "o", "base patterns"),
    "orientation": (PALETTE["blue_main"], "s", "orientation"),
    "rand": (PALETTE["teal"], "^", "random patches"),
    "frag": (PALETTE["violet"], "D", "fragmented"),
}


def save_pub(fig, stem):
    stem = Path(stem)
    for folder in (stem.parent, VEC):
        dest = folder / stem.name
        fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main():
    arch = {c["case"]: c for c in json.load(open(R / "moments_vs_archive.json"))["cases"]}
    stat = {r["case"]: r for r in json.load(open(R / "static_two_solve_estimator.json"))["results"]["N12000"]}
    fp0 = arch["full_active"]["fp"]
    cases = [c for k, c in arch.items() if k != "full_active" and k in stat]

    rows = []
    for c in cases:
        s = stat[c["case"]]
        fp_static = fp0 * s["M0n"] / s["Lambda"]
        rows.append({
            "case": c["case"],
            "group": c["group"],
            "Lambda_E": c["Lam"],
            "M1n_FEM": c["M1n"],
            "M0n_FEM": c["M0n"],
            "M0n_static": s["M0n"],
            "fp_FEM_Hz": c["fp"],
            "fp_static_Hz": fp_static,
        })
    with (SRC / "panel_fig3_moment_validation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    x1 = np.array([r["Lambda_E"] for r in rows])
    y1 = np.array([r["M1n_FEM"] for r in rows])
    r2 = 1.0 - np.sum((y1 - x1) ** 2) / np.sum((y1 - y1.mean()) ** 2)

    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(178 * mm, 62 * mm), constrained_layout=True)
    axa, axb, axc = axes
    seen = set()
    for r in rows:
        col, mk, lab = GRP[r["group"]]
        kw = dict(s=22, c=col, marker=mk, zorder=3, edgecolors="white", linewidths=0.3)
        axa.scatter(r["Lambda_E"], r["M1n_FEM"], **kw,
                    label=lab if r["group"] not in seen else None)
        axb.scatter(r["M0n_static"], r["M0n_FEM"], **kw)
        axc.scatter(r["fp_FEM_Hz"], r["fp_static_Hz"], **kw)
        seen.add(r["group"])

    for ax, lo, hi in ((axa, -0.02, 1.05), (axb, -0.02, 1.05)):
        ax.plot([0, 1], [0, 1], color=PALETTE["neutral_dark"], ls="--", lw=0.8, zorder=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    axa.set(xlabel=r"electrical leverage $\Lambda_E$",
            ylabel=r"$M_1$ from FEM spectrum")
    axa.set_title("First-moment identity", loc="left")
    axa.text(0.08, 0.90, rf"$R^2={r2:.5f}$", transform=axa.transAxes, fontsize=6,
             color=PALETTE["neutral_dark"])
    axa.legend(loc="lower right", fontsize=5.5, handletextpad=0.3, markerscale=0.85)

    axb.set(xlabel=r"$M_0$ from independent static solve",
            ylabel=r"$M_0$ from FEM spectrum")
    axb.set_title("Independent $M_0$ vs FEM", loc="left")

    lo, hi = 34, 82
    axc.plot([lo, hi], [lo, hi], color=PALETTE["neutral_dark"], ls="--", lw=0.8, zorder=1)
    axc.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal",
            xlabel=r"FEM peak $f_p$ (Hz)",
            ylabel=r"$f_p^{0}\,M_0/M_1$ from two static solves (Hz)")
    axc.set_title("Mid-band peak from statics", loc="left")

    for ax, lab in zip(axes, "abc"):
        ax.text(-0.22, 1.08, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")

    stem = R / "moment_validation"
    save_pub(fig, stem)
    save_pub(fig, VEC / "figure3_moment_validation")
    plt.close(fig)
    print("wrote", stem, "n=", len(rows), "R2_M1=", round(r2, 6))


if __name__ == "__main__":
    main()
