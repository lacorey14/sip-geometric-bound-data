#!/usr/bin/env python3
"""Figure 2: why a Debye loss peak cannot sit below 1/τ0."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
VEC = ROOT / "figures_vector"
R.mkdir(exist_ok=True)
VEC.mkdir(exist_ok=True)

PALETTE = {
    "blue_main": "#0F4D92",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
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


def save_pub(fig, stem):
    for folder in (stem.parent, VEC):
        dest = folder / stem.name
        fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def debye_loss(u, r, a):
    """σ'' term vs u = ω τ0, with r = τ/τ0."""
    x = u * r
    return a * x / (1.0 + x * x)


def debye_dloss(u, r, a):
    """dσ'' / d(ω τ0)."""
    x = u * r
    return a * r * (1.0 - x * x) / (1.0 + x * x) ** 2


def main():
    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(178 * mm, 62 * mm))
    axa, axb = axes

    u = np.logspace(-1.15, 1.05, 800)
    modes = [
        (0.35, 0.45, PALETTE["teal"], r"$\tau=0.35\,\tau_0$"),
        (0.70, 0.35, PALETTE["violet"], r"$\tau=0.70\,\tau_0$"),
        (0.95, 0.20, PALETTE["neutral_mid"], r"$\tau=0.95\,\tau_0$"),
    ]
    L = np.zeros_like(u)
    dL = np.zeros_like(u)
    for r, a, col, lab in modes:
        Li = debye_loss(u, r, a)
        dLi = debye_dloss(u, r, a)
        L += Li
        dL += dLi
        axa.plot(u, Li, color=col, lw=0.85, ls="--", label=lab, zorder=2)
        axb.plot(u, dLi, color=col, lw=0.85, ls="--", zorder=2)

    ipeak = int(np.argmax(L))
    up = u[ipeak]

    for ax in (axa, axb):
        ax.axvspan(u.min(), 1.0, color=PALETTE["red_strong"], alpha=0.10, lw=0)
        ax.axvline(1.0, color=PALETTE["red_strong"], ls="--", lw=0.9, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(u.min(), u.max())
        ax.set_xlabel(r"$\omega\,\tau_0$")

    axa.plot(u, L, color=PALETTE["blue_main"], lw=1.4, zorder=3, label="sum")
    axa.plot([up], [L[ipeak]], "o", ms=4.5, color=PALETTE["blue_main"],
             markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    axa.set_ylabel(r"$\sigma''$")
    axa.set_ylim(0, 1.22 * L.max())
    axa.set_title("Each Debye term is still rising", loc="left")
    axa.legend(loc="upper left", fontsize=5.5, ncol=1,
               handlelength=1.4, labelspacing=0.15, borderaxespad=0.15)
    axa.annotate("still rising", xy=(0.28, debye_loss(0.28, 0.35, 0.45)
                                     + debye_loss(0.28, 0.70, 0.35)
                                     + debye_loss(0.28, 0.95, 0.20)),
                 xytext=(0.12, 0.72 * L.max()),
                 fontsize=6, color=PALETTE["red_strong"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=0.8))
    axa.annotate(r"peak $\tau_p\leq\tau_0$", xy=(up, L[ipeak]),
                 xytext=(up * 1.55, 1.08 * L[ipeak]),
                 fontsize=7, color=PALETTE["blue_main"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["blue_main"], lw=0.8))

    axb.plot(u, dL, color=PALETTE["blue_main"], lw=1.4, zorder=3)
    axb.axhline(0.0, color=PALETTE["neutral_dark"], lw=0.6, zorder=1)
    axb.plot([up], [0.0], "o", ms=4.5, color=PALETTE["blue_main"],
             markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    axb.set_ylabel(r"$d\sigma''/d(\omega\tau_0)$")
    ymax = 1.15 * np.max(np.abs(dL))
    axb.set_ylim(-ymax, ymax)
    axb.set_title(r"No stationary point for $\omega\tau_0<1$", loc="left")
    axb.text(0.22, 0.88, r"$d\sigma''/d\omega>0$", transform=axb.transAxes,
             color=PALETTE["red_strong"], fontsize=6.5, ha="center")
    axb.text(1.12, 0.08, r"$1/\tau_0$", color=PALETTE["red_strong"], fontsize=6.5,
             transform=axb.get_xaxis_transform())

    for ax, lab in zip(axes, "ab"):
        ax.text(-0.16, 1.10, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.96))
    stem = R / "figure4_peak_exclusion"
    save_pub(fig, stem)
    save_pub(fig, VEC / "figure4_peak_exclusion")
    plt.close(fig)
    print("wrote", stem, "peak at omega tau0 =", float(up))


if __name__ == "__main__":
    main()
