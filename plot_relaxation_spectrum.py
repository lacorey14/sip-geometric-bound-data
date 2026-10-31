#!/usr/bin/env python3
"""Figure S2: Debye weights versus relaxation time (archived modes only)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
VEC = ROOT / "figures_vector"
VEC.mkdir(exist_ok=True)

PALETTE = {
    "blue_main": "#0F4D92",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
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
    "legend.frameon": False,
})

TAU0_MS = 5.0  # C0 R / (2 sigma_w) for the archived sphere


def save_pub(fig, stem: Path) -> None:
    for folder in (stem.parent, VEC):
        dest = folder / stem.name
        fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def stems(ax, x, y, color, label=None, z=3):
    markerline, stemlines, baseline = ax.stem(
        x, y, linefmt="-", markerfmt="o", basefmt=" ", label=label
    )
    plt.setp(stemlines, color=color, lw=1.0, zorder=z)
    plt.setp(markerline, color=color, markersize=4.0, zorder=z + 1,
             markeredgecolor="white", markeredgewidth=0.3)
    return markerline


def main() -> None:
    modal = json.load(open(R / "modal_spectrum.json"))
    contact = json.load(open(R / "two_mode_contact_fits.json"))
    tau_p = float(contact["limits"]["G_0S"]["tau"])
    tau_c = float(contact["limits"]["ideal_connected"]["tau"])

    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(178 * mm, 68 * mm), constrained_layout=True)
    axa, axb = axes

    cases = [
        ("polar_par", PALETTE["blue_main"], "polar cap"),
        ("hemisphere_par", PALETTE["red_strong"], "hemisphere"),
        ("six_caps", PALETTE["violet"], "six caps"),
    ]
    for name, col, lab in cases:
        d = modal[name]
        tau = np.array(d["tau_ms"]) / TAU0_MS
        a = np.array(d["a_top"])
        keep = a >= 5e-4
        stems(axa, tau[keep], a[keep], col, label=lab)
    axa.axvline(1.0, color=PALETTE["red_strong"], ls="--", lw=0.9, zorder=1)
    axa.set(xlabel=r"$\tau/\tau_0$", ylabel=r"Debye weight $A$",
            xlim=(0.35, 1.15), ylim=(0, 1.05))
    axa.text(0.98, 0.98, r"$\tau_0$", color=PALETTE["red_strong"],
             ha="right", va="top", fontsize=6, transform=axa.get_xaxis_transform())
    axa.legend(loc="upper left", fontsize=5.5)
    axa.set_title("Passivation: weights stay below the bound", loc="left")

    usable = [row for row in contact["table"]
              if row["G_true"] >= 0.004 and row["f_s_Hz"] >= 2.0]
    cmap = plt.cm.ScalarMappable(
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "gseq", [PALETTE["teal"], PALETTE["blue_main"]]),
        norm=mpl.colors.LogNorm(vmin=usable[0]["G_true"], vmax=usable[-1]["G_true"]),
    )
    for row in usable:
        col = cmap.to_rgba(row["G_true"])
        tau_s = (1.0 / (2 * math.pi * row["f_s_Hz"])) / tau_p
        tau_f = (1.0 / (2 * math.pi * row["f_f_Hz"])) / tau_p
        stems(axb, [tau_f, tau_s], [row["a_f"], row["a_s"]], col)
    axb.axvline(1.0, color=PALETTE["red_strong"], ls="--", lw=0.9, zorder=1)
    axb.axvline(tau_c / tau_p, color=PALETTE["neutral_dark"], ls=":", lw=1.0, zorder=1)
    axb.set(xlabel=r"$\tau/\tau_{\mathrm{particle}}$", ylabel=r"Debye weight $A$",
            xlim=(0.6, 6.2), ylim=(0, 0.034))
    axb.text(3.35, 0.0315, "particle limit", color=PALETTE["red_strong"], fontsize=6)
    axb.text(3.35, 0.0285, "cluster bound", color=PALETTE["neutral_dark"], fontsize=6)
    cax = fig.colorbar(cmap, ax=axb, fraction=0.046, pad=0.04)
    cax.set_label(r"$G$ (S)", fontsize=6)
    axb.set_title("Finite contact: times and weights both change", loc="left")

    for ax, lab in zip(axes, "ab"):
        ax.text(-0.18, 1.08, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")

    stem = R / "figureS2_relaxation_weights"
    save_pub(fig, stem)
    save_pub(fig, VEC / "figureS2_relaxation_weights")
    plt.close(fig)
    print("wrote", stem)


if __name__ == "__main__":
    main()
