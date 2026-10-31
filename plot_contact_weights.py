#!/usr/bin/env python3
"""Post-process archived two-Debye contact fits: A_s(G) and A_s τ_s.

Does not refit. Reads results/two_mode_contact_fits.json only.
"""
from __future__ import annotations

import csv
import json
import math
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
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
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

F_BAND = 2.0  # Hz; archived spectra start at 2.5 Hz


def save_pub(fig, stem: Path) -> None:
    for folder in (stem.parent, VEC):
        dest = folder / stem.name
        fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    raw = json.load(open(R / "two_mode_contact_fits.json"))
    lim_iso = raw["limits"]["G_0S"]
    lim_wire = raw["limits"]["ideal_connected"]
    m1_iso = float(raw["M1_disc"])
    m1_wire = float(raw["M1_ideal"])
    dm1 = m1_wire - m1_iso
    a_iso = float(lim_iso["a"])
    a_wire = float(lim_wire["a"])
    tau_iso = float(lim_iso["tau"])
    tau_wire = float(lim_wire["tau"])

    rows = []
    for t in raw["table"]:
        fs = float(t["f_s_Hz"])
        ff = float(t["f_f_Hz"])
        a_s = float(t["a_s"])
        a_f = float(t["a_f"])
        tau_s = 1.0 / (2.0 * math.pi * fs)
        tau_f = 1.0 / (2.0 * math.pi * ff)
        a_s_tau = a_s * tau_s
        in_band = fs >= F_BAND
        # Smallest G: two-mode labels swap (slow peak sits on the particle line).
        label_ok = not (t["G_true"] < 1e-4 and fs > 20.0)
        rows.append({
            "G_true_S": t["G_true"],
            "G_nom_S": t["G_nom"],
            "a_f": a_f,
            "f_f_Hz": ff,
            "tau_f_s": tau_f,
            "a_s": a_s,
            "f_s_Hz": fs,
            "tau_s_s": tau_s,
            "a_s_tau_s": a_s_tau,
            "a_f_tau_f": a_f * tau_f,
            "sum_M1": a_f * tau_f + a_s_tau,
            "sum_M1_over_M1_wire": (a_f * tau_f + a_s_tau) / m1_wire,
            "tau_s_over_tau_particle": tau_s / tau_iso,
            "in_band": int(in_band),
            "label_ok": int(label_ok),
            "usable": int(in_band and label_ok),
        })

    fields = list(rows[0].keys())
    with (SRC / "contact_mode_weights.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "M1_iso": m1_iso,
        "M1_wire": m1_wire,
        "Delta_M1": dm1,
        "A_iso": a_iso,
        "A_wire": a_wire,
        "tau_iso_s": tau_iso,
        "tau_wire_s": tau_wire,
        "tau_cluster_over_tau_particle": tau_wire / tau_iso,
        "band_floor_Hz": F_BAND,
        "note": (
            "As G->0, tau_s diverges and A_s -> 0; A_s*tau_s stays near "
            "Delta_M1 = M1_wire - M1_iso if the slow peak is resolved. "
            "In-band fits (f_s>=2 Hz) have A_s*tau_s -> M1_wire because the "
            "slow mode becomes the cluster mode. Below-band fits break the "
            "sum rule and are not used for A_s*tau_s."
        ),
    }
    (R / "contact_mode_weights.json").write_text(json.dumps({
        "limits": summary, "table": rows,
    }, indent=2))

    usable = [r for r in rows if r["usable"]]
    below = [r for r in rows if r["label_ok"] and not r["in_band"]]
    print("limits: M1_iso={:.4e}  M1_wire={:.4e}  Delta_M1={:.4e}".format(
        m1_iso, m1_wire, dm1))
    print("G_true     f_s     A_s       A_s*tau_s   sumM1/M1w  usable")
    for r in rows:
        print("{:8.4g} {:7.2f} {:8.3e} {:10.3e} {:9.3f}  {}".format(
            r["G_true_S"], r["f_s_Hz"], r["a_s"], r["a_s_tau_s"],
            r["sum_M1_over_M1_wire"], r["usable"]))

    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(120 * mm, 58 * mm), constrained_layout=True)
    axa, axb = axes

    g = np.array([r["G_true_S"] for r in usable])
    a_s = np.array([r["a_s"] for r in usable])
    a_s_tau = np.array([r["a_s_tau_s"] for r in usable])

    axa.axhline(a_iso, color=PALETTE["red_strong"], ls="--", lw=0.9)
    axa.axhline(a_wire, color=PALETTE["neutral_dark"], ls=":", lw=1.0)
    axa.plot(g, a_s, color=PALETTE["blue_main"], lw=0.8)
    axa.scatter(g, a_s, s=26, c=PALETTE["blue_main"], zorder=3,
                edgecolors="white", linewidths=0.3)
    axa.set(xscale="log", xlabel=r"true contact $G$ (S)",
            ylabel=r"slow-mode amplitude $A_s$",
            xlim=(2e-3, 3), ylim=(0, 0.034))
    axa.text(2.2e-3, a_iso + 0.0012, "disconnected $A$",
             color=PALETTE["red_strong"], fontsize=6)
    axa.text(2.2e-3, a_wire - 0.0026, "ideal-wire $A$",
             color=PALETTE["neutral_dark"], fontsize=6)
    axa.set_title(r"$A_s$ falls as $G$ falls", loc="left")

    axb.axhline(m1_wire, color=PALETTE["neutral_dark"], ls=":", lw=1.0)
    axb.axhline(dm1, color=PALETTE["teal"], ls="--", lw=0.9)
    axb.axhline(m1_iso, color=PALETTE["red_strong"], ls="--", lw=0.9)
    axb.plot(g, a_s_tau, color=PALETTE["blue_main"], lw=0.8)
    axb.scatter(g, a_s_tau, s=26, c=PALETTE["blue_main"], zorder=3,
                edgecolors="white", linewidths=0.3)
    if below:
        gb = np.array([r["G_true_S"] for r in below])
        yb = np.array([r["a_s_tau_s"] for r in below])
        axb.scatter(gb, yb, s=22, facecolors="none", edgecolors=PALETTE["neutral_mid"],
                    linewidths=0.7, zorder=3)
    axb.set(xscale="log", xlabel=r"true contact $G$ (S)",
            ylabel=r"$A_s\tau_s$", xlim=(1e-4, 3), ylim=(0, 2.15e-4))
    axb.text(1.2e-4, m1_wire + 4e-6, r"$M_1^{\mathrm{wire}}$",
             color=PALETTE["neutral_dark"], fontsize=6)
    axb.text(1.2e-4, dm1 - 1.4e-5, r"$\Delta M_1$",
             color=PALETTE["teal"], fontsize=6)
    axb.text(1.2e-4, m1_iso + 4e-6, r"$M_1^{\mathrm{iso}}$",
             color=PALETTE["red_strong"], fontsize=6)
    axb.set_title(r"$A_s\tau_s$ stays finite", loc="left")

    for ax, lab in zip(axes, "ab"):
        ax.text(-0.22, 1.08, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")

    stem = R / "figure_contact_weights"
    save_pub(fig, stem)
    save_pub(fig, VEC / "figureS1_contact_weights")
    plt.close(fig)
    print("wrote", stem, "and", SRC / "contact_mode_weights.csv")


if __name__ == "__main__":
    main()
