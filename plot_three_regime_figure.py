#!/usr/bin/env python3
"""GRL-oriented three-regime test of the geometric SIP bound (Python only)."""
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
SRC.mkdir(exist_ok=True)
VEC.mkdir(exist_ok=True)

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_light": "#CFCECE",
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
    "lines.markersize": 4.5,
})

C0, RADIUS, SIGMA_W = 0.4, 5e-3, 0.2
TAU0 = C0 * RADIUS / (2 * SIGMA_W)
F0 = 1.0 / (2 * math.pi * TAU0)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collect():
    arch = json.load(open(R / "moments_vs_archive.json"))["cases"]
    full = next(c for c in arch if c["case"] == "full_active")
    f_ref = full["f_bar"]
    pass_rows = []
    for c in arch:
        if c["case"] == "full_active":
            continue
        pass_rows.append({
            "case": c["case"],
            "group": c["group"],
            "Lambda_E": c["Lam"],
            "tau_over_tau0": f_ref / c["f_bar"],
            "source": "FEM_archive_sum_rule",
        })
    cloud = []
    for x in json.load(open(R / "slowness_bound_check.json")):
        if x["Lambda"] <= 1e-6:
            continue
        cloud.append({
            "Lambda_E": x["Lambda"],
            "tau_over_tau0": x["Lambda"] / x["M0n"],
            "kind": x["kind"],
            "source": "static_single_layer",
        })

    semi_src = ROOT / "inputs" / "dimensionless_regime_map.json"
    d = json.load(open(semi_src))
    interior = []
    for s in d["summaries"]:
        chi = s["chi_sigma_s_over_sigma_w"]
        beta = s["beta_Ceff_over_Cdl"]
        tau = 1.0 / (2 * math.pi * s["peak_frequency_hz"] * beta)
        interior.append({
            "chi": chi,
            "beta": beta,
            "tau_over_tau0": tau / TAU0,
            "theory": 1.0 + 2.0 / chi,
            "source": "archived_FEM_sweep",
        })

    contact_raw = json.load(open(R / "two_mode_contact_fits.json"))
    tau_p = contact_raw["limits"]["G_0S"]["tau"]
    tau_c = contact_raw["limits"]["ideal_connected"]["tau"]
    contact = []
    for row in contact_raw["table"]:
        reliable = row["G_true"] >= 0.0008 and row["f_s_Hz"] >= 2.0
        contact.append({
            "G_true_S": row["G_true"],
            "f_slow_Hz": row["f_s_Hz"],
            "tau_slow_over_tau_particle": (1.0 / (2 * math.pi * row["f_s_Hz"])) / tau_p,
            "reliable": int(reliable),
            "source": "two_Debye_fit_of_archived_spectra",
        })
    meta = {
        "tau0_s": TAU0,
        "f0_Hz": F0,
        "passivation_reference": "full_active FEM sum-rule frequency",
        "f_ref_Hz": f_ref,
        "tau_particle_contact_s": tau_p,
        "tau_cluster_contact_s": tau_c,
        "tau_cluster_over_tau_particle": tau_c / tau_p,
    }
    write_csv(SRC / "panel_a_passivation.csv", pass_rows,
              ["case", "group", "Lambda_E", "tau_over_tau0", "source"])
    write_csv(SRC / "panel_a_random_cloud.csv", cloud,
              ["Lambda_E", "tau_over_tau0", "kind", "source"])
    write_csv(SRC / "panel_b_interior.csv", interior,
              ["chi", "beta", "tau_over_tau0", "theory", "source"])
    write_csv(SRC / "panel_c_contact.csv", contact,
              ["G_true_S", "f_slow_Hz", "tau_slow_over_tau_particle", "reliable", "source"])
    (SRC / "figure_meta.json").write_text(json.dumps(meta, indent=2))
    return pass_rows, cloud, interior, contact, meta


def save_pub(fig, stem):
    for folder in (stem.parent, VEC):
        dest = folder / stem.name
        fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def save_panel(ax, stem, size_mm=(56, 62)):
    fig_p = ax.figure
    fig_p.canvas.draw()
    bbox = ax.get_tightbbox(fig_p.canvas.get_renderer()).transformed(fig_p.dpi_scale_trans.inverted())
    fig_p.savefig(stem.with_suffix(".svg"), bbox_inches=bbox.expanded(1.04, 1.06))
    fig_p.savefig(stem.with_suffix(".pdf"), bbox_inches=bbox.expanded(1.04, 1.06))


def _two_grains(ax, y=0.52, r=0.17, fc="#D9E6F2", ec=None):
    from matplotlib.patches import Circle
    if ec is None:
        ec = PALETTE["blue_main"]
    c1, c2 = (0.28, y), (0.66, y)
    ax.add_patch(Circle(c1, r, facecolor=fc, edgecolor=ec, lw=1.15, zorder=2))
    ax.add_patch(Circle(c2, r, facecolor=fc, edgecolor=ec, lw=1.15, zorder=2))
    return c1[0] + r, c2[0] - r, y


def draw_schematic():
    from matplotlib.patches import Circle, Wedge
    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(178 * mm, 54 * mm))
    titles = [
        r"Passivation: still geometric",
        r"Ideal wire: new geometric bound",
        r"Resistance: can exceed the particle limit",
    ]
    for ax, title, lab in zip(axes, titles, "abc"):
        ax.set_xlim(0, 1)
        ax.set_ylim(0.02, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=6.5, pad=3)
        ax.text(-0.02, 1.12, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")

    # (a) one grain, passive cap
    ax = axes[0]
    ax.add_patch(Circle((0.40, 0.50), 0.26, facecolor="#D9E6F2",
                        edgecolor=PALETTE["blue_main"], lw=1.15, zorder=2))
    ax.add_patch(Wedge((0.40, 0.50), 0.26, 48, 132, facecolor="#E8D0D0",
                       edgecolor=PALETTE["red_strong"], lw=0.9, zorder=3))
    ax.text(0.40, 0.78, "passive", ha="center", fontsize=6, color=PALETTE["red_strong"])
    ax.text(0.40, 0.38, "active", ha="center", fontsize=6, color=PALETTE["blue_main"])
    ax.annotate("", xy=(0.84, 0.70), xytext=(0.62, 0.58),
                arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=0.95))
    ax.text(0.68, 0.76, r"$\tau<\tau_0$", color=PALETTE["red_strong"], fontsize=7.5)
    ax.text(0.40, 0.10, "ordered times only shorten", ha="center", fontsize=6,
            color=PALETTE["neutral_dark"])

    # (b) two grains, ideal wire
    ax = axes[1]
    xL, xR, y = _two_grains(ax, fc="#D9E6F2", ec=PALETTE["blue_main"])
    ax.plot([xL, xR], [y, y], color=PALETTE["neutral_black"], lw=3.4,
            solid_capstyle="round", zorder=3)
    ax.text(0.47, 0.76, "ideal wire", ha="center", fontsize=6, color=PALETTE["neutral_black"])
    ax.annotate("", xy=(0.88, 0.30), xytext=(0.74, 0.42),
                arrowprops=dict(arrowstyle="->", color=PALETTE["neutral_dark"], lw=0.95))
    ax.text(0.52, 0.18, r"$\tau_0\to\tau_{\mathrm{cluster}}$", fontsize=7.5,
            color=PALETTE["neutral_dark"])
    ax.text(0.47, 0.10, "slowest time cannot decrease", ha="center", fontsize=6,
            color=PALETTE["neutral_dark"])

    # (c) two grains, finite contact (or interior resistance)
    ax = axes[2]
    xL, xR, y = _two_grains(ax, fc="#E4EEF0", ec=PALETTE["teal"])
    xs = np.linspace(xL, xR, 9)
    amp = 0.055
    ys = y + amp * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
    ax.plot(xs, ys, color=PALETTE["teal"], lw=1.25, zorder=3, solid_capstyle="round")
    ax.text(0.47, 0.78, r"finite $G$ or $\chi$", ha="center", fontsize=6,
            color=PALETTE["teal"])
    ax.annotate("", xy=(0.88, 0.70), xytext=(0.74, 0.56),
                arrowprops=dict(arrowstyle="->", color=PALETTE["teal"], lw=0.95))
    ax.text(0.66, 0.78, r"$\tau>\tau_0$", color=PALETTE["teal"], fontsize=7.5)
    ax.text(0.47, 0.10, "resistive, not geometric", ha="center", fontsize=6,
            color=PALETTE["neutral_dark"])

    fig.tight_layout(w_pad=1.4, rect=(0, 0, 1, 0.96))
    return fig


def main():
    pass_rows, cloud, interior, contact, meta = collect()
    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(178 * mm, 72 * mm), constrained_layout=True)
    axa, axb, axc = axes

    cx = np.array([r["Lambda_E"] for r in cloud])
    cy = np.array([r["tau_over_tau0"] for r in cloud])
    axa.scatter(cx, cy, s=6, c=PALETTE["neutral_light"], linewidths=0, zorder=1)
    grp = {
        "base": (PALETTE["red_strong"], "o", "base patterns"),
        "orientation": (PALETTE["blue_main"], "s", "orientation"),
        "rand": (PALETTE["teal"], "^", "random patches"),
        "frag": (PALETTE["violet"], "D", "fragmented"),
    }
    seen = set()
    for r in pass_rows:
        col, mk, lab = grp[r["group"]]
        axa.scatter(r["Lambda_E"], r["tau_over_tau0"], s=22, c=col, marker=mk, zorder=3,
                    edgecolors="white", linewidths=0.3,
                    label=lab if r["group"] not in seen else None)
        seen.add(r["group"])
    axa.axhline(1.0, color=PALETTE["red_strong"], ls="--", lw=0.9, zorder=2)
    axa.set(xlabel=r"electrical leverage $\Lambda_E$", ylabel=r"$\bar\tau/\tau_0$",
            xlim=(-0.02, 1.02), ylim=(0, 1.15))
    axa.text(0.04, 1.03, "geometric limit", color=PALETTE["red_strong"], fontsize=6)
    axa.legend(loc="lower left", fontsize=5.5, handletextpad=0.3, borderaxespad=0.2,
               markerscale=0.85)
    axa.set_title("Passivation stays below the limit", loc="left")

    chi_line = np.logspace(-0.05, 3.05, 200)
    axb.plot(chi_line, 1 + 2 / chi_line, color=PALETTE["neutral_dark"], lw=1.1, zorder=2,
             label=r"$1+2/\chi$")
    axb.axhline(1.0, color=PALETTE["red_strong"], ls="--", lw=0.9, zorder=1)
    chi = np.array([r["chi"] for r in interior])
    y = np.array([r["tau_over_tau0"] for r in interior])
    # one marker per chi (collapse beta: they collapse by construction)
    for xval in sorted(set(chi)):
        axb.scatter(xval, np.median(y[chi == xval]), s=28, c=PALETTE["blue_main"],
                    zorder=3, edgecolors="white", linewidths=0.3)
    axb.set(xscale="log", xlabel=r"interior contrast $\chi=\sigma_s/\sigma_w$",
            ylabel=r"$\tau_p/\tau_0$", xlim=(0.8, 1200), ylim=(0, 3.4))
    axb.text(80, 1.08, "geometric limit", color=PALETTE["red_strong"], fontsize=6)
    axb.text(1.05, 3.05, r"theory $1+2/\chi$", color=PALETTE["neutral_dark"], fontsize=6)
    axb.set_title("Interior resistance exceeds the limit", loc="left")

    g = np.array([r["G_true_S"] for r in contact])
    ys = np.array([r["tau_slow_over_tau_particle"] for r in contact])
    ok = np.array([r["reliable"] for r in contact], dtype=bool)
    axc.axhline(1.0, color=PALETTE["red_strong"], ls="--", lw=0.9, zorder=1)
    axc.axhline(meta["tau_cluster_over_tau_particle"], color=PALETTE["neutral_dark"],
                ls=":", lw=1.0, zorder=1)
    axc.plot(g[ok], ys[ok], color=PALETTE["teal"], lw=0.8, zorder=2)
    axc.scatter(g[ok], ys[ok], s=28, c=PALETTE["teal"], zorder=3,
                edgecolors="white", linewidths=0.3)
    axc.set(xscale="log", xlabel=r"true contact $G$ (S)",
            ylabel=r"$\tau_\mathrm{slow}/\tau_\mathrm{particle}$",
            ylim=(0.6, 6.6), xlim=(3e-4, 3))
    axc.text(4e-3, 1.15, "particle limit", color=PALETTE["red_strong"], fontsize=6)
    axc.text(4e-3, meta["tau_cluster_over_tau_particle"] + 0.18, "ideal cluster",
             color=PALETTE["neutral_dark"], fontsize=6)
    axc.set_title("Finite contact exceeds the particle limit", loc="left")

    for ax, lab in zip(axes, "abc"):
        ax.text(-0.18, 1.08, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")

    stem = R / "figure_three_regimes"
    save_pub(fig, stem)
    save_pub(fig, VEC / "figure2_three_regimes")
    for ax, name in zip(axes, ("panel_a_passivation", "panel_b_interior", "panel_c_contact")):
        save_panel(ax, VEC / name)
    fig_s = draw_schematic()
    save_pub(fig_s, R / "figure_schematic")
    save_pub(fig_s, VEC / "figure1_schematic")
    plt.close(fig)
    plt.close(fig_s)
    print("wrote", stem, "and vector bundle in", VEC, "n_pass", len(pass_rows), "n_cloud", len(cloud))


if __name__ == "__main__":
    main()
