#!/usr/bin/env python3
"""
Read-only two-mode analysis of the ARCHIVED finite-contact two-particle spectra.

Sum-rule picture: for any contact conductance G > 0 the low-frequency asymptote is the
equipotential-cluster one (M1 -> M1_ideal) and the amplitude weight M0 stays between the
disconnected and ideal values, so the extra weight M1_ideal - M1_disc must be carried by a
slow, contact-limited mode.  Here the archived spectrum of each case is fitted with

    sigma(w) = sigma0 + a_f * i w tau_f/(1 + i w tau_f) + a_s * i w tau_s/(1 + i w tau_s)

and compared with (i) the single-mode fits of the disconnected and ideal limits,
(ii) the sum rule  a_f tau_f + a_s tau_s = M1_ideal,  (iii) tau_s * G_true vs C0*A.

The archived contact assembly uses G_true = 0.4 * G_nominal. Frequencies span only
2.5-382 Hz, so slow modes below ~2 Hz are not resolved and are excluded from interpretation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "inputs" / "phase_diagram.csv"
OUT = ROOT / "results"
C0, R = 0.4, 5e-3
CA = C0 * 4 * math.pi * R**2  # F, interfacial capacitance of one particle


def model(p, w, n_modes):
    s0 = p[0]
    y = np.full(w.shape, s0, dtype=complex)
    for k in range(n_modes):
        a, lt = p[1 + 2 * k], p[2 + 2 * k]
        z = 1j * w * 10**lt
        y += a * z / (1 + z)
    return y


def fit(f, sre, sim, n_modes, starts):
    w = 2 * np.pi * f
    scale = np.max(np.abs(sim))
    best = None
    for st in starts:
        p0 = [sre[0]] + st
        lo = [sre[0] - 0.01] + [0, -4] * n_modes
        hi = [sre[0] + 0.01] + [0.1, 1] * n_modes

        def resid(p):
            y = model(p, w, n_modes)
            return np.concatenate([(y.real - sre) / scale, (y.imag - sim) / scale])

        try:
            r = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def main():
    rows = list(csv.DictReader(open(SRC)))
    cases = {}
    for r in rows:
        cases.setdefault(r["case_id"], []).append(r)
    out = {}
    single_starts = [[0.004, -2.0 + d] for d in (-0.5, 0.0, 0.5)]
    for cid, rr in cases.items():
        rr.sort(key=lambda r: float(r["frequency_hz"]))
        f = np.array([float(r["frequency_hz"]) for r in rr])
        sre = np.array([float(r["sigma_re_S_m"]) for r in rr])
        sim = np.array([float(r["sigma_im_S_m"]) for r in rr])
        g_nom = rr[0]["contact_conductance_S"]
        if cid in ("G_0S", "ideal_connected"):
            b = fit(f, sre, sim, 1, single_starts)
            p = b.x
            out[cid] = dict(kind="single", sigma0=p[0], a=p[1], tau=10 ** p[2], a_tau=p[1] * 10 ** p[2],
                            f_tau_Hz=1 / (2 * math.pi * 10 ** p[2]), rms_resid=float(math.sqrt(2 * b.cost / (2 * len(f)))))
    M1_ideal = out["ideal_connected"]["a_tau"]
    M1_disc = out["G_0S"]["a_tau"]
    M0_ideal, M0_disc = out["ideal_connected"]["a"], out["G_0S"]["a"]
    print(f"disconnected: a={M0_disc:.4e} tau={out['G_0S']['tau']*1e3:.3f} ms ({out['G_0S']['f_tau_Hz']:.1f} Hz)  M1={M1_disc:.4e}")
    print(f"ideal       : a={M0_ideal:.4e} tau={out['ideal_connected']['tau']*1e3:.3f} ms ({out['ideal_connected']['f_tau_Hz']:.1f} Hz)  M1={M1_ideal:.4e}")
    print(f"(single-mode fit rms residual: disc={out['G_0S']['rms_resid']:.2e} ideal={out['ideal_connected']['rms_resid']:.2e} of peak)")
    print("\nG_nom[S]  G_true[S]  Lambda_true | single-mode fit: f_tau  rms | two-mode: a_f  f_f[Hz]  a_s  f_s[Hz]  rms | a_f*tau_f+a_s*tau_s / M1_ideal | tau_s*G_true/(C0*A)")
    table = []
    two_starts = []
    for lf in (-2.5, -2.0, -1.7):
        for ls in (-1.5, -1.0, -0.5, 0.0):
            two_starts.append([0.003, lf, 0.004, ls])
    for cid, rr in cases.items():
        if cid in ("G_0S", "ideal_connected"):
            continue
        rr.sort(key=lambda r: float(r["frequency_hz"]))
        f = np.array([float(r["frequency_hz"]) for r in rr])
        sre = np.array([float(r["sigma_re_S_m"]) for r in rr])
        sim = np.array([float(r["sigma_im_S_m"]) for r in rr])
        g_nom = float(rr[0]["contact_conductance_S"]); g_true = 0.4 * g_nom
        one = fit(f, sre, sim, 1, single_starts)
        two = fit(f, sre, sim, 2, two_starts)
        scale = np.max(np.abs(sim))
        rms1 = math.sqrt(2 * one.cost / (2 * len(f))); rms2 = math.sqrt(2 * two.cost / (2 * len(f)))
        p = two.x
        modes = sorted([(10 ** p[2], p[1]), (10 ** p[4], p[3])])      # fast first
        (tf, af), (ts, as_) = modes
        sumM1 = af * tf + as_ * ts
        lam = g_true / (2 * math.pi * 31.83 * CA)  # Lambda_true = G/(omega0 C0 A) with f0 = sigma/(pi R C0)
        row = dict(G_nom=g_nom, G_true=g_true, Lambda_true=lam,
                   one_mode_f_Hz=1 / (2 * math.pi * 10 ** one.x[2]), one_mode_rms=rms1,
                   a_f=af, f_f_Hz=1 / (2 * math.pi * tf), a_s=as_, f_s_Hz=1 / (2 * math.pi * ts),
                   two_mode_rms=rms2, sum_M1_over_M1_ideal=sumM1 / M1_ideal,
                   tauG_over_CA=ts * g_true / CA, slow_weight_share=as_ / (af + as_))
        table.append(row)
        print(f"{g_nom:7g} {g_true:9.4g} {lam:9.3f} |  {row['one_mode_f_Hz']:7.2f} {rms1:.1e} | {af:.2e} {row['f_f_Hz']:7.2f} {as_:.2e} {row['f_s_Hz']:8.2f} {rms2:.1e} | {row['sum_M1_over_M1_ideal']:.3f} | {row['tauG_over_CA']:.3f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "two_mode_contact_fits.json").write_text(json.dumps(dict(limits=out, M1_ideal=M1_ideal, M1_disc=M1_disc, table=table), indent=2, default=float))


if __name__ == "__main__":
    main()
