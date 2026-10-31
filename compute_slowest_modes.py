#!/usr/bin/env python3
"""Recompute the slowest single-layer eigenvalue for the 17 passivation patterns.

The eigenvalue is of the compressed single-layer operator on the active face
of the unit sphere, with zero net flux. It is not a finite-element eigenmode.
The archived values in results/modal_spectrum.json were produced with the same
discretization (N=3000). This script does not read or rerun finite-element spectra.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
N = 3000


def fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = math.pi * (1 + 5**0.5) * i
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def self_term(w):
    g = math.acos(1 - w / (2 * math.pi))
    t = np.linspace(0, g, 4001)[1:]
    s = np.sin(t / 2)
    f = (1 / s - np.log1p(1 / s)) * np.sin(t)
    trap = getattr(np, "trapezoid", np.trapz)
    return 0.5 * float(trap(np.concatenate([[2.0], f]), np.concatenate([[0.0], t])))


def patterns(u):
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    out = {
        "hemisphere_par": x <= 0,
        "polar_par": np.abs(x) >= 0.5,
        "polar_perp": np.abs(y) >= 0.5,
    }
    c6 = 1 - 2 / 12
    out["six_caps"] = ~((np.abs(x) >= c6) | (np.abs(y) >= c6) | (np.abs(z) >= c6))
    for a in (15, 30, 45, 60, 75):
        ax = np.array([math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0])
        out[f"angle_{a}"] = np.abs(u @ ax) >= 0.5
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z, -1, 1))
    scores = {
        "frag_checker_octants": x * y * z,
        "frag_azimuth_4": np.cos(4 * phi),
        "frag_latitude_4": np.cos(4 * theta),
        "frag_mixed_harmonic": np.sin(3 * theta) * np.cos(3 * phi)
        + 0.55 * np.sin(5 * theta) * np.sin(2 * phi),
    }
    for k in range(4):
        rng = np.random.default_rng(20260781 + k)
        lin = u @ rng.normal(size=3)
        q = rng.normal(size=(3, 3))
        q = 0.5 * (q + q.T)
        quad = np.einsum("ni,ij,nj->n", u, q, u)
        d = rng.normal(size=(3, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        cub = np.prod(u @ d.T, axis=1)
        scores[f"rand_random_{k + 1}"] = lin + 0.7 * quad + 0.5 * cub
    for name, s in scores.items():
        thr = np.sort(s)[::-1][len(s) // 2 - 1]
        out[name] = s >= thr
    return out


def slowest_mu(u, w, s_self, active):
    idx = np.where(active)[0]
    m = len(idx)
    P = u[idx]
    A = np.empty((m, m))
    for s0 in range(0, m, 500):
        e0 = min(s0 + 500, m)
        d = np.linalg.norm(P[s0:e0, None, :] - P[None, :, :], axis=2)
        s = np.maximum(d / 2, 1e-12)
        A[s0:e0] = w * (1 / s - np.log1p(1 / s)) / (4 * math.pi)
    A[np.arange(m), np.arange(m)] = s_self
    H = np.eye(m) - 1.0 / m
    ev, U = np.linalg.eigh(H)
    Z = U[:, ev > 0.5]
    mu = np.linalg.eigvalsh(Z.T @ A @ Z)
    return float(mu.max())


def main():
    archived = json.loads((ROOT / "results" / "modal_spectrum.json").read_text())
    u = fibonacci(N)
    w = 4 * math.pi / N
    s_self = self_term(w)
    worst = 0.0
    print(f"{'case':22s} {'2mu':>8s} {'archived':>8s} {'abs diff':>10s}")
    for name, mask in patterns(u).items():
        mu2 = 2.0 * slowest_mu(u, w, s_self, mask)
        ref = float(archived[name]["mu_max2"])
        err = abs(mu2 - ref)
        worst = max(worst, err)
        print(f"{name:22s} {mu2:8.4f} {ref:8.4f} {err:10.3e}")
    if worst > 1e-6:
        raise SystemExit(f"recomputed 2 mu_max differs from the archive by {worst}")
    print(f"matched archived 2 mu_max for {len(archived)} patterns; max abs diff {worst:.3e}")


if __name__ == "__main__":
    main()
