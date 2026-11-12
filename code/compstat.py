"""Deterministic steady state across parameter variations around the calibrated point.

Starts from data/ou_best.npz and its underwriting standard qf. For each variation, compares the closed-form
B_pd(0,1), d_first, and B_fd(0,1) of the characterization section with the computed solution (implicit upwind
time stepping, 200 nodes per unit of debt on [0, 1.2], tolerances 1e-7 / 1e-8). Variations run in parallel.
Writes data/compstat.npz and data/compstat.log; tab_compstat.py writes the table.
Usage: python code/compstat.py
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from paths import DATA
QF = float(np.load(os.path.join(DATA, "ou_best.npz"))["qf"])
NODES = 200
BMAX = 1.2
MAXIT = 30000


def base_params():
    """Calibrated structural parameters from data/ou_best.npz."""
    a = np.load(os.path.join(DATA, "ou_best.npz"))
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = a["x"].astype(float)
    return dict(g0=g0, g1=g1, g2=g2, zt=zt, rho=rho, kappa=kappa, delta=delta)


def closed_forms(p):
    """Model parameters and the closed-form F, B_pd(0,1), d_first, and B_fd(0,1) at z = 0."""
    from pdmodel import Params
    lam = float(np.log(1.04))
    par = Params(nu=2.0, rho=p["rho"], r=lam, delta=p["delta"], kappa=p["kappa"], g0=p["g0"], g1=p["g1"], g2=p["g2"],
                 ztilde=p["zt"], fixed_cost_form="level", S_cap=1.0, q_floor=QF)
    g0, g1, g2, zt, kappa, delta = p["g0"], p["g1"], p["g2"], p["zt"], p["kappa"], p["delta"]
    F = max(1 - g2 * (1.0 - np.exp(zt)), par.F_min)
    G = g0 * g1 * F
    K = (1 - kappa) * (delta + lam)
    Bpd = ((g1 / (g1 - 1)) * (1 - F) * G ** (1 / (g1 - 1)) / K ** (g1 / (g1 - 1))) ** ((g1 - 1) / g1)
    dfirst = ((1 - F) / ((g1 - 1) * g0 * F)) ** (1 / g1)
    Bfd1 = G / K
    return par, dict(F=F, Bpd=Bpd, dfirst=dfirst, Bfd1=Bfd1)


def solve_one(item):
    """Solve the deterministic model for one variation and locate the computed thresholds on the grid."""
    label, p = item
    from pdmodel import Model, ZProcess
    from pdtimestep import solve_timestep
    t0 = time.time()
    par, cf = closed_forms(p)
    nB = int(round(BMAX * NODES)) + 1
    m = Model(par, ZProcess.none(0.0), 0.0, BMAX, nB)
    res = solve_timestep(m, None, None, dt=0.01, maxit=MAXIT, print_every=10 ** 9, verbose=False, dt_adapt=False,
                         tolV=1e-7, tolq=1e-8, q_init="static")
    # steady state, first default node, floor, and full default on the grid
    d, q, S = res.d[:, 0], res.q[:, 0], res.S[:, 0]
    i1 = np.where(d > 1e-9)[0]
    i0 = np.where(S > 1e-10)[0]
    ifl = np.where(q < QF)[0]
    ifd = np.where(d >= 1 - 1e-9)[0]
    out = dict(label=label, **{k: float(v) for k, v in p.items()}, **cf,
               Bstar=float(m.B[i0[-1]]) if i0.size else np.nan,
               Bfirst=float(m.B[i1[0]]) if i1.size else np.nan, dfirst_num=float(d[i1[0]]) if i1.size else np.nan,
               Bfloor=float(m.B[ifl[0]]) if ifl.size else np.nan, Bfull=float(m.B[ifd[0]]) if ifd.size else np.nan,
               converged=bool(res.converged), resV=float(res.resV), resq=float(res.resq), iters=int(res.iters),
               secs=time.time() - t0)
    return out


def variations():
    """The baseline and two values on each side for gamma0, kappa, gamma2, rho, and delta."""
    b = base_params()
    V = [("baseline", dict(b))]
    for f in (0.8, 1.2):
        p = dict(b); p["g0"] = b["g0"] * f; V.append((f"gamma0 x {f}", p))
    for v in (0.45, 0.65):
        p = dict(b); p["kappa"] = v; V.append((f"kappa = {v}", p))
    for f in (0.5, 1.5):
        p = dict(b); p["g2"] = b["g2"] * f; V.append((f"gamma2 x {f}", p))
    for v in (0.050, 0.070):
        p = dict(b); p["rho"] = v; V.append((f"rho = {v}", p))
    for v in (0.12, 0.22):
        p = dict(b); p["delta"] = v; V.append((f"delta = {v}", p))
    return V


# LaTeX row labels for the table
LABELS = {
    "baseline": r"Calibrated point",
    "gamma0 x 0.8": r"$\gamma_0 \times 0.8$", "gamma0 x 1.2": r"$\gamma_0 \times 1.2$",
    "kappa = 0.45": r"$\kappa = 0.45$", "kappa = 0.65": r"$\kappa = 0.65$",
    "gamma2 x 0.5": r"$\gamma_2 \times 0.5$", "gamma2 x 1.5": r"$\gamma_2 \times 1.5$",
    "rho = 0.05": r"$\rho = 0.05$", "rho = 0.07": r"$\rho = 0.07$",
    "delta = 0.12": r"$\delta = 0.12$", "delta = 0.22": r"$\delta = 0.22$",
}


if __name__ == "__main__":
    t0 = time.time()
    V = variations()
    with Pool(min(11, os.cpu_count() or 1)) as pool:
        rows = pool.map(solve_one, V)
    lines = []
    for r in rows:
        lines.append(f"{r['label']:14s} Bpd {r['Bpd']:.4f} B* node {r['Bstar']:.4f} first node {r['Bfirst']:.4f} "
                     f"d_first {r['dfirst']:.4f} / {r['dfirst_num']:.4f}  Bfd {r['Bfd1']:.3f} floor {r['Bfloor']:.3f} full {r['Bfull']:.3f} "
                     f"conv {r['converged']} res {r['resV']:.1e}/{r['resq']:.1e} {r['iters']} steps {r['secs']:.0f}s")
    lines.append(f"total {time.time() - t0:.0f}s")
    open(os.path.join(DATA, "compstat.log"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    np.savez(os.path.join(DATA, "compstat.npz"), rows=np.array(rows, dtype=object), allow_pickle=True)
