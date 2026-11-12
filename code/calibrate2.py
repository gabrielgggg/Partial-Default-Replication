"""
calibrate2.py -- calibration library: joint moment matching of the OU model to Arellano,
Mateos-Planas and Rios-Rull (2023, JPE). The nine free parameters are those of their Table 3
(phi0, gamma, phi1, z*, beta, kappa, delta, rho_z, sigma_eps), the 11 target moments and weights
those of Table 4 and footnote 22. nu = 2 and r = ln(1.04) are fixed; the fixed cost has the level form.

The package scripts use the parameter mapping (unpack) and init_from_file. The underwriting
standard q_floor is read from the environment variable PD_QFLOOR at import (see
pdutil.load_calibrate2). The calibration search itself (evaluate and the __main__ block, a
Nelder-Mead search) is not run by the package.

Moments: exact stationary (ergodic) moments of the discretized chain for the state functions
(debt/y, debt service/y, debt due/y, spread sd) and a 16-path x 25000-year simulation with annual
accounting for the partial-default and output moments (pdmoments.combined_moments).

Each evaluation: warm start (V, q of the previous point interpolated onto the new z grid), implicit
upwind time stepping for a fixed number of steps, moments, loss.
"""

import sys
import time
import json

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from pdmodel import Model, Params, ZProcess
from pdtimestep import solve_timestep
from pdmoments import combined_moments, loss, table_combined, AMR_DATA, AMR_MODEL, TARGETS
from pddiag import resolution_report
from amr_values import REPORTED_PRECISION

TAG = sys.argv[1] if len(sys.argv) > 1 else "calib2"
MAXEVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 400
NZ, BMAX, DB = 25, 1.0, 0.005
NB = int(round(BMAX / DB)) + 1
DT, MAXIT = 0.015, 3000
NAMES = ["g0", "g1", "g2", "ztilde", "rho", "kappa", "delta", "rho_z", "sig_eps"]
R_FIX = np.log(1.04)
import os
Q_FLOOR = float(os.environ.get("PD_QFLOOR", "0.0"))   # underwriting standard: no issuance at q < Q_FLOOR
PATIENCE = int(os.environ.get("PD_PATIENCE", "60"))   # stop after this many evaluations without improvement

state = dict(V=None, q=None, z=None, n=0, best=(np.inf, None))

# Stopping rule: the loss reaches the benchmark L_STOP, the loss of the AMR model column under the AMR
# weights, or every target agrees with the data to the precision reported by AMR (half a unit of the
# last printed digit).
L_STOP = loss(AMR_MODEL)


def within_reported(M):
    """Targets that the moments M match to the precision AMR report."""
    return [k for k in TARGETS if abs(M[k] - AMR_DATA[k]) <= 0.5 * REPORTED_PRECISION[k]]


class GoodEnough(Exception):
    """Raised by evaluate to stop the search."""
    pass


def unpack(x):
    """Params and the OU ZProcess (NZ nodes, +-2.5 unconditional sd) from the 9-vector x."""
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = map(float, x)
    mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
    par = Params(nu=2.0, rho=rho, r=R_FIX, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2, ztilde=zt,
                 fixed_cost_form="level", S_cap=1.0, q_floor=Q_FLOOR)
    zp = ZProcess.ou(mu, sigma, NZ, zmin=-2.5 * sd, zmax=2.5 * sd)
    return par, zp


def feasible(x):
    """Parameter bounds of the search."""
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = x
    return (g0 > 0 and g1 > 1.0 and g2 >= 0 and rho > R_FIX + 0.001 and 0.0 <= kappa <= 1.0
            and 0.02 <= delta <= 0.5 and 0.5 <= rho_z <= 0.99 and sig_eps > 0.005)


def warm_start(m):
    """Previous (V, q) interpolated onto the current z grid (B grid is fixed)."""
    V0, q0, z_old = state["V"], state["q"], state["z"]
    if np.allclose(z_old, m.z):
        return V0, q0
    f = lambda F: np.stack([np.interp(m.z, z_old, F[i]) for i in range(m.nB)])
    return f(V0), np.clip(f(q0), 1e-6, 1.0)


def init_from_file(path, m):
    """(V, q) of a saved solution interpolated onto the grid of model m; beyond the saved Bmax, V is
    extended linearly and q geometrically."""
    a = np.load(path)
    B_old, z_old = a["B"], a["z"]
    BB, ZZ = np.meshgrid(np.clip(m.B, B_old[0], B_old[-1]), np.clip(m.z, z_old[0], z_old[-1]), indexing="ij")
    pts = np.stack([BB.ravel(), ZZ.ravel()], axis=1)
    V0 = RegularGridInterpolator((B_old, z_old), a["V"])(pts).reshape(m.nB, m.nz)
    q0 = RegularGridInterpolator((B_old, z_old), a["q"])(pts).reshape(m.nB, m.nz)
    ext = m.B > B_old[-1]
    if ext.any():
        slope = (a["V"][-1] - a["V"][-2]) / (B_old[-1] - B_old[-2])
        V0[ext] = (np.interp(m.z, z_old, a["V"][-1])[None, :]
                   + np.interp(m.z, z_old, slope)[None, :] * (m.B[ext] - B_old[-1])[:, None])
        ratio = np.clip(np.interp(m.z, z_old, a["q"][-1] / a["q"][-2]), 0.5, 1.0)
        q0[ext] = (np.interp(m.z, z_old, a["q"][-1])[None, :]
                   * ratio[None, :] ** ((m.B[ext] - B_old[-1])[:, None] / (B_old[-1] - B_old[-2])))
    return V0, np.clip(q0, 1e-6, 1.0)


def evaluate(x, verbose=True):
    """Loss at x: solve from a warm start, compute the moments, log the row, keep the best point."""
    if not feasible(x):
        return 1e3
    par, zp = unpack(x)
    m = Model(par, zp, 0.0, BMAX, NB)
    t0 = time.time()
    # solve from the previous solution (or a saved one at the first evaluation)
    if state["V"] is None:
        V0, q0 = init_from_file("out/calib1_best.npz", m)
    else:
        V0, q0 = warm_start(m)
    res = solve_timestep(m, V0, q0, dt=DT, maxit=MAXIT, print_every=10 ** 9, verbose=False, tolV=1e-7, tolq=1e-8)
    state["V"], state["q"], state["z"] = res.V, res.q, m.z.copy()
    # moments and loss
    M, info = combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
    L = loss(M)
    E = info["ergodic"]
    E["mass_unresolved"] = resolution_report(res, verbose=False).get("ergodic_mass_unresolved", np.nan)
    # record the best point and log the evaluation
    state["n"] += 1
    if L < state["best"][0]:
        state["best"] = (L, dict(x=list(map(float, x)), M=M))
        state["n_best"] = state["n"]
        np.savez(f"out/{TAG}_best.npz", V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z,
                 x=np.array(x, dtype=float), names=np.array(NAMES), qf=par.q_floor)
        with open(f"out/{TAG}_best.md", "w", encoding="utf-8") as f:
            f.write(f"# {TAG}: best point after {state['n']} evaluations, loss {L:.4f}\n\n"
                    + "| " + " | ".join(NAMES) + " |\n|" + "---|" * len(NAMES) + "\n| "
                    + " | ".join(f"{v:.4f}" for v in x) + " |\n\n" + table_combined(M, info) + "\n")
    line = ("| " + str(state["n"]) + " | " + " | ".join(f"{v:.4f}" for v in x) + f" | {L:.4f} | {res.resV:.0e}/{res.resq:.0e} | "
            + " | ".join(f"{M[k]:.3f}" if abs(M[k]) >= 0.1 else f"{M[k]:.4f}" for k in TARGETS)
            + f" | {M['spread_mean']:.4f} | {E['mass_top']:.4f} | {E['spread_var_share_top']:.2f} | {E['mass_unresolved']:.4f} | {time.time() - t0:.0f}s |")
    with open(f"out/{TAG}_log.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if verbose:
        print(line, flush=True)
    # stopping rules: the benchmark loss is reached, or no improvement for PATIENCE evaluations
    if L <= L_STOP or len(within_reported(M)) == len(TARGETS):
        raise GoodEnough(L)
    if state["n"] - state.get("n_best", 0) >= PATIENCE:
        raise GoodEnough(L)
    return L


HEADER = ("| n | " + " | ".join(NAMES) + " | loss | resV/resq | " + " | ".join(TARGETS)
          + " | spread mean | mass top 10% B | spread var share top | time |\n"
          + "|---|" + "---|" * (len(NAMES) + 2 + len(TARGETS) + 4))

if __name__ == "__main__":
    # Nelder-Mead search in scaled deviations from x0
    x0 = np.array([0.0330, 2.090, 0.1341, -0.1034, -np.log(0.954), 0.70, 0.12, 0.875, 0.052])
    if len(sys.argv) > 3:
        x0 = np.load(sys.argv[3])["x"].astype(float)
        assert x0.size == 9
    scale = np.array([0.005, 0.25, 0.04, 0.03, 0.006, 0.05, 0.02, 0.025, 0.006])
    with open(f"out/{TAG}_log.md", "a", encoding="utf-8") as f:
        f.write(f"# joint calibration run {TAG}\n\ngrid: Bmax={BMAX} dB={DB} nz={NZ}; dt={DT} steps={MAXIT}; q_floor={Q_FLOOR}; "
                f"targets: {json.dumps({k: AMR_DATA[k] for k in TARGETS})}\n\nx0 = {x0.tolist()}, scale = {scale.tolist()}\n\n" + HEADER + "\n")
    f = lambda y: evaluate(x0 + scale * y)
    try:
        r = minimize(f, np.zeros(9), method="Nelder-Mead",
                     options=dict(maxfev=MAXEVAL, xatol=0.02, fatol=1e-3, adaptive=True,
                                  initial_simplex=np.vstack([np.zeros(9), np.eye(9)])))
        msg = str(r.message)
    except GoodEnough:
        b = state["best"]
        if b[0] <= L_STOP or len(within_reported(b[1]["M"])) == len(TARGETS):
            msg = f"stopped: benchmark reached (loss {b[0]:.4f} <= {L_STOP:.4f}, or all targets within reported precision)"
        else:
            msg = f"stopped: no improvement for {PATIENCE} evaluations (best loss {b[0]:.4f} at evaluation {state.get('n_best', 0)})"
    best = state["best"]
    print("best loss", best[0], "at", best[1]["x"], msg)
    with open(f"out/{TAG}_log.md", "a", encoding="utf-8") as fh:
        fh.write(f"\nfinished ({msg}); best loss {best[0]:.4f} at x = {best[1]['x']}\n")
