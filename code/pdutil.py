"""pdutil.py -- shared helpers for the scripts that solve and analyze the diffusion model: paths, the
9-vector of calibrated parameters and its mapping to Params and ZProcess, a Result rebuilt from a saved
solution, the diagnostics of the underwriting standard, and small table and JSON helpers."""
import json
import os
import sys

import numpy as np

from amr_values import AMR_LOSS  # noqa: F401

from paths import DATA, LOGS as OUT
PKG = os.path.dirname(os.path.abspath(__file__))

R_FIX = float(np.log(1.04))
NAMES = ["g0", "g1", "g2", "ztilde", "rho", "kappa", "delta", "rho_z", "sig_eps"]


def load_calibrate2(floor):
    """calibrate2 reads the underwriting standard from the environment and parses sys.argv at import:
    set both up, import, and check. Must be the first import of calibrate2 in the process."""
    os.environ["PD_QFLOOR"] = repr(float(floor))
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        import calibrate2 as C
    finally:
        sys.argv = argv
    if abs(C.Q_FLOOR - float(floor)) > 1e-12:
        raise RuntimeError(f"calibrate2 was imported with q_floor {C.Q_FLOOR}, wanted {floor}")
    return C


def unpack(x, floor, nz):
    """Parameters and productivity process of the diffusion model from the 9-vector (as calibrate2.unpack)."""
    from pdmodel import Params, ZProcess
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = map(float, x)
    mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
    par = Params(nu=2.0, rho=rho, r=R_FIX, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2, ztilde=zt,
                 fixed_cost_form="level", S_cap=1.0, q_floor=float(floor))
    zp = ZProcess.ou(mu, sigma, nz, zmin=-2.5 * sd, zmax=2.5 * sd)
    return par, zp


def feasible(x):
    """Parameter bounds of the calibration search (as calibrate2.feasible)."""
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = x
    return (g0 > 0 and g1 > 1.0 and g2 >= 0 and rho > R_FIX + 0.001 and 0.0 <= kappa <= 1.0
            and 0.02 <= delta <= 0.5 and 0.5 <= rho_z <= 0.99 and sig_eps > 0.005)


def result_from_npz(a, floor=None):
    """Rebuild Model, policies, generator and Result from a saved solution (V, q, B, z, x, qf)."""
    from pdmodel import Model, Result, ZProcess, ergodic_density
    x = a["x"].astype(float)
    floor = float(a["qf"]) if floor is None else float(floor)
    B, z = a["B"], a["z"]
    par, zp0 = unpack(x, floor, z.size)
    zp = ZProcess.ou(zp0.mu, zp0.sigma, z.size, zmin=float(z[0]), zmax=float(z[-1]))
    assert np.allclose(zp.z, z)
    h = np.diff(B)
    m = Model(par, zp, 0.0, float(B[-1]), B.size) if np.allclose(h, h[0]) else Model(par, zp, 0.0, float(B[-1]), 0, B=B)
    V, q = a["V"], a["q"]
    S, c, d, u, H, which, S_free, d_free = m.policies(V, q, 28)
    A = m.assemble(S)
    g = a["g"] if "g" in a and a["g"].size == V.size and np.all(np.isfinite(a["g"])) else ergodic_density(m, A)
    coupon = par.delta + par.lam
    resV = float(np.max(np.abs(par.rho * V.ravel() - u.ravel() - A @ V.ravel())))
    resq = float(np.max(np.abs(par.xi(d.ravel()) * q.ravel() - (1 - d.ravel()) * coupon - A @ q.ravel())))
    res = Result(m, V, q, S, c, d, which, S_free, d_free, A, g, True, 0, resV, resq)
    return m, res


def standard_diag(res, floor):
    """How the underwriting standard shapes the solution: binding mass, full-default issuing (the explosion),
    the zero-drift stop by productivity state."""
    m = res.model
    B, z, q, d, S = m.B, m.z, res.q, res.d, res.S
    g = res.g / res.g.sum()
    # nodes where the standard binds (price below the floor, zero issuance, d < 1) and nodes in full
    # default that still accumulate debt
    iss = res.issuance
    binding = (q < floor - 1e-9) & (np.abs(iss) < 1e-12) & (d < 1 - 1e-9) if floor > 0 else np.zeros_like(q, bool)
    full_issuing = (d >= 1 - 1e-9) & (S > 1e-10)
    # in each z row, the first node where the drift turns from positive to non-positive, classified
    # as on the floor, in partial default, or without default
    rows = []
    mass_floor_rows = 0.0
    n_floor_rows = 0
    for j in range(z.size):
        k = np.where((S[:-1, j] > 0) & (S[1:, j] <= 0))[0]
        if not k.size:
            rows.append((float(z[j]), np.nan, np.nan, np.nan, "none"))
            continue
        i = k[0]
        on_floor = floor > 0 and abs(q[i, j] - floor) < 0.003
        lab = "floor" if on_floor else ("pd" if d[i, j] > 1e-9 else "nodef")
        if on_floor:
            mass_floor_rows += g[:, j].sum(); n_floor_rows += 1
        rows.append((float(z[j]), float(B[i]), float(q[i, j]), float(d[i, j]), lab))
    return dict(binding_mass=float(g[binding].sum()),
                full_issuing_nodes=int(full_issuing.sum()),
                full_issuing_max_q=float(q[full_issuing].max()) if full_issuing.any() else float("nan"),
                full_issuing_mass=float(g[full_issuing].sum()),
                mass_full_default=float(g[d >= 1 - 1e-9].sum()),
                mass_top10=float(g[B >= 0.9 * B[-1]].sum()),
                mass_q_below_floor=float(g[q < floor].sum()) if floor > 0 else 0.0,
                floor_rows=n_floor_rows, floor_rows_mass=float(mass_floor_rows),
                max_B_with_mass=float(B[np.where(g.sum(1) > 1e-6)[0].max()]),
                atom0=float(g[0].sum()), EB=float((g * B[:, None]).sum()), pdef=float(g[d > 1e-9].sum()),
                locus=rows)


def diag_line(D):
    """One-line summary of standard_diag."""
    return (f"binding {D['binding_mass']:.4f} | full-default issuing nodes {D['full_issuing_nodes']} (max q {D['full_issuing_max_q']:.2f}, "
            f"mass {D['full_issuing_mass']:.1e}) | mass full default {D['mass_full_default']:.1e} | mass top 10% {D['mass_top10']:.1e} | "
            f"floor rows {D['floor_rows']} (mass {D['floor_rows_mass']:.3f}) | max B with mass {D['max_B_with_mass']:.3f}")


def locus_table(D, every=4):
    """Markdown table of the zero-drift locus from standard_diag, every `every`-th z row."""
    lines = ["| z | zero-drift B | q there | d there | regime |", "|---|---|---|---|---|"]
    for r in D["locus"][::every]:
        lines.append(f"| {r[0]:+.3f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.2f} | {r[4]} |")
    return "\n".join(lines)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
