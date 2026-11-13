"""calutil.py -- shared helpers of the calibration and of the scripts that solve the diffusion model: the folder of
the calibration files, the parameter vector and its mapping to Params and ZProcess, and the diagnostics of the
underwriting standard."""
import json
import os

import numpy as np

from amr_values import AMR_LOSS, AMR_KAPPA, AMR_DELTA_ANNUAL, AMR_R_GROSS  # noqa: F401
from paths import DATA

CAL = os.path.join(DATA, "calibration")            # start.npz, and everything the search and its confirmation write
os.makedirs(CAL, exist_ok=True)

R_FIX = float(np.log(AMR_R_GROSS))
NAMES = ["g0", "g1", "g2", "ztilde", "rho", "kappa", "delta", "rho_z", "sig_eps"]
# Fixed at the values of AMR. Their recovery factor is the present value, at the risk-free rate, of the new claims per
# unit of missed payment, in their model and in ours, so it carries over unchanged. Their coupons decay by the factor
# 0.88 a year; ours decay at the rate delta, so delta = -ln(0.88) gives the same decay over a year (as r = ln(1.04)).
KAPPA_FIX = float(AMR_KAPPA)
DELTA_FIX = float(-np.log(AMR_DELTA_ANNUAL))
FREE = [0, 1, 2, 3, 4, 7, 8]                       # positions of the seven calibrated parameters in the 9-vector
FREE_NAMES = [NAMES[i] for i in FREE]


def full_x(xf):
    """The 9-vector of the model from the seven calibrated parameters."""
    x = np.empty(9)
    x[FREE] = np.asarray(xf, float)
    x[5], x[6] = KAPPA_FIX, DELTA_FIX
    return x


def unpack(x, floor, nz, band=0.0, lband=0.1, closed_d=True):
    """Parameters and endowment process of the diffusion model from the 9-vector.
    band, lband: the smooth underwriting standard (pdmodel.Params.q_band, l_band); band 0 is the hard standard."""
    from pdmodel import Params, ZProcess
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = map(float, x)
    mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
    par = Params(nu=2.0, rho=rho, r=R_FIX, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2, ztilde=zt,
                 fixed_cost_form="level", S_cap=1.0, q_floor=float(floor), q_band=float(band), l_band=float(lband), closed_d=bool(closed_d))
    zp = ZProcess.ou(mu, sigma, nz, zmin=-2.5 * sd, zmax=2.5 * sd)
    return par, zp


def feasible(x):
    """Parameter bounds of the calibration search."""
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = x
    return (g0 > 0 and g1 > 1.0 and g2 >= 0 and rho > R_FIX + 0.001 and 0.0 <= kappa <= 1.0
            and 0.02 <= delta <= 0.5 and 0.5 <= rho_z <= 0.99 and sig_eps > 0.005)


def standard_diag(res, floor):
    """How the underwriting standard shapes the solution: the mass on which it binds, issuance in full default,
    and the zero-drift locus by endowment node."""
    m = res.model
    B, z, q, d, S = m.B, m.z, res.q, res.d, res.S
    g = res.g / res.g.sum()
    iss = res.issuance
    # hard standard: binding where the price is below the level, issuance is zero, and d < 1
    binding = (q < floor - 1e-9) & (np.abs(iss) < 1e-12) & (d < 1 - 1e-9) if floor > 0 else np.zeros_like(q, bool)
    full_issuing = (d >= 1 - 1e-9) & (S > 1e-10)
    # in each z row, the first node where the drift turns from positive to non-positive, classified as on the
    # standard, in partial default, or without default
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
    par = m.par
    if floor > 0 and par.q_band > 0:                 # smooth standard: the cap L(q) on issuance is active
        w = np.clip((q - floor) / par.q_band, 0.0, 1.0 - 1e-12)
        cap = np.where(q < floor, 0.0, par.l_band * w / (1.0 - w))
        binding = (q < floor + par.q_band) & (iss >= cap - 1e-9) & (d < 1 - 1e-9)
    gB = g.sum(1)
    return dict(binding_mass=float(g[binding].sum()),
                mass_top2=float(gB[-2:].sum()),
                mass_band=float(g[(q >= floor) & (q < floor + par.q_band)].sum()) if floor > 0 else 0.0,
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
    return (f"standard binds on {D['binding_mass']:.4f} | mass on the last 2 debt nodes {D['mass_top2']:.1e} | full-default issuing nodes {D['full_issuing_nodes']} (max q {D['full_issuing_max_q']:.2f}, "
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
