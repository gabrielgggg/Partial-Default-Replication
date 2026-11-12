"""Calibrated diffusion model with saving.

The debt grid extends below zero (B <= 0: a risk-free asset at q^rf, no default), at the calibrated parameters
of data/ou_main.npz, with the solver variants pdmodel_sav.py and pdtimestep_sav.py. Writes
data/saving/saving_<tag>.npz, a checkpoint saving_<tag>_ckpt.npz, and prints state moments next to the
solution without saving.
Usage: python code/run_saving.py [Bmin] [dB] [maxit] [tag] [init.npz]
  defaults: Bmin -0.4, dB 0.005 (the paper's step), maxit 12000, tag main, warm start from data/ou_main.npz
"""
import os, sys, time
from paths import DATA
HERE = os.path.join(DATA, "saving")
os.makedirs(HERE, exist_ok=True)
import numpy as np
import pdmodel_sav, pdtimestep_sav                 # solver variants for grids below zero debt
from pdmodel_sav import Model, ZProcess, Params
from pdtimestep_sav import solve_timestep

a = np.load(os.path.join(DATA, "ou_main.npz"))
Bmin = float(sys.argv[1]) if len(sys.argv) > 1 else -0.4
dB = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
maxit = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
tag = sys.argv[4] if len(sys.argv) > 4 else "main"

# the paper's parameter mapping (as in calibrate2.unpack) and the same OU grid as the main solution
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = map(float, a["x"])
mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
par = Params(nu=2.0, rho=rho, r=float(np.log(1.04)), delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2,
             ztilde=zt, fixed_cost_form="level", S_cap=1.0, q_floor=float(a["qf"]))
z0 = a["z"]
zp = ZProcess.ou(mu, sigma, z0.size, zmin=float(z0[0]), zmax=float(z0[-1]))
assert np.allclose(zp.z, z0)
# debt grid on [Bmin, Bmax] with zero as a node
Bmax = float(a["B"][-1])
nB = int(round((Bmax - Bmin) / dB)) + 1
m = Model(par, zp, Bmin, Bmax, nB)
i0 = int(np.argmin(np.abs(m.B)))
assert abs(m.B[i0]) < 1e-9, "zero must be a grid node"
print(f"grid: {nB} debt nodes on [{Bmin}, {Bmax}] (step {dB}), {m.nz} endowment nodes, zero at node {i0}", flush=True)

# warm start: a saved solution or checkpoint (argument 5, a path) on its own grid, else the paper's solution on
# B >= 0; interpolated onto this grid and extended linearly below its lowest node (q = 1 there)
init = sys.argv[5] if len(sys.argv) > 5 else None
w = np.load(init) if init else a
B0, Vw, qw = w["B"], w["V"], w["q"]
print("warm start from", init or "data/ou_main.npz", f"(debt grid [{B0[0]}, {B0[-1]}])", flush=True)
V0 = np.empty((nB, m.nz)); q0 = np.ones((nB, m.nz))
inside = m.B >= B0[0] - 1e-12
for j in range(m.nz):
    V0[inside, j] = np.interp(m.B[inside], B0, Vw[:, j])
    q0[inside, j] = np.interp(m.B[inside], B0, qw[:, j])
    slope = (Vw[1, j] - Vw[0, j]) / (B0[1] - B0[0])
    V0[~inside, j] = Vw[0, j] + slope * (m.B[~inside] - B0[0])
q0[m.B < -1e-12, :] = 1.0

# time-step to the stationary point, with checkpoints
t0 = time.time()
res = solve_timestep(m, V0, q0, dt=0.015, maxit=maxit, tolV=1e-7, tolq=1e-6, verbose=True, print_every=500,
                     save_path=os.path.join(HERE, f"saving_{tag}_ckpt.npz"))
print(f"solve: {time.time() - t0:.0f}s, converged {res.converged}, resV {res.resV:.1e}, resq {res.resq:.1e}", flush=True)
np.savez(os.path.join(HERE, f"saving_{tag}.npz"), V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g,
         B=m.B, z=m.z, x=a["x"], qf=a["qf"], resV=res.resV, resq=res.resq, converged=res.converged)


def state_moments(B, z, q, d, g, coupon, r_delta):
    """Ergodic moments of debt, assets, default, and spreads (spreads only where B >= 0)."""
    g = g / g.sum(); y = np.exp(z)[None, :]; BB = B[:, None] * np.ones_like(g)
    spread = coupon / q - r_delta
    out = {
        "mass B<0": float(g[BB < -1e-12].sum()),
        "mass at B=0": float(g[np.abs(BB) <= 1e-12].sum()),
        "mass at bottom node": float(g[0, :].sum()),
        "mean B (net)": float((g * BB).sum()),
        "mean B (gross, max(B,0))": float((g * np.maximum(BB, 0)).sum()),
        "mean B/y (net)": float((g * BB / y).sum()),
        "mean B/y (gross)": float((g * np.maximum(BB, 0) / y).sum()),
        "mean assets | B<0": float((g * np.where(BB < 0, -BB, 0)).sum() / max(g[BB < 0].sum(), 1e-300)),
        "share in default (inst.)": float(g[d > 1e-9].sum()),
        "mean d | d>0": float((g * d)[d > 1e-9].sum() / max(g[d > 1e-9].sum(), 1e-300)),
    }
    w = g * (BB >= -1e-12)                       # spreads where a bond market exists (B >= 0)
    w = w / w.sum()
    ms = float((w * spread).sum())
    out["spread mean (B>=0)"] = ms
    out["spread s.d. (B>=0)"] = float(np.sqrt((w * (spread - ms) ** 2).sum()))
    return out


# compare with the solution without saving
coupon = par.lam + par.delta; rd = par.r + par.delta
base = state_moments(a["B"], a["z"], a["q"], a["d"], a["g"], coupon, rd)
new = state_moments(m.B, m.z, res.q, res.d, res.g, coupon, rd)
print(f"\n{'state moment':30s} {'paper (no saving)':>18s} {'with saving':>14s}")
for k in new:
    print(f"{k:30s} {base.get(k, float('nan')):18.4f} {new[k]:14.4f}")
