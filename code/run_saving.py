"""The calibrated model with saving.

The debt grid extends below zero: with B < 0 the sovereign holds a risk-free asset at the price q^rf and does
not default. Parameters, standard, endowment grid, and debt step are those of the main solution; the debt grid
is [-1, 1]. The solver is the one of the main solution with the changes of saving_solver.py. The time stepping
starts from the main solution, extended linearly in V below zero debt.

Reads data/ou_main.npz. Writes data/ou_saving.npz (V, q, S, c, d, g, B, z, x, qf, band, lband, residuals).
Usage: python code/run_saving.py
"""
import os
import time

import numpy as np

import saving_solver
from paths import DATA
from calutil import R_FIX

saving_solver.build()
from pdmodel_sav import Model, ZProcess, Params      # noqa: E402
from pdtimestep_sav import solve_timestep            # noqa: E402

BMIN, DB, MAXIT = -1.0, 0.005, 12000

# parameters and endowment grid of the main solution
a = np.load(os.path.join(DATA, "ou_main.npz"))
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = map(float, a["x"])
mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
par = Params(nu=2.0, rho=rho, r=R_FIX, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2,
             ztilde=zt, fixed_cost_form="level", S_cap=1.0, q_floor=float(a["qf"]),
             q_band=float(a["band"]), l_band=float(a["lband"]), closed_d=True)
z0 = a["z"]
zp = ZProcess.ou(mu, sigma, z0.size, zmin=float(z0[0]), zmax=float(z0[-1]))
assert np.allclose(zp.z, z0)
# debt grid on [BMIN, Bmax] with zero as a node
Bmax = float(a["B"][-1])
nB = int(round((Bmax - BMIN) / DB)) + 1
m = Model(par, zp, BMIN, Bmax, nB)
i0 = int(np.argmin(np.abs(m.B)))
assert abs(m.B[i0]) < 1e-9, "zero must be a grid node"
print(f"grid: {nB} debt nodes on [{BMIN}, {Bmax}] (step {DB}), {m.nz} endowment nodes, zero at node {i0}", flush=True)

# start: the main solution on B >= 0, V extended linearly below zero debt, q = 1 there
B0, Vw, qw = a["B"], a["V"], a["q"]
V0 = np.empty((nB, m.nz)); q0 = np.ones((nB, m.nz))
inside = m.B >= B0[0] - 1e-12
for j in range(m.nz):
    V0[inside, j] = np.interp(m.B[inside], B0, Vw[:, j])
    q0[inside, j] = np.interp(m.B[inside], B0, qw[:, j])
    slope = (Vw[1, j] - Vw[0, j]) / (B0[1] - B0[0])
    V0[~inside, j] = Vw[0, j] + slope * (m.B[~inside] - B0[0])
q0[m.B < -1e-12, :] = 1.0

t0 = time.time()
# time-step ramp and tie rule of the main solution
res = solve_timestep(m, V0, q0, maxit=MAXIT, tolV=1e-7, tolq=1e-6, verbose=True, print_every=500,
                     dt=0.015, dt_adapt=True, dt_grow=1.02, dt_max=0.5, sticky=1e-6, freeze_flips=3)
print(f"solve: {time.time() - t0:.0f}s, converged {res.converged}, resV {res.resV:.1e}, resq {res.resq:.1e}", flush=True)
np.savez(os.path.join(DATA, "ou_saving.npz"), V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g,
         B=m.B, z=m.z, x=a["x"], qf=a["qf"], resV=res.resV, resq=res.resq, converged=res.converged,
         band=a["band"], lband=a["lband"])
