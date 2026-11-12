"""Re-solve the calibrated point with the underwriting standard at another level, or without it (floor 0).

Uses the parameters of data/ou_main.npz on the main grid, warm-started from the main solution. Reports the
moments, the diagnostics of the standard, and the zero-drift locus. Writes data/floorcf/qf<floor>.{npz,md,log}.
Usage: python code/floorscan.py <floor> [steps=8000]
"""
import os
import sys
import time

import numpy as np

from pdutil import DATA, NAMES, load_calibrate2, unpack, standard_diag, diag_line, locus_table

FLOOR = float(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
C = load_calibrate2(FLOOR)
from pdmodel import Model, ergodic_density
from pdtimestep import solve_timestep
from pdmoments import combined_moments, table_combined, loss
from pddiag import resolution_report

OUT = os.path.join(DATA, "floorcf"); os.makedirs(OUT, exist_ok=True)
tag = f"qf{FLOOR:.2f}"
MAIN = os.path.join(DATA, "ou_main.npz")
a = np.load(MAIN)
x = a["x"].astype(float)
t0 = time.time()
log = open(os.path.join(OUT, f"{tag}.log"), "w")


def say(*a_):
    """Print a line and append it to the log."""
    s = " ".join(str(v) for v in a_); print(s, flush=True); log.write(s + "\n"); log.flush()


# main grid with this floor, warm-started from the main solution
par, zp = unpack(x, FLOOR, a["z"].size)
m = Model(par, zp, 0.0, float(a["B"][-1]), a["B"].size)
assert np.allclose(m.B, a["B"]) and np.allclose(m.z, a["z"])
say(f"{tag}: floor {FLOOR}; nB {m.nB} nz {m.nz}; {STEPS} steps from the main solution (floor {float(a['qf'])})")
res = solve_timestep(m, a["V"].copy(), a["q"].copy(), dt=0.015, maxit=STEPS, print_every=1000, verbose=True, tolV=1e-7, tolq=1e-8)
if res.g is None:
    res.g = ergodic_density(m, res.A)
say(f"{tag}: {res.iters} steps, resV {res.resV:.1e} resq {res.resq:.1e}, {time.time() - t0:.0f}s")
# moments, loss, and diagnostics of the standard
rr = resolution_report(res, verbose=False)
M, info = combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
E = info["ergodic"]; L = loss(M); unres = rr.get("ergodic_mass_unresolved", np.nan)
D = standard_diag(res, FLOOR)
# save the solution and build the report
f_sol = os.path.join(OUT, f"{tag}.npz")
np.savez(f_sol, V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z, x=x, names=np.array(NAMES),
         qf=par.q_floor, resV=res.resV, resq=res.resq, iters=res.iters)
lines = [f"# {tag}: underwriting standard at {FLOOR}\n",
         f"{res.iters} steps, resV {res.resV:.1e}, resq {res.resq:.1e}, {time.time() - t0:.0f}s; loss {L:.4f}; mass top {E['mass_top']:.4f}; unresolved {unres:.4f}\n",
         table_combined(M, info), "", diag_line(D), "",
         f"E[B] {D['EB']:.4f}; P(d>0) {D['pdef']:.4f}; mass with q<0.7 {float((res.g / res.g.sum())[res.q < 0.7].sum()):.4f}; atom at 0 {D['atom0']:.4f}", "",
         "## zero-drift locus", "", locus_table(D)]
say("\n".join(lines[3:]))
open(os.path.join(OUT, f"{tag}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
say("done", f"{time.time() - t0:.0f}s")
