"""Solve the calibrated point on the paper's main grid and report its moments.

Reads the selected parameters and floor from data/ou_best.npz, solves on 201 debt nodes on [0, 1] and 75
productivity nodes from a warm start interpolated from ou_best, and computes the moments, the loss, and the
robustness-table row. Writes data/ou_main.npz, data/main_moments.md, data/main_row.tex, and data/main75.log.
Usage: python code/main75.py [steps=9000]
"""
import os
import sys
import time

import numpy as np

from pdutil import DATA, NAMES, load_calibrate2, unpack, standard_diag, diag_line, locus_table

# selected parameters and floor
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
src_path = os.path.join(DATA, "ou_best.npz")
src = np.load(src_path)
x = src["x"].astype(float); QF = float(src["qf"])
C = load_calibrate2(QF)
from pdmodel import Model, ergodic_density
from pdtimestep import solve_timestep
from pdmoments import combined_moments, table_combined, loss
from pddiag import resolution_report

t0 = time.time()
log = open(os.path.join(DATA, "main75.log"), "w")


def say(*a_):
    """Print a line and append it to the log."""
    s = " ".join(str(v) for v in a_); print(s, flush=True); log.write(s + "\n"); log.flush()


# warm start from the selected solution interpolated onto the 75-node grid
par, zp = unpack(x, QF, 75)
m = Model(par, zp, 0.0, 1.0, 201)
V0, q0 = C.init_from_file(src_path, m)
say(f"main75: floor {QF}, from {os.path.basename(src_path)} ({src['z'].size} z nodes), {steps} steps")
# time-step to the stationary point
res = solve_timestep(m, V0, q0, dt=0.015, maxit=steps, print_every=1000, verbose=True, tolV=1e-7, tolq=1e-8)
if res.g is None:
    res.g = ergodic_density(m, res.A)
say(f"main75: {res.iters} steps, resV {res.resV:.1e} resq {res.resq:.1e}, {time.time() - t0:.0f}s")
# simulated and ergodic moments, the loss, and the policy diagnostics
rr = resolution_report(res, verbose=False)
M, info = combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
E = info["ergodic"]; L = loss(M); unres = rr.get("ergodic_mass_unresolved", np.nan)
D = standard_diag(res, QF)
# save the solution, the robustness-table row, and the moments report
np.savez(os.path.join(DATA, "ou_main.npz"), V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z, x=x,
         names=np.array(NAMES), qf=QF, resV=res.resV, resq=res.resq)
row = (f"main grid & {M['debt_y_mean']:.3f} & {M['debt_y_sd']:.3f} & {M['spread_sd']:.3f} & {M['pd_freq']:.3f} & "
       f"{M['pd_mean_cond']:.3f} & {L:.3f} & {E['mass_top']:.4f} & {unres:.3f} & {res.resV:.0e} / {res.resq:.0e} \\\\")
open(os.path.join(DATA, "main_row.tex"), "w", newline="\n").write(row + "\n")
md = [f"# main solution (nz = 75) at the selected point, floor {QF}: {res.iters} steps, resV {res.resV:.1e}, resq {res.resq:.1e}", "",
      "| " + " | ".join(NAMES) + " |", "|" + "---|" * len(NAMES), "| " + " | ".join(f"{v:.4f}" for v in x) + " |", "",
      table_combined(M, info), "", f"mass unresolved {unres:.4f}", "", diag_line(D), "", locus_table(D)]
open(os.path.join(DATA, "main_moments.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
say("\n".join(md))
say("done", f"{time.time() - t0:.0f}s")
