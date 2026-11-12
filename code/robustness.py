"""Re-solve the main solution (data/ou_main.npz) on other grids for the robustness table.

Cases: fine (step 0.0025, 401 nodes), wide (barrier 1.5, 301 nodes), nonuni (non-uniform grid built by
pdgrid.build_grid from the bond price of the main solution), and z30 and z35 (endowment range of 3 and 3.5
stationary standard deviations instead of 2.5, at the node spacing of the main grid: 89 and 105 nodes). Each
case reports the moments, the loss, and the table row. Writes data/rob75/<case>.{npz,md,log} and
data/rob75/<case>_row.tex.
Usage: python code/robustness.py <fine|wide|nonuni|z30|z35> [steps=6000]   (the stored z30 and z35 use 9000 steps)
"""
import os
import sys
import time

import numpy as np

from pdutil import PKG, DATA, load_calibrate2, unpack, standard_diag, diag_line

case = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
MAIN = os.path.join(DATA, "ou_main.npz")
a = np.load(MAIN)
x = a["x"].astype(float); QF = float(a["qf"])
C = load_calibrate2(QF)
from pdmodel import Model, ZProcess, ergodic_density
from pdtimestep import solve_timestep
from pdmoments import combined_moments, table_combined, loss
from pddiag import resolution_report

OUT = os.path.join(DATA, "rob75"); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
log = open(os.path.join(OUT, f"{case}.log"), "w")


def say(*a_):
    """Print a line and append it to the log."""
    s = " ".join(str(v) for v in a_); print(s, flush=True); log.write(s + "\n"); log.flush()


# build the case's debt grid and warm-start from the main solution
par, zp = unpack(x, QF, 75)
if case == "fine":
    label = r"debt step halved"; m = Model(par, zp, 0.0, 1.0, 401); V0, q0 = C.init_from_file(MAIN, m)
elif case == "wide":
    label = r"barrier at $B=1.5$"; m = Model(par, zp, 0.0, 1.5, 301); V0, q0 = C.init_from_file(MAIN, m)
elif case == "nonuni":
    from pdgrid import build_grid, warm_start
    nodes = build_grid(a["B"], a["q"].min(axis=1), 1.0, h0=0.005, hmin=2.5e-4)
    m = Model(par, zp, 0.0, 1.0, 0, B=nodes); label = r"non-uniform debt grid"
    V0, q0 = warm_start(a["B"], a["z"], a["V"], a["q"], m.B, m.z)
elif case in ("z30", "z35"):
    # wider endowment range: more nodes below the fixed-cost threshold ln y*
    width, nz = {"z30": (3.0, 89), "z35": (3.5, 105)}[case]
    sd = zp.sigma / np.sqrt(2 * zp.mu)
    zpw = ZProcess.ou(zp.mu, zp.sigma, nz, zmin=-width * sd, zmax=width * sd)
    label = rf"endowment range $\pm{width:g}$ s.d."; m = Model(par, zpw, 0.0, 1.0, 201); V0, q0 = C.init_from_file(MAIN, m)
else:
    raise SystemExit("case: fine | wide | nonuni | z30 | z35")
say(f"{case}: {label}; nB {m.nB} nz {m.nz}; {steps} steps from the main solution; floor {QF}")
# time-step to the stationary point
res = solve_timestep(m, V0, q0, dt=0.015, maxit=steps, print_every=1000, verbose=True, tolV=1e-7, tolq=1e-8)
if res.g is None:
    res.g = ergodic_density(m, res.A)
say(f"{case}: {res.iters} steps, resV {res.resV:.1e} resq {res.resq:.1e}, {time.time() - t0:.0f}s")
# moments, loss, diagnostics, and the table row
rr = resolution_report(res, verbose=False)
M, info = combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
E = info["ergodic"]; L = loss(M); unres = rr.get("ergodic_mass_unresolved", np.nan)
D = standard_diag(res, QF)
row = (f"{label} & {M['debt_y_mean']:.3f} & {M['debt_y_sd']:.3f} & {M['spread_sd']:.3f} & {M['pd_freq']:.3f} & "
       f"{M['pd_mean_cond']:.3f} & {L:.3f} & {E['mass_top']:.4f} & {unres:.3f} & {res.resV:.0e} / {res.resq:.0e} \\\\")
say(row); say(diag_line(D))
# save the solution, the row, and the report
np.savez(os.path.join(OUT, f"{case}.npz"), V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z, x=x,
         qf=QF, resV=res.resV, resq=res.resq, iters=res.iters)
open(os.path.join(OUT, f"{case}_row.tex"), "w", newline="\n").write(row + "\n")
open(os.path.join(OUT, f"{case}.md"), "w", encoding="utf-8").write(
    f"# {case}: {label}\n\n{res.iters} steps, resV {res.resV:.1e}, resq {res.resq:.1e}, {time.time() - t0:.0f}s\n\n"
    + table_combined(M, info) + f"\n\nmass unresolved {unres:.4f}\n\n{diag_line(D)}\n")
say("done", f"{time.time() - t0:.0f}s")
