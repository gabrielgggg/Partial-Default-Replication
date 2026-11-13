"""robustness.py -- the reported calibration (data/ou_main.npz) on other grids, and without the underwriting standard.

Cases:
  fine          debt step halved (401 nodes), borrowing limit 1.0
  wide          debt step 0.005, borrowing limit 1.5
  nz51, nz101   the main debt grid with 51 or 101 endowment nodes
  z30, z35      endowment range of 3 or 3.5 stationary standard deviations instead of 2.5, at the node spacing of
                the main grid (89 or 105 nodes)
  nonuni        non-uniform debt grid built by pdgrid.build_grid from the bond price of the main solution
  nostd, nostd2 no standard, with the borrowing limit at 1 and at 2
The standard stays at the level and band of the main solution, except in nostd and nostd2. Each solve starts from the
main solution interpolated onto the new grid, with the solver settings and tolerances of the confirmation step of the
calibration (cal_core.SOLVER, STRICT) and at most `maxit` steps. Without a standard the iteration does not settle,
and those two cases take `maxit` steps of the plain scheme.

Reads data/ou_main.npz. Writes data/robustness/<case>.npz (the solution and its residuals node by node),
<case>_row.tex (the row of the robustness table), <case>.md (moments and diagnostics), and <case>.log.

Usage: python robustness.py <case> [maxit=8000]   (the stored nonuni, z30, and z35 use maxit = 6000)
"""
import os
import sys
import time

import numpy as np

from cal_core import SOLVER, STRICT
from calutil import unpack, diag_line
from paths import DATA
from pdgrid import build_grid, warm_start
from pdmodel import Model, ZProcess, ergodic_density
from pdmoments import table_combined
from pdtimestep import solve_timestep
from solve_main import MAIN, summarize

OUT = os.path.join(DATA, "robustness")
os.makedirs(OUT, exist_ok=True)
case = sys.argv[1]
maxit = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
t0 = time.time()
log = open(os.path.join(OUT, f"{case}.log"), "w")


def say(*a_):
    """Print a line and append it to the log."""
    s = " ".join(str(v) for v in a_)
    print(s, flush=True); log.write(s + "\n"); log.flush()


a = np.load(MAIN)
x = a["x"].astype(float)
qf, band, lband = float(a["qf"]), float(a["band"]), float(a["lband"])
par, zp = unpack(x, qf, a["z"].size, band, lband)


def zproc(nz):
    """The endowment process on nz nodes over the range of the main grid."""
    return ZProcess.ou(zp.mu, zp.sigma, nz, zmin=zp.z[0], zmax=zp.z[-1])


if case == "fine":
    label, m = r"debt step halved", Model(par, zproc(a["z"].size), 0.0, 1.0, 401)
elif case == "wide":
    label, m = r"limit at $\bar B=1.5$", Model(par, zproc(a["z"].size), 0.0, 1.5, 301)
elif case in ("nz51", "nz101"):
    nz = int(case[2:])
    label, m = rf"$n_z={nz}$", Model(par, zproc(nz), 0.0, 1.0, 201)
elif case in ("z30", "z35"):            # wider endowment range, same step: 89 or 105 nodes
    width, nz = {"z30": (3.0, 89), "z35": (3.5, 105)}[case]
    sd = zp.sigma / np.sqrt(2 * zp.mu)
    label = rf"endowment range $\pm{width:g}$ s.d."
    m = Model(par, ZProcess.ou(zp.mu, zp.sigma, nz, zmin=-width * sd, zmax=width * sd), 0.0, 1.0, 201)
elif case == "nonuni":                  # non-uniform debt grid from the resolution rule
    nodes = build_grid(a["B"], a["q"].min(axis=1), 1.0, h0=0.005, hmin=2.5e-4)
    label, m = r"non-uniform debt grid", Model(par, zproc(a["z"].size), 0.0, 1.0, 0, B=nodes)
elif case in ("nostd", "nostd2"):       # no standard, on [0, 1] and on [0, 2]
    qf = 0.0
    par, _ = unpack(x, 0.0, a["z"].size, 0.0, lband)
    Bmax = 1.0 if case == "nostd" else 2.0
    label, m = rf"no standard, $B\le{Bmax:g}$", Model(par, zproc(a["z"].size), 0.0, Bmax, int(round(Bmax / 0.005)) + 1)
else:
    raise SystemExit("case: fine | wide | nz51 | nz101 | z30 | z35 | nonuni | nostd | nostd2")

say(f"{case}: {label}; nB {m.nB} nz {m.nz}; standard {qf} band {band}; at most {maxit} steps from the main solution")
V0, q0 = warm_start(a["B"], a["z"], a["V"], a["q"], m.B, m.z)
if case.startswith("nostd"):
    # Without a standard the iteration does not settle, and the rule that holds the branch of alternating nodes
    # would freeze a large part of the grid. These runs use the plain scheme: a fixed small time step, no held
    # branches, a fixed number of steps.
    res = solve_timestep(m, V0, q0, print_every=500, verbose=True, dt=0.015, sweeps=True, **dict(STRICT, maxit=maxit))
else:
    res = solve_timestep(m, V0, q0, print_every=200, verbose=True, **SOLVER, **dict(STRICT, maxit=maxit))
if res.g is None:
    res.g = ergodic_density(m, res.A)
say(f"{case}: {res.iters} steps, converged {res.converged}, resV {res.resV:.1e} resq {res.resq:.1e}, {time.time() - t0:.0f}s")
s = summarize(res, qf, label, say)
np.savez(os.path.join(OUT, f"{case}.npz"), V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z, x=x,
         names=a["names"], qf=qf, band=band, lband=lband, resV=res.resV, resq=res.resq, iters=res.iters, loss=s["L"],
         resV_mass=s["resV_mass"], resq_mass=s["resq_mass"], rV=s["rV"], rq=s["rq"])
open(os.path.join(OUT, f"{case}_row.tex"), "w", newline="\n").write(s["row"] + "\n")
open(os.path.join(OUT, f"{case}.md"), "w", encoding="utf-8").write(
    f"# {case}: {label}\n\n{res.iters} steps, resV {res.resV:.1e}, resq {res.resq:.1e}, {time.time() - t0:.0f}s\n\n"
    + table_combined(s["M"], s["info"]) + f"\n\nmass unresolved {s['unres']:.4f}\n\n" + diag_line(s["D"]) + "\n")
say("done", f"{time.time() - t0:.0f}s")
