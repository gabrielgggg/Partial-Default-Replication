"""Solve the model without the underwriting standard on a wide debt grid.

Debt grid [0, Bmax] with step 0.005 (401 nodes at the default Bmax = 2), 75 productivity nodes, warm start from
data/floorcf/qf0.00.npz. Writes data/floorcf/qf0.00_wide2.{npz,md} with the moments, the loss, and the
zero-drift locus.
Usage: python code/nofloor_wide.py [Bmax=2.0] [steps=8000]
"""
import os
import sys
import time

import numpy as np

from pdutil import DATA, load_calibrate2, unpack, standard_diag, diag_line, locus_table
C = load_calibrate2(0.0)
from pdmodel import Model, ergodic_density
from pdtimestep import solve_timestep
from pdmoments import combined_moments, table_combined, loss

SRC = os.path.join(DATA, "floorcf", "qf0.00.npz")
OUT = os.path.join(DATA, "floorcf", "qf0.00_wide2")
Bmax = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
a = np.load(SRC)
x = a["x"].astype(float)
# wide grid without the floor, warm start from the no-floor solution on the main grid
par, zp = unpack(x, 0.0, 75)
nB = int(round(Bmax / 0.005)) + 1
m = Model(par, zp, 0.0, Bmax, nB)
V0, q0 = C.init_from_file(SRC, m)
t0 = time.time()
# time-step to the stationary point and save
res = solve_timestep(m, V0, q0, dt=0.015, maxit=steps, print_every=1000, verbose=True, tolV=1e-7, tolq=1e-8)
if res.g is None:
    res.g = ergodic_density(m, res.A)
np.savez(OUT + ".npz", V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z, x=x, qf=0.0,
         resV=res.resV, resq=res.resq)
# moments, loss, and diagnostics
M, info = combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
D = standard_diag(res, 0.0)
md = [f"{res.iters} steps, resV {res.resV:.1e} resq {res.resq:.1e}, {time.time() - t0:.0f}s", table_combined(M, info),
      f"loss {loss(M)}", diag_line(D), locus_table(D)]
open(OUT + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
