"""Equilibria of the self-fulfilling example: the deterministic model at its own parameters.

The example has a small first default and an impatience wedge of about 200 basis points (rho = 0.06), on 341
debt nodes on [0, 1.7]. The branching march of pdmarch_branch marches every locally consistent zero-drift
candidate up and down. A candidate is usable if its branch reaches B = 0 without a violation of the upwind HJB
test at or below its steady state. The usable solutions include the no-default steady state and
partial-default steady states.

Reads no file. Writes data/wedge/rho0.060_illus.npz: the debt grid and, for each usable solution k, sol<k>_V,
_q, _S, _d, _c, and sol<k>_meta = (B*, q*, d*, number of violations, HJB residual, pricing residual).
Usage: python code/wedge_search.py
"""
import os
import time

import numpy as np

from paths import DATA
from pdmodel import Model, Params, ZProcess
from pdmarch_branch import solve_deterministic, to_result

RHO, NODES, BMAX = 0.06, 200, 1.7           # discount rate, nodes per unit of debt, top of the grid
os.makedirs(os.path.join(DATA, "wedge"), exist_ok=True)
par = Params(rho=RHO, g0=0.075, g1=1.5, g2=0.001, ztilde=-1.0, S_cap=1.0, q_floor=0.7)
nB = int(round(BMAX * NODES)) + 1
m = Model(par, ZProcess.none(0.0), 0.0, BMAX, nB)

t0 = time.time()
print(f"rho {RHO:.4f} (wedge {1e4 * (RHO - par.r):.0f} bp), {nB} nodes on [0,{BMAX}], q_floor {par.q_floor}", flush=True)
# march all candidates; keep the usable ones
good, sols = solve_deterministic(m, verbose=False)
print(f"{len(good)} usable of {len(sols)} marched candidates, {time.time() - t0:.0f}s", flush=True)
out = {}
for k, s in enumerate(good):
    res = to_result(m, s)
    kind = "no-default steady state" if (s["qstar"] > 0.999 and s["dstar"] < 1e-6) else "PARTIAL-DEFAULT steady state"
    print(f"  [{k}] B*={s['Bstar']:.4f} q*={s['qstar']:.4f} d*={s['dstar']:.4f} violations {len(s['viol'])} "
          f"first above: {s['Bhat']}  residuals {res.resV:.1e}/{res.resq:.1e}  min S {res.S.min():+.4f} max S {res.S.max():+.4f}  {kind}", flush=True)
    out[f"sol{k}_V"] = res.V[:, 0]; out[f"sol{k}_q"] = res.q[:, 0]; out[f"sol{k}_S"] = res.S[:, 0]
    out[f"sol{k}_d"] = res.d[:, 0]; out[f"sol{k}_c"] = res.c[:, 0]
    out[f"sol{k}_meta"] = np.array([s["Bstar"], s["qstar"], s["dstar"], len(s["viol"]), res.resV, res.resq])
np.savez(os.path.join(DATA, "wedge", "rho0.060_illus.npz"), B=m.B, rho=RHO, n_good=len(good), **out)
print("done", f"{time.time() - t0:.0f}s", flush=True)
