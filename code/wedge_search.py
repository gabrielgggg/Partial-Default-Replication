"""Search for self-fulfilling partial-default steady states in the deterministic model at a given discount rate.

Uses the calibrated parameters (data/ou_best.npz, z = 0, its standard qf), or with `illus` the illustrative
parameterization, with rho set to <rho> (the impatience wedge is rho - r). The branching march of pdmarch_branch
marches every locally consistent zero-drift candidate up and down. A candidate is a global equilibrium if its
branch reaches B = 0 without a violation of the upwind HJB test.
Writes data/wedge/rho<rho>[_illus][_n<nodes>].npz (all usable solutions) and .log.
Usage: python code/wedge_search.py <rho> [nodes=200] [Bmax=1.2] [illus]
"""
import os
import sys
import time

import numpy as np

from paths import DATA as PDATA
from pdmodel import Model, Params, ZProcess
from pdmarch_branch import solve_deterministic, to_result

DATA = os.path.join(PDATA, "wedge")
os.makedirs(DATA, exist_ok=True)
rho = float(sys.argv[1])
nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
Bmax = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
# calibrated parameters with rho replaced, or the illustrative parameterization
a = np.load(os.path.join(PDATA, "ou_best.npz"))
g0, g1, g2, zt, rho0, kappa, delta, rho_z, sig_eps = a["x"].astype(float)
lam = float(np.log(1.04))
par = Params(nu=2.0, rho=rho, r=lam, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2, ztilde=zt,
             fixed_cost_form="level", S_cap=1.0, q_floor=float(a["qf"]))
tag = f"rho{rho:.3f}"
if "illus" in sys.argv:        # the illustrative parameterization (small first default)
    par = Params(rho=rho, g0=0.075, g1=1.5, g2=0.001, ztilde=-1.0, S_cap=1.0, q_floor=0.7)
    tag += "_illus"
if nodes != 200:
    tag += f"_n{nodes}"
nB = int(round(Bmax * nodes)) + 1
m = Model(par, ZProcess.none(0.0), 0.0, Bmax, nB)
log = open(os.path.join(DATA, f"{tag}.log"), "w")


def say(*s_):
    """Print a line and append it to the log."""
    t = " ".join(str(v) for v in s_)
    print(t, flush=True); log.write(t + "\n"); log.flush()


t0 = time.time()
say(f"rho {rho:.4f} (wedge {1e4 * (rho - lam):.0f} bp), {nB} nodes on [0,{Bmax}], q_floor {par.q_floor}")
# march all candidates; keep the global equilibria
good, sols = solve_deterministic(m, verbose=False)
say(f"{len(good)} usable of {len(sols)} marched candidates, {time.time() - t0:.0f}s")
# classify and save each equilibrium
out = {}
for k, s in enumerate(good):
    res = to_result(m, s)
    kind = "no-default steady state" if (s["qstar"] > 0.999 and s["dstar"] < 1e-6) else "PARTIAL-DEFAULT steady state"
    say(f"  [{k}] B*={s['Bstar']:.4f} q*={s['qstar']:.4f} d*={s['dstar']:.4f} violations {len(s['viol'])} "
        f"first above: {s['Bhat']}  residuals {res.resV:.1e}/{res.resq:.1e}  min S {res.S.min():+.4f} max S {res.S.max():+.4f}  {kind}")
    out[f"sol{k}_V"] = res.V[:, 0]; out[f"sol{k}_q"] = res.q[:, 0]; out[f"sol{k}_S"] = res.S[:, 0]
    out[f"sol{k}_d"] = res.d[:, 0]; out[f"sol{k}_c"] = res.c[:, 0]
    out[f"sol{k}_meta"] = np.array([s["Bstar"], s["qstar"], s["dstar"], len(s["viol"]), res.resV, res.resq])
np.savez(os.path.join(DATA, f"{tag}.npz"), B=m.B, rho=rho, n_good=len(good), **out)
say("done", f"{time.time() - t0:.0f}s")
