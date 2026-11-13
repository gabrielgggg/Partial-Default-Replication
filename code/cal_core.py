"""cal_core.py -- one evaluation of the calibration objective, and the bookkeeping of a search.

Seven parameters are calibrated (calutil.FREE_NAMES). The recovery factor and the maturity are fixed at the values of
Arellano, Mateos-Planas, and Rios-Rull (2023) (calutil.KAPPA_FIX, DELTA_FIX). The underwriting standard is not a
free parameter either: for every parameter vector it is the smallest level, on the menu LEVELS, at which the solution
is admissible. evaluate(task) finds that level and returns the loss there.

Admissible (RULES):
  'top'       the solve converges and the last two debt nodes carry no ergodic mass
  'top_nofd'  the same, and no ergodic mass sits on nodes where the sovereign issues in full default

The loss is the one of the paper: the eleven targets and the weights of amr_values.py, with output net of default
costs (pdmoments.NET_OUTPUT), on the 201 x 75 grid.

The solver is the implicit upwind time stepping of the paper (pdtimestep.solve_timestep) with the settings SOLVER:
the standard is smooth over a band of prices above its level (pdmodel.issue_cap), the time step grows along a ramp,
and a node where two upwind branches tie keeps its branch. With them the residuals fall to the tolerance, and a
solve stops when they do. Two settings make a step cheaper without changing its result: the default intensity is
taken from its closed form where that applies, and the two linear systems are solved by line sweeps.

Imported by cal_cma.py, cal_confirm.py, cal_report.py, solve_main.py, and robustness.py. Reads
data/calibration/start.npz; a search writes data/calibration/<tag>_evals.md and <tag>_best.{npz,md,json} (Book).
"""
import json
import os
import time

import numpy as np

from amr_values import AMR_DATA, AMR_MODEL, TARGETS
from calutil import CAL, NAMES, FREE, FREE_NAMES, full_x, feasible, save_json  # noqa: F401

SCALE9 = np.array([0.005, 0.25, 0.04, 0.03, 0.006, 0.05, 0.02, 0.025, 0.006])   # one search unit of each parameter
SCALE = SCALE9[FREE]
BMAX, NB = 1.0, 201
BAND, LBAND = 0.03, 0.1                    # smooth standard: band of prices above the level, and the scale of the cap
LEVELS = np.round(np.arange(0.30, 0.9001, 0.025), 3)     # menu of levels of the standard
RULES = ("top", "top_nofd")
RULE = "top_nofd"                          # the rule of the paper
TOP_NODES, TOP_TOL = 2, 1e-10              # 'no mass' on the last two debt nodes
FD_TOL = 1e-6                              # 'no mass' on nodes that issue in full default
MAX_SOLVES = 6                             # solves per evaluation, at most
SOLVER = dict(dt=0.015, dt_adapt=True, dt_grow=1.02, dt_max=0.5, sticky=1e-6, freeze_flips=3, sweeps=True)
SEARCH = dict(maxit=2000, tolV=1e-6, tolq=1e-6)          # tolerances of the search
STRICT = dict(maxit=5000, tolV=1e-7, tolq=1e-8)          # tolerances of the confirmation
BAD = 1e3                                  # value of a failed or infeasible evaluation
UNTARGETED = ["dc_y_mean_cond", "dc_y_sd_cond", "spread_mean", "corr_spread_y", "corr_spread_debt", "c_sd_rel"]
START = os.path.join(CAL, "start.npz")     # starting point of the search: parameters, standard, and solution


def on_grid(m, warm):
    """(V, q) of a saved solution on the grid of model m (same debt grid; the endowment grid may differ)."""
    Vw, qw, zw = warm
    if Vw.shape == (m.nB, m.nz) and np.allclose(zw, m.z):
        return Vw.copy(), qw.copy()
    f = lambda F: np.stack([np.interp(m.z, zw, F[i]) for i in range(m.nB)])
    return f(Vw), np.clip(f(qw), 1e-6, 1.0)


def solve_level(x, level, nz, V0, q0, rule, strict=False):
    """Solve at one level of the standard. Returns (res, D, ok, why): the solution, its diagnostics, whether it is
    admissible under `rule`, and the reason if not."""
    from calutil import unpack, standard_diag
    from pdgrid import warm_start
    from pdmodel import Model, ergodic_density
    from pdtimestep import solve_timestep
    par, zp = unpack(x, level, nz, BAND, LBAND)
    m = Model(par, zp, 0.0, BMAX, NB)
    if V0 is None:
        a = np.load(START)
        V0, q0 = warm_start(a["B"], a["z"], a["V"], a["q"], m.B, m.z)
    elif V0.shape != (m.nB, m.nz):
        raise ValueError("warm start on another grid")
    seen = dict(top=0, fd=0, why="")

    def stop(model, S, d, A, it):
        """Give up on a level that is plainly inadmissible: mass at the top, or on full-default issuing nodes, at
        three checks in a row."""
        if it < 300:
            return False
        g = ergodic_density(model, A)
        seen["top"] = seen["top"] + 1 if g[-TOP_NODES:].sum() > 1e-3 else 0
        seen["fd"] = seen["fd"] + 1 if rule == "top_nofd" and g[(d >= 1 - 1e-9) & (S > 1e-10)].sum() > 1e-3 else 0
        if seen["top"] >= 3:
            seen["why"] = "mass at the top of the debt grid (stopped early)"
        elif seen["fd"] >= 3:
            seen["why"] = "issuance in full default (stopped early)"
        return bool(seen["why"])

    res = solve_timestep(m, V0, q0, print_every=10 ** 9, verbose=False, stop_fn=stop, **SOLVER, **(STRICT if strict else SEARCH))
    if res.g is None:
        res.g = ergodic_density(m, res.A)
    D = standard_diag(res, level)
    D = {k: v for k, v in D.items() if k != "locus"}
    g = res.g / res.g.sum()
    D.update(n_frozen=int(res.n_frozen), frozen_mass=float(g[res.frozen].sum()), iters=int(res.iters),
             direct_solves=int(res.direct_solves), sweep_passes=int(res.passes),
             resV=float(res.resV), resq=float(res.resq), converged=bool(res.converged))
    if seen["why"]:
        return res, D, False, seen["why"]
    if not res.converged:
        return res, D, False, f"not converged (residuals {res.resV:.0e}, {res.resq:.0e})"
    if D["mass_top2"] > TOP_TOL:
        return res, D, False, f"mass {D['mass_top2']:.1e} on the last two debt nodes"
    if rule == "top_nofd" and D["full_issuing_mass"] > FD_TOL:
        return res, D, False, f"mass {D['full_issuing_mass']:.1e} issuing in full default"
    return res, D, True, ""


def evaluate(task):
    """Find the smallest admissible standard at a parameter vector, and score the solution there.

    task: x (the nine parameters), rule, nz, paths, years, strict (optional), level (where to start on the menu),
    warm (None, or the tuple (V, q, z) of a solution at a nearby point and at `level`).
    The search over levels starts at `level`: down while admissible, or up until admissible. Each solve starts from
    the one before. At most MAX_SOLVES solves."""
    t0 = time.time()
    x = np.asarray(task["x"], float)
    r = dict(x=x, ok=False, loss=BAD, err="", trail=[])
    try:
        if not feasible(x):
            raise ValueError("parameters out of bounds")
        from pdmoments import combined_moments, loss, table_combined
        from calutil import unpack, diag_line
        from pdmodel import Model
        rule, nz, strict = task["rule"], int(task["nz"]), bool(task.get("strict", False))
        i = int(np.argmin(np.abs(LEVELS - float(task.get("level", 0.7)))))
        V0 = q0 = None
        if task.get("warm") is not None:
            par, zp = unpack(x, LEVELS[i], nz, BAND, LBAND)
            V0, q0 = on_grid(Model(par, zp, 0.0, BMAX, NB), task["warm"])
        best, direction = None, 0
        for _ in range(MAX_SOLVES):
            res, D, ok, why = solve_level(x, float(LEVELS[i]), nz, V0, q0, rule, strict)
            r["trail"].append(dict(level=float(LEVELS[i]), ok=ok, why=why, steps=D["iters"]))
            if ok:
                best = (res, D, float(LEVELS[i]))
                if direction > 0 or i == 0:
                    break
                direction, V0, q0 = -1, res.V, res.q
                i -= 1
            else:
                if direction < 0 or i == len(LEVELS) - 1:
                    break
                direction = +1
                if not getattr(res, "stopped", False) and np.all(np.isfinite(res.V)):
                    V0, q0 = res.V, res.q
                i += 1
        if best is None:
            raise ValueError("no admissible standard: " + "; ".join(f"{t['level']:.3f} {t['why']}" for t in r["trail"]))
        res, D, level = best
        M, info = combined_moments(res, n_paths=int(task["paths"]), T_years=int(task["years"]), steps_per_year=100, seed=1)
        L = float(loss(M))
        if not np.isfinite(L):
            raise ValueError("loss is not finite")
        m = res.model
        r.update(ok=True, loss=L, M={k: float(v) for k, v in M.items()}, D=D, level=level, rule=rule, nz=nz,
                 resV=D["resV"], resq=D["resq"], table=table_combined(M, info), diag=diag_line(D),
                 sol=dict(V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z))
    except Exception as e:                       # a failed solve is a bad point, not the end of the search
        r["err"] = repr(e)
    r["secs"] = time.time() - t0
    return r


def closer_than_amr(M, keys):
    """How many of the moments `keys` are at least as close to the data as the model column of AMR."""
    return sum(abs(M[k] - AMR_DATA[k]) <= abs(AMR_MODEL[k] - AMR_DATA[k]) + 1e-12 for k in keys)


def trail_text(trail):
    """The levels tried in an evaluation, with the outcome and the number of steps of each solve."""
    return ", ".join(f"{t['level']:.3f} {'ok' if t['ok'] else 'no'} ({t['steps']})" for t in trail)


HEADER = ("| n | gen | " + " | ".join(FREE_NAMES) + " | standard | loss | residuals | " + " | ".join(TARGETS)
          + " | spread mean | standard binds | full default issuing | max B with mass | tie nodes (mass) | levels tried (steps) | time |\n"
          + "|---|---|" + "---|" * (len(FREE_NAMES) + 3 + len(TARGETS) + 7))


class Book:
    """Log and best point of one search. Files in data/calibration/: <tag>_evals.md (one line per evaluation) and
    <tag>_best.{npz,md,json}."""

    def __init__(self, tag, meta, fresh=False):
        self.tag = tag
        self.log = os.path.join(CAL, f"{tag}_evals.md")
        self.n = 0
        self.best = (np.inf, None)                 # (loss, evaluation number)
        self.x = None                              # the nine parameters of the best point
        self.warm, self.level = None, None         # its solution (V, q, z) and its standard
        f = os.path.join(CAL, f"{tag}_best.npz")
        if not fresh and os.path.exists(f):
            a = np.load(f)
            j = json.load(open(f[:-4] + ".json", encoding="utf-8"))
            self.best = (float(a["loss"]), int(j["n"]))
            self.x, self.warm, self.level = a["x"].astype(float), (a["V"], a["q"], a["z"]), float(a["qf"])
        if not fresh and os.path.exists(self.log):           # continue the numbering of the run that was stopped
            self.n = sum(1 for s in open(self.log, encoding="utf-8") if s[:2] == "| " and s[2:3].isdigit())
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(f"\n# {tag}: {meta}\n\ntargets: {json.dumps({k: AMR_DATA[k] for k in TARGETS})}\n\n" + HEADER + "\n")

    def record(self, r, gen=""):
        """Log one evaluation, update the best point, and return the value of the objective."""
        self.n += 1
        x = r["x"]
        head = f"| {self.n} | {gen} | " + " | ".join(f"{v:.4f}" for v in x[FREE])
        if not r["ok"]:
            line = head + f" | failed: {r['err']} | {r['secs']:.0f}s |"
        else:
            M, D = r["M"], r["D"]
            line = (head + f" | {r['level']:.3f} | {r['loss']:.4f} | {r['resV']:.0e}/{r['resq']:.0e} | "
                    + " | ".join(f"{M[k]:.3f}" if abs(M[k]) >= 0.1 else f"{M[k]:.4f}" for k in TARGETS)
                    + f" | {M['spread_mean']:.4f} | {D['binding_mass']:.4f} | {D['full_issuing_mass']:.1e} | {D['max_B_with_mass']:.3f} | "
                    + f"{D['n_frozen']} ({D['frozen_mass']:.1e}) | {trail_text(r['trail'])} | {r['secs']:.0f}s |")
            if r["loss"] < self.best[0]:
                self.best = (r["loss"], self.n)
                self.x, self.level = x.copy(), r["level"]
                self.warm = (r["sol"]["V"], r["sol"]["q"], r["sol"]["z"])
                self._save(r)
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return r["loss"] if r["ok"] else BAD

    def _save(self, r):
        """Write the best point: its solution, its report, and its record."""
        base = os.path.join(CAL, f"{self.tag}_best")
        x, M = r["x"], r["M"]
        np.savez(base + ".npz", **r["sol"], x=x, names=np.array(NAMES), qf=r["level"], band=BAND, lband=LBAND,
                 resV=r["resV"], resq=r["resq"], loss=r["loss"])
        with open(base + ".md", "w", encoding="utf-8") as fh:
            fh.write(f"# {self.tag}: best point after {self.n} evaluations, loss {r['loss']:.4f}, standard {r['level']:.3f} "
                     f"(rule '{r['rule']}')\n\n| " + " | ".join(NAMES) + " |\n|" + "---|" * len(NAMES) + "\n| "
                     + " | ".join(f"{v:.4f}" for v in x) + " |\n\n" + r["table"] + "\n\n" + r["diag"]
                     + f"\n\nlevels tried (steps): {trail_text(r['trail'])}\n")
        save_json(base + ".json", dict(tag=self.tag, n=self.n, loss=r["loss"], level=r["level"], rule=r["rule"],
                                       x=list(map(float, x)), M=M, nz=r["nz"], resV=r["resV"], resq=r["resq"],
                                       diag=r["D"], trail=r["trail"]))
