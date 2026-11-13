"""solve_main.py -- the main solution at the reported calibration, and its report.

data/ou_main.npz holds the reported calibration (the nine parameters x, the level qf of the underwriting standard,
its band and lband) and the main solution on the grid of 201 debt nodes and 75 endowment nodes (V, q, the policies
S, c, d, and the ergodic density g).

Without options the script first solves the model again: at the stored parameters, standard, and band, with the
solver settings and the strict tolerances of the confirmation step of the calibration (cal_core.SOLVER, STRICT),
starting from the solution of data/calibration/start.npz as the confirmation does (cal_core.on_grid, solve_level).
It overwrites data/ou_main.npz. A re-solve reproduces the stored solution up to the solver tolerance and the branch
held at tie nodes: where two upwind branches tie, the solver keeps the branch the node had, and which one that is
depends on the path of the iteration. If the solve does not end at an admissible solution, the script stops and
leaves every file as it was.

It then evaluates the stored solution with its saved policies and writes
  data/main_moments.md   the moments next to the targets, and the loss
  data/main_row.tex      the row of the main grid in the robustness table
  data/main_check.md     diagnostics of the standard and the zero-drift locus by endowment node
The saved policies are used, and not policies recomputed from (V, q), because the two differ at the tie nodes.

Usage: python solve_main.py               solve again, then evaluate
       python solve_main.py --evaluate    evaluate the stored solution only
"""
import argparse
import os
import time

import numpy as np

from cal_core import BMAX, NB, BAND, LBAND, RULE, START, on_grid, solve_level
from calutil import NAMES, unpack, standard_diag, diag_line, locus_table
from paths import DATA
from pddiag import resolution_report
from pdmodel import Model, Result, u_fn
from pdmoments import combined_moments, table_combined, loss

MAIN = os.path.join(DATA, "ou_main.npz")


def summarize(res, qf, label, say):
    """Residuals node by node, diagnostics of the standard, moments, and loss of a solution, and its row of the
    robustness table. Reports through say() and returns the pieces in a dict."""
    m = res.model
    par = m.par
    # residuals node by node: on nodes with ergodic mass, and where they exceed 1e-6
    u = np.vectorize(lambda cc: u_fn(cc, par.nu))(res.c)
    rV = np.abs(par.rho * res.V.ravel() - u.ravel() - res.A @ res.V.ravel()).reshape(res.V.shape)
    rq = np.abs(par.xi(res.d.ravel()) * res.q.ravel() - (1 - res.d.ravel()) * (par.delta + par.lam) - res.A @ res.q.ravel()).reshape(res.V.shape)
    g = res.g / res.g.sum()
    mass = g > 1e-10
    resV_mass, resq_mass = float(rV[mass].max()), float(rq[mass].max())
    big = (rq > 1e-6).any(1)
    say(f"residuals on the {int(mass.sum())} nodes with mass above 1e-10: {resV_mass:.1e} / {resq_mass:.1e}; largest debt with such mass "
        f"{m.B[np.where(mass.any(1))[0].max()]:.3f}; nodes with pricing residual above 1e-6: {int((rq > 1e-6).sum())}"
        + (f", all at debt of {m.B[np.where(big)[0].min()]:.3f} or more" if big.any() else ""))
    D = standard_diag(res, qf)
    rr = resolution_report(res, verbose=False)
    M, info = combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
    E = info["ergodic"]
    L = loss(M)
    unres = rr.get("ergodic_mass_unresolved", np.nan)
    row = (f"{label} & {M['debt_y_mean']:.3f} & {M['debt_y_sd']:.3f} & {M['spread_sd']:.3f} & {M['pd_freq']:.3f} & "
           f"{M['pd_mean_cond']:.3f} & {L:.3f} & {E['mass_top']:.4f} & {unres:.3f} & {resV_mass:.0e} / {resq_mass:.0e} \\\\")
    say(f"loss {L:.4f}  mass top {E['mass_top']:.4f}  unresolved {unres:.4f}")
    say(f"standard binds on {D['binding_mass']:.4f}; mass on the last two nodes {D['mass_top2']:.1e}; issuing in full default "
        f"{D['full_issuing_mass']:.1e}; max B with mass {D['max_B_with_mass']:.3f}"
        + (f"; tie nodes {int(res.n_frozen)} (mass {float(g[res.frozen].sum()):.1e})" if hasattr(res, "frozen") else ""))
    say(row)
    return dict(M=M, info=info, L=L, D=D, unres=unres, row=row, rV=rV, rq=rq, resV_mass=resV_mass, resq_mass=resq_mass)


def resolve(a):
    """Strict solve at the parameters and the standard of the stored solution `a`, from the solution of the start
    file. Returns the contents of the new ou_main.npz, without the loss."""
    x, qf = a["x"].astype(float), float(a["qf"])
    if (float(a["band"]), float(a["lband"])) != (BAND, LBAND):
        raise SystemExit("the stored band of the standard is not the one of cal_core")
    nz = a["z"].size
    s = np.load(START)
    par, zp = unpack(x, qf, nz, BAND, LBAND)
    V0, q0 = on_grid(Model(par, zp, 0.0, BMAX, NB), (s["V"], s["q"], s["z"]))
    t0 = time.time()
    res, D, ok, why = solve_level(x, qf, nz, V0, q0, RULE, strict=True)
    print(f"solve at the standard {qf}: {D['iters']} steps, residuals {D['resV']:.1e} / {D['resq']:.1e}, tie nodes "
          f"{D['n_frozen']} (mass {D['frozen_mass']:.1e}), {time.time() - t0:.0f}s", flush=True)
    if not ok:
        raise SystemExit(f"the solve is not admissible: {why}; data/ou_main.npz is unchanged")
    m = res.model
    return dict(V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=m.B, z=m.z, x=x, names=np.array(NAMES), qf=qf,
                band=BAND, lband=LBAND, resV=D["resV"], resq=D["resq"])


def stored_result(a):
    """Result of the stored solution `a` with its saved policies (the solver holds the branch of tie nodes)."""
    x = a["x"].astype(float)
    par, zp = unpack(x, float(a["qf"]), a["z"].size, float(a["band"]), float(a["lband"]))
    m = Model(par, zp, 0.0, BMAX, NB)
    S, _c, d, _u, _H, which, Sf, df = m.policies(a["V"], a["q"], 28)
    print(f"policies recomputed from (V, q) differ from the saved ones by at most {np.abs(S - a['S']).max():.1e} (S), "
          f"{np.abs(d - a['d']).max():.1e} (d), at {int((np.abs(S - a['S']) > 1e-8).sum())} nodes", flush=True)
    return Result(m, a["V"], a["q"], a["S"], a["c"], a["d"], which, Sf, df, m.assemble(a["S"]), a["g"], True, 0,
                  float(a["resV"]), float(a["resq"]))


def check_text(D, qf, band, nB, nz, resV, resq):
    """Text of main_check.md: diagnostics of the standard and the zero-drift locus by endowment node."""
    return (f"# zero-drift locus and ergodic set of the main solution: standard {qf}, band {band}, grid {nB} x {nz}, "
            f"residuals {resV:.1e} / {resq:.1e}\n\n" + diag_line(D) + "\n\n"
            + f"E[B] {D['EB']:.4f}; P(d>0) {D['pdef']:.4f}; atom at 0 {D['atom0']:.4f}; mass with q below the standard "
            + f"{D['mass_q_below_floor']:.4f}; mass in the band {D['mass_band']:.4f}\n\n## zero-drift locus\n\n"
            + locus_table(D, every=1) + "\n")


def main():
    """Solve again unless --evaluate, then evaluate the stored solution and write its report."""
    ap = argparse.ArgumentParser(description="The main solution at the reported calibration, and its report.")
    ap.add_argument("--evaluate", action="store_true", help="evaluate the stored solution only, without solving again")
    args = ap.parse_args()
    a = dict(np.load(MAIN))
    if not args.evaluate:
        a = resolve(a)
    t0 = time.time()
    qf, band = float(a["qf"]), float(a["band"])
    res = stored_result(a)
    m = res.model
    s = summarize(res, qf, "main grid", lambda t: print(t, flush=True))
    if not args.evaluate:
        np.savez(MAIN, **a, loss=s["L"])
    open(os.path.join(DATA, "main_row.tex"), "w", newline="\n").write(s["row"] + "\n")
    open(os.path.join(DATA, "main_moments.md"), "w", encoding="utf-8").write(
        f"# main: main grid\n\n{res.iters} steps, resV {res.resV:.1e}, resq {res.resq:.1e}, {time.time() - t0:.0f}s\n\n"
        + table_combined(s["M"], s["info"]) + f"\n\nmass unresolved {s['unres']:.4f}\n")
    txt = check_text(s["D"], qf, band, m.nB, m.nz, res.resV, res.resq)
    open(os.path.join(DATA, "main_check.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
