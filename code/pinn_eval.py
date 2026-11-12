"""
pinn_eval.py: evaluates a neural-network solution against the paper's finite-difference solution data/ou_main.npz.
Input: a <tag>_grid.npz file written by pinn_solve.py (the network's V, q, policies, and stationary residuals on
the reference grid); optionally a second one (--alt) summarized in the pinnAlt* macros.

Reports: errors of V and q (absolute and relative to their ranges), of the policies, of the default
thresholds by productivity; agreement of the default region and of the drift sign; the stationary residuals
of the network (continuous, by autodiff) next to the discrete residuals of the finite-difference solution;
the calibration moments with each solution's policies by the paper's procedure (exact ergodic moments from
the upwind generator on the grid, simulated path moments with annual accounting; pdmoments with output net
of default costs), and the ergodic debt densities.

Writes data/pinn_eval.md (the report), data/pinn_main.npz (the network solution with its
ergodic distribution, for Figure B.3), output/tab_pinn_rows.tex (rows of Table B.2), and
output/numbers_pinn.tex (macros).  Usage:
    python code/pinn_eval.py data/pinn/polish_grid.npz --label polish --alt data/pinn/base_grid.npz
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

from paths import DATA, OUT
import pdmoments as PM
from pdmodel import Model, Params, ZProcess, ergodic_density

assert PM.NET_OUTPUT

ap = argparse.ArgumentParser()
ap.add_argument("grid")
ap.add_argument("--fd-time", type=float, default=732.0, help="wall time of the finite-difference main solve in seconds, quoted in the report")
ap.add_argument("--no-sim", action="store_true", help="skip the simulated path moments (quick look)")
ap.add_argument("--label", default="", help="label written into the report (default: the file name)")
ap.add_argument("--no-write", action="store_true", help="print the report only; leave the paper's files (pinn_eval.md, pinn_main.npz, rows, macros) alone")
ap.add_argument("--alt", default="", help="a second network solution, summarized in macros pinnAlt*")
args = ap.parse_args()

# both solutions must share the grid and the parameters
a = np.load(os.path.join(DATA, "ou_main.npz"))
g = np.load(args.grid)
B, z = a["B"], a["z"]
assert np.allclose(g["B"], B) and np.allclose(g["z"], z), "the network solution is not on the reference grid"
assert np.allclose(g["x"], a["x"]) and abs(float(g["qf"]) - float(a["qf"])) < 1e-12, "parameters differ"
label = args.label or os.path.basename(args.grid).replace("_grid.npz", "")

# ----------------------------------------------------------------------------- model and generators
# the finite-difference model on the reference grid, used for the generators and ergodic densities of both solutions
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = map(float, a["x"])
mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
par = Params(nu=2.0, rho=rho, r=float(np.log(1.04)), delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2,
             ztilde=zt, fixed_cost_form="level", S_cap=1.0, q_floor=float(a["qf"]))
zp = ZProcess.ou(mu, sigma, z.size, zmin=float(z[0]), zmax=float(z[-1]))
assert np.allclose(zp.z, z)
m = Model(par, zp, 0.0, float(B[-1]), B.size)
coupon = par.delta + par.lam


def result(S, d, c, q):
    """A Result-like object with the given policies and the ergodic distribution of the upwind generator."""
    S = S.copy(); S[0] = np.maximum(S[0], 0.0); S[-1] = np.minimum(S[-1], 0.0)
    A = m.assemble(S)
    gg = ergodic_density(m, A)
    res = SimpleNamespace(model=m, S=S, d=d, c=c, q=q, g=gg, A=A)
    return res


# generators and ergodic densities of both solutions; report header
t0 = time.time()
fd = result(a["S"], a["d"], a["c"], a["q"])
nn = result(g["S"], g["d"], g["c"], g["q"])
lines = [f"# network solution `{label}` against the finite-difference solution (data/ou_main.npz)", ""]
lines.append(f"network: {int(g['width'])} x {int(g['depth'])} tanh, {int(g['n'])} interior points per step, outer steps {int(g['outer'])}, "
             f"final dt {float(g['dt']):.3f}, wall time {float(g['elapsed']):.0f} s; finite differences: 201 x 75 grid, 9000 implicit steps, {args.fd_time:.0f} s")

# ----------------------------------------------------------------------------- errors
# pointwise errors of values and policies, drift-sign and default-region agreement
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
def mx(x): return float(np.abs(x).max())

errV, errq, errS, errd, errc = (g[k] - a[k] for k in ("V", "q", "S", "d", "c"))
rngV = float(a["V"].max() - a["V"].min()); rngq = float(a["q"].max() - a["q"].min())
E = dict(V_rms=rms(errV), V_max=mx(errV), q_rms=rms(errq), q_max=mx(errq), S_rms=rms(errS), S_max=mx(errS),
         d_rms=rms(errd), d_max=mx(errd), c_rms=rms(errc), c_max=mx(errc),
         sign=float(np.mean(np.sign(g["S"]) == np.sign(a["S"]))),
         sign_big=float(np.mean((np.sign(g["S"]) == np.sign(a["S"]))[np.abs(a["S"]) > 1e-3])),
         region=float(np.mean((g["d"] > 1e-9) == (a["d"] > 1e-9))))
# consumption-equivalent value error: dV / V_c with V_c ~ u'(c)/rho at the ergodic mean consumption (approximate scale)
cbar = float((fd.g * a["c"]).sum() / fd.g.sum())
E["V_rms_ce"] = E["V_rms"] / (cbar ** (-par.nu) / par.rho) / cbar      # fraction of consumption
lines += ["", "## errors on the grid (network minus finite differences)", "",
          f"- V: rms {E['V_rms']:.4f}, max {E['V_max']:.4f} (range of V {rngV:.2f}; rms as a share of the range {E['V_rms'] / rngV:.2e}; "
          f"as a share of consumption, dV / (u'(c) / rho) at mean consumption, {E['V_rms_ce']:.2e})",
          f"- q: rms {E['q_rms']:.4f}, max {E['q_max']:.4f} (range {rngq:.2f}; rms as a share of the range {E['q_rms'] / rngq:.2e})",
          f"- S: rms {E['S_rms']:.4f}, max {E['S_max']:.4f}; d: rms {E['d_rms']:.4f}, max {E['d_max']:.4f}; c: rms {E['c_rms']:.4f}, max {E['c_max']:.4f}",
          f"- drift sign agreement {E['sign']:.4f} (nodes with |S_FD| > 1e-3: {E['sign_big']:.4f}); default region agreement {E['region']:.4f}"]


def thresholds(d):
    """First default node in debt for each endowment (NaN where there is none)."""
    out = np.full(z.size, np.nan)
    for j in range(z.size):
        i = np.where(d[:, j] > 1e-9)[0]
        if i.size:
            out[j] = B[i[0]]
    return out


tn, tf = thresholds(g["d"]), thresholds(a["d"])
ok = np.isfinite(tn) & np.isfinite(tf)
E["thr_mean"] = float(np.mean(np.abs(tn - tf)[ok])); E["thr_max"] = float(np.max(np.abs(tn - tf)[ok]))
lines.append(f"- first-default threshold by productivity: mean |difference| {E['thr_mean']:.4f}, max {E['thr_max']:.4f} (grid step {B[1] - B[0]:.3f}); "
             f"{int(ok.sum())} of {z.size} productivity nodes have a threshold in both")
lines.append("- thresholds (z, FD, network): " + "; ".join(f"{z[j]:+.2f}: {tf[j]:.3f} / {tn[j]:.3f}" for j in range(0, z.size, 12)))

# residuals: network continuous (autodiff) vs finite-difference discrete
Vf, qf_, df = a["V"].ravel(), a["q"].ravel(), a["d"].ravel()
uf = (a["c"] ** (1 - par.nu) / (1 - par.nu)).ravel()
rV_fd = (par.rho * Vf - uf - fd.A @ Vf).reshape(B.size, z.size)
rq_fd = (par.xi(df) * qf_ - (1 - df) * coupon - fd.A @ qf_).reshape(B.size, z.size)
R = dict(nn_rV_rms=rms(g["rV"]), nn_rV_max=mx(g["rV"]), nn_rq_rms=rms(g["rq"]), nn_rq_max=mx(g["rq"]),
         fd_rV_rms=rms(rV_fd), fd_rV_max=mx(rV_fd), fd_rq_rms=rms(rq_fd), fd_rq_max=mx(rq_fd))
lines += ["", "## stationary residuals", "",
          f"- network (continuous equations, autodiff, on the grid): HJB rms {R['nn_rV_rms']:.2e} max {R['nn_rV_max']:.2e}; price rms {R['nn_rq_rms']:.2e} max {R['nn_rq_max']:.2e}",
          f"- finite differences (discrete equations): HJB rms {R['fd_rV_rms']:.2e} max {R['fd_rV_max']:.2e}; price rms {R['fd_rq_rms']:.2e} max {R['fd_rq_max']:.2e}"]
# where the network's residuals are large: share of the squared residual within 0.02 of the FD threshold
near = np.zeros_like(a["d"], dtype=bool)
for j in range(z.size):
    if np.isfinite(tf[j]):
        near[:, j] = np.abs(B - tf[j]) <= 0.02
lines.append(f"- share of the network's squared HJB residual within 0.02 of the FD thresholds: {float((g['rV'] ** 2)[near].sum() / (g['rV'] ** 2).sum()):.2f} "
             f"(price: {float((g['rq'] ** 2)[near].sum() / (g['rq'] ** 2).sum()):.2f}); those nodes are {near.mean():.3f} of the grid")
# squared residual shares on the outer z rows, the floor band, and the zero-drift band; drift-sign errors; V_B at low debt
zedge = np.zeros_like(near); zedge[:, :3] = True; zedge[:, -3:] = True
floor = np.abs(a["q"] - par.q_floor) < 0.02
zero = np.zeros_like(near)
for j in range(z.size):
    for i in np.where(np.sign(a["S"][1:, j]) != np.sign(a["S"][:-1, j]))[0]:
        zero[max(0, i - 3):i + 4, j] = True
r2V, r2q = g["rV"] ** 2, g["rq"] ** 2
LOC = dict(V_zedge=float(r2V[zedge].sum() / r2V.sum()), q_zedge=float(r2q[zedge].sum() / r2q.sum()),
           V_floor=float(r2V[floor].sum() / r2V.sum()), q_floor=float(r2q[floor].sum() / r2q.sum()),
           V_zero=float(r2V[zero].sum() / r2V.sum()), q_zero=float(r2q[zero].sum() / r2q.sum()),
           zedge_frac=float(zedge.mean()), floor_frac=float(floor.mean()), zero_frac=float(zero.mean()))
bad = np.sign(g["S"]) != np.sign(a["S"])
LOC["bad_zero"] = float((bad & zero).sum() / max(bad.sum(), 1))
LOC["bad_medS"] = float(np.median(np.abs(a["S"][bad]))) if bad.any() else 0.0
VB_fd = np.gradient(a["V"], B, axis=0)
lowB = B < 0.3
LOC["VB_rel_low"] = float(np.sqrt(np.mean(((g["V_B"] - VB_fd) / VB_fd)[lowB] ** 2)))
lines.append(f"- squared residual shares: three outer z rows ({LOC['zedge_frac']:.3f} of the grid) HJB {LOC['V_zedge']:.2f} price {LOC['q_zedge']:.2f}; "
             f"floor band |q - q_floor| < 0.02 ({LOC['floor_frac']:.3f}) HJB {LOC['V_floor']:.2f} price {LOC['q_floor']:.2f}; "
             f"zero-drift band +-3 nodes ({LOC['zero_frac']:.3f}) HJB {LOC['V_zero']:.2f} price {LOC['q_zero']:.2f}")
lines.append(f"- drift-sign disagreements: {LOC['bad_zero']:.2f} of them within 3 nodes of the FD zero-drift locus; median |S_FD| there {LOC['bad_medS']:.4f}")
lines.append(f"- slope V_B at B < 0.3: rms relative error {LOC['VB_rel_low']:.4f} (against the central difference of the FD value)")
# ripple of the policies: the drift error where the true policies are smooth (no default, six nodes below the threshold, interior z)
smooth = np.zeros_like(near)
for j in range(3, z.size - 3):
    i = np.where(a["d"][:, j] > 1e-9)[0]
    top = (i[0] - 6) if i.size else B.size - 6
    if top > 4:
        smooth[2:top, j] = True
LOC["S_smooth"] = rms((g["S"] - a["S"])[smooth]); LOC["S_smooth_typ"] = float(np.median(np.abs(a["S"][smooth])))
LOC["c_smooth"] = rms((g["c"] - a["c"])[smooth])
lines.append(f"- ripple: in the smooth no-default region ({smooth.mean():.2f} of the grid) the drift differs by {LOC['S_smooth']:.4f} rms "
             f"({100 * LOC['S_smooth'] / LOC['S_smooth_typ']:.0f} percent of the median |S| there, {LOC['S_smooth_typ']:.3f}); consumption by {LOC['c_smooth']:.4f}")

# ----------------------------------------------------------------------------- moments
def all_moments(res):
    """Calibration moments, details, and loss for a solution (ergodic moments only with --no-sim)."""
    if args.no_sim:
        Eg = PM.ergodic_moments(res)
        M = {k: Eg[k] for k in PM.ERGODIC_KEYS}
        return M, dict(ergodic=Eg, sim_mean={}, sim_sd={}), float("nan")
    M, info = PM.combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
    return M, info, PM.loss(M)


# moments of both solutions by the same procedure, and summary statistics of the ergodic densities
M_fd, I_fd, L_fd = all_moments(fd)
M_nn, I_nn, L_nn = all_moments(nn)
wf = fd.g / fd.g.sum(); wn = nn.g / nn.g.sum()
dens = dict(fd_EB=float((wf * B[:, None]).sum()), nn_EB=float((wn * B[:, None]).sum()),
            fd_pdef=float(wf[a["d"] > 1e-9].sum()), nn_pdef=float(wn[g["d"] > 1e-9].sum()),
            fd_atom=float(wf[0].sum()), nn_atom=float(wn[0].sum()))
lines += ["", "## moments (paper's procedure, output net of default costs)", "",
          "| moment | data (AMR) | FD | network |", "|---|---|---|---|"]
keys = PM.TARGETS + ["spread_mean", "corr_spread_y", "corr_spread_debt", "c_sd_rel"]
for k in keys:
    if k in M_fd and k in M_nn:
        lines.append(f"| {k} | {PM.AMR_DATA[k]:.3f} | {M_fd[k]:.3f} | {M_nn[k]:.3f} |")
lines.append(f"| loss (AMR weights) | | {L_fd:.3f} | {L_nn:.3f} |")
lines.append(f"| ergodic mean debt | | {dens['fd_EB']:.3f} | {dens['nn_EB']:.3f} |")
lines.append(f"| ergodic P(d > 0) | | {dens['fd_pdef']:.3f} | {dens['nn_pdef']:.3f} |")
lines.append(f"| atom at B = 0 | | {dens['fd_atom']:.3f} | {dens['nn_atom']:.3f} |")
lines.append(f"| mass in the top tenth of the grid | | {I_fd['ergodic']['mass_top']:.4f} | {I_nn['ergodic']['mass_top']:.4f} |")
lines.append("")
lines.append(f"evaluation time {time.time() - t0:.0f} s")
txt = "\n".join(lines)
print(txt)
if args.no_write:
    sys.exit(0)
open(os.path.join(DATA, "pinn_eval.md"), "w", encoding="utf-8").write(txt + "\n")

# ----------------------------------------------------------------------------- figure data, table rows, macros
# data for Figure B.3
np.savez(os.path.join(DATA, "pinn_main.npz"), **{k: g[k] for k in g.files}, g_nn=nn.g, g_fd=fd.g, label=label)


def f3(v): return f"{v:.3f}"


# Table B.2 rows: selected moments, loss, and mass at zero debt for both solutions
rows = []
names = [("debt_y_mean", "Debt to output, mean"), ("debt_y_sd", "Debt to output, s.d."), ("spread_sd", "Spread, s.d."),
         ("pd_freq", "Partial default frequency"), ("pd_mean_cond", "Partial default, mean"), ("spread_mean", "Spread, mean")]
for k, lab in names:
    if k in M_fd and k in M_nn:
        rows.append(f"{lab} & {f3(M_fd[k])} & {f3(M_nn[k])} \\\\")
rows.append(f"Calibration loss & {f3(L_fd)} & {f3(L_nn)} \\\\")
rows.append(f"Mass at zero debt & {f3(dens['fd_atom'])} & {f3(dens['nn_atom'])} \\\\")
open(os.path.join(OUT, "tab_pinn_rows.tex"), "w", newline="\n").write("\n".join(rows) + "\n")


def sci(v):
    """Scientific notation in LaTeX, one significant digit."""
    e = int(np.floor(np.log10(abs(v)))); mant = v / 10 ** e
    return f"{mant:.0f}\\times10^{{{e}}}" if round(mant) != 1 else f"10^{{{e}}}"


# macros: network architecture and training settings, errors, residuals, moments, and where the errors sit
mac = {
    "pinnLabel": label, "pinnWidth": int(g["width"]), "pinnDepth": int(g["depth"]),
    "pinnParams": f"{2 * int(g['width']) + int(g['width']) + (int(g['depth']) - 1) * (int(g['width']) ** 2 + int(g['width'])) + 2 * int(g['width']) + 2:,}",
    "pinnPoints": f"{int(g['n']):,}", "pinnOuter": f"{int(g['outer']):,}", "pinnDtMax": f"{float(g['dt']):.2f}",
    "pinnInner": int(g["inner"]) if "inner" in g.files else 50, "pinnDtZero": f"{float(g['dt0']) if 'dt0' in g.files else 0.02:.2f}",
    "pinnDtEvery": int(g["dt_every"]) if "dt_every" in g.files else 100,
    "pinnDevice": "GPU" if ("device" in g.files and "cuda" in str(g["device"])) else "CPU",
    "pinnMinutes": f"{float(g['elapsed']) / 60:.0f}", "fdMinutes": f"{args.fd_time / 60:.0f}",
    "pinnRmsV": f"{E['V_rms']:.3f}", "pinnMaxV": f"{E['V_max']:.3f}", "pinnRmsVrel": sci(E["V_rms"] / rngV),
    "pinnRmsVce": f"{100 * E['V_rms_ce']:.2f}", "pinnRmsQ": f"{E['q_rms']:.3f}", "pinnMaxQ": f"{E['q_max']:.3f}",
    "pinnRmsS": f"{E['S_rms']:.3f}", "pinnRmsD": f"{E['d_rms']:.3f}", "pinnRmsC": f"{E['c_rms']:.4f}",
    "pinnThrMean": f"{E['thr_mean']:.3f}", "pinnThrMax": f"{E['thr_max']:.3f}",
    "pinnSign": f"{100 * E['sign']:.1f}", "pinnSignBig": f"{100 * E['sign_big']:.1f}", "pinnRegion": f"{100 * E['region']:.1f}",
    "pinnResV": sci(R["nn_rV_rms"]), "pinnResVmax": sci(R["nn_rV_max"]), "pinnResq": sci(R["nn_rq_rms"]), "pinnResqmax": sci(R["nn_rq_max"]),
    "fdResVrms": sci(R["fd_rV_rms"]), "fdResqrms": sci(R["fd_rq_rms"]),
    "pinnLoss": f3(L_nn), "fdLossRecomputed": f3(L_fd),
    "pinnDebtY": f3(M_nn.get("debt_y_mean", np.nan)), "fdDebtY": f3(M_fd.get("debt_y_mean", np.nan)),
    "pinnPdFreq": f3(M_nn.get("pd_freq", np.nan)), "fdPdFreq": f3(M_fd.get("pd_freq", np.nan)),
    "pinnSpreadSd": f3(M_nn.get("spread_sd", np.nan)), "fdSpreadSd": f3(M_fd.get("spread_sd", np.nan)),
    "pinnAtom": f3(dens["nn_atom"]), "fdAtom": f3(dens["fd_atom"]),
}
mac.update({"pinnResVzedge": f"{100 * LOC['V_zedge']:.0f}", "pinnZedgeFrac": f"{100 * LOC['zedge_frac']:.0f}",
            "pinnResqKinks": f"{100 * (LOC['q_floor'] + LOC['q_zero']):.0f}", "pinnKinksFrac": f"{100 * (LOC['floor_frac'] + LOC['zero_frac']):.0f}",
            "pinnBadZero": f"{100 * LOC['bad_zero']:.0f}", "pinnBadMedS": f"{LOC['bad_medS']:.3f}",
            "pinnVBlow": f"{100 * LOC['VB_rel_low']:.0f}",
            "pinnRippleS": f"{LOC['S_smooth']:.3f}", "pinnRipplePct": f"{100 * LOC['S_smooth'] / LOC['S_smooth_typ']:.0f}"})
# second network solution, if given
if args.alt:
    h = np.load(args.alt)
    assert np.allclose(h["B"], B) and np.allclose(h["z"], z)
    alt = result(h["S"], h["d"], h["c"], h["q"])
    M_alt, I_alt, L_alt = all_moments(alt)
    wa = alt.g / alt.g.sum()
    VB_alt = float(np.sqrt(np.mean(((h["V_B"] - VB_fd) / VB_fd)[lowB] ** 2)))
    mac.update({"pinnAltRmsV": f"{rms(h['V'] - a['V']):.3f}", "pinnAltRmsQ": f"{rms(h['q'] - a['q']):.3f}",
                "pinnAltSign": f"{100 * float(np.mean(np.sign(h['S']) == np.sign(a['S']))):.1f}",
                "pinnAltLoss": f3(L_alt), "pinnAltEB": f3(float((wa * B[:, None]).sum())), "pinnAltAtom": f3(float(wa[0].sum())),
                "pinnAltDebtY": f3(M_alt.get("debt_y_mean", np.nan)), "pinnAltVBlow": f"{100 * VB_alt:.0f}"})
    print(f"alt {args.alt}: loss {L_alt:.3f}, E[B] {(wa * B[:, None]).sum():.3f}, atom {wa[0].sum():.3f}, V_B rel err at B<0.3 {VB_alt:.4f}")
mac["fdEB"] = f3(dens["fd_EB"]); mac["pinnEB"] = f3(dens["nn_EB"])
out = ["% Numbers of the neural-network solution (code/pinn_eval.py); regenerate, do not edit."]
out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
open(os.path.join(OUT, "numbers_pinn.tex"), "w", newline="\n").write("\n".join(out) + "\n")
print("wrote data/pinn_eval.md, data/pinn_main.npz, output/tab_pinn_rows.tex, output/numbers_pinn.tex")
