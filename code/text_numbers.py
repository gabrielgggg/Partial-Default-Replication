r"""text_numbers.py: the computed numbers quoted in the text of the paper, recomputed from stored results and written
as LaTeX macros, and the bodies of the calibration, fit, and robustness tables.

Reads (in data/):
  ou_main.npz, main_moments.md, main_check.md, main_row.tex              main solution
  irf.npz                                                                impulse responses
  det_fine15.npz, illustrations_fine15.log                               deterministic solution
  jump_floor.npz, jumpbuf_floor.npz, jump3s_floor.npz                    chain illustrations
  robustness/{fine,nonuni,wide,nz51,nz101,z30,z35}_row.tex, robustness/nonuni.{log,npz},
  robustness/{z30,z35}.{npz,md}, robustness/{nostd,nostd2}.md            other grids, and no underwriting standard
  calibration/cma_*_gens.md, calibration/cma_*_best.npz,
  calibration/confirm_*.json                                             calibration searches and the selected point
  wedge/rho0.060_illus.npz, pdss/sweep_0.06.txt, pdss/threshold_nu2.0.txt, sfregion.npz
                                                                         self-fulfilling example (its own parameters,
                                                                         closed forms in example_closed_forms.py)
  ou_saving.npz                                                          solution with saving
  approach_fig_out.txt, grid_check_out.txt                               the pessimistic approach to capacity
  and the parameter values of Arellano, Mateos-Planas, and Rios-Rull (2023) from amr_values.py

Writes:
  output/numbers_text.tex       macros \tx... (calibrated model) and \ex... (self-fulfilling example)
  output/tab_calibration.tex, output/tab_calibration_fixed.tex, output/tab_fit.tex, output/tab_robfloor_rows.tex
  logs/numbers_text_report.md   every value, with the checks of statements the text makes in words

Each put(...) defines one macro. Each check(...) verifies a statement the text makes in words (not as a macro)
and records OK or FAIL in the report.

Usage: python code/text_numbers.py
"""
import glob
import json
import math
import os
import re
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

import example_closed_forms as ecf
from amr_values import AMR_PARAMS
from generator import model_from_solution
from paths import DATA, OUT, LOGS

CAL = os.path.join(DATA, "calibration")
ROB = os.path.join(DATA, "robustness")

M = {}          # macro name -> LaTeX value
R = []          # report lines
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
         11: "eleven", 12: "twelve"}


def put(name, value, note):
    """Define macro \\name with a LaTeX value; note goes to the report."""
    M[name] = value
    R.append(f"| `\\{name}` | {value} | {note} |")


def check(claim, ok, detail):
    """Record whether a statement in the text holds (OK or FAIL) in the report."""
    R.append(f"| (words) | {'OK' if ok else '**FAIL**'} | {claim}: {detail} |")


def thousands(v):
    """Integer with LaTeX thousands separators."""
    return f"{v:,.0f}".replace(",", "{,}")


def sci(v):
    """Order of magnitude in LaTeX: 10^{e} or m\\times10^{e}, one significant digit."""
    e = int(math.floor(math.log10(abs(v))))
    m = round(v / 10 ** e)
    if m == 10:
        m, e = 1, e + 1
    return f"10^{{{e}}}" if m == 1 else f"{m}\\times10^{{{e}}}"


def read(p):
    """Contents of a text file."""
    return open(p, encoding="utf-8").read()


def rnd(v, k):
    """Round half away from zero at k decimals (0.475 -> 0.48), as a string."""
    return str(Decimal(repr(round(float(v), 9))).quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP))


def trim(s):
    """Drop trailing zeros of a plain decimal for the text: 0.100 -> 0.1, 1.00 -> 1; anything else unchanged."""
    return s.rstrip("0").rstrip(".") if re.fullmatch(r"-?\d+\.\d+", s) else s


# --------------------------------------------------------------------------- parameters and closed forms
a = np.load(os.path.join(DATA, "ou_main.npz"))
x = a["x"].astype(float)
g0, g1, g2, zt, rho, kap, dl, rz, se = x
r = lam = math.log(1.04)
qf, band, lband = float(a["qf"]), float(a["band"]), float(a["lband"])    # underwriting standard: level, band, scale of the cap
B, z, q, S, c, d, g = (a[k] for k in ("B", "z", "q", "S", "c", "d", "g"))
w = g / g.sum()
cs = dl + lam
sdz = se / math.sqrt(1 - rz ** 2)
F0 = 1 - g2 * (1 - math.exp(zt))
dfirst = ((1 - F0) / ((g1 - 1) * g0 * F0)) ** (1 / g1)
xi = lambda d: r + dl - kap * cs * d
qr = lambda d: (1 - d) * cs / xi(d)
s_first = cs * (1 / qr(dfirst) - 1)
floor_spread = cs * (1 / qf - 1)
wedge = rho - r
jshow_ = int(np.where(z < zt)[0].max())       # the lowest endowment level of the figure ou_slices
R += ["# Numbers in the text", "", "| macro | value | source / note |", "|---|---|---|"]
put("txSfirstBp", thousands(1e4 * s_first), "stationary spread at the first default")
put("txFloorSpreadBp", thousands(1e4 * floor_spread), "spread at the underwriting standard")
put("txFloorSpreadRound", thousands(round(1e4 * floor_spread, -1)), "spread at the underwriting standard, to 10 bp")
put("txWedgeBp", f"{1e4 * wedge:.0f}", "rho - r")
put("txRNThresholdBp", thousands(1e4 * s_first), "risk-neutral threshold Theta* = s_first (gamma_1 > 2: min of Theta, attained at d_first here)")
put("txQrFirst", f"{qr(dfirst):.3f}", "q_r(d_first)")
put("txZlow", f"{z[jshow_]:.2f}", f"lowest endowment level shown in the figure ou_slices: the highest node below ln y*, {z[jshow_] / sdz:.2f} s.d.")
put("txLnYstar", f"{zt:.3f}", "ln y*")

# Theta is minimized at d_first (gamma_1 > 2), so the text uses s_first as the threshold
ds = np.linspace(dfirst, 0.99, 20001)
Theta = (1 - kap) * cs * ds / (1 - ds) * (1 + (g1 - 1) * (dfirst / ds) ** g1) / g1
check("risk-neutral threshold equals s_first at the calibration", abs(Theta.min() - s_first) < 1e-6,
      f"min Theta {1e4 * Theta.min():.1f} bp at d {ds[Theta.argmin()]:.3f}; s_first {1e4 * s_first:.1f} bp")

# --------------------------------------------------------------------------- main solution
put("txPdefInst", f"{w[d > 1e-9].sum():.2f}", "instantaneous probability of default")
ad = dl - kap * cs * d
ell = S + ad * B[:, None]
Fz = 1 - g2 * (np.exp(z) - math.exp(zt)) * (np.exp(z) >= math.exp(zt))
# nodes in partial default where the cap L(q) of the standard on issuance is active
w_ = np.clip((q - qf) / band, 0.0, 1.0 - 1e-12)
cap_ = np.where(q < qf, 0.0, lband * w_ / (1.0 - w_))
bind = (q < qf + band) & (ell >= cap_ - 1e-9) & (d > 0) & (d < 1 - 1e-9)
put("txBindPct", f"{100 * w[bind].sum():.1f}", "ergodic mass on which the standard's cap on issuance is active, percent")
put("txFullDefPct", f"{100 * w[d >= 1 - 1e-9].sum():.1f}", "ergodic mass in full default, percent")
put("txStdBand", f"{band:g}", "band of the standard")
put("txStdLband", f"{lband:g}", "scale of the cap of the standard")
dint = np.minimum(1, ((1 - kap * q) * cs * B[:, None] / (g0 * g1 * Fz * np.exp(z))) ** (1 / (g1 - 1)))
with np.errstate(invalid="ignore", divide="ignore"):
    ex = d / dint - 1
bm = bind & (w > 0)
ex_mean = (ex[bm] * w[bm]).sum() / w[bm].sum()
ex_max = np.nanmax(ex[bm])
put("txExMeanPct", f"{100 * ex_mean:.0f}", "default above the market-price rule where the standard binds, mean, percent")
put("txExMaxPct", f"{100 * ex_max:.0f}", "same, maximum over nodes with mass, percent")
j0_ = int(np.argmin(np.abs(z))); i0_ = int(np.where(d[:, j0_] > 1e-9)[0][0])
Bpd0_ = g0 * g1 * F0 * dfirst ** (g1 - 1) / ((1 - kap) * cs)       # B_pd(0, q_rf), the deterministic capacity
put("txThrZeroB", f"{B[i0_]:.3f}", "first default node at the mean endowment (z = 0 column), diffusion model")
put("txThrZeroQ", f"{q[i0_, j0_]:.2f}", "price at that node")
check("the mean-endowment threshold lies below the deterministic capacity, with a price below par",
      B[i0_] < Bpd0_ and q[i0_, j0_] < 0.95, f"threshold node {B[i0_]:.3f}, B_pd(0,1) {Bpd0_:.3f}, q {q[i0_, j0_]:.3f}")
cmin = 0.05 * np.exp(z)[None, :]
check("consumption floor and drift cap slack at the main solution", (c > cmin * 1.01).all() and np.abs(S).max() < 0.99,
      f"min c/c_min {(c / cmin).min():.1f}, max |S| {np.abs(S).max():.3f}")

# zero-drift locus and ergodic set, from main_check.md
mc = read(os.path.join(DATA, "main_check.md"))
loc = [(float(m_.group(1)), float(m_.group(2))) for m_ in re.finditer(r"^\| ([+-]\d\.\d+) \| (\d\.\d+) \|", mc, re.M)]
locB = np.array([b for _, b in loc]); locz = np.array([zz for zz, _ in loc])
lowz_ = locz < 0          # the part of the locus swept by the ergodic set (the rows at high endowment and high debt carry no mass)
locB, locz = locB[lowz_], locz[lowz_]
put("txLocusMax", rnd(locB.max(), 2), f"maximum of the zero-drift locus at z < 0, at z {locz[locB.argmax()]:+.3f}")
put("txLocusLow", rnd(locB[locz.argmin()], 2), "zero-drift debt at the lowest node")
nolocus = [float(m_.group(1)) for m_ in re.finditer(r"^\| ([+-]\d\.\d+) \| (?:nan|0\.9\d\d|0\.000) ", mc, re.M) if float(m_.group(1)) > 0]
check("above about one s.d. above the mean the drift is negative everywhere", abs(min(nolocus) - sdz) < 0.03,
      f"no zero-drift point from z {min(nolocus):+.3f}, +1 s.d. {sdz:+.3f}")
put("txErgMaxB", f"{float(re.search(r'max B with mass ([0-9.]+)', mc).group(1)):.2f}", "largest debt with ergodic mass")

# residuals of the main solution, node by node, with the saved policies
m_, par_, A_, S_, c_, d_ = model_from_solution(a)
u_ = c_ ** (1 - par_.nu) / (1 - par_.nu)
RV = (par_.rho * a["V"].ravel() - u_.ravel() - A_ @ a["V"].ravel()).reshape(B.size, z.size)
Rq = (par_.xi(d_.ravel()) * a["q"].ravel() - (1 - d_.ravel()) * cs - A_ @ a["q"].ravel()).reshape(B.size, z.size)
put("txMainResV", sci(np.abs(RV).max()), "largest HJB residual of the main solution")
put("txMainResq", sci(np.abs(Rq).max()), "largest pricing residual of the main solution")
check("the main solution meets the tolerances at every node", np.abs(RV).max() < 1e-7 and np.abs(Rq).max() < 1.5e-8,
      f"{np.abs(RV).max():.1e} / {np.abs(Rq).max():.1e}")
pol_ = m_.policies(a["V"], a["q"], 28)
tie_ = np.abs(pol_[0] - a["S"]) > 1e-8
put("txTieNodes", str(int(tie_.sum())), "nodes whose upwind branch is held (policies recomputed from V, q differ)")
put("txTieMass", sci(w[tie_].sum()) if w[tie_].sum() > 0 else "0", "their ergodic mass")
# the selected point of the calibration: the strict re-solve with the smallest loss among those that converged
# (the rule of cal_adopt.py)
conf_ = [json.load(open(f_, encoding="utf-8")) for f_ in sorted(glob.glob(os.path.join(CAL, "confirm_*.json")))]
jc_ = min((j_ for j_ in conf_ if j_["diag"]["converged"]), key=lambda j_: j_["loss"])
put("txMainSteps", str(jc_["trail"][-1]["steps"]), "time steps of the solve of the selected point at the selected standard")

# moments
mm = read(os.path.join(DATA, "main_moments.md"))
put("txPdFreq", f"{float(re.search(r'\| pd_freq \|[^|]*\|[^|]*\| ([0-9.]+)', mm).group(1)):.2f}", "annual frequency of partial default")

# --------------------------------------------------------------------------- impulse responses
ir = np.load(os.path.join(DATA, "irf.npz"))
tt = ir["t"]; dt = tt[1] - tt[0]
yb = ir["control_yn"].mean()                 # mean output, net of default costs
dB = ir["shock_B"] - ir["control_B"]; kp = int(np.argmax(dB))
di = (ir["shock_issue"] - ir["control_issue"]) / yb
dc = (ir["shock_defc"] - ir["control_defc"]) / yb
dd = ir["shock_d"] - ir["control_d"]
put("txIrfIssue", f"{100 * di[:kp].sum() * dt:.1f}", "issuance cumulated to the debt peak, % of output")
put("txIrfArrears", f"{100 * kap * dc[:kp].sum() * dt:.1f}", "arrears cumulated to the peak, % of output")
put("txIrfSpreadBp", f"{1e4 * (ir['shock_spread'][0] - ir['control_spread'][0]):.0f}", "spread on impact, bp")

# --------------------------------------------------------------------------- the deterministic illustration
df15 = np.load(os.path.join(DATA, "det_fine15.npz"))
Bd, Sd, qd = df15["B"], df15["S"], df15["q"]
beta, alpha, Sig1, Q1, Bs = (float(df15[k]) for k in ("beta", "alpha", "Sig1", "Q1", "Bpd"))


def err(lo, hi):
    """Largest relative errors of the two-term drift and price gap at B* + x, x in [lo, hi]."""
    k = (Bd - Bs >= lo) & (Bd - Bs <= hi); xx = Bd[k] - Bs
    return (np.abs(Sd[k] / (-beta * xx ** (1 / 3) - Sig1 * xx ** (2 / 3)) - 1).max(),
            np.abs((1 - qd[k]) / (alpha * xx ** (2 / 3) + Q1 * xx) - 1).max())


put("txCubeDriftErr", f"{math.ceil(100 * err(0.01, 0.05)[0])}", "max error of the two-term drift on x in [0.01, 0.05], percent")
put("txCubePriceErrNear", f"{math.ceil(100 * err(0.02, 0.05)[1])}", "max error of the two-term price gap on [0.02, 0.05], percent")
log15 = read(os.path.join(DATA, "illustrations_fine15.log"))
mres = re.search(r"residuals ([0-9.e+-]+) / ([0-9.e+-]+)", log15)
put("txDetResV", sci(float(mres.group(1))), "final HJB residual, [0,1.5] grid")
put("txDetResq", sci(float(mres.group(2))), "final pricing residual, [0,1.5] grid")
check("second drift term half the first at x = 0.1", abs(abs(Sig1) * 0.1 ** (1 / 3) / beta - 0.5) < 0.1,
      f"ratio {abs(Sig1) * 0.1 ** (1 / 3) / beta:.2f}")
lead_time = lambda x0: 3 / (2 * beta) * x0 ** (2 / 3)   # leading-order return time, integral of dx / (beta x^{1/3})
put("txReturnOnePct", rnd(lead_time(0.01), 1), "years to return to capacity from 1 percent of output above it, leading order 3x^{2/3}/(2 beta)")
put("txReturnTenPct", rnd(lead_time(0.1), 1), "years to return to capacity from 10 percent of output above it, leading order")

# --------------------------------------------------------------------------- the chain illustrations
# the two-state chain of the figure jump_policies: both states rest at the node below the low state's threshold
jf = np.load(os.path.join(DATA, "jump_floor.npz"))
assert jf["z"].size == 2, "the chain illustration of the paper has two states"
wj = jf["g"] / jf["g"].sum(); Bj = jf["B"]; cj = jf["c"]
rhoj, lamj, nuj, etaj = (float(jf[k]) for k in ("rho", "lam", "nu", "eta"))
upj = lambda x_: x_ ** (-nuj)
kA = int(np.argmax(wj.sum(1)))
firstL_ = Bj[np.where(jf["d"][:, 0] > 1e-9)[0][0]]; firstH_ = Bj[np.where(jf["d"][:, 1] > 1e-9)[0][0]]
put("txChainFirstL", f"{firstL_:.3f}", "two-state chain of the figure: the low state's first default node")
put("txChainFirstH", f"{firstH_:.3f}", "its high state's first default node")
put("txChainGap", rnd(firstH_ - Bj[kA], 1), "distance of the high state's rest point from its own threshold")
impA_ = (rhoj - lamj) * upj(cj[kA, 1]); precA_ = etaj * (upj(cj[kA, 0]) - upj(cj[kA, 1]))
put("txChainImp", rnd(impA_, 4), "there, high state: impatience term (rho - r) u'(c_H)")
put("txChainPrec", rnd(precA_, 4), "there, high state: precautionary term eta (u'(c_L) - u'(c_H))")
check("figure chain: both states rest at the node below the low state's threshold, with all the mass",
      wj[kA].sum() > 0.999 and abs(firstL_ - Bj[kA] - (Bj[1] - Bj[0])) < 1e-9, f"mass at the node {wj[kA].sum():.4f}")
check("figure chain: impatience exceeds precaution at the rest point of the high state, so the kink holds it", impA_ > precA_ > 0, f"{impA_:.4f} vs {precA_:.4f}")
check("figure chain: no state defaults on the ergodic set", wj[jf["d"] > 1e-9].sum() < 1e-6, f"mass in default {wj[jf['d'] > 1e-9].sum():.1e}")
# the chain with a precautionary buffer: the high state rests where the Euler equation holds
jb = np.load(os.path.join(DATA, "jumpbuf_floor.npz"))
wb = jb["g"] / jb["g"].sum(); Bb = jb["B"]; cb = jb["c"]; etab = float(jb["eta"])
kH = int(np.where(jb["S"][:, 1] <= 1e-10)[0][0])
imp_ = (rhoj - lamj) * upj(cb[kH, 1]); prec_ = etab * (upj(cb[kH, 0]) - upj(cb[kH, 1]))
put("txBufSd", f"{abs(float(jb['z'][0])):.2f}", "buffer chain: size of the states")
put("txBufEta", f"{etab:.2f}", "buffer chain: switching rate")
put("txBufHighB", f"{Bb[kH]:.2f}", "buffer chain: debt at which the drift of the high state vanishes")
put("txBufLowB", f"{Bb[int(np.argmax(wb[:, 0]))]:.2f}", "buffer chain: node at which the low state rests")
put("txBufImp", rnd(imp_, 4), "there: impatience term (rho - r) u'(c_H)")
put("txBufPrec", rnd(prec_, 4), "there: precautionary term eta (u'(c_L) - u'(c_H))")
check("buffer chain: the Euler equation holds at the high state's rest node to within five percent", abs(imp_ / prec_ - 1) < 0.05, f"{imp_:.5f} vs {prec_:.5f}")
check("buffer chain: no state defaults on the ergodic set", wb[jb["d"] > 1e-9].sum() < 1e-10, "")
# the three-state chain with a rare jump to the low state
j3 = np.load(os.path.join(DATA, "jump3s_floor.npz"))
w3 = j3["g"] / j3["g"].sum()
put("txThreePdef", f"{w3[j3['d'] > 1e-9].sum():.2f}", "three-state chain: ergodic mass in default")
check("three-state chain: all default in the low state", w3[:, 1:][j3["d"][:, 1:] > 1e-9].sum() < 1e-10, "")

put("txSpreadSd", f"{float(re.search(r'\| spread_sd \|[^|]*\|[^|]*\| ([0-9.]+)', mm).group(1)):.3f}", "spread s.d. with the standard")

# --------------------------------------------------------------------------- calibration bookkeeping
ends = []
for f_ in sorted(glob.glob(os.path.join(CAL, "cma_*_gens.md"))):
    e_ = re.search(r"stopped \(deadline\) after (\d+) generations, (\d+) evaluations, (\d+) restarts; best loss ([0-9.]+)", read(f_))
    ends.append((int(e_.group(1)), int(e_.group(2)), float(e_.group(4))))
put("txCalSearches", WORDS[len(ends)], "population searches")
put("txCalEvals", thousands(sum(e_[1] for e_ in ends)), "evaluations in total")
# the starting point has no search values: when no search improves on it, its strict values stand in
put("txCalSearchLoss", f"{jc_['search_loss'] or jc_['loss']:.3f}", "loss of the best point at the tolerances of the search")
put("txCalSearchStd", f"{jc_['search_level'] or jc_['level']:g}", "standard selected for it at the tolerances of the search")
gap_ = max(np.abs(np.load(f_)["x"] / x - 1).max() for f_ in glob.glob(os.path.join(CAL, "cma_*_best.npz")))
check("the searches end at nearly the same parameters", gap_ < 0.2, f"largest proportional gap {gap_:.2f}")
put("txStdLevel", f"{qf:g}", "level of the standard")
put("txKappa", f"{kap:.2f}", "recovery factor")
put("txDelta", f"{dl:.3f}", "maturity rate")

# --------------------------------------------------------------------------- the self-fulfilling example
put("exSfirstBp", f"{1e4 * ecf.s(ecf.dfirst):.0f}", "example: s_first")
put("exOmegaBp", f"{1e4 * ecf.Omega(ecf.dfirst):.0f}", "example: Omega(d_first)")
put("exGxi", f"{(ecf.g1 - 1) * ecf.xi(ecf.dfirst):.3f}", "example: (gamma_1 - 1) xi(d_first)")
put("exWedgeBp", f"{1e4 * (ecf.rho - ecf.r):.0f}", "example: wedge")
# the two steady states of the figure det_multiplicity: no default, and the partial default with the deepest default
W = np.load(os.path.join(DATA, "wedge", "rho0.060_illus.npz"))
metas = {k: W[k] for k in W.files if k.endswith("_meta")}
nd = [(k, v) for k, v in metas.items() if v[1] > 0.999 and v[2] < 1e-6]
pdm = [(k, v) for k, v in metas.items() if v[2] > 0.15]
put("exNoDefB", rnd(nd[0][1][0], 2), "example: no-default steady state")
k_, v_ = max(pdm, key=lambda t: t[1][2])
put("exPdB", rnd(v_[0], 2), f"example: partial-default steady state ({k_})")
put("exPdD", f"{100 * v_[2]:.0f}", "example: its default intensity, percent")
put("exPdQ", rnd(v_[1], 2), "example: its price")
put("exPdSpreadBp", f"{1e4 * ecf.cs * (1 / v_[1] - 1):.0f}", "example: its spread, (delta+lam)(1/q-1)")
dser = W[k_.replace("_meta", "_d")]
put("exPdStartB", rnd(W['B'][int(np.where(dser > 1e-9)[0][0])], 2), "example: debt at which default starts on that branch")
# continuum of steady states reached from zero debt, and the smallest wedge that reaches one at nu = 2
mrg = re.search(r"B0 in \[([0-9.]+), ([0-9.]+)\]", read(os.path.join(DATA, "pdss", "sweep_0.06.txt")))
put("exContLo", rnd(float(mrg.group(1)), 2), "example: continuum reached from zero debt, lowest")
put("exContHi", rnd(float(mrg.group(2)), 2), "example: same, highest")
mt = re.search(r"threshold wedge in \[([0-9.]+), ([0-9.]+)\]", read(os.path.join(DATA, "pdss", "threshold_nu2.0.txt")))
put("exThrNuTwo", f"{(float(mt.group(1)) + float(mt.group(2))) / 2:.0f}", f"example: smallest reachable wedge, nu 2.0 (bracket {mt.group(1)}-{mt.group(2)})")
# the region of wedges with a self-fulfilling steady state, by volatility and by curvature
S_ = np.load(os.path.join(DATA, "sfregion.npz"))
lo_, hi_ = S_["w_lo"], S_["w_hi"]
okl = np.isfinite(lo_)
put("exSfCloseSd", f"{100 * S_['sd_close'][0]:.0f}",
    f"shocks: s.d. of log endowment (%) at which the region closes (between {100*S_['sd_close'][0]:.2f} and {100*S_['sd_close'][1]:.2f})")
check("shocks: the lower boundary rises with volatility", bool(np.all(np.diff(lo_[okl]) > 0)), f"{np.round(lo_[okl], 1)}")
fh_ = np.isfinite(hi_) & okl
check("shocks: the upper boundary falls with volatility", bool(np.all(np.diff(hi_[fh_]) < 0)), f"{np.round(hi_[fh_], 1)}")
check("shocks: the region starts at s_first", abs(lo_[0] - float(S_["s_first"])) < 0.5, f"{lo_[0]:.1f}")
nu_m = 0.5 * (S_["nu_lo"] + S_["nu_hi"])
check("curvature: the threshold rises with nu", bool(np.all(np.diff(nu_m) > 0)), f"{np.round(nu_m, 1)}")
ratio_ = float(M["exThrNuTwo"]) / float(M["exSfirstBp"])
check("introduction: risk aversion of two roughly triples the threshold wedge", abs(ratio_ - 3) < 0.25, f"ratio {ratio_:.2f}")

# --------------------------------------------------------------------------- tables
# calibration: symbol, description, value of Arellano, Mateos-Planas, and Rios-Rull, calibrated value
cal = [("\\gamma_0", "variable default cost, level", AMR_PARAMS["g0"], f"{g0:.4f}"),
       ("\\gamma_1", "variable default cost, curvature", AMR_PARAMS["g1"], f"{g1:.2f}"),
       ("\\gamma_2", "fixed default cost, slope", AMR_PARAMS["g2"], f"{g2:.3f}"),
       ("y^*", "fixed default cost, threshold (level)", AMR_PARAMS["ystar"], f"{math.exp(zt):.3f}"),
       ("\\rho", "discount rate (factor $e^{-\\rho}$)", AMR_PARAMS["rho"], f"{rho:.3f}"),
       ("\\rho_z", "annual persistence of the endowment", AMR_PARAMS["rho_z"], f"{rz:.3f}"),
       ("\\sigma_\\epsilon", "annual innovation s.d.", AMR_PARAMS["sig_eps"], f"{se:.3f}")]
fixed = [("\\kappa", "recovery of defaulted coupons", f"{kap:.2f}", f"{kap:.2f}"),
         ("\\delta", "maturity rate ($-\\ln 0.88$)", f"{dl:.3f}", f"{dl:.3f}")]
open(os.path.join(OUT, "tab_calibration.tex"), "w", encoding="utf-8", newline="\n").write(
    "% generated by text_numbers.py from data/ou_main.npz -- do not edit\n"
    + "".join(f"\\quad ${s_}$ & {d_} & {trim(a_)} & {trim(b_)} \\\\\n" for s_, d_, a_, b_ in cal))
open(os.path.join(OUT, "tab_calibration_fixed.tex"), "w", encoding="utf-8", newline="\n").write(
    "% generated by text_numbers.py from data/ou_main.npz -- do not edit\n"
    + "".join(f"\\quad ${s_}$ & {d_} & {a_} & {b_} \\\\\n" for s_, d_, a_, b_ in fixed))
# fit: targeted moments (key, label, weight, digits) and untargeted moments (key, label, digits)
FIT = [("pd_freq", "Partial default: frequency", "1", 2), ("pd_mean_cond", "Partial default: mean $\\mid d>0$", "1", 2),
       ("pd_sd_cond", "Partial default: s.d. $\\mid d>0$", "1", 2), ("debt_y_mean", "Debt to output: mean", "1", 2),
       ("debt_y_sd", "Debt to output: s.d.", "1", 2), ("ds_y_mean", "Debt service to output: mean", "1", 3),
       ("ds_y_sd", "Debt service to output: s.d.", "1", 3), ("dd_y_mean", "Debt due to output: mean", "3", 3),
       ("spread_sd", "Spread: s.d.", "1.25", 3), ("y_persist", "Output: persistence", "1", 2), ("y_sd", "Output: s.d.", "1", 2)]
UNT = [("dc_y_mean_cond", "Defaulted coupons to output: mean $\\mid d>0$", 3), ("dc_y_sd_cond", "Defaulted coupons to output: s.d. $\\mid d>0$", 3),
       ("spread_mean", "Spread: mean", 3), ("corr_spread_y", "Spread: correlation with output", 2),
       ("corr_spread_debt", "Spread: correlation with debt", 2), ("c_sd_rel", "Consumption s.d. relative to output s.d.", 2)]
tab = {mm_.group(1): (float(mm_.group(2)), float(mm_.group(3)), float(mm_.group(4))) for mm_ in
       re.finditer(r"^\| (\w+)(?: \(untargeted\))? \| (-?[0-9.]+) \| (-?[0-9.]+) \| (-?[0-9.]+) \|", mm, re.M)}


def f_(v, k):
    """Number with k decimals, in math mode when negative."""
    s_ = f"{v:.{k}f}"
    return f"${s_}$" if v < 0 else s_


L = ["% generated by text_numbers.py from data/main_moments.md -- do not edit", "\\multicolumn{5}{l}{\\emph{Targeted}} \\\\"]
L += [f"{lab} & {wt} & {f_(tab[k][0], dig)} & {f_(tab[k][1], dig)} & {f_(tab[k][2], dig)} \\\\" for k, lab, wt, dig in FIT]
L += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Untargeted}} \\\\"]       # no loss row: the text reports the two losses
L += [f"{lab} & & {f_(tab[k][0], dig)} & {f_(tab[k][1], dig)} & {f_(tab[k][2], dig)} \\\\" for k, lab, dig in UNT]
open(os.path.join(OUT, "tab_fit.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
# robustness: the row of the main solution and one row per other grid
rr = [read(os.path.join(DATA, "main_row.tex")).strip()]
for k in ("fine", "nonuni", "wide", "nz51", "nz101", "z30", "z35"):
    rr.append(read(os.path.join(ROB, f"{k}_row.tex")).strip())
# drop the last field (final residuals): the residuals of the main solution are reported once, in the text
rr = [r_.rsplit("&", 1)[0].rstrip() + " \\\\" for r_ in rr]
# drop the column with the ergodic mass in the top tenth of the debt grid, after checking that it is zero in every row
cells_ = [[c_.strip() for c_ in r_.rstrip().rstrip('\\').split('&')] for r_ in rr]
assert all(float(c_[7]) == 0.0 for c_ in cells_), 'mass in the top tenth of the debt grid is not zero in every row'
rr = [' & '.join(c_[:7] + c_[8:]) + ' \\\\' for c_ in cells_]
open(os.path.join(OUT, "tab_robfloor_rows.tex"), "w", encoding="utf-8", newline="\n").write(
    "% generated by text_numbers.py from data/main_row.tex and data/robustness/*_row.tex -- do not edit\n" + "\n".join(rr) + "\n")
put("txNonuniNodes", re.search(r"nB (\d+)", read(os.path.join(ROB, "nonuni.log"))).group(1), "nodes of the non-uniform robustness grid")
hn = np.diff(np.load(os.path.join(ROB, "nonuni.npz"))["B"])
put("txNonuniStepMin", rnd(hn.min(), 4), "smallest step of the non-uniform robustness grid")
put("txNonuniStepMax", rnd(hn.max(), 3), "largest step of the non-uniform robustness grid")

# wider endowment ranges: nodes and ergodic mass below ln y*, and the s.d. of output
ysd_ = lambda md: float(re.search(r"\| y_sd \|[^|]*\|[^|]*\| ([0-9.]+)", read(md)).group(1))
put("txYsdMain", f"{ysd_(os.path.join(DATA, 'main_moments.md')):.3f}", "s.d. of output, main grid")
for tag_, nm_ in (("z30", "Zthree"), ("z35", "ZthreeFive")):
    w_ = np.load(os.path.join(ROB, f"{tag_}.npz"))
    gw_ = w_["g"] / w_["g"].sum(); low_ = w_["z"] < zt
    put(f"tx{nm_}Nodes", str(w_["z"].size), f"{tag_}: endowment nodes")
    put(f"tx{nm_}Below", str(int(low_.sum())), f"{tag_}: nodes below ln y*")
    put(f"tx{nm_}BelowPct", f"{100 * gw_[:, low_].sum():.1f}", f"{tag_}: ergodic mass below ln y*, percent")
    put(f"tx{nm_}Ysd", f"{ysd_(os.path.join(ROB, f'{tag_}.md')):.3f}", f"{tag_}: s.d. of output")
    check(f"{tag_}: no mass in full default or near the top of the debt grid",
          gw_[w_["d"] > 0.999].sum() < 5e-3 and gw_[w_["B"] > 0.9].sum() < 1e-8,
          f"full default {gw_[w_['d'] > 0.999].sum():.1e}, B > 0.9 {gw_[w_['B'] > 0.9].sum():.1e}")

# --------------------------------------------------------------------------- more numbers for the text
# the main solution: the region without a fixed cost, the ergodic density of debt, full default
nofix = z < zt
put("txNoFixedNodes", WORDS.get(int(nofix.sum()), str(int(nofix.sum()))), "endowment nodes below ln y*")
put("txNoFixedMassPct", f"{100 * w[:, nofix].sum():.1f}", "ergodic mass below ln y*, percent")
dens_B = w.sum(axis=1)
hump = (B > 0.5) & (B < 0.75)
put("txHumpB", f"{B[hump][dens_B[hump].argmax()]:.2f}", "debt at the second mode of the ergodic density")
put("txAtomZeroPct", f"{100 * w[0].sum():.0f}", "ergodic mass at zero debt, percent")
put("txFdMaxQ", f"{q[(d >= 1 - 1e-9) & (w > 1e-10)].max():.2f}", "highest price of debt in full default on the ergodic set (mass above 1e-10)")

# impulse responses
arr_ = kap * dc
put("txIrfPeakPct", f"{100 * dB[kp] / yb:.1f}", "debt peak, percent of mean output")
put("txIrfIssueYears", rnd(tt[np.where(di > arr_)[0].max()], 1), "years during which new issues exceed arrears")

# without the standard, on [0, 1] and on [0, 2]
nf = read(os.path.join(ROB, "nostd.md"))
w2 = read(os.path.join(ROB, "nostd2.md"))


def grab(pattern, txt):
    """The number matched by the group of pattern in txt."""
    return float(re.search(pattern, txt).group(1))


LOSS, SPREAD_SD = r"loss \(AMR weights\) \| \| \| ([0-9.]+)", r"\| spread_sd \|[^|]*\|[^|]*\| (-?[0-9.]+)"
MAX_B, MAX_Q, FULL = r"max B with mass ([0-9.]+)", r"max q ([0-9.]+)", r"mass full default ([0-9.e+-]+)"
put("txNoStdLossOne", f"{grab(LOSS, nf):.3f}", "loss without the standard, grid to 1")
put("txNoStdSpreadSdOne", f"{grab(SPREAD_SD, nf):.3f}", "spread s.d. without the standard, grid to 1")
put("txNoStdMaxQOne", f"{grab(MAX_Q, nf):.2f}", "highest price of full-default issuing, grid to 1")
put("txNoStdLossTwo", f"{grab(LOSS, w2):.3f}", "loss without the standard, grid to 2")
put("txNoStdSpreadSdTwo", f"{grab(SPREAD_SD, w2):.3f}", "spread s.d. without the standard, grid to 2")
put("txNoStdMaxBTwo", f"{grab(MAX_B, w2):.2f}", "largest debt with mass, grid to 2")
put("txNoStdMaxQTwo", f"{grab(MAX_Q, w2):.2f}", "highest price of full-default issuing, grid to 2")
put("txNoStdFdPctTwo", f"{100 * grab(FULL, w2):.1f}", "mass in full default, percent, grid to 2")
put("txNoStdTopPctOne", f"{100 * grab(r'mass top 10% ([0-9.e+-]+)', nf):.1f}", "mass in the top tenth of [0,1] without the standard, percent")

# the model with saving
sv = np.load(os.path.join(DATA, "ou_saving.npz"))
Bs_, zs_, gs_ = sv["B"], sv["z"], sv["g"] / sv["g"].sum()
neg_ = Bs_ < -1e-12
put("txSavNodes", str(Bs_.size), "debt nodes of the saving grid")
put("txSavResV", sci(float(sv["resV"])), "HJB residual of the saving solution")
put("txSavResq", sci(float(sv["resq"])), "pricing residual of the saving solution")
put("txSavBottomPct", rnd(100 * gs_[0].sum(), 2), "ergodic mass at the bottom node of the saving grid, percent")
put("txSavAssetsPct", f"{100 * gs_[neg_].sum():.0f}", "ergodic mass with assets, percent")
put("txSavAssetsMean", rnd(-(gs_[neg_] * (Bs_[neg_, None] / np.exp(zs_)[None, :])).sum() / gs_[neg_].sum(), 2), "mean assets to output, given assets")
Ss_ = sv["S"]; i0s_ = int(np.argmin(np.abs(Bs_)))
neg_drift_at_zero = Ss_[i0s_] < 0
put("txSavZsd", rnd(zs_[neg_drift_at_zero].min() / sdz, 1), "endowment above which debt falls through zero, in s.d.")

# the pessimistic approach to capacity in the deterministic model
ap_ = read(os.path.join(DATA, "approach_fig_out.txt"))
gc_ = read(os.path.join(DATA, "grid_check_out.txt"))
m_ = re.search(r"B\* ([0-9.]+), B_2 ([0-9.]+), stretch width [0-9.]+ \(([0-9.]+)% of output\)", ap_)
put("txApB", f"{float(m_.group(1)):.3f}", "capacity"); put("txApBtwo", f"{float(m_.group(2)):.3f}", "start of the default stretch")
put("txApWidthPct", rnd(float(m_.group(3)), 1), "width of the stretch, percent of output")
m_ = re.search(r"min price ([0-9.]+) at B [0-9.]+; spread there (\d+) bp", ap_)
put("txApMinQ", f"{float(m_.group(1)):.3f}", "lowest price on the approach"); put("txApSpreadBp", m_.group(2), "spread there, bp")
put("txApYears", rnd(float(re.search(r"time in default on the approach: ([0-9.]+) years", ap_).group(1)), 1), "years in default on the approach")
m_ = re.search(r"pessimistic ([0-9.]+) years, basin ([0-9.]+) years", ap_)
put("txApTimePess", rnd(float(m_.group(1)), 1), "years from zero debt to capacity, pessimistic"); put("txApTimeBasin", rnd(float(m_.group(2)), 1), "same, basin")
put("txApLossPct", rnd(float(re.search(r"consumption-equivalent loss ([0-9.]+)%", ap_).group(1)), 2), "value loss at zero debt, percent of consumption")
m_ = re.search(r"mirror: max\|R_V\| [0-9.e+-]+ \(away from B\*: ([0-9.e+-]+)\), max\|R_q\| [0-9.e+-]+ \(away: ([0-9.e+-]+)\); nodes with d>0 below B\*: (\d+)", gc_)
put("txApResV", sci(float(m_.group(1))), "pessimistic candidate: HJB residual away from B*"); put("txApResq", sci(float(m_.group(2))), "same, pricing")
put("txApNodes", m_.group(3), "nodes of the stretch on the fine grid")
m_ = re.search(r"basin: max\|R_V\| [0-9.e+-]+ \(away from B\*: ([0-9.e+-]+)\), max\|R_q\| [0-9.e+-]+ \(away: ([0-9.e+-]+)\)", gc_)
put("txApBasinResV", sci(float(m_.group(1))), "basin candidate: HJB residual away from B*"); put("txApBasinResq", sci(float(m_.group(2))), "same, pricing")
put("txApAfterNodes", WORDS.get(int(re.search(r"after: nodes with d>0 below B\*: (\d+)", gc_).group(1)), "?"), "default nodes below B* after time stepping from the pessimistic candidate")

# --------------------------------------------------------------------------- more statements of the text
check("the standard does not exclude partial-default steady states at the calibration", qr(dfirst) > qf, f"q_r(d_first) {qr(dfirst):.4f} vs standard {qf}")
check("debt rises for about ten years after the shock", round(tt[kp]) == 10, f"peak at {tt[kp]:.1f} y")
check("new issues are the larger flow for about five years", 4.5 < tt[np.where(di > arr_)[0].max()] < 5.9, f"{tt[np.where(di > arr_)[0].max()]:.1f} y")
check("the default intensity is still elevated after two decades", dd[-1] > 0.4 * dd[0], f"impact {dd[0]:.3f}, end {dd[-1]:.3f}")
check("without the standard the moments depend on the limit", abs(float(M["txNoStdLossTwo"]) - float(M["txNoStdLossOne"])) > 0.05, f"loss {M['txNoStdLossOne']} on [0,1], {M['txNoStdLossTwo']} on [0,2]")
check("the highest price of debt in full default on the ergodic set is below the standard", float(M["txFdMaxQ"]) < qf, f"{M['txFdMaxQ']} vs {qf}")

# --------------------------------------------------------------------------- write
with open(os.path.join(OUT, "numbers_text.tex"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("% generated by text_numbers.py -- do not edit; rerun the script\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{trim(v)}}}\n")
fails = sum("**FAIL**" in l_ for l_ in R)
R.insert(1, f"\n{len(M)} macros; {sum('(words)' in l_ for l_ in R)} statements in words checked, {fails} failed.\n")
open(os.path.join(LOGS, "numbers_text_report.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
print(f"{len(M)} macros, {fails} failed checks")
