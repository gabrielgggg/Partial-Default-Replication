r"""text_numbers.py: the computed numbers quoted in the text of the paper, recomputed from saved
solution files and written as LaTeX macros, plus the calibration, fit, and robustness tables (Tables 1, 2, and C.1).

Sources:
  data/ou_best.npz, data/ou_main.npz, data/main_moments.md, data/main_check.md, data/main_row.tex,
  data/rob75/*_row.tex, data/rob75/nonuni.{npz,log}, data/irf.npz, data/det_fine15.npz,
  data/illustrations_fine15.log, data/jump_floor.npz, output/numbers.tex, data/compstat.log,
  data/floorcf/qf0.00.md, data/floorcf/qf0.00_wide2.md
  illustrative self-fulfilling example (its own parameters): example_closed_forms.py,
  data/wedge/rho0.060_illus.npz, data/pdss/{sweep_0.06,threshold_*}.txt, data/sfregion.npz

Outputs:
  output/numbers_text.tex       macros \tx... (main solution) and \ex... (illustrative example)
  output/tab_calibration.tex, output/tab_fit.tex, output/tab_robfloor_rows.tex   table bodies
  data/ou_main_residuals.npz    HJB and pricing residuals by node of the main solution
  logs/numbers_text_report.md   every value, with the checks of claims the text states in words

Each put(...) defines one macro. Each check(...) verifies a statement the text makes in words (not as a macro)
and records OK or FAIL in the report.

Usage: python code/text_numbers.py
"""
import glob
import math
import os
import re
import sys

import numpy as np

from paths import DATA, OUT, LOGS

M = {}          # macro name -> LaTeX value
R = []          # report lines
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
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
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(float(v), 9))).quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP))


def trim(s):
    """Drop trailing zeros of a plain decimal (0.100 -> 0.1, 1.00 -> 1). Anything else is unchanged."""
    return s.rstrip("0").rstrip(".") if re.fullmatch(r"-?\d+\.\d+", s) else s


# --------------------------------------------------------------------------- parameters and closed forms
# calibrated parameters, the first default d_first, the stationary spread there, the spread at the floor, and the wedge
ob = np.load(os.path.join(DATA, "ou_best.npz"))
x = ob["x"].astype(float)
g0, g1, g2, zt, rho, kap, dl, rz, se = x
r = lam = math.log(1.04)
qf = float(ob["qf"])
cs = dl + lam
mu = -math.log(rz)
sdz = se / math.sqrt(1 - rz ** 2)
F0 = 1 - g2 * (1 - math.exp(zt))
dfirst = ((1 - F0) / ((g1 - 1) * g0 * F0)) ** (1 / g1)
xi = lambda d: r + dl - kap * cs * d
qr = lambda d: (1 - d) * cs / xi(d)
s_first = cs * (1 / qr(dfirst) - 1)
floor_spread = cs * (1 / qf - 1)
wedge = rho - r
R += ["# Numbers in the text", "", "| macro | value | source / note |", "|---|---|---|"]
put("txFirstRatio", f"{(1 - F0) / ((g1 - 1) * g0 * F0):.2f}", "bracket of (17) at z = 0")
put("txGammaOneShort", f"{g1:.1f}", "gamma_1, one decimal")
put("txSfirstBp", thousands(1e4 * s_first), "stationary spread at the first default")
put("txFloorSpreadBp", thousands(1e4 * floor_spread), "spread at the floor")
put("txFloorSpreadRound", thousands(round(1e4 * floor_spread, -1)), "spread at the floor, to 10 bp")
put("txWedgeBp", f"{1e4 * wedge:.0f}", "rho - r")
put("txRNThresholdBp", thousands(1e4 * s_first), "risk-neutral threshold Theta* = s_first (gamma_1 > 2: min of Theta, attained at d_first here)")
put("txQrFirst", f"{qr(dfirst):.3f}", "q_r(d_first)")
put("txChainAutocorr", f"{math.exp(-mu / 2):.2f}", "three-state chain: exp(-eta), eta = mu/2")
put("txOUAutocorr", f"{rz:.2f}", "rho_z")
put("txZlow", None, "")    # set below from the grid
put("txLnYstar", f"{zt:.3f}", "ln y*")

# the risk-neutral threshold Theta attains its minimum at d_first (the text uses s_first as the threshold)
ds = np.linspace(dfirst, 0.99, 20001)
Theta = (1 - kap) * cs * ds / (1 - ds) * (1 + (g1 - 1) * (dfirst / ds) ** g1) / g1
check("risk-neutral threshold equals s_first at the calibration", abs(Theta.min() - s_first) < 1e-6,
      f"min Theta {1e4 * Theta.min():.1f} bp at d {ds[Theta.argmin()]:.3f}; s_first {1e4 * s_first:.1f} bp")
check("the standard excludes partial-default steady states", qr(dfirst) < qf, f"q_r(d_first) {qr(dfirst):.4f} vs floor {qf}")

# --------------------------------------------------------------------------- main solution
# endowment grid, prices at zero debt, default probability, where the standard binds, and the threshold at z = 0
a = np.load(os.path.join(DATA, "ou_main.npz"))
B, z, q, S, c, d, g = (a[k] for k in ("B", "z", "q", "S", "c", "d", "g"))
w = g / g.sum()
M["txZlow"] = f"{z[0]:.2f}"
R[[i for i, l in enumerate(R) if "txZlow" in l][0]] = f"| `\\txZlow` | {z[0]:.2f} | lowest endowment node |"
check("only the lowest node lies below the fixed-cost threshold", z[0] < zt < z[1], f"z0 {z[0]:.4f} < ln y* {zt:.4f} < z1 {z[1]:.4f}")
put("txLowNodeMassPct", f"{100 * w[:, 0].sum():.1f}", "ergodic mass on the lowest node, percent")
put("txQzeroMin", f"{q[0].min():.2f}", "price at zero debt, min over z")
put("txQzeroMax", f"{q[0].max():.2f}", "price at zero debt, max over z")
put("txPdefInst", f"{w[d > 1e-9].sum():.2f}", "instantaneous probability of default")
# binding nodes, and default there relative to the intensity implied by the market price
ad = dl - kap * cs * d
ell = S + ad * B[:, None]
Fz = 1 - g2 * (np.exp(z) - math.exp(zt)) * (np.exp(z) >= math.exp(zt))
bind = (q < qf) & (np.abs(ell) < 1e-7) & (d > 0)
dint = np.minimum(1, ((1 - kap * q) * cs * B[:, None] / (g0 * g1 * Fz * np.exp(z))) ** (1 / (g1 - 1)))
with np.errstate(invalid="ignore", divide="ignore"):
    ex = d / dint - 1
bm = bind & (w > 0)
ex_mean = (ex[bm] * w[bm]).sum() / w[bm].sum()
ex_max = np.nanmax(ex[bm])
check("standard binds on just under one percent of the mass", 0.009 <= w[bind].sum() < 0.01, f"{100 * w[bind].sum():.2f}%")
bmass = (w * bind).sum(axis=0)
check("binding mass all more than one s.d. below the mean endowment", bmass[z >= -sdz].sum() < 1e-3 * bmass.sum(),
      f"share of binding mass at z >= -1 s.d.: {bmass[z >= -sdz].sum() / bmass.sum():.1e}; "
      f"highest z with binding mass > 1e-6: {z[bmass > 1e-6].max():+.3f}, s.d. {sdz:.3f}")
check("default exceeds the market-price rule by two percent on average", round(100 * ex_mean) == 2, f"{100 * ex_mean:.1f}%")
check("and by up to about four percent", round(100 * ex_max) == 4, f"{100 * ex_max:.1f}% (nodes with positive mass)")
check("full default not reached on the grid", (d < 1 - 1e-9).all(), f"max d {d.max():.3f}")
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
put("txLocusMax", rnd(locB.max(), 2), f"maximum of the zero-drift locus, at z {locz[locB.argmax()]:+.3f}")
put("txLocusLow", rnd(locB[locz.argmin()], 2), "zero-drift debt at the lowest node")
check("locus peaks about one s.d. below the mean", abs(locz[locB.argmax()] + sdz) < 0.03, f"peak at z {locz[locB.argmax()]:+.3f}, -1 s.d. {-sdz:+.3f}")
nolocus = [float(m_.group(1)) for m_ in re.finditer(r"^\| ([+-]\d\.\d+) \| nan", mc, re.M)]
check("above about one s.d. above the mean the drift is negative everywhere", abs(min(nolocus) - sdz) < 0.03,
      f"no zero-drift point from z {min(nolocus):+.3f}, +1 s.d. {sdz:+.3f}")
put("txErgMaxB", f"{float(re.search(r'max B with mass ([0-9.]+)', mc).group(1)):.2f}", "largest debt with ergodic mass")

# HJB and pricing residuals by node, stored; their maxima on and off the floor locus (floor_residuals.py)
from generator import model_from_solution
m_, par_, A_, S_, c_, d_ = model_from_solution(a)
u_ = c_ ** (1 - par_.nu) / (1 - par_.nu)
RV = (par_.rho * a["V"].ravel() - u_.ravel() - A_ @ a["V"].ravel()).reshape(B.size, z.size)
Rq = (par_.xi(d_.ravel()) * a["q"].ravel() - (1 - d_.ravel()) * cs - A_ @ a["q"].ravel()).reshape(B.size, z.size)
np.savez(os.path.join(DATA, "ou_main_residuals.npz"), RV=RV, Rq=Rq, B=B, z=z)
import floor_residuals as FR
rep = FR.report()
off_V = max(rep["zero"]["RV"], rep["else"]["RV"]); off_q = max(rep["zero"]["Rq"], rep["else"]["Rq"])
put("txFloorBand", f"{FR.BAND:.2f}", "floor locus: |q - q_floor| below this, or a floor crossing between debt neighbors")
put("txOffResV", sci(off_V), "largest HJB residual off the floor locus")
put("txOffResq", sci(off_q), "largest pricing residual off the floor locus (at the zero-drift locus)")
put("txElseResq", sci(rep["else"]["Rq"]), "largest pricing residual off both the floor and the zero-drift locus")
put("txFloorResV", sci(rep["floor"]["RV"]), "largest HJB residual on the floor locus")
put("txFloorResq", sci(rep["floor"]["Rq"]), "largest pricing residual on the floor locus")
put("txFloorMassPct", f"{100 * rep['floor']['mass']:.1f}", "ergodic mass on the floor locus, percent")
check("off the floor locus the pricing residual peaks at the zero-drift locus", rep["zero"]["Rq"] >= rep["else"]["Rq"],
      f"zero-drift locus {rep['zero']['Rq']:.1e}, elsewhere {rep['else']['Rq']:.1e}")

# simulated moments
mm = read(os.path.join(DATA, "main_moments.md"))
put("txPdFreq", f"{float(re.search(r'\| pd_freq \|[^|]*\|[^|]*\| ([0-9.]+)', mm).group(1)):.2f}", "annual frequency of partial default")

# --------------------------------------------------------------------------- impulse responses
# issuance and arrears cumulated to the debt peak, spread on impact, and the timing statements
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
yrs_iss = tt[np.where(di > dc)[0].max()]
check("debt rises for about six years, by some six percent of output", round(tt[kp]) == 6 and round(100 * dB[kp] / yb) == 6,
      f"peak {100 * dB[kp] / yb:.2f}% at {tt[kp]:.1f} y")
check("new issues exceed defaulted coupons for five and a half years", abs(yrs_iss - 5.5) < 0.05, f"{yrs_iss:.2f} y")
check("default intensity elevated for a decade", dd[int(10 / dt)] > 0.5 * dd[0], f"impact {dd[0]:.3f}, 10 y {dd[int(10 / dt)]:.3f}")

# --------------------------------------------------------------------------- the deterministic illustration
# cube-root expansion at B*: coefficients, errors of the two-term laws, residuals, steady-state node, return times
df15 = np.load(os.path.join(DATA, "det_fine15.npz"))
Bd, Sd, qd = df15["B"], df15["S"], df15["q"]
beta, alpha, Sig1, Q1, Bs = (float(df15[k]) for k in ("beta", "alpha", "Sig1", "Q1", "Bpd"))
put("txSigOverBeta", f"{Sig1 / beta:.1f}", "Sigma_1 / beta")


def err(lo, hi):
    """Max relative errors of the two-term drift and price-gap expansions for B - B* in [lo, hi]."""
    k = (Bd - Bs >= lo) & (Bd - Bs <= hi); xx = Bd[k] - Bs
    return (np.abs(Sd[k] / (-beta * xx ** (1 / 3) - Sig1 * xx ** (2 / 3)) - 1).max(),
            np.abs((1 - qd[k]) / (alpha * xx ** (2 / 3) + Q1 * xx) - 1).max())


put("txCubeDriftErr", f"{math.ceil(100 * err(0.01, 0.05)[0])}", "max error of the two-term drift on x in [0.01, 0.05], percent")
put("txCubePriceErrNear", f"{math.ceil(100 * err(0.02, 0.05)[1])}", "max error of the two-term price gap on [0.02, 0.05], percent")
put("txCubePriceErrFar", f"{math.ceil(100 * err(0.02, 0.2)[1])}", "max error of the two-term price gap on [0.02, 0.2], percent")
log15 = read(os.path.join(DATA, "illustrations_fine15.log"))
mres = re.search(r"residuals ([0-9.e+-]+) / ([0-9.e+-]+)", log15)
put("txDetResV", sci(float(mres.group(1))), "final HJB residual, [0,1.5] grid")
put("txDetResq", sci(float(mres.group(2))), "final pricing residual, [0,1.5] grid")
check("second drift term half the first at x = 0.1", abs(abs(Sig1) * 0.1 ** (1 / 3) / beta - 0.5) < 0.1,
      f"ratio {abs(Sig1) * 0.1 ** (1 / 3) / beta:.2f}")
gd = df15["g"]
iss = int(np.argmax(gd))
put("txDetSSNode", f"{Bd[iss]:.3f}", "node where the drift vanishes (the atom of the stationary distribution), [0,1.5] grid")
check("the drift vanishes at the atom node and is positive just below it", abs(Sd[iss]) < 1e-10 and Sd[iss - 1] > 0,
      f"S {Sd[iss - 1]:+.5f} / {Sd[iss]:+.5f} / {Sd[iss + 1]:+.5f} at {Bd[iss - 1]:.3f} / {Bd[iss]:.3f} / {Bd[iss + 1]:.3f}; mass {gd[iss] / gd.sum():.6f}")


def return_time(x0):
    """Time to reach the zero-drift node from B* + x0 along the computed drift."""
    k0 = int(np.argmin(np.abs(Bd - (Bs + x0))))
    sel = np.arange(iss + 1, k0 + 1)
    return float(np.sum((Bd[sel] - Bd[sel - 1]) / np.abs(Sd[sel])))


lead_time = lambda x0: 3 / (2 * beta) * x0 ** (2 / 3)   # leading-order return time, integral of dx / (beta x^{1/3})
put("txReturnOnePct", rnd(lead_time(0.01), 1), "years to return to capacity from 1 percent of output above it, leading order 3x^{2/3}/(2 beta)")
put("txReturnTenPct", rnd(lead_time(0.1), 1), "years to return to capacity from 10 percent of output above it, leading order")
check("the computed drift returns more slowly than the leading term (Sigma_1 < 0)",
      return_time(0.01) > lead_time(0.01) and return_time(0.1) > lead_time(0.1),
      f"computed {return_time(0.01):.2f} / {return_time(0.1):.2f} y vs leading {lead_time(0.01):.2f} / {lead_time(0.1):.2f} y")

# --------------------------------------------------------------------------- the three-state chain
# mass and price of the high state at the low state's threshold node, and the Euler equation there
jf = np.load(os.path.join(DATA, "jump_floor.npz"))
wj = jf["g"] / jf["g"].sum(); Bj = jf["B"]
iL = int(np.argmin(np.abs(Bj - float(re.search(r"jumpStatL\}\{([0-9.]+)", read(os.path.join(OUT, "numbers.tex"))).group(1)))))
put("txJumpMassHigh", f"{wj[iL, 2]:.2f}", "ergodic mass of the high state at the low state's threshold node")
check("price of the high state there within one percent of q_rf", jf["q"][iL, 2] > 0.99, f"q {jf['q'][iL, 2]:.4f}")
# the Euler equation (16) at the high state's atom: H -> M at eta, H -> L at eps
rhoj, lamj, nuj, etaj, epsj = (float(jf[k]) for k in ("rho", "lam", "nu", "eta", "eps"))
upj = lambda x_: x_ ** (-nuj)
cL_, cM_, cH_ = jf["c"][iL]
imp_ = (rhoj - lamj) * upj(cH_)
prec_ = etaj * (upj(cM_) - upj(cH_)) + epsj * (upj(cL_) - upj(cH_))
put("txJumpEulerImp", rnd(imp_, 3), "high state at the low state's threshold node: impatience term (rho - lambda) u'(c_H)")
put("txJumpEulerPrec", rnd(prec_, 3), "same node: precautionary term eta (u'(c_M) - u'(c_H)) + eps (u'(c_L) - u'(c_H))")
check("impatience exceeds precaution at the high state's atom, so the kink pins it", imp_ > prec_ > 0, f"{imp_:.4f} vs {prec_:.4f}")

# --------------------------------------------------------------------------- comparative statics
# from data/compstat.log: first default node and steady state relative to B_pd in grid steps, and any stalled case
cl = read(os.path.join(DATA, "compstat.log"))
cs_rows = re.findall(r"^(\S.*?)\s+Bpd ([0-9.]+) B\* node ([0-9.]+) first node ([0-9.]+) d_first ([0-9.]+) / ([0-9.]+).*?conv (\w+) res ([0-9.e+-]+)/([0-9.e+-]+) (\d+) steps", cl, re.M)
h = 0.005
first_off = [(lab.strip(), (float(bp) - float(fn)) / h) for lab, bp, bn, fn, *_ in cs_rows]
ss_off = [(lab.strip(), (float(bp) - float(bn)) / h) for lab, bp, bn, *_ in cs_rows]
dgap = max(abs(float(r_[4]) - float(r_[5])) for r_ in cs_rows)
put("txCompDgap", f"{math.ceil(dgap * 1000) / 1000:.3f}", "max |d_first - computed d at the first default node|")
worst = max(first_off, key=lambda t: abs(t[1]))
put("txCompFirstWorst", f"{worst[1]:.1f}", f"first default node below B_pd, worst case ({worst[0]}), grid steps")
others = [abs(v) for lab, v in first_off if lab != worst[0]]
check("first default node within one grid step of B_pd in every other case", max(others) <= 1.0 + 1e-9, f"max {max(others):.2f} steps")
oth_ss = [v for lab, v in ss_off if lab != worst[0]]
put("txCompSSMin", f"{min(oth_ss):.1f}", "steady state below B_pd, grid steps, min over the other cases")
put("txCompSSMax", f"{max(oth_ss):.1f}", "same, max")
put("txCompSSWorst", f"{dict(ss_off)[worst[0]]:.1f}", f"steady state below B_pd at {worst[0]}, grid steps")
check("last positive-drift node two nodes below the first default node in every case",
      all(abs((float(fn) - float(bn)) / h - 2) < 1e-6 for lab, bp, bn, fn, *_ in cs_rows), "")
cs_bpd = {lab.strip(): float(bp) for lab, bp, *_ in cs_rows}
put("txCsCapDeltaLo", f"{cs_bpd['delta = 0.12']:.3f}", "B_pd(0, q_rf) at delta = 0.12 (Table B.1)")
put("txCsCapDeltaHi", f"{cs_bpd['delta = 0.22']:.3f}", "B_pd(0, q_rf) at delta = 0.22 (Table B.1)")
nc = [r_ for r_ in cs_rows if r_[6] != "True"]
for r_ in nc:
    put("txCompStallLabel", r_[0].strip(), "the case that did not converge")
    put("txCompStallResq", sci(float(r_[8])), "its final pricing residual")
    put("txCompStallSteps", thousands(float(r_[9])), "its steps")

# --------------------------------------------------------------------------- without the standard
# full-default issuing, spread volatility, and loss with q_floor = 0, and the check that the grid to 2 suffices
w2 = read(os.path.join(DATA, "floorcf", "qf0.00_wide2.md"))
nf = read(os.path.join(DATA, "floorcf", "qf0.00.md"))
put("txNoStdMaxQ", f"{float(re.search(r'max q ([0-9.]+)', w2).group(1)):.2f}", "highest price of full-default issuing, grid to 2")
put("txNoStdMaxQRound", f"{float(re.search(r'max q ([0-9.]+)', w2).group(1)):.1f}", "same, one decimal")
mfd = float(re.search(r"full-default issuing nodes \d+ \(max q [0-9.]+, mass ([0-9.e+-]+)\)", w2).group(1))
put("txNoStdMass", sci(mfd), "ergodic mass on full-default issuing nodes, grid to 2")
put("txNoStdResq", rnd(float(re.search(r"resq ([0-9.e+-]+)", w2).group(1)), 1), "final pricing residual, grid to 2")
put("txNoStdSpreadSd", f"{float(re.search(r'\| spread_sd \|[^|]*\|[^|]*\| ([0-9.]+)', w2).group(1)):.3f}", "spread s.d. without the standard, grid to 2")
put("txNoStdLoss", f"{float(re.search(r'loss \(AMR weights\) \| \| \| ([0-9.]+)', w2).group(1)):.3f}", "loss without the standard, grid to 2")
put("txSpreadSd", f"{float(re.search(r'\| spread_sd \|[^|]*\|[^|]*\| ([0-9.]+)', mm).group(1)):.3f}", "spread s.d. with the standard")


def mom(txt):
    """Model moments from a moments table in markdown."""
    return {k: float(v) for k, v in re.findall(r"^\| (\w+)(?: \(untargeted\))? \|[^|]*\|[^|]*\| (-?[0-9.]+) \|", txt, re.M)}


m2, m1 = mom(w2), mom(nf)
dmax = max(abs(m2[k] - m1[k]) for k in m1 if k in m2)
check("moments on [0,2] within a few thousandths of [0,1], without the standard", dmax < 0.01, f"max gap {dmax:.4f}")

# --------------------------------------------------------------------------- illustrative self-fulfilling example
# closed forms of the example (example_closed_forms.py)
exm = {}
try:
    import example_closed_forms as ecf  # noqa: E402  (prints its own summary)
    put("exDfirst", f"{ecf.dfirst:.2f}", "example: first default")
    put("exSfirstBp", f"{1e4 * ecf.s(ecf.dfirst):.0f}", "example: s_first")
    put("exOmegaBp", f"{1e4 * ecf.Omega(ecf.dfirst):.0f}", "example: Omega(d_first)")
    put("exGxi", f"{(ecf.g1 - 1) * ecf.xi(ecf.dfirst):.3f}", "example: (gamma_1 - 1) xi(d_first)")
    put("exWedgeBp", f"{1e4 * (ecf.rho - ecf.r):.0f}", "example: wedge")
    okd = np.array([ecf.Theta(dd_) <= ecf.rho - ecf.r < ecf.Omega(dd_) and ecf.qr(dd_) >= ecf.qf for dd_ in ecf.ds])
    put("exRNBlo", rnd(ecf.Bd(ecf.ds[okd].min()), 2), "example: risk-neutral steady states, lowest debt")
    put("exRNBhi", rnd(ecf.Bd(ecf.ds[okd].max()), 2), "example: same, highest debt")
except Exception as e:  # noqa: BLE001
    R.append(f"| (example closed forms) | **FAIL** | {e} |")
# the two steady states of Figure 5, selected as in make_figures.fig_multiplicity
wf = os.path.join(DATA, "wedge", "rho0.060_illus.npz")
if os.path.exists(wf):
    W = np.load(wf)
    metas = {k: W[k] for k in W.files if k.endswith("_meta")}
    nd = [(k, v) for k, v in metas.items() if v[1] > 0.999 and v[2] < 1e-6]
    pdm = [(k, v) for k, v in metas.items() if v[2] > 0.15]
    if nd:
        put("exNoDefB", rnd(nd[0][1][0], 2), "example: no-default steady state")
    if pdm:
        k_, v_ = max(pdm, key=lambda t: t[1][2])
        put("exPdB", rnd(v_[0], 2), f"example: partial-default steady state ({k_})")
        put("exPdD", f"{100 * v_[2]:.0f}", "example: its default intensity, percent")
        put("exPdQ", rnd(v_[1], 2), "example: its price")
        put("exPdSpreadBp", f"{1e4 * ecf.cs * (1 / v_[1] - 1):.0f}", "example: its spread, (delta+lam)(1/q-1)")
        dser = W[k_.replace("_meta", "_d")]
        i0 = int(np.where(dser > 1e-9)[0][0])
        put("exPdStartB", rnd(W['B'][i0], 2), "example: debt at which default starts on that branch")
# continuum of steady states reached from zero debt, and the threshold wedges by curvature nu
sw = os.path.join(DATA, "pdss", "sweep_0.06.txt")
if os.path.exists(sw):
    t_ = read(sw)
    mrg = re.search(r"B0 in \[([0-9.]+), ([0-9.]+)\]", t_)
    if mrg:
        put("exContLo", rnd(float(mrg.group(1)), 2), "example: continuum reached from zero debt, lowest")
        put("exContHi", rnd(float(mrg.group(2)), 2), "example: same, highest")
for nu, tag, nm in ((2.0, "threshold_nu2.0", "exThrNuTwo"), (1.0, "threshold_nu1.0", "exThrNuOne"),
                    (0.25, "threshold_wide_nu0.25", "exThrNuQuarter"), (0.1, "threshold_wide_nu0.1", "exThrNuTenth")):
    f_ = os.path.join(DATA, "pdss", f"{tag}.txt")
    if os.path.exists(f_):
        mt = re.search(r"threshold wedge in \[([0-9.]+), ([0-9.]+)\]", read(f_))
        if mt:
            put(nm, f"{(float(mt.group(1)) + float(mt.group(2))) / 2:.0f}", f"example: smallest reachable wedge, nu {nu} (bracket {mt.group(1)}-{mt.group(2)})")
# region with shocks (Figure B.4): where it closes and how its boundaries move
sfr = os.path.join(DATA, "sfregion.npz")
if os.path.exists(sfr):
    S_ = np.load(sfr)
    lo_, hi_, sd_ = S_["w_lo"], S_["w_hi"], S_["sd"]
    okl = np.isfinite(lo_)
    put("exSfCloseSd", f"{100 * S_['sd_close'][0]:.0f}",
        f"shocks: s.d. of log endowment (%) at which the region closes (between {100*S_['sd_close'][0]:.2f} and {100*S_['sd_close'][1]:.2f})")
    check("shocks: the lower boundary rises with volatility", bool(np.all(np.diff(lo_[okl]) > 0)), f"{np.round(lo_[okl], 1)}")
    fh_ = np.isfinite(hi_) & okl
    check("shocks: the upper boundary falls with volatility", bool(np.all(np.diff(hi_[fh_]) < 0)), f"{np.round(hi_[fh_], 1)}")
    check("shocks: the region starts at s_first", abs(lo_[0] - float(S_["s_first"])) < 0.5, f"{lo_[0]:.1f}")
    nu_m = 0.5 * (S_["nu_lo"] + S_["nu_hi"])
    check("curvature: the threshold rises with nu", bool(np.all(np.diff(nu_m) > 0)), f"{np.round(nu_m, 1)}")
# statements in the introduction
if "exThrNuTwo" in M and "exSfirstBp" in M:
    ratio_ = float(M["exThrNuTwo"]) / float(M["exSfirstBp"])
    check("introduction: risk aversion of two roughly triples the threshold wedge", abs(ratio_ - 3) < 0.25, f"ratio {ratio_:.2f}")
check("introduction: new issues add several times as much to debt as arrears, to the peak",
      float(M["txIrfIssue"]) / float(M["txIrfArrears"]) > 3, f"ratio {float(M['txIrfIssue']) / float(M['txIrfArrears']):.1f}")

# --------------------------------------------------------------------------- tables
# Table 1: calibration (target value, then the calibrated value)
cal = [("\\gamma_0", "variable default cost, level", "0.0476", f"{g0:.4f}"),
       ("\\gamma_1", "variable default cost, curvature", "2", f"{g1:.2f}"),
       ("\\gamma_2", "fixed default cost, slope", "0.12", f"{g2:.3f}"),
       ("y^*", "fixed default cost, threshold (level)", "0.938", f"{math.exp(zt):.3f}"),
       ("\\rho", "discount rate (factor $e^{-\\rho}$)", "0.047", f"{rho:.3f}"),
       ("\\kappa", "recovery of defaulted coupons", "0.7", f"{kap:.3f}"),
       ("\\delta", "maturity rate (survival $1-\\delta$)", "0.12", f"{dl:.3f}"),
       ("\\rho_z", "annual persistence of the endowment", "0.875", f"{rz:.3f}"),
       ("\\sigma_\\epsilon", "annual innovation s.d.", "0.052", f"{se:.3f}")]
open(os.path.join(OUT, "tab_calibration.tex"), "w", encoding="utf-8").write(
    "% generated by text_numbers.py from data/ou_best.npz -- do not edit\n"
    + "".join(f"\\quad ${s_}$ & {d_} & {trim(a_)} & {trim(b_)} \\\\\n" for s_, d_, a_, b_ in cal))
# Table 2: targeted moments (label, weight, digits) and untargeted moments, read from data/main_moments.md
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
loss_m = float(re.search(r"loss \(AMR weights\) \| \| \| ([0-9.]+)", mm).group(1))


def f_(v, k):
    """Number with k decimals, in math mode when negative."""
    s_ = f"{v:.{k}f}"
    return f"${s_}$" if v < 0 else s_


L = ["% generated by text_numbers.py from data/main_moments.md -- do not edit", "\\multicolumn{5}{l}{\\emph{Targeted}} \\\\"]
L += [f"{lab} & {wt} & {f_(tab[k][0], dig)} & {f_(tab[k][1], dig)} & {f_(tab[k][2], dig)} \\\\" for k, lab, wt, dig in FIT]
L += ["\\midrule", f"Loss & & & \\calLossAMR & {loss_m:.3f} \\\\", "\\midrule", "\\multicolumn{5}{l}{\\emph{Untargeted}} \\\\"]
L += [f"{lab} & & {f_(tab[k][0], dig)} & {f_(tab[k][1], dig)} & {f_(tab[k][2], dig)} \\\\" for k, lab, dig in UNT]
open(os.path.join(OUT, "tab_fit.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
# Table C.1: robustness rows (main solution, finer, non-uniform, and wider grids), and the non-uniform grid steps
rr = [read(os.path.join(DATA, "main_row.tex")).strip()]
for k in ("fine", "nonuni", "wide", "z30", "z35"):
    rr.append(read(os.path.join(DATA, "rob75", f"{k}_row.tex")).strip())
# drop the last field (final residuals): the text reports the residuals of the main solution in Section 4.2
rr = [r_.rsplit("&", 1)[0].rstrip() + " \\\\" for r_ in rr]
open(os.path.join(OUT, "tab_robfloor_rows.tex"), "w", encoding="utf-8").write(
    "% generated by text_numbers.py from data/main_row.tex and data/rob75/*_row.tex -- do not edit\n" + "\n".join(rr) + "\n")
nl_ = re.search(r"nB (\d+)", read(os.path.join(DATA, "rob75", "nonuni.log")))
put("txNonuniNodes", nl_.group(1) if nl_ else "--", "nodes of the non-uniform robustness grid")
hn = np.diff(np.load(os.path.join(DATA, "rob75", "nonuni.npz"))["B"])
put("txNonuniStepMin", rnd(hn.min(), 4), "smallest step of the non-uniform robustness grid")
put("txNonuniStepMax", rnd(hn.max(), 3), "largest step of the non-uniform robustness grid")

# wider endowment ranges (robustness.py z30, z35): nodes and ergodic mass below ln y*, and the s.d. of output
ysd_ = lambda md: float(re.search(r"\| y_sd \|[^|]*\|[^|]*\| ([0-9.]+)", read(md)).group(1))
put("txYsdMain", f"{ysd_(os.path.join(DATA, 'main_moments.md')):.3f}", "s.d. of output, main grid")
for tag_, nm_ in (("z30", "Zthree"), ("z35", "ZthreeFive")):
    w_ = np.load(os.path.join(DATA, "rob75", f"{tag_}.npz"))
    gw_ = w_["g"] / w_["g"].sum(); low_ = w_["z"] < zt
    put(f"tx{nm_}Nodes", str(w_["z"].size), f"{tag_}: endowment nodes")
    put(f"tx{nm_}Below", str(int(low_.sum())), f"{tag_}: nodes below ln y*")
    put(f"tx{nm_}BelowPct", f"{100 * gw_[:, low_].sum():.1f}", f"{tag_}: ergodic mass below ln y*, percent")
    put(f"tx{nm_}Ysd", f"{ysd_(os.path.join(DATA, 'rob75', f'{tag_}.md')):.3f}", f"{tag_}: s.d. of output")
    check(f"{tag_}: no mass in full default or near the top of the debt grid",
          gw_[w_["d"] > 0.999].sum() < 1e-8 and gw_[w_["B"] > 0.9].sum() < 1e-8,
          f"full default {gw_[w_['d'] > 0.999].sum():.1e}, B > 0.9 {gw_[w_['B'] > 0.9].sum():.1e}")

# --------------------------------------------------------------------------- write
# macros file and the report; every macro must have a value
miss = [k for k, v in M.items() if v is None]
assert not miss, miss
with open(os.path.join(OUT, "numbers_text.tex"), "w", encoding="utf-8") as fh:
    fh.write("% generated by code/text_numbers.py\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{trim(v)}}}\n")
fails = sum("**FAIL**" in l_ for l_ in R)
R.insert(1, f"\n{len(M)} macros; {sum('(words)' in l_ for l_ in R)} claims in words checked, {fails} failed.\n")
open(os.path.join(LOGS, "numbers_text_report.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
print(f"{len(M)} macros, {fails} failed checks")
