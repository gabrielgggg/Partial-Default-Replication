"""Zero-drift locus and ergodic set of a saved diffusion solution.

Reports where the standard binds and where the ergodic mass lies, the zero-drift locus by endowment state, and,
with --irf, the issuance and arrears decomposition after the shock. Prints the report and, with --md, writes it
to a file.
Usage: python code/cal_check.py <solution.npz> [--md report.md] [--irf irf.npz]
"""
import argparse
import os

import numpy as np

from pdutil import load_calibrate2, result_from_npz, standard_diag, diag_line, locus_table

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--md", default="")
ap.add_argument("--irf", default="")
args = ap.parse_args()

# load the solution and compute the diagnostics
a = np.load(args.npz)
floor = float(a["qf"])
C = load_calibrate2(floor)
m, res = result_from_npz(a)
D = standard_diag(res, floor)

# report
lines = [f"# zero-drift locus and ergodic set of {os.path.basename(args.npz)}: floor {floor}, grid {a['B'].size} x {a['z'].size}, "
         f"residuals {res.resV:.1e} / {res.resq:.1e}\n",
         diag_line(D), "",
         f"E[B] {D['EB']:.4f}; P(d>0) {D['pdef']:.4f}; atom at 0 {D['atom0']:.4f}; mass with q below floor {D['mass_q_below_floor']:.4f}", "",
         "## zero-drift locus", "", locus_table(D)]
# impulse-response summary: debt peak, spread and d on impact, issuance against defaulted coupons
if args.irf and os.path.exists(args.irf):
    f = np.load(args.irf); tt = f["t"]; dt = tt[1] - tt[0]; yb = f["control_y"].mean()
    dB = f["shock_B"] - f["control_B"]; kp = int(np.argmax(dB))
    di = (f["shock_issue"] - f["control_issue"]) / yb; dc = (f["shock_defc"] - f["control_defc"]) / yb; dd = f["shock_d"] - f["control_d"]
    kap = float(a["x"][5])
    lines += ["", "## impulse response to z - 0.10", "",
              f"debt peak {100 * dB[kp] / yb:.2f}% of mean output at {tt[kp]:.1f} years; spread on impact {1e4 * (f['shock_spread'][0] - f['control_spread'][0]):.0f} bp; "
              f"d on impact {dd[0]:.3f}, at 2y {dd[int(2 / dt)]:.3f}, at 5y {dd[int(5 / dt)]:.3f}",
              f"cumulated to the peak: extra issuance {100 * di[:kp].sum() * dt:.2f}% of output, extra defaulted coupons {100 * dc[:kp].sum() * dt:.2f}% "
              f"(arrears {100 * kap * dc[:kp].sum() * dt:.2f}%); issuance exceeds defaulted coupons until year "
              f"{tt[np.where(di > dc)[0].max()] if (di > dc).any() else 0:.1f}"]
txt = "\n".join(lines) + "\n"
print(txt)
if args.md:
    open(args.md, "w", encoding="utf-8").write(txt)
