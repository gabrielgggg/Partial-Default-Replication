"""Impulse responses of the main solution to z - 0.10, by the forward equation.

Reads data/ou_main.npz. Writes data/irf.npz (read by make_figures.py) and data/irf_summary.md with the
issuance and arrears decomposition.
Usage: python code/irf.py
"""
import os

import numpy as np

from pdutil import DATA

a = np.load(os.path.join(DATA, "ou_main.npz"))
from generator import expected_paths

# expected paths after the shock and in the control, saved for the figure
tt, out = expected_paths(a)
np.savez(os.path.join(DATA, "irf.npz"), t=tt, **{f"{s}_{k}": v for s in out for k, v in out[s].items()})
# differences from the control, scaled by mean output
sh, co = out["shock"], out["control"]; yb = co["y"].mean(); dt = tt[1] - tt[0]
dB = sh["B"] - co["B"]; kp = int(np.argmax(dB))
di = (sh["issue"] - co["issue"]) / yb; dc = (sh["defc"] - co["defc"]) / yb; dd = sh["d"] - co["d"]
kap = float(a["x"][5])
# summary and the table by year
lines = [f"# impulse response to z - 0.10 at the main solution (floor {float(a['qf'])})", "",
         f"output on impact {100 * (sh['y'][0] - co['y'][0]) / yb:.2f}%; consumption {100 * (sh['c'][0] - co['c'][0]) / yb:.2f}%",
         f"debt peak {100 * dB[kp] / yb:.2f}% of mean output at {tt[kp]:.1f} years; spread on impact {1e4 * (sh['spread'][0] - co['spread'][0]):.0f} bp",
         f"default intensity: impact {dd[0]:.3f}, 2y {dd[int(2 / dt)]:.3f}, 5y {dd[int(5 / dt)]:.3f}, 10y {dd[int(10 / dt)]:.3f}",
         f"cumulated to the peak: extra issuance {100 * di[:kp].sum() * dt:.2f}% of output; extra defaulted coupons {100 * dc[:kp].sum() * dt:.2f}% "
         f"(arrears {100 * kap * dc[:kp].sum() * dt:.2f}%)",
         f"issuance exceeds defaulted coupons until year {tt[np.where(di > dc)[0].max()] if (di > dc).any() else 0:.1f}", "",
         "| year | issuance | defaulted coupons | d | debt |", "|---|---|---|---|---|"]
for T in (0, 1, 2, 3, 5, 8, 10):
    i = int(T / dt); lines.append(f"| {T} | {100 * di[i]:+.2f} | {100 * dc[i]:+.2f} | {dd[i]:+.3f} | {100 * dB[i] / yb:+.2f} |")
txt = "\n".join(lines) + "\n"
open(os.path.join(DATA, "irf_summary.md"), "w", encoding="utf-8").write(txt)
print(txt)
