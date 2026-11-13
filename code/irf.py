"""Impulse responses of the main solution to a fall of 0.10 in the log endowment, by the forward equation.

Two sets of expected paths over 21 years, each for the shocked economy and for the control without the shock:
  data/irf.npz         from the ergodic distribution (keys t, shock_<field>, control_<field>);
  data/irf_states.npz  from the mean endowment with debt at its ergodic mean, 25 percent below it, and 25 percent
                       above it (keys t, EB, ybar, <i>_shock_<field>, <i>_control_<field> for i = 0, 1, 2).
Fields: y, yn (output net of default costs), c, d, B, spread, issue, defc.

Reads data/ou_main.npz. Writes data/irf.npz and data/irf_states.npz (read by make_figures.py and text_numbers.py).
Usage: python code/irf.py
"""
import os

import numpy as np

from generator import expected_paths, expected_paths_from_states
from paths import DATA

T = 21.0                       # the figures show 20 years; the paths run past the frame
MULTS = [1.0, 0.75, 1.25]      # initial debt, as multiples of its ergodic mean

a = np.load(os.path.join(DATA, "ou_main.npz"))
tt, out = expected_paths(a, T=T)
np.savez(os.path.join(DATA, "irf.npz"), t=tt, **{f"{s}_{k}": v for s in out for k, v in out[s].items()})

g = a["g"] / a["g"].sum()
EB = float((g.sum(1) * a["B"]).sum())
tt, res, ybar = expected_paths_from_states(a, [(m * EB, 0.0) for m in MULTS], T=T)
np.savez(os.path.join(DATA, "irf_states.npz"), t=tt, EB=EB, ybar=ybar,
         **{f"{i}_{s}_{k}": v for i, r in enumerate(res) for s in r for k, v in r[s].items()})
print(f"impulse responses to {T:g} years; mean debt {EB:.3f}, mean net output {ybar:.4f}")
