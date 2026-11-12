"""Collect the self-fulfilling region (data/pdss/threshold_*.txt, data/sf_region/sd*.npz) into data/sfregion.npz.

Curvature: nu and the bisection bracket of the smallest wedge with a self-fulfilling path (bp).
Volatility: s.d. of the log endowment, lower and upper boundaries of the existence region (bp; upper = inf above
the wedge grid, 600 bp), and the s.d. at which the region closes.
Usage: python code/collect_sf.py
"""
import glob
import os
import re

import numpy as np

from paths import DATA

# curvature: parse the threshold bracket from each pdss_threshold.py log
logs = {0.02: "threshold_wide_nu0.02.txt", 0.05: "threshold_wide_nu0.05.txt", 0.1: "threshold_wide_nu0.1.txt",
        0.25: "threshold_wide_nu0.25.txt", 0.5: "threshold_wide_nu0.5.txt", 1.0: "threshold_nu1.0.txt",
        2.0: "threshold_nu2.0.txt", 4.0: "threshold_nu4.0.txt"}
nu, lo_b, hi_b = [], [], []
for v, f in sorted(logs.items()):
    m = re.search(r"threshold wedge in \[([\d.]+), ([\d.]+)\]", open(os.path.join(DATA, "pdss", f)).read())
    nu.append(v); lo_b.append(float(m.group(1))); hi_b.append(float(m.group(2)))
# volatility: first upward and last downward transition in the wedge for each s.d.
rows = []
for f in glob.glob(os.path.join(DATA, "sf_region", "sd*.npz")):
    a = np.load(f)
    W, cnt, tr, kinds = a["W"], a["counts"], a["trans"], a["kinds"]
    ups = [t for t, k in zip(tr, kinds) if k == "up"]; downs = [t for t, k in zip(tr, kinds) if k == "down"]
    lo = ups[0] if ups else np.nan
    hi = downs[-1] if downs else (np.inf if cnt.any() else np.nan)
    rows.append((float(a["sd"]), lo, hi, int(cnt.max())))
rows.sort()
sd = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows]); hi = np.array([r[2] for r in rows])
# the region closes between the last s.d. with a lower boundary and the next one
last = int(np.max(np.nonzero(np.isfinite(lo))[0]))
sd_close = (sd[last], sd[last + 1])
np.savez(os.path.join(DATA, "sfregion.npz"), nu=np.array(nu), nu_lo=np.array(lo_b), nu_hi=np.array(hi_b), s_first=46.8,
         sd=sd, w_lo=lo, w_hi=hi, n_max=np.array([r[3] for r in rows]), sd_close=np.array(sd_close), wedge_grid_max=600.0)
