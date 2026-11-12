"""Existence region of self-fulfilling partial default (risk-neutral sovereign, illustrative parameters, block
construction with optimal waiting) in the plane (unconditional s.d. of the log endowment, impatience wedge).

For each s.d., counts equilibria on a wedge grid, then bisects every transition between "some" and "none".
Writes data/sf_region/sd<sd>.npz with the counts and the transition wedges.
Usage: python code/region.py sd [sd ...]
"""
import os
import sys
import time

import numpy as np

from rn_diffusion import RN
from rn_scan_free import classify

# OU mean reversion, the first default intensity, and the grids in d and in the wedge
MU = 0.225
from paths import DATA
OUT = os.path.join(DATA, "sf_region")
os.makedirs(OUT, exist_ok=True)
D_FIRST = 0.08932
U = np.linspace(0.0, 1.0, 60)
D_GRID = D_FIRST + 1e-4 + (0.6 - D_FIRST) * U ** 2          # dense near the first default
W_GRID = np.unique(np.round(np.geomspace(30, 600, 26)))     # bp


def n_eq(sd, w):
    """Number of equilibria at s.d. sd and wedge w (bp), with their B0 and mean spreads."""
    m = RN(0.039 + w * 1e-4, sd * np.sqrt(2 * MU))
    res = [m.check_free(d) for d in D_GRID]
    eq = [r for r in res if classify(r) == "EQ"]
    return len(eq), ([r["B0"] for r in eq], [r["spread_mean"] for r in eq])


def bisect(sd, lo, hi, n_lo, iters=7):
    """lo, hi wedges with (n_lo > 0) != (n_hi > 0); returns the transition wedge"""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n, _ = n_eq(sd, mid)
        if (n > 0) == (n_lo > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    for sd in [float(s) for s in sys.argv[1:]]:
        t0 = time.time()
        counts, info = [], []
        for w in W_GRID:
            n, inf = n_eq(sd, w)
            counts.append(n); info.append(inf)
        counts = np.array(counts)
        trans = []
        for k in range(len(W_GRID) - 1):
            if (counts[k] > 0) != (counts[k + 1] > 0):
                trans.append((bisect(sd, W_GRID[k], W_GRID[k + 1], counts[k]), "up" if counts[k + 1] > 0 else "down"))
        np.savez(os.path.join(OUT, f"sd{sd:.4f}.npz"), sd=sd, W=W_GRID, counts=counts,
                 trans=np.array([t for t, _ in trans]), kinds=np.array([k for _, k in trans]))
        print(f"sd {sd:.4f}: counts {counts.tolist()} transitions {[(round(t, 1), k) for t, k in trans]} ({time.time()-t0:.0f}s)", flush=True)
