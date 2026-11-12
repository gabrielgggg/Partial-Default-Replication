"""Scan the risk-neutral block construction with optimal waiting at zero debt (RN.check_free).

For each OU volatility sigma and each wedge on a fixed grid, checks the family members on a grid of d_ref and
prints the number of equilibria, their range of d_ref, B0, waiting mass, and spreads, and the reasons the other
members fail. --nsd sets the width of the z grid in unconditional s.d.
Usage: python code/rn_scan_free.py sigma [sigma ...] [--nsd x]
"""
import sys
import time
from collections import Counter

import numpy as np

from rn_diffusion import RN

WEDGES_BP = [40, 47, 50, 55, 60, 70, 80, 100, 125, 150, 200, 250, 300, 400]
D_GRID = np.linspace(0.0894, 0.6, 52)


def classify(r):
    """First failed condition of a check_free result, or "EQ" if all hold."""
    if r.get("collapsed"):
        return "locus collapses"
    if r["price_res"] > 1e-8:
        return "price root not found"
    if not r["floor"]:
        return "below floor"
    if not r["any_jump"]:
        return "never jumps from zero"
    if r["buyback_min"] < -1e-12:
        return "buyback-and-wait beats B0"
    if r["Dmin_free"] < -1e-10:
        return "standing still better on (0,B0)"
    if r["SBmax"] >= 0:
        return "no attraction"
    mm = r.get("march")
    if mm is None or not mm["ok"]:
        return "continuation fails"
    return "EQ"


if __name__ == "__main__":
    args = sys.argv[1:]
    nsd = 3.0
    if "--nsd" in args:
        k = args.index("--nsd"); nsd = float(args[k + 1]); del args[k:k + 2]
    for sigma in [float(s) for s in args]:
        t0 = time.time()
        for w in WEDGES_BP:
            m = RN(0.039 + w * 1e-4, sigma, n_sd=nsd)
            res = [m.check_free(d) for d in D_GRID]
            cls = [classify(r) for r in res]
            eq = [r for r, c in zip(res, cls) if c == "EQ"]
            cnt = Counter(c for c in cls if c != "EQ")
            sd = sigma / np.sqrt(2 * 0.225) if sigma > 0 else 1.0
            print(f"sigma {sigma:.3f} nsd {nsd} wedge {w:4d} bp: {len(eq):2d}/{len(D_GRID)} EQ"
                  + (f", d_ref [{min(r['d_ref'] for r in eq):.3f},{max(r['d_ref'] for r in eq):.3f}], B0 [{min(r['B0'] for r in eq):.3f},{max(r['B0'] for r in eq):.3f}]"
                     f", wait mass [{min(r['frac_wait'] for r in eq):.2f},{max(r['frac_wait'] for r in eq):.2f}], z*/sd [{min(r['zstar'] for r in eq)/sd:+.2f},{max(r['zstar'] for r in eq)/sd:+.2f}]"
                     f", spread [{1e4*min(r['spread_mean'] for r in eq):.0f},{1e4*max(r['spread_mean'] for r in eq):.0f}] bp" if eq else "")
                  + " | " + "; ".join(f"{k}: {v}" for k, v in cnt.items()), flush=True)
        print(f"sigma {sigma}: {time.time()-t0:.0f}s", flush=True)
