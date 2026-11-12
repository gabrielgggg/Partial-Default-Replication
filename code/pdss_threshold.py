"""Smallest wedge at which some path from zero debt is captured by a partial-default steady state.

Illustrative parameterization with CRRA curvature nu. Bisects on the discount rate between THR_LO and THR_HI
(wedges, defaults half of s_first and 0.03) for THR_IT steps (default 7); --wide scans more starting prices and
consumption levels. Prints the bracket of the threshold wedge.
Usage: python code/pdss_threshold.py nu [--wide]
"""
import os, sys, time
import numpy as np
os.environ["PDSS_NU"] = sys.argv[1]
WIDE = "--wide" in sys.argv
Q0S = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.994, 0.997, 0.999, 0.9997] if WIDE else [0.990, 0.994, 0.997, 0.999, 0.9997]
CMAX = 1.60 if WIDE else 1.30
from pdss_shoot import illus
from pdss_fwd2 import fwd
from pdss_band import att_boundary


def exists(rho):
    """Search starting points for a captured path, bisecting deep-to-shallow transitions in c(0)."""
    m = illus(rho)
    Batt = att_boundary(m)
    for q0 in Q0S:
        cs = np.arange(1.0005, CMAX, 0.002)
        ks = []
        for c in cs:
            r = fwd(m, q0, c, Bend=2.0)
            if r["cls"] == "CAPT":
                return True, (q0, c, r["B"])
            ks.append("DEEP" if (r["cls"] in ("BRK", "OVER") and r["B"] > Batt) else "SHAL" if r["cls"] in ("BRK", "OVER") else r["cls"])
        idx = [i for i in range(len(ks) - 1) if ks[i] == "DEEP" and ks[i + 1] == "SHAL"]
        if not idx:
            continue
        lo, hi = cs[idx[0]], cs[idx[0] + 1]
        for _ in range(48):
            mid = 0.5 * (lo + hi); r = fwd(m, q0, mid, Bend=2.0)
            if r["cls"] == "CAPT":
                return True, (q0, mid, r["B"])
            if r["cls"] in ("BRK", "OVER") and r["B"] > Batt:
                lo = mid
            else:
                hi = mid
    return False, None


# bracket in the wedge, then bisect while a captured path exists at the upper end
nu = float(sys.argv[1])
m0 = illus(0.05)
sf = m0.spread(m0.qr(m0.d_first()))
lo, hi = m0.r + float(os.environ.get("THR_LO", 0.5 * sf)), m0.r + float(os.environ.get("THR_HI", 0.03))
t0 = time.time()
ok_hi, info = exists(hi)
print(f"nu {nu}: s_first {1e4 * sf:.0f} bp; at wedge {1e4 * (hi - m0.r):.0f} bp exists={ok_hi} {info}", flush=True)
if ok_hi:
    for _ in range(int(os.environ.get("THR_IT", 7))):
        mid = 0.5 * (lo + hi)
        ok, info = exists(mid)
        print(f"   wedge {1e4 * (mid - m0.r):6.1f} bp: {'exists' if ok else 'none'} {info if ok else ''}  ({time.time() - t0:.0f}s)", flush=True)
        if ok:
            hi = mid
        else:
            lo = mid
    print(f"nu {nu}: threshold wedge in [{1e4 * (lo - m0.r):.1f}, {1e4 * (hi - m0.r):.1f}] bp (s_first {1e4 * sf:.1f} bp)", flush=True)
