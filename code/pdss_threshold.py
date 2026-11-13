"""Smallest wedge at which some path from zero debt is captured by a partial-default steady state.

Deterministic model at the parameters of the self-fulfilling example, with CRRA curvature nu. Bisects on the
discount rate; at each rate, searches starting prices q(0) and consumption levels c(0) at zero debt for a path
that a partial-default steady state captures. --wide scans more starting prices and consumption levels (used
for nu below one).

Reads no file. Writes data/pdss/threshold_nu<nu>.txt, or threshold_wide_nu<nu>.txt with --wide: the steps of
the bisection and the bracket of the threshold wedge (read by collect_sf.py and text_numbers.py).
Usage: python code/pdss_threshold.py nu [--wide]
"""
import os
import sys
import time

import numpy as np

from paths import DATA
from pdss_fwd2 import fwd
from pdss_shoot import illus

NU = float(sys.argv[1])
WIDE = "--wide" in sys.argv
Q0S = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.994, 0.997, 0.999, 0.9997] if WIDE else [0.990, 0.994, 0.997, 0.999, 0.9997]
CMAX = 1.60 if WIDE else 1.30
os.makedirs(os.path.join(DATA, "pdss"), exist_ok=True)
log = open(os.path.join(DATA, "pdss", f"threshold_{'wide_' if WIDE else ''}nu{NU}.txt"), "w")


def say(s):
    """Print a line and append it to the output file."""
    print(s, flush=True); log.write(s + "\n"); log.flush()


def W_att(m, B0, q0):
    """Largest wedge at which the candidate (B0, q0) attracts."""
    d0 = m.dstar(B0, q0); a0 = m.a(d0)
    Pd = (1 - m.kappa * q0) * m.cs
    dB = d0 / ((m.g1 - 1) * B0); dq = -d0 / (m.g1 - 1) * m.kappa / (1 - m.kappa * q0)
    PB, Pq = Pd * dB, m.xi(d0) + Pd * dq
    return PB * a0 * B0 / (q0 * Pq) if Pq > 0 else np.inf


def att_boundary(m):
    """Smallest candidate B0 whose attraction bound W_att exceeds the wedge."""
    w = m.rho - m.r
    for B0 in np.arange(0.2, 2.0, 0.0025):
        for q0 in m.candidates(B0):
            if W_att(m, B0, q0) > w:
                return B0
    return np.inf


def exists(rho):
    """Search starting points for a captured path, bisecting deep-to-shallow transitions in c(0)."""
    m = illus(rho, NU)
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
m0 = illus(0.05, NU)
sf = m0.spread(m0.qr(m0.d_first()))
if NU < 0.1:        # the threshold is close to s_first at low curvature: a narrower bracket, one step fewer
    lo, hi, steps = m0.r + 0.0045, m0.r + 0.008, 6
else:
    lo, hi, steps = m0.r + 0.5 * sf, m0.r + 0.03, 7
t0 = time.time()
ok_hi, info = exists(hi)
say(f"nu {NU}: s_first {1e4 * sf:.0f} bp; at wedge {1e4 * (hi - m0.r):.0f} bp exists={ok_hi} {info}")
if ok_hi:
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        ok, info = exists(mid)
        say(f"   wedge {1e4 * (mid - m0.r):6.1f} bp: {'exists' if ok else 'none'} {info if ok else ''}  ({time.time() - t0:.0f}s)")
        if ok:
            hi = mid
        else:
            lo = mid
    say(f"nu {NU}: threshold wedge in [{1e4 * (lo - m0.r):.1f}, {1e4 * (hi - m0.r):.1f}] bp (s_first {1e4 * sf:.1f} bp)")
