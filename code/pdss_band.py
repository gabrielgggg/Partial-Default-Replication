"""Search for a band of starting points captured by a partial-default steady state.

For each starting price q(0), finds the transition in c(0) between paths that stall beyond the attraction
boundary (deep) and paths that stall before it (shallow), and bisects it. Any capture met on the way proves a
captured band. Prints the band, or its absence to the bisection tolerance.
Usage: python code/pdss_band.py <illus|calib> rho [rho ...]
"""
import sys
import numpy as np
from pdss_shoot import illus, calib
from pdss_fwd2 import fwd
from pdss_linear import W_att


def att_boundary(m):
    """Smallest candidate B0 whose attraction bound W_att exceeds the wedge."""
    w = m.rho - m.r
    for B0 in np.arange(0.2, 2.0, 0.0025):
        for q0 in m.candidates(B0):
            if W_att(m, B0, q0) > w:
                return B0
    return np.inf


def kind(m, q0, c0, Batt):
    """Classify a start as CAPT, DEEP or SHAL (stalls beyond or before Batt), or the fwd class."""
    r = fwd(m, q0, c0, Bend=2.0)
    if r["cls"] == "CAPT":
        return "CAPT", r
    if r["cls"] in ("BRK", "OVER"):
        return ("DEEP" if r["B"] > Batt else "SHAL"), r
    return r["cls"], r


if __name__ == "__main__":
    which = sys.argv[1]
    for rho in [float(v) for v in sys.argv[2:]]:
        m = illus(rho) if which == "illus" else calib(rho)
        Batt = att_boundary(m)
        print(f"== {which} rho {rho:.4f} wedge {1e4 * (rho - m.r):.0f} bp, attraction boundary B0 {Batt:.4f}", flush=True)
        for q0 in [0.985, 0.99, 0.994, 0.997, 0.999, 0.9997]:
            # coarse grid in c(0), then bisect the first deep-to-shallow transition
            cs = np.arange(1.0005, 1.30, 0.002)
            ks = [kind(m, q0, c, Batt)[0] for c in cs]
            caps = [c for c, k in zip(cs, ks) if k == "CAPT"]
            if caps:
                print(f"   q0 {q0}: captured on the coarse grid, c0 {min(caps):.4f}-{max(caps):.4f}"); continue
            idx = [i for i in range(len(ks) - 1) if ks[i] == "DEEP" and ks[i + 1] == "SHAL"]
            if not idx:
                print(f"   q0 {q0}: no deep-to-shallow transition ({sorted(set(ks))})"); continue
            lo, hi = cs[idx[0]], cs[idx[0] + 1]
            found = None
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                k, r = kind(m, q0, mid, Batt)
                if k == "CAPT":
                    found = (mid, r); break
                if k == "DEEP":
                    lo, rlo = mid, r
                else:
                    hi, rhi = mid, r
            if found:
                print(f"   q0 {q0}: CAPTURE at c0 {found[0]:.10f}, steady state B0 {found[1]['B']:.4f} d0 {found[1]['d']:.3f}")
            else:
                print(f"   q0 {q0}: no capture down to width {hi - lo:.1e}; deep side stalls at B {rlo['B']:.4f} (d {rlo['d']:.3f}), "
                      f"shallow side at B {rhi['B']:.4f} (d {rhi['d']:.3f})")
