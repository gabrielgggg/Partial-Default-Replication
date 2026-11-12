"""Scan starting points at zero debt for capture by a partial-default steady state.

Integrates forward from a grid of (q(0), c(0)), records which starts are captured, and checks the right branch
at a few captured steady states. Writes data/pdss/<which>_rho<rho>.npz and prints a summary.
Usage: python code/pdss_sweep.py <illus|calib> rho
"""
import os, sys, time
import numpy as np
from pdss_shoot import illus, calib
from pdss_fwd2 import fwd, right

which, rho = sys.argv[1], float(sys.argv[2])
m = illus(rho) if which == "illus" else calib(rho)
# grid of starting points, dense in q(0) near 1
qs = np.concatenate([np.linspace(0.90, 0.99, 10), np.linspace(0.991, 0.99999, 12)])
cs = np.linspace(1.0005, 1.30, 300)
t0 = time.time()
res = []
for q0 in qs:
    for c0 in cs:
        r = fwd(m, q0, c0, Bend=2.0)
        res.append((q0, c0, {"CAPT": 1, "BRK": 2, "OVER": 3, "QOUT": 4, "NEG": 5}[r["cls"]], r["B"], r["q"], r["d"]))
# captured starts and the steady states they reach
res = np.array(res)
cap = res[res[:, 2] == 1]
lines = [f"{which} rho {rho:.4f} wedge {1e4 * (rho - m.r):.0f} bp d_first {m.d_first():.3f}: {len(cap)} of {len(res)} starts captured"]
okR = []
if len(cap):
    B0s = cap[:, 3]
    # right-branch check on a few captured steady states (land points approximate B0 from the left)
    for Bl in np.unique(np.round(np.quantile(B0s, [0, 0.5, 1]), 3)):
        cands = m.candidates(Bl)
        if not cands:
            continue
        qq = min(cands, key=lambda v: abs(v - cap[np.argmin(np.abs(B0s - Bl)), 4]))
        loc = m.local(Bl, qq)
        R = right(m, loc) if loc["s1"] < 0 else dict(cls="repelling", B=Bl)
        okR.append((Bl, R["cls"], R["B"]))
    lines.append(f"   steady states reached: B0 in [{B0s.min():.3f}, {B0s.max():.3f}], d0 in [{cap[:, 5].min():.3f}, {cap[:, 5].max():.3f}], "
                 f"q(0) in [{cap[:, 0].min():.5f}, {cap[:, 0].max():.5f}], c(0) in [{cap[:, 1].min():.4f}, {cap[:, 1].max():.4f}]")
    lines.append("   right branches: " + "; ".join(f"B0 {b:.3f} -> {c} at {e:.3f}" for b, c, e in okR))
lines.append(f"   {time.time() - t0:.0f}s")
from paths import DATA
np.savez(os.path.join(DATA, "pdss", f"{which}_rho{rho:.4f}.npz"), res=res, rho=rho)
print("\n".join(lines), flush=True)
