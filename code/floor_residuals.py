"""floor_residuals.py: where the residuals of the main diffusion solution sit. Nodes are split into
  floor locus:       the price crosses q_floor between the node and a neighbor in debt, or |q - q_floor| < band
                     (where the standard switches issuance on and off)
  zero-drift locus:  within one node (diagonals included) of a sign change of the drift in debt or productivity
                     (the stationary points, where the upwind direction switches)
  elsewhere:         all other nodes
and reports the largest HJB and pricing residuals and the ergodic mass on each. Reads data/ou_main.npz and
data/ou_main_residuals.npz (written by text_numbers.py, which imports report()). Prints only.
Usage: python code/floor_residuals.py [band]"""
import os
import sys

import numpy as np

from paths import DATA
BAND = 0.02


def neighbors_differ(m):
    """True where a boolean map differs from a neighbor in either direction."""
    out = np.zeros_like(m)
    for ax in (0, 1):
        dif = np.diff(m.astype(np.int8), axis=ax) != 0
        sl_lo = [slice(None)] * 2; sl_hi = [slice(None)] * 2
        sl_lo[ax] = slice(None, -1); sl_hi[ax] = slice(1, None)
        out[tuple(sl_lo)] |= dif; out[tuple(sl_hi)] |= dif
    return out


def loci(q, S, qf, band=BAND):
    """Boolean maps of the floor locus, the zero-drift locus (off the floor locus), and the remaining nodes."""
    below = q < qf
    cross = np.zeros_like(below)
    cross[1:] |= below[1:] != below[:-1]; cross[:-1] |= below[1:] != below[:-1]
    floor = cross | (np.abs(q - qf) < band)
    core = neighbors_differ(S > 1e-12) | (np.abs(S) <= 1e-12)
    zero = core.copy()                       # within one node of a sign change, diagonals included
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            zero |= np.roll(np.roll(core, di, axis=0), dj, axis=1)
    zero &= ~floor
    return floor, zero, ~floor & ~zero


def report(path_sol="ou_main.npz", path_res="ou_main_residuals.npz", band=BAND):
    """Nodes, ergodic mass, and largest absolute HJB and pricing residuals on each locus."""
    a = np.load(os.path.join(DATA, path_sol)); r = np.load(os.path.join(DATA, path_res))
    w = a["g"] / a["g"].sum()
    RV, Rq = np.abs(r["RV"]), np.abs(r["Rq"])
    out = {}
    for nm, m in zip(("floor", "zero", "else"), loci(a["q"], a["S"], float(a["qf"]), band)):
        out[nm] = dict(nodes=int(m.sum()), mass=float(w[m].sum()), RV=float(RV[m].max()), Rq=float(Rq[m].max()))
    return out


if __name__ == "__main__":
    band = float(sys.argv[1]) if len(sys.argv) > 1 else BAND
    for nm, v in report(band=band).items():
        print(f"{nm:6s} nodes {v['nodes']:5d} mass {v['mass']:.4f}  max RV {v['RV']:.1e}  max Rq {v['Rq']:.1e}")
