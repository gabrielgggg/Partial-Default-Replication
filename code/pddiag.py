"""
pddiag.py -- resolution diagnostic of a solution on the debt grid.

The policy comes from u'(c) = -V_B/q, so an error dV_B in the one-sided difference moves the
drift by dS = dV_B / (q^2 |u''(c)|), with dV_B ~ V_BB dB / 2 for first-order upwind. The
ratio

    r(B,z) = (|V_BB| dB / 2) / (q^2 |u''(c)| max(|S|, eps))

is the discretization error in the drift relative to the drift itself. Where r > ~0.5 the
scheme cannot determine the sign of the drift and iterative solvers chatter; the solution
(and in particular the ergodic distribution) is trustworthy only where r is small.
"""

from __future__ import annotations

import numpy as np

from pdmodel import Result


def resolution_ratio(res: Result, eps=1e-4):
    """The ratio r(B, z) above, with V_BB from second differences (non-uniform grids allowed)."""
    m = res.model
    par = m.par
    V, q, c, S = res.V, res.q, res.c, res.S
    hf, hb = m.hf[:, None], m.hb[:, None]
    VBB = np.zeros_like(V)
    VBB[1:-1] = 2.0 * ((V[2:] - V[1:-1]) / hf[1:-1] - (V[1:-1] - V[:-2]) / hb[1:-1]) / (hf[1:-1] + hb[1:-1])
    VBB[0] = VBB[1]; VBB[-1] = VBB[-2]
    upp = par.nu * c ** (-par.nu - 1.0)          # |u''(c)| for CRRA
    h = m.hcell[:, None]
    r = 0.5 * np.abs(VBB) * h / (q ** 2 * upp * np.maximum(np.abs(S), eps))
    return r


def resolution_report(res: Result, thresh=0.5, verbose=True):
    """First B per z where r > thresh away from steady states, and the ergodic mass on such nodes."""
    m = res.model
    r = resolution_ratio(res)
    out = {}
    Blim = np.full(m.nz, np.nan)
    for j in range(m.nz):
        bad = np.where(r[:, j] > thresh)[0]
        # ignore nodes right at a steady state (|S| ~ 0): require |S| > 1e-3 to count
        bad = [i for i in bad if abs(res.S[i, j]) > 1e-3]
        if bad:
            Blim[j] = m.B[min(bad)]
    out["r"] = r
    out["B_resolution_limit"] = Blim
    if res.g is not None:
        unresolved = (r > thresh) & (np.abs(res.S) > 1e-3)
        out["ergodic_mass_unresolved"] = float(res.g[unresolved].sum())
    if verbose:
        print(f"  resolution: first B per z where drift error / |S| > {thresh}: {np.array2string(Blim, precision=3)}")
        if res.g is not None:
            print(f"  ergodic mass on unresolved nodes: {out['ergodic_mass_unresolved']:.2e}")
    return out
