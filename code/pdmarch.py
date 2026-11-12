"""Exact marching solution of the deterministic (fixed z) discrete equilibrium.

In the deterministic case the upwind discretization is triangular once the steady-state node
i* is known: a node with S < 0 uses the backward difference and depends only on the node
below it; a node with S > 0 depends only on the node above it. The solution proceeds in steps.

  1. Find the candidate steady states: nodes where the zero-drift policy is self-consistent
     (q* = q_ss(d*), d* = argmax of zero-drift consumption at that price).
  2. March outward from each candidate, solving at every node the 2-equation system

        rho V_i = H(V_i, q_i)                               (HJB, one-sided difference)
        xi(d_i) q_i = (1-d_i)(lam+delta) + S_i * (one-sided difference of q)

     The HJB equation is monotone in V_i given q_i (unique root, bisection); the price
     equation is then a scalar equation in q_i which is scanned on (0,1] to find *all* roots
     (local multiplicity) and continued along the root closest to the previous node.
  3. Verify at every node that the assumed one-sided candidate is the global maximum of the
     three candidates (forward / backward / zero drift) given the marched neighbours.

The result solves the same discrete system as the iterative solvers and serves as a ground
truth for the deterministic case. Node solves are compiled with numba.
Library module, imported by pdmarch_branch.py.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pdmodel import Model, Result, best_node, u_fn

MAXR = 8  # max roots stored per node


@njit(cache=True)
def _xi(d, p):
    """Effective discount on the bond, r + delta - kappa (delta + lam) d."""
    return p[4] + p[2] - p[3] * (p[2] + p[5]) * d


@njit(cache=True)
def _vroot(q, V_nb, mode, B, z, y, p, dB, n_golden):
    """Root of rho V - H(VB(V), q) = 0 (monotone in V). Returns (V, H, S, c, d)."""
    rho = p[1]
    lo = V_nb - 1.0
    hi = V_nb + 1.0
    # expand the bracket until the residual changes sign, then bisect
    for _ in range(60):
        VB = (lo - V_nb) / dB if mode == -1 else (V_nb - lo) / dB
        g = rho * lo - best_node(VB, q, B, z, y, p, mode, n_golden)[0]
        if g <= 0.0:
            break
        lo -= 2.0 * (V_nb - lo + 1.0)
    for _ in range(60):
        VB = (hi - V_nb) / dB if mode == -1 else (V_nb - hi) / dB
        g = rho * hi - best_node(VB, q, B, z, y, p, mode, n_golden)[0]
        if g >= 0.0:
            break
        hi += 2.0 * (hi - V_nb + 1.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        VB = (mid - V_nb) / dB if mode == -1 else (V_nb - mid) / dB
        g = rho * mid - best_node(VB, q, B, z, y, p, mode, n_golden)[0]
        if g > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-13 * (1.0 + abs(mid)):
            break
    V = 0.5 * (lo + hi)
    VB = (V - V_nb) / dB if mode == -1 else (V_nb - V) / dB
    H, S, c, d = best_node(VB, q, B, z, y, p, mode, n_golden)
    return V, H, S, c, d


@njit(cache=True)
def _price_res(q, V_nb, q_nb, mode, B, z, y, p, dB, n_golden):
    """Residual of the price equation at q, with V solved from the HJB; NaN if infeasible."""
    V, H, S, c, d = _vroot(q, V_nb, mode, B, z, y, p, dB, n_golden)
    coupon = p[5] + p[2]
    qB = (q - q_nb) / dB if mode == -1 else (q_nb - q) / dB
    r = _xi(d, p) * q - (1.0 - d) * coupon - S * qB
    if H < -1.0e29:
        r = np.nan
    return r, V, S, c, d


@njit(cache=True)
def node_solutions(V_nb, q_nb, mode, B, z, y, p, dB, nq, n_golden):
    """All roots in q of the price equation at a node (given the upwind neighbour).

    Returns (nroots, qr, Vr, Sr, cr, dr) arrays of length MAXR.
    """
    qr = np.zeros(MAXR); Vr = np.zeros(MAXR); Sr = np.zeros(MAXR); cr = np.zeros(MAXR); dr = np.zeros(MAXR)
    n = 0
    # scan q on a grid dense near 1 and bisect every sign change
    qprev = 1.0e-4
    rprev = _price_res(qprev, V_nb, q_nb, mode, B, z, y, p, dB, n_golden)[0]
    for k in range(1, nq + 1):
        qk = 1.0 if k == nq else 1.0 - (1.0 - 1.0e-4) * (1.0 - k / nq) ** 2
        rk = _price_res(qk, V_nb, q_nb, mode, B, z, y, p, dB, n_golden)[0]
        if np.isfinite(rprev) and np.isfinite(rk) and rprev * rk <= 0.0 and n < MAXR:
            a = qprev; b = qk; ra = rprev
            for _ in range(60):
                mid = 0.5 * (a + b)
                rm = _price_res(mid, V_nb, q_nb, mode, B, z, y, p, dB, n_golden)[0]
                if not np.isfinite(rm):
                    break
                if rm * ra <= 0.0:
                    b = mid
                else:
                    a = mid; ra = rm
                if b - a < 1e-14:
                    break
            qq = 0.5 * (a + b)
            r, V, S, c, d = _price_res(qq, V_nb, q_nb, mode, B, z, y, p, dB, n_golden)
            if abs(r) < 1e-7:
                qr[n] = qq; Vr[n] = V; Sr[n] = S; cr[n] = c; dr[n] = d
                n += 1
        qprev = qk; rprev = rk
    return n, qr, Vr, Sr, cr, dr


@njit(cache=True)
def zero_drift_fixed_points(B, z, y, p, nq, n_golden):
    """All self-consistent zero-drift prices at a node: q = (1-d(q)) coupon / xi(d(q))."""
    qr = np.zeros(MAXR); Vr = np.zeros(MAXR); cr = np.zeros(MAXR); dr = np.zeros(MAXR); Hr = np.zeros(MAXR)
    coupon = p[5] + p[2]
    n = 0

    def F(q):
        H, S, c, d = best_node(0.0, q, B, z, y, p, 0, n_golden)
        if H < -1.0e29:
            return np.nan
        return (1.0 - d) * coupon / _xi(d, p) - q

    qprev = 1.0e-4
    fprev = F(qprev)
    for k in range(1, nq + 1):
        qk = 1.0 if k == nq else 1.0 - (1.0 - 1.0e-4) * (1.0 - k / nq) ** 2
        fk = F(qk)
        if np.isfinite(fprev) and np.isfinite(fk) and fprev * fk <= 0.0 and n < MAXR:
            a = qprev; b = qk; fa = fprev
            for _ in range(60):
                mid = 0.5 * (a + b)
                fm = F(mid)
                if fm * fa <= 0.0:
                    b = mid
                else:
                    a = mid; fa = fm
                if b - a < 1e-14:
                    break
            qq = 0.5 * (a + b)
            H, S, c, d = best_node(0.0, qq, B, z, y, p, 0, n_golden)
            qr[n] = qq; cr[n] = c; dr[n] = d; Hr[n] = H; Vr[n] = u_fn(c, p[0]) / p[1]
            n += 1
        qprev = qk; fprev = fk
    return n, qr, Vr, cr, dr, Hr


def march(m: Model, istar: int, fp_index: int | None = None, verbose=False, nq=200, n_golden=40):
    """March outward from steady-state node istar for every zero-drift fixed point there
    (or only the fp_index-th). Returns a list of solution dicts."""
    assert m.nz == 1
    nB = m.nB
    B = m.B; z = m.z[0]; y = m.y[0]; p = m.p; dB = m.dB
    nfp, qf, Vf, cf, df, Hf = zero_drift_fixed_points(B[istar], z, y, p, 400, n_golden)
    results = []
    fp_range = range(nfp) if fp_index is None else [fp_index]
    for f in fp_range:
        V = np.full(nB, np.nan); q = np.full(nB, np.nan); S = np.zeros(nB); c = np.full(nB, np.nan)
        d = np.full(nB, np.nan); which = np.zeros(nB, dtype=np.int8); nsol = np.zeros(nB, dtype=int)
        # start at the steady state, march up (backward differences), then down (forward differences)
        V[istar] = Vf[f]; q[istar] = qf[f]; c[istar] = cf[f]; d[istar] = df[f]
        nsol[istar] = nfp
        stop_up = stop_dn = None
        for i in range(istar + 1, nB):
            n, qr, Vr, Sr, cr, dr = node_solutions(V[i - 1], q[i - 1], -1, B[i], z, y, p, dB, nq, n_golden)
            nsol[i] = n
            if n == 0:
                stop_up = i; break
            k = int(np.argmin(np.abs(qr[:n] - q[i - 1])))
            V[i] = Vr[k]; q[i] = qr[k]; S[i] = Sr[k]; c[i] = cr[k]; d[i] = dr[k]; which[i] = -1
        for i in range(istar - 1, -1, -1):
            n, qr, Vr, Sr, cr, dr = node_solutions(V[i + 1], q[i + 1], +1, B[i], z, y, p, dB, nq, n_golden)
            nsol[i] = n
            if n == 0:
                stop_dn = i; break
            k = int(np.argmin(np.abs(qr[:n] - q[i + 1])))
            V[i] = Vr[k]; q[i] = qr[k]; S[i] = Sr[k]; c[i] = cr[k]; d[i] = dr[k]; which[i] = +1
        # global check: the chosen one-sided candidate must beat the other two at every node
        viol = []
        for i in range(nB):
            if not np.isfinite(V[i]):
                continue
            cands = {}
            if i < nB - 1 and np.isfinite(V[i + 1]):
                cands[1] = best_node((V[i + 1] - V[i]) / dB, q[i], B[i], z, y, p, 1, n_golden)[0]
            if i > 0 and np.isfinite(V[i - 1]):
                cands[-1] = best_node((V[i] - V[i - 1]) / dB, q[i], B[i], z, y, p, -1, n_golden)[0]
            cands[0] = best_node(0.0, q[i], B[i], z, y, p, 0, n_golden)[0]
            best = max(cands, key=cands.get)
            if cands[best] > cands[int(which[i])] + 1e-9:
                viol.append((i, int(which[i]), best, cands[best] - cands[int(which[i])]))
        results.append(dict(istar=istar, Bstar=B[istar], qstar=qf[f], dstar=df[f], cstar=cf[f],
                            V=V, q=q, S=S, c=c, d=d, which=which, nsol=nsol, viol=viol,
                            stop_up=stop_up, stop_dn=stop_dn,
                            consistent=(len(viol) == 0 and stop_up is None and stop_dn is None)))
        if verbose:
            print(f"  istar={istar} (B={B[istar]:.3f}) q*={qf[f]:.4f} d*={df[f]:.3f}: stop_up={stop_up} "
                  f"stop_dn={stop_dn} violations={len(viol)} multi-root nodes={int(np.sum(nsol > 1))}")
    return results


def scan_steady_states(m: Model, verbose=True, n_golden=40, nq=100):
    """Locally consistent zero-drift nodes: zero drift beats forward/backward once the two
    neighbours are marched one step. Returns a list of (istar, fp_index, q*, d*, c*)."""
    nB = m.nB
    B = m.B; z = m.z[0]; y = m.y[0]; p = m.p; dB = m.dB
    out = []
    for istar in range(nB):
        nfp, qf, Vf, cf, df, Hf = zero_drift_fixed_points(B[istar], z, y, p, 400, n_golden)
        for f in range(nfp):
            ok = True
            if istar < nB - 1:
                n, qr, Vr, Sr, cr, dr = node_solutions(Vf[f], qf[f], -1, B[istar + 1], z, y, p, dB, nq, n_golden)
                if n:
                    k = int(np.argmin(np.abs(qr[:n] - qf[f])))
                    HF = best_node((Vr[k] - Vf[f]) / dB, qf[f], B[istar], z, y, p, 1, n_golden)[0]
                    if HF > Hf[f] + 1e-9:
                        ok = False
                else:
                    ok = False
            if ok and istar > 0:
                n, qr, Vr, Sr, cr, dr = node_solutions(Vf[f], qf[f], +1, B[istar - 1], z, y, p, dB, nq, n_golden)
                if n:
                    k = int(np.argmin(np.abs(qr[:n] - qf[f])))
                    HB = best_node((Vf[f] - Vr[k]) / dB, qf[f], B[istar], z, y, p, -1, n_golden)[0]
                    if HB > Hf[f] + 1e-9:
                        ok = False
                else:
                    ok = False
            if ok:
                out.append((istar, f, qf[f], df[f], cf[f]))
    if verbose:
        print(f"scan: {len(out)} locally consistent steady-state candidates:")
        for i, f, qs, ds, cs in out:
            print(f"   B={B[i]:.4f}  q*={qs:.4f}  d*={ds:.4f}  c*={cs:.4f}")
    return out


def solve_deterministic(m: Model, verbose=True, n_golden=40, nq=200):
    """Scan + march. Returns (list of consistent solutions, all candidates)."""
    cands = scan_steady_states(m, verbose=verbose, n_golden=n_golden)
    sols = []
    for istar, f, qs, ds, cs in cands:
        rs = march(m, istar, fp_index=f, verbose=verbose, nq=nq, n_golden=n_golden)
        sols.extend(rs)
    good = [s for s in sols if s["consistent"]]
    if verbose:
        print(f"{len(good)} globally consistent solution(s) out of {len(sols)} marched candidates")
    return good, sols


def to_result(m: Model, sol: dict) -> Result:
    """Wrap a marched solution in a Result with policies, generator, and HJB and price residuals."""
    V = np.asarray(sol["V"]).reshape(-1, 1); q = np.asarray(sol["q"]).reshape(-1, 1)
    S, c, d, u, Hval, which, S_free, d_free = m.policies(V, q)
    A = m.assemble(S)
    par = m.par
    RV = par.rho * V.ravel() - u.ravel() - A @ V.ravel()
    Rq = par.xi(d.ravel()) * q.ravel() - (1 - d.ravel()) * (par.lam + par.delta) - A @ q.ravel()
    return Result(m, V, q, S, c, d, which, S_free, d_free, A, None, True, 0,
                  float(np.max(np.abs(RV))), float(np.max(np.abs(Rq))), [], 0.0)
