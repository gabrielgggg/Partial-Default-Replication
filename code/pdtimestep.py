"""
pdtimestep.py -- the implicit time-stepping solver that iterates V and q jointly to a stationary
point (the scheme described in Appendix B of the paper). Each step takes the policy explicitly from
(V^k, q^k) and solves for V^{k+1} and q^{k+1} implicitly in the generator A(S^k):

    [(rho + 1/dt) I - A] V^{k+1} = u + V^k / dt
    [(xi(d) + 1/dt) I - A] q^{k+1} = (1-d)(lam+delta) + q^k / dt

This is a robust but slow global relaxation. When it converges, it converges to the limit of the
finite-horizon equilibrium. The residuals of the discrete system are monitored, and a
non-converging oscillation is reported. Oscillation occurs where the upwind scheme cannot resolve
the sign of the drift (see pddiag.py).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pdmodel import Model, Result, ergodic_density, best_node, u_fn


def _zero_drift_override(model, mask, S, c, d, u, which, q, n_golden):
    """Replace the policy at the masked nodes by the zero-drift candidate (if feasible)."""
    p = model.p
    nu = p[0]
    nfix = 0
    for i, j in zip(*np.nonzero(mask)):
        H, S_, c_, d_ = best_node(0.0, q[i, j], model.B[i], model.z[j], model.y[j], p, 0, n_golden)
        if np.isfinite(H) and H > -1e29:
            S[i, j] = 0.0; c[i, j] = c_; d[i, j] = d_; u[i, j] = u_fn(c_, nu); which[i, j] = 0
            nfix += 1
    return nfix


def solve_timestep(model: Model, V0=None, q0=None, *, dt=0.001, maxit=200000, tolV=1e-8, tolq=1e-9,
                   n_golden=28, verbose=True, print_every=5000, q_init="one", ergodic=True,
                   dt_adapt=False, dt_max=0.05, save_path=None,
                   chatter_fix=False, chatter_window=30, chatter_flips=3, chatter_hold=200) -> Result:
    """Joint implicit time stepping of V and q from (V0, q0) with step dt, until both residuals are
    below tolerance or maxit steps. Returns a Result (with the ergodic density if ergodic and nz > 1).

    chatter_fix: nodes whose drift sign flipped at least `chatter_flips` times within the last
    `chatter_window` steps are assigned the zero-drift candidate for the next `chatter_hold` steps
    (sliding-mode / Filippov convention at a price cliff: the state rests at the cliff node).
    save_path: if given, the current iterate is saved there every `print_every` steps."""
    par = model.par
    N = model.N
    S_prev = None
    flip_hist = np.zeros((model.nB, model.nz), dtype=np.int32)   # recent flips (decayed window)
    flip_times = []                                               # list of flip masks (window)
    hold = np.zeros((model.nB, model.nz), dtype=np.int32)         # remaining steps of forced zero drift
    n_forced_last = 0
    # starting point
    if V0 is None or q0 is None:
        V0_, q0_ = model.initial_guess(q_init)
        V = V0_ if V0 is None else np.array(V0, dtype=np.float64).copy()
        q = q0_ if q0 is None else np.array(q0, dtype=np.float64).copy()
    else:
        V, q = np.array(V0, dtype=np.float64).copy(), np.array(q0, dtype=np.float64).copy()
    I = sp.identity(N, format="csc")
    coupon = par.lam + par.delta
    t0 = time.perf_counter()
    hist = []
    converged = False
    rV_prev = np.inf
    it = 0
    for it in range(1, maxit + 1):
        # policies given (V, q): consumption from the first-order condition, d and S by maximizing
        # the Hamiltonian
        S, c, d, u, Hval, which, S_free, d_free = model.policies(V, q, n_golden)
        # optional: hold chattering nodes at zero drift
        if chatter_fix:
            if S_prev is not None:
                flips = (np.sign(S) != np.sign(S_prev)) & (np.abs(S) > 1e-7) & (np.abs(S_prev) > 1e-7)
                flip_times.append(flips)
                flip_hist += flips
                if len(flip_times) > chatter_window:
                    flip_hist -= flip_times.pop(0)
                newly = (flip_hist >= chatter_flips) & (hold == 0)
                hold[newly] = chatter_hold
            forced = hold > 0
            n_forced_last = int(forced.sum())
            if n_forced_last:
                _zero_drift_override(model, forced, S, c, d, u, which, q, n_golden)
            hold[forced] -= 1
            S_prev = S.copy()
        # assemble the upwind generator and compute the residuals of the HJB and pricing equations
        A = model.assemble(S)
        Vf, qf, uf, df = V.ravel(), q.ravel(), u.ravel(), d.ravel()
        xi = par.xi(df)
        rV = float(np.max(np.abs(par.rho * Vf - uf - A @ Vf)))
        rq = float(np.max(np.abs(xi * qf - (1.0 - df) * coupon - A @ qf)))
        if it % print_every == 0 or it == 1:
            hist.append((it, dt, rV, rq))
            if verbose:
                print(f"step {it:7d} dt {dt:.3g} resV {rV:.2e} resq {rq:.2e} S[{S.min():+.4f},{S.max():+.4f}] "
                      f"q[{q.min():.3f},{q.max():.3f}]" + (f" forced0 {n_forced_last}" if chatter_fix else "")
                      + f" t={time.perf_counter() - t0:.0f}s", flush=True)
            if save_path:
                np.savez(save_path, V=V, q=q, S=S, c=c, d=d, which=which, B=model.B, z=model.z, it=it, rV=rV, rq=rq)
        # stop when both residuals are below tolerance
        if rV < tolV and rq < tolq:
            converged = True
            break
        # optional adaptive step: grow while the HJB residual falls, shrink otherwise
        if dt_adapt:
            if rV < rV_prev:
                dt = min(dt * 1.02, dt_max)
            else:
                dt = max(dt * 0.7, 1e-4)
            rV_prev = rV
        # implicit step for V, then for q (q kept in (0, 1])
        Vn = spla.spsolve((par.rho + 1.0 / dt) * I - A, uf + Vf / dt)
        qn = spla.spsolve(sp.diags(xi + 1.0 / dt, format="csc") - A, (1.0 - df) * coupon + qf / dt)
        V = Vn.reshape(model.nB, model.nz)
        q = np.clip(qn, 1e-12, 1.0).reshape(model.nB, model.nz)
    elapsed = time.perf_counter() - t0
    # policies, generator, and ergodic density at the final (V, q)
    S, c, d, u, Hval, which, S_free, d_free = model.policies(V, q, n_golden)
    if chatter_fix and (hold > 0).any():
        _zero_drift_override(model, hold > 0, S, c, d, u, which, q, n_golden)
    A = model.assemble(S)
    if verbose:
        print(f"{'converged' if converged else 'NOT converged'} after {it} steps, {elapsed:.0f}s; resV {rV:.2e} resq {rq:.2e}")
    g = ergodic_density(model, A) if (ergodic and model.nz > 1) else None
    return Result(model, V, q, S, c, d, which, S_free, d_free, A, g, converged, it, rV, rq, hist, elapsed)
