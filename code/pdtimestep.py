"""
pdtimestep.py -- the implicit time-stepping solver that iterates V and q jointly to a stationary
point (the scheme of the paper, described in its appendix on the numerical implementation). Each
step takes the policy explicitly from (V^k, q^k) and solves for V^{k+1} and q^{k+1} implicitly in
the generator A(S^k):

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
                   dt_adapt=False, dt_max=0.05, dt_grow=1.02, save_path=None, sticky=0.0,
                   freeze_flips=0, freeze_window=30, stop_fn=None, stop_every=100, sweeps=False,
                   chatter_fix=False, chatter_window=30, chatter_flips=3, chatter_hold=200) -> Result:
    """Joint implicit time stepping of V and q from (V0, q0) with step dt, until both residuals are
    below tolerance or maxit steps. Returns a Result (with the ergodic density if ergodic and nz > 1).

    dt_adapt: the time step grows by the factor dt_grow each step, from dt up to dt_max (small steps first, while
    the policy settles).
    sticky: a node keeps its upwind branch from the step before unless another branch has a Hamiltonian higher by
    more than `sticky` (pdmodel.policies); 0 is the plain comparison.
    freeze_flips: a node whose branch changed at least this many times within the last `freeze_window` steps keeps
    the branch it has from then on (0: off). Counting starts when the time step has reached dt_max, so that the
    changes of the early transient do not count. These are nodes where two branches tie and each choice moves the price
    enough to favor the other. The number of such nodes is returned as Result.n_frozen, their mask as Result.frozen.
    sweeps: solve the two linear systems of a step by line sweeps (Model.solve_shifted), with the sparse direct solve
    as the fallback; False (the default): the direct solve. Result.direct_solves counts the fallbacks, Result.passes the sweeps.
    stop_fn(model, S, d, A, it) -> bool, called every `stop_every` steps: return True to abandon the solve (the
    result then has converged = False and Result.stopped = True).
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
    which = None
    stopped = False
    n_direct = n_passes = 0
    tau = np.full((model.nB, model.nz), float(sticky))
    frozen = np.zeros((model.nB, model.nz), dtype=bool)
    changes = np.zeros((model.nB, model.nz), dtype=np.int32)
    change_hist = []
    for it in range(1, maxit + 1):
        # policies given (V, q): consumption from the first-order condition, d and S by maximizing
        # the Hamiltonian
        w_old = which
        S, c, d, u, Hval, which, S_free, d_free = model.policies(V, q, n_golden, which, tau)
        # hold the branch of nodes that keep alternating
        if freeze_flips and w_old is not None and (not dt_adapt or dt >= dt_max):   # only once the ramp is over
            ch = which != w_old
            change_hist.append(ch); changes += ch
            if len(change_hist) > freeze_window:
                changes -= change_hist.pop(0)
            new = (changes >= freeze_flips) & ~frozen
            if new.any():
                frozen |= new; tau[new] = 1.0e30
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
        if stop_fn is not None and it % stop_every == 0 and stop_fn(model, S, d, A, it):
            stopped = True
            break
        if dt_adapt:
            dt = min(dt * dt_grow, dt_max)         # a ramp: small steps first, while the policy settles
            rV_prev = rV
        # implicit step for V, then for q (q kept in (0, 1])
        if sweeps:
            V, n1 = model.solve_shifted(par.rho + 1.0 / dt, S, uf + Vf / dt, V, A)
            qn, n2 = model.solve_shifted(xi + 1.0 / dt, S, (1.0 - df) * coupon + qf / dt, q, A)
            n_direct += (n1 < 0) + (n2 < 0); n_passes += max(n1, 0) + max(n2, 0)
        else:
            V = spla.spsolve((par.rho + 1.0 / dt) * I - A, uf + Vf / dt).reshape(model.nB, model.nz)
            qn = spla.spsolve(sp.diags(xi + 1.0 / dt, format="csc") - A, (1.0 - df) * coupon + qf / dt)
        q = np.clip(qn, 1e-12, 1.0).reshape(model.nB, model.nz)
    elapsed = time.perf_counter() - t0
    # policies, generator, and ergodic density at the final (V, q)
    S, c, d, u, Hval, which, S_free, d_free = model.policies(V, q, n_golden, which, tau)
    if chatter_fix and (hold > 0).any():
        _zero_drift_override(model, hold > 0, S, c, d, u, which, q, n_golden)
    A = model.assemble(S)
    if verbose:
        print(f"{'converged' if converged else 'NOT converged'} after {it} steps, {elapsed:.0f}s; resV {rV:.2e} resq {rq:.2e}")
    g = ergodic_density(model, A) if (ergodic and model.nz > 1) else None
    res = Result(model, V, q, S, c, d, which, S_free, d_free, A, g, converged, it, rV, rq, hist, elapsed)
    res.frozen = frozen; res.n_frozen = int(frozen.sum()); res.stopped = stopped
    res.direct_solves = int(n_direct); res.passes = int(n_passes)
    return res
