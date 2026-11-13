"""Deterministic and finite-state illustrations at the calibrated parameters.

The parameters and the underwriting standard are those of the main solution (data/ou_main.npz); only the
endowment process differs. Each mode solves one model by the implicit upwind time stepping of the main solution.

  det      deterministic (z = 0), 200 nodes per unit of debt on [0, 1.2]   -> data/det_floor.npz
  fine     deterministic, 500 nodes per unit on [0, 1]                     -> data/det_fine.npz
  fine15   deterministic, 500 nodes per unit on [0, 1.5]                   -> data/det_fine15.npz
  jump     two states -0.05, +0.05 that switch at the rate mu / 2          -> data/jump_floor.npz
  jumpbuf  two states -0.06, +0.06 that switch at the rate 0.15            -> data/jumpbuf_floor.npz
  jump3s   three states -0.05, 0, +0.05: adjacent switches at mu / 2 and a direct jump to the low state
           at the rate 0.03                                                -> data/jump3s_floor.npz
The chains are on 201 debt nodes on [0, 1]; mu is the mean reversion of the calibrated endowment process.
Every output also stores the closed-form constants at z = 0 (Bpd, dfirst, Bfd1, cbar, beta, alpha, sq, Sig1, Q1).

Reads data/ou_main.npz. Writes the file of the mode and the log data/illustrations_<mode>.log.
Usage: python code/illustrations.py <det|fine|fine15|jump|jumpbuf|jump3s>
"""
import os
import sys
import time

import numpy as np

from paths import DATA
from calutil import R_FIX
from pdmodel import Model, Params, ZProcess, diagnostics, ergodic_density, u_fn
from pdtimestep import solve_timestep

MODES = ("det", "fine", "fine15", "jump", "jumpbuf", "jump3s")
if len(sys.argv) != 2 or sys.argv[1] not in MODES:
    raise SystemExit("usage: python code/illustrations.py <" + "|".join(MODES) + ">")
MODE = sys.argv[1]

# calibrated parameters and standard; OU mean reversion mu and volatility sigma from the annual AR(1)
a = np.load(os.path.join(DATA, "ou_main.npz"))
QF, BAND, LBAND = float(a["qf"]), float(a["band"]), float(a["lband"])
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = a["x"].astype(float)
r = lam = R_FIX
mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
par = Params(nu=2.0, rho=rho, r=r, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2, ztilde=zt,
             fixed_cost_form="level", S_cap=1.0, q_floor=QF, q_band=BAND, l_band=LBAND, closed_d=True)
# time-step ramp and tie rule of the main solution
STEP = dict(dt=0.015, dt_adapt=True, dt_grow=1.02, dt_max=0.5, sticky=1e-6, freeze_flips=3)
log = open(os.path.join(DATA, f"illustrations_{MODE}.log"), "w")


def say(*a_):
    """Print a line and append it to the log."""
    s = " ".join(str(x) for x in a_)
    print(s, flush=True)
    log.write(s + "\n"); log.flush()


def with_density(m, res):
    """Attach the stationary density if the time stepper did not return a usable one."""
    if res.g is None:
        res.g = ergodic_density(m, res.A)
    if not np.all(np.isfinite(res.g)) or res.g.sum() <= 0:
        # atoms at absorbing nodes: implicit Euler on the forward equation from a uniform start converges to them
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        N = m.N
        g = np.ones(N) / N
        M = spla.splu((sp.identity(N, format="csc") - 200.0 * res.A.T).tocsc())
        for _ in range(400):
            g = M.solve(g); g = np.maximum(g, 0); g /= g.sum()
        res.g = g.reshape(m.nB, m.nz)
    return res


say(f"parameters: g0 {g0:.4f} g1 {g1:.4f} g2 {g2:.4f} ztilde {zt:.4f} rho {rho:.4f} kappa {kappa:.4f} "
    f"delta {delta:.4f} r=lam {r:.4f}; OU mu {mu:.4f} sigma {sigma:.4f} sd {sd:.4f}; q_floor {QF} band {BAND}")

# closed forms at z = 0: thresholds, the first default, and the coefficients of the cube-root and square-root laws
F = max(1 - g2 * (1.0 - np.exp(zt)), par.F_min)
G = g0 * g1 * F
K = (1 - kappa) * (delta + lam)
Bpd = ((g1 / (g1 - 1)) * (1 - F) * G ** (1 / (g1 - 1)) / K ** (g1 / (g1 - 1))) ** ((g1 - 1) / g1)
dfirst = ((1 - F) / ((g1 - 1) * g0 * F)) ** (1 / g1)
Bfd1 = G / K
cbar = 1.0 - lam * Bpd
nu = 2.0
a_df = delta - kappa * (delta + lam) * dfirst
Pi0 = (1 - kappa) * (delta + lam) * dfirst
beta = (3 * a_df * Bpd * Pi0 * cbar / nu) ** (1 / 3)
alpha = 3 * Pi0 / (2 * beta)
sq = np.sqrt(2 * (rho - lam) * cbar / nu)
RB = -(1 - dfirst) * (delta + lam) + a_df
Sig1 = -(2 * a_df * Bpd * alpha + beta ** 2 / cbar + 3 * (cbar / nu) * (RB + rho) / beta) / 4
Q1 = -Pi0 * Sig1 / beta ** 2
say(f"closed forms at z=0: F {F:.5f}  B_pd(0,1) {Bpd:.4f}  d_first {dfirst:.4f}  B_fd(0,1) {Bfd1:.4f}  "
    f"cbar {cbar:.4f}  beta {beta:.4f}  alpha {alpha:.4f}  Sigma1 {Sig1:.4f}  Q1 {Q1:.4f}  sqrt-law coefficient {sq:.4f}  a {par.a:.4f}  xi(1)/a {par.xi(1.0) / par.a:.3f}")
consts = dict(Bpd=Bpd, dfirst=dfirst, Bfd1=Bfd1, cbar=cbar, beta=beta, alpha=alpha, sq=sq, rho=rho, lam=lam, nu=nu, Sig1=Sig1, Q1=Q1)


def residual_fields(res):
    """HJB and pricing residuals of the solution, node by node."""
    u = np.vectorize(lambda cc: u_fn(cc, par.nu))(res.c)
    coupon = par.delta + par.lam
    rV = np.abs(par.rho * res.V.ravel() - u.ravel() - res.A @ res.V.ravel()).reshape(res.V.shape)
    rq = np.abs(par.xi(res.d.ravel()) * res.q.ravel() - (1 - res.d.ravel()) * coupon - res.A @ res.q.ravel()).reshape(res.V.shape)
    return rV, rq


def report_residuals(m, res, tag):
    """Log the residuals on the nodes with mass and the largest pricing residual; return both fields."""
    rV, rq = residual_fields(res)
    w = res.g / res.g.sum()
    mass = w > 1e-10
    i, j = np.unravel_index(np.argmax(rq), rq.shape)
    say(f"  [{tag}] residuals on nodes with mass above 1e-10 ({int(mass.sum())} nodes): {rV[mass].max():.1e} / {rq[mass].max():.1e}; "
        f"largest pricing residual {rq[i, j]:.1e} at B {m.B[i]:.3f}, z {m.z[j]:+.3f} (q {res.q[i, j]:.3f}, d {res.d[i, j]:.3f}, mass {w[i, j]:.1e}); "
        f"{int((rq > 1e-6).sum())} nodes above 1e-6, from B {m.B[np.where((rq > 1e-6).any(1))[0].min()] if (rq > 1e-6).any() else np.nan:.3f}; "
        f"tie nodes {int(getattr(res, 'n_frozen', 0))}")
    return rV, rq


def report_1d(m, res, tag):
    """Log, for each state, the first default node, the last node with positive drift, and where full default
    and the standard start."""
    for j in range(m.nz):
        d, q, S = res.d[:, j], res.q[:, j], res.S[:, j]
        i1 = np.where(d > 1e-9)[0]; ifd = np.where(d >= 1 - 1e-9)[0]; ifl = np.where(q < QF)[0]
        i0 = np.where(S > 1e-10)[0]
        say(f"  [{tag}] z={m.z[j]:+.4f}: first default node {m.B[i1[0]] if i1.size else np.nan:.4f} "
            f"(d there {d[i1[0]] if i1.size else np.nan:.4f}); last positive-drift node {m.B[i0[-1]] if i0.size else np.nan:.4f}; "
            f"full default from {m.B[ifd[0]] if ifd.size else np.nan}; standard binds from B = {m.B[ifl[0]] if ifl.size else np.nan}; "
            f"q(Bmax) {q[-1]:.4f}; drift range [{S.min():+.4f}, {S.max():+.4f}]")


def det(nodes_per_unit, Bmax, tag, maxit):
    """Solve the deterministic model on [0, Bmax] and save the policies, the density, and the closed forms."""
    t0 = time.time()
    nB = int(round(Bmax * nodes_per_unit)) + 1
    m = Model(par, ZProcess.none(0.0), 0.0, Bmax, nB)
    res = solve_timestep(m, None, None, maxit=maxit, print_every=5000, verbose=True, **STEP,
                         tolV=1e-7, tolq=1e-8, q_init="static")
    res = with_density(m, res)
    say(f"deterministic ({nodes_per_unit}/unit, Bmax {Bmax}): converged {res.converged}, residuals {res.resV:.1e} / {res.resq:.1e}, "
        f"{res.iters} steps, {time.time() - t0:.0f}s")
    report_1d(m, res, tag)
    i0 = np.where(res.S[:, 0] > 1e-10)[0]
    Bstar = m.B[i0[-1]] if i0.size else np.nan
    np.savez(os.path.join(DATA, f"{tag}.npz"), V=res.V[:, 0], q=res.q[:, 0], S=res.S[:, 0], c=res.c[:, 0], d=res.d[:, 0],
             B=m.B, Bstar=Bstar, issuance=res.issuance[:, 0], g=res.g[:, 0], **consts)


def chain(zp, eta, eps, name):
    """Solve the model with the finite-state endowment process zp on [0, 1] and save it as data/<name>.npz."""
    t0 = time.time()
    Bmax = 1.0
    nB = int(round(Bmax / 0.005)) + 1
    mj = Model(par, zp, 0.0, Bmax, nB)
    res = solve_timestep(mj, None, None, maxit=40000, print_every=5000, verbose=True, **STEP,
                         tolV=1e-7, tolq=1e-8, q_init="static")
    res = with_density(mj, res)
    dg = diagnostics(res, verbose=False)
    say(f"jump: converged {res.converged} residuals {res.resV:.1e} / {res.resq:.1e} in {res.iters} steps, {time.time() - t0:.0f}s")
    say(f"  Bbar(z) = {np.array2string(dg['Bbar'], precision=3)}; ceiling binds {dg['ceiling_binds']}; "
        f"E[B] {dg.get('mean_B'):.3f}; P(d>0) {dg.get('frac_default'):.3f}; mass top 3 nodes {dg.get('mass_top3'):.2e}")
    report_1d(mj, res, "jump")
    rV, rq = report_residuals(mj, res, "jump")
    w = res.g / res.g.sum()
    for j in range(mj.nz):
        say(f"  state z={mj.z[j]:+.3f}: mass {w[:, j].sum():.3f}, mass in default {w[res.d[:, j] > 1e-9, j].sum():.3f}, "
            f"E[B|state] {(w[:, j] * mj.B).sum() / w[:, j].sum():.3f}, mode node {mj.B[np.argmax(w[:, j])]:.3f} (mass {w[:, j].max():.3f})")
    np.savez(os.path.join(DATA, f"{name}.npz"), V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=mj.B, z=mj.z,
             issuance=res.issuance, eta=eta, eps=eps, resV=res.resV, resq=res.resq, rV=rV, rq=rq, **consts)


if MODE == "det":
    det(200, 1.2, "det_floor", 30000)
elif MODE == "fine":
    det(500, 1.0, "det_fine", 60000)
elif MODE == "fine15":
    det(500, 1.5, "det_fine15", 60000)
elif MODE in ("jump", "jumpbuf"):
    # jump: both states rest at the node below the low state's default threshold. jumpbuf: the high state stops
    # strictly inside the default-free set, where the Euler equation holds.
    sdj, eta = (0.05, mu / 2) if MODE == "jump" else (0.06, 0.15)
    say(f"jump: two states z = +-{sdj:.4f}, eta = {eta:.4f}, Bmax 1.0, 201 nodes")
    chain(ZProcess.jump2(-sdj, sdj, eta), eta, 0.0, f"{MODE}_floor")
else:
    sdj, eta, eps = 0.05, mu / 2, 0.03
    zs = np.array([-sdj, 0.0, sdj])
    Lam = np.array([[-eta, eta, 0.0], [eps, -(eps + eta), eta], [eps, eta, -(eps + eta)]])
    say(f"jump: three states z = {zs}, eta = {eta:.4f}, eps = {eps}, Bmax 1.0, 201 nodes")
    chain(ZProcess("jump", zs, Lambda=Lam), eta, eps, "jump3s_floor")
say("done")
