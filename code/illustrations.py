"""Deterministic and three-state illustrations at the selected point (data/ou_best.npz: x and the floor).

  det     200 nodes per unit of debt on [0, 1.2]  -> data/det_floor.npz
  fine    500 nodes per unit on [0, 1]            -> data/det_fine.npz   (the cube-root check)
  fine15  500 nodes per unit on [0, 1.5]          -> data/det_fine15.npz
  jump    three states -sd, 0, +sd (sd 0.08), adjacent switches at mu/2, rare jump to the low state at 0.03,
          201 nodes on [0, 1]                     -> data/jump_floor.npz
Each output also stores the closed-form constants (Bpd, dfirst, Bfd1, cbar, beta, alpha, Sigma1, Q1).
The log goes to data/illustrations_<mode>.log.
Usage: python code/illustrations.py <det|fine|fine15|jump>
"""
import os
import sys
import time

import numpy as np

from pdutil import DATA, R_FIX
from pdmodel import Model, Params, ZProcess, diagnostics, ergodic_density
from pdtimestep import solve_timestep

# selected parameters; OU mean reversion mu and volatility sigma from the annual AR(1)
MODE = sys.argv[1] if len(sys.argv) > 1 else "det"
a = np.load(os.path.join(DATA, "ou_best.npz"))
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = a["x"].astype(float)
QF = float(a["qf"])
r = lam = R_FIX
mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
par = Params(nu=2.0, rho=rho, r=r, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2, ztilde=zt,
             fixed_cost_form="level", S_cap=1.0, q_floor=QF)
log = open(os.path.join(DATA, f"illustrations_{MODE}.log"), "w")


def say(*a_):
    """Print a line and append it to the log."""
    s = " ".join(str(v) for v in a_); print(s, flush=True); log.write(s + "\n"); log.flush()


def with_density(m, res):
    """Attach the ergodic density; fall back to implicit power iteration if the direct solve fails."""
    if res.g is None:
        res.g = ergodic_density(m, res.A)
    if not np.all(np.isfinite(res.g)) or res.g.sum() <= 0:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        N = m.N; g = np.ones(N) / N
        M = spla.splu((sp.identity(N, format="csc") - 200.0 * res.A.T).tocsc())
        for _ in range(400):
            g = M.solve(g); g = np.maximum(g, 0); g /= g.sum()
        res.g = g.reshape(m.nB, m.nz)
    return res


say(f"parameters: g0 {g0:.4f} g1 {g1:.4f} g2 {g2:.4f} ztilde {zt:.4f} rho {rho:.4f} kappa {kappa:.4f} delta {delta:.4f} r=lam {r:.4f}; "
    f"OU mu {mu:.4f} sigma {sigma:.4f} sd {sd:.4f}; q_floor {QF}")
# closed forms at z = 0
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
qr_first = (1 - dfirst) * (delta + lam) / (r + delta - kappa * (delta + lam) * dfirst)
s_first = (delta + lam) * (1 / qr_first - 1)
say(f"closed forms at z=0: F {F:.5f}  B_pd(0,1) {Bpd:.4f}  d_first {dfirst:.4f}  B_fd(0,1) {Bfd1:.4f}  cbar {cbar:.4f}  beta {beta:.4f}  "
    f"alpha {alpha:.4f}  Sigma1 {Sig1:.4f}  Q1 {Q1:.4f}  sqrt-law {sq:.4f}  a {par.a:.4f}  xi(1)/a {par.xi(1.0) / par.a:.3f}  "
    f"q_r(d_first) {qr_first:.4f}  s_first {1e4 * s_first:.0f} bp  floor spread {1e4 * (delta + lam) * (1 / QF - 1):.0f} bp  wedge {1e4 * (rho - r):.0f} bp")
consts = dict(Bpd=Bpd, dfirst=dfirst, Bfd1=Bfd1, cbar=cbar, beta=beta, alpha=alpha, sq=sq, rho=rho, lam=lam, nu=nu, Sig1=Sig1, Q1=Q1)


def report_1d(m, res, tag):
    """Log, for each z, the first default node, the steady state, full default, and where the floor binds."""
    for j in range(m.nz):
        d, q, S = res.d[:, j], res.q[:, j], res.S[:, j]
        i1 = np.where(d > 1e-9)[0]; ifd = np.where(d >= 1 - 1e-9)[0]; ifl = np.where(q < QF)[0]; i0 = np.where(S > 1e-10)[0]
        say(f"  [{tag}] z={m.z[j]:+.4f}: first default node {m.B[i1[0]] if i1.size else np.nan:.4f} (d there {d[i1[0]] if i1.size else np.nan:.4f}); "
            f"last positive-drift node {m.B[i0[-1]] if i0.size else np.nan:.4f}; full default from {m.B[ifd[0]] if ifd.size else np.nan}; "
            f"floor binds from B = {m.B[ifl[0]] if ifl.size else np.nan}; q(Bmax) {q[-1]:.4f}; drift range [{S.min():+.4f}, {S.max():+.4f}]")


def det(nodes_per_unit, Bmax, tag, maxit):
    """Solve the deterministic model on [0, Bmax] and save the policies, density, and closed forms."""
    t0 = time.time()
    nB = int(round(Bmax * nodes_per_unit)) + 1
    m = Model(par, ZProcess.none(0.0), 0.0, Bmax, nB)
    res = solve_timestep(m, None, None, dt=0.01, maxit=maxit, print_every=5000, verbose=True, dt_adapt=False, tolV=1e-7, tolq=1e-8, q_init="static")
    res = with_density(m, res)
    say(f"deterministic ({nodes_per_unit}/unit, Bmax {Bmax}): converged {res.converged}, residuals {res.resV:.1e} / {res.resq:.1e}, {res.iters} steps, {time.time() - t0:.0f}s")
    report_1d(m, res, tag)
    # steady state: the last node with positive drift
    i0 = np.where(res.S[:, 0] > 1e-10)[0]
    Bstar = m.B[i0[-1]] if i0.size else np.nan
    np.savez(os.path.join(DATA, f"{tag}.npz"), V=res.V[:, 0], q=res.q[:, 0], S=res.S[:, 0], c=res.c[:, 0], d=res.d[:, 0],
             B=m.B, Bstar=Bstar, issuance=res.issuance[:, 0], g=res.g[:, 0], resV=res.resV, resq=res.resq, converged=res.converged, **consts)


if MODE == "det":
    det(200, 1.2, "det_floor", 30000)
elif MODE == "fine":
    det(500, 1.0, "det_fine", 60000)
elif MODE == "fine15":
    det(500, 1.5, "det_fine15", 60000)
elif MODE == "jump":
    t0 = time.time()
    Bmax = 1.0; nB = int(round(Bmax / 0.005)) + 1
    eta = mu / 2; sdj = 0.08; eps = 0.03
    # three-state chain: adjacent switches at rate eta, jump to the low state at rate eps
    zs = np.array([-sdj, 0.0, sdj])
    Lam = np.array([[-eta, eta, 0.0], [eps, -(eps + eta), eta], [eps, eta, -(eps + eta)]])
    mj = Model(par, ZProcess("jump", zs, Lambda=Lam), 0.0, Bmax, nB)
    say(f"jump: three states z = {zs}, eta = {eta:.4f}, eps = {eps}, Bmax {Bmax}, {nB} nodes")
    res = solve_timestep(mj, None, None, dt=0.01, maxit=40000, print_every=5000, verbose=True, dt_adapt=False, tolV=1e-7, tolq=1e-8, q_init="static")
    res = with_density(mj, res)
    dg = diagnostics(res, verbose=False)
    say(f"jump: converged {res.converged} residuals {res.resV:.1e} / {res.resq:.1e} in {res.iters} steps, {time.time() - t0:.0f}s")
    say(f"  E[B] {dg.get('mean_B'):.3f}; P(d>0) {dg.get('frac_default'):.3f}; mass top 3 nodes {dg.get('mass_top3'):.2e}")
    report_1d(mj, res, "jump")
    # ergodic mass, default mass, and mean debt by state
    w = res.g / res.g.sum()
    for j in range(mj.nz):
        say(f"  state z={mj.z[j]:+.3f}: mass {w[:, j].sum():.3f}, mass in default {w[res.d[:, j] > 1e-9, j].sum():.3f}, "
            f"E[B|state] {(w[:, j] * mj.B).sum() / w[:, j].sum():.3f}, mode node {mj.B[np.argmax(w[:, j])]:.3f} (mass {w[:, j].max():.3f})")
    np.savez(os.path.join(DATA, "jump_floor.npz"), V=res.V, q=res.q, S=res.S, c=res.c, d=res.d, g=res.g, B=mj.B, z=mj.z,
             issuance=res.issuance, eta=eta, eps=eps, resV=res.resV, resq=res.resq, converged=res.converged, **consts)
else:
    raise SystemExit("mode: det | fine | fine15 | jump")
say("done")
