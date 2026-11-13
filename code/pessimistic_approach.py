"""Pessimistic approach and no-default basin in the deterministic model at the calibrated parameters.

At z = 0 two paths lead into capacity B* = B_pd(0, 1), where default starts at the price q = 1. The pessimistic
approach is the left branch of the ODE system in (S, q, V), started from the cube-root expansion at B*; it passes
through a stretch of partial default below B*. The no-default basin has q = 1 and starts from the square-root law
at B*. Both share the right branch above B*.

Reads data/ou_main.npz (the calibrated parameters and the underwriting standard).
  python code/pessimistic_approach.py          draws output/pessimistic_approach.pdf: price, default intensity, and
                                               drift on both paths near B*
  python code/pessimistic_approach.py --solve  writes the numbers the text quotes:
      data/approach_fig_out.txt   width of the default stretch, lowest price, time in default, travel times from
                                  zero debt, and the consumption-equivalent loss at zero debt
      data/grid_check_out.txt     residuals of the discretized model at both candidates on a grid of 4,251 debt
                                  nodes on [0, 0.85], and where 4,000 time steps from the pessimistic candidate end
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from paths import DATA, OUT

# calibrated parameters and the static pieces at z = 0
_a = np.load(os.path.join(DATA, "ou_main.npz"))
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = (float(v) for v in _a["x"])
r = lam = float(np.log(1.04)); nu = 2.0; qfloor = float(_a["qf"])
band = float(_a["band"]); lband = float(_a["lband"])
F0 = 1 - g2 * (1 - np.exp(zt)); cs = delta + lam
a = lambda d: delta - kappa * cs * d
xi = lambda d: r + a(d)
phi = lambda d: (1 - g0 * d ** g1) * F0 if d > 0 else 1.0
R = lambda d, B, q: phi(d) - (1 - d) * cs * B + q * a(d) * B
def dint(B, q): return min(1.0, ((1 - kappa * q) * cs * B / (g0 * g1 * F0)) ** (1 / (g1 - 1)))
def dstar(B, q):
    di = dint(B, q); return di if R(di, B, q) > R(0.0, B, q) else 0.0
def Bpd(q): return brentq(lambda B: R(dint(B, q), B, q) - R(0.0, B, q), 1e-9, 5.0)
u = lambda c: c ** (1 - nu) / (1 - nu); up = lambda c: c ** (-nu)
psi = lambda c: c / nu
# capacity and the coefficients of the cube-root expansion there
dfirst = ((1 - F0) / ((g1 - 1) * g0 * F0)) ** (1 / g1)
Bstar = Bpd(1.0); cbar = 1 - r * Bstar
Pi0 = (1 - kappa) * cs * dfirst
beta = (3 * a(dfirst) * Bstar * Pi0 * psi(cbar)) ** (1 / 3); alpha = 3 * Pi0 / (2 * beta)
Sig1 = -0.25 * (beta ** 2 * (nu + 1) / cbar + 3 * psi(cbar) * (rho - r + Pi0) / beta); Q1 = -Pi0 * Sig1 / beta ** 2
Vstar = u(cbar) / rho


def rhs(x, y, sgn):
    """ODE in x = |B - B*| on the side sgn (+1 above B*, -1 below); y = (S, q, V)."""
    B = Bstar + sgn * x; S, q, V = y
    d = dstar(B, q)
    RB = -(1 - d) * cs + q * a(d); Rq = a(d) * B
    c = R(d, B, q) + q * S
    qp = (xi(d) * q - (1 - d) * cs) / S
    cp = (-rho * q - RB - Rq * qp) * psi(c) / (S * q)
    Sp = (cp - RB - Rq * qp - qp * S) / q
    Vp = -q * up(c)
    return [sgn * Sp, sgn * qp, sgn * Vp]


def branch(sgn, xmax, x0=1e-7):
    """The branch through B* on the side sgn, started from the cube-root expansion at distance x0."""
    chi = sgn * x0 ** (1 / 3)
    y0 = [-beta * chi - Sig1 * chi ** 2, 1 - alpha * chi ** 2 - Q1 * chi ** 3, Vstar + sgn * x0 * (-up(cbar))]
    return solve_ivp(rhs, (x0, xmax), y0, args=(sgn,), method="LSODA", max_step=5e-4, rtol=1e-10, atol=1e-13, dense_output=True)


def basin(xmax, x0=1e-9):
    """The no-default path below B* (q = 1), started from the square-root law; y = (S, V), x = B* - B."""
    def f(x, y):
        S, V = y; B = Bstar - x; c = 1 - r * B + S
        return [-(r * S - (rho - r) * psi(c)) / S, up(c)]
    return solve_ivp(f, (x0, xmax), [np.sqrt(2 * (rho - r) * psi(cbar) * x0), Vstar + x0 * up(cbar)], method="LSODA", max_step=5e-4, rtol=1e-10, atol=1e-13, dense_output=True)


Bmax = 0.85
right = branch(+1, Bmax - Bstar); left = branch(-1, Bstar); bas = basin(Bstar)
HEAD = f"B* {Bstar:.5f}; left branch reaches B=0 with S {left.y[0,-1]:.4f}; basin S(0) {bas.y[0,-1]:.4f}"


def approach_numbers():
    """Lines with the default stretch, the travel times, and the value loss of the pessimistic approach."""
    xs = np.linspace(1e-7, Bstar, 200001)
    S_m = left.sol(xs)[0]; S_b = bas.sol(xs)[0]
    q_m = left.sol(xs)[1]
    d_m = np.array([dstar(Bstar - x, qq) for x, qq in zip(xs, q_m)])
    x2 = xs[np.argmax(d_m == 0)]                    # start of the no-default part moving down
    tau_stretch = np.trapezoid(1 / S_m[xs <= x2], xs[xs <= x2])
    T_m = np.trapezoid(1 / S_m, xs); T_b = np.trapezoid(1 / S_b, xs)
    Vm0 = left.sol(Bstar)[2]; Vb0 = bas.sol(Bstar)[1]
    c0 = 1 + bas.sol(Bstar)[0]
    return [HEAD,
            f"B* {Bstar:.4f}, B_2 {Bstar - x2:.4f}, stretch width {x2:.4f} ({100*x2:.2f}% of output)",
            f"min price {q_m.min():.4f} at B {Bstar - xs[np.argmin(q_m)]:.4f}; spread there {1e4*cs*(1/q_m.min()-1):.0f} bp",
            f"time in default on the approach: {tau_stretch:.3f} years ({12*tau_stretch:.1f} months)",
            f"time from zero debt to capacity: pessimistic {T_m:.2f} years, basin {T_b:.2f} years",
            f"V(0): pessimistic {Vm0:.5f}, basin {Vb0:.5f}; consumption-equivalent loss {(Vb0 - Vm0) * rho / up(c0) * 100:.4f}% ",
            f"d on the stretch: from {d_m[d_m>0].min():.3f} to {d_m[d_m>0].max():.3f}; S at B_2: pessimistic {left.sol(x2)[0]:.4f}, basin {bas.sol(x2)[0]:.4f}"]


def grid_check():
    """Lines with the residuals of the discretized model at both candidates, and the result of time stepping
    from the pessimistic candidate."""
    from pdmodel import Model, Params, ZProcess
    from pdtimestep import solve_timestep
    nB = 4251                       # h = 2e-4 on [0, 0.85]
    par = Params(nu=nu, rho=rho, r=r, delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2, ztilde=zt,
                 fixed_cost_form="level", S_cap=1.0, q_floor=qfloor, q_band=band, l_band=lband)
    m = Model(par, ZProcess.none(0.0), 0.0, Bmax, nB)
    Bg = m.B
    lines = [HEAD]

    def on_grid(which):
        """V, q, and the static default rule of a candidate on the grid: 'mirror' (pessimistic) or 'basin'."""
        V = np.empty(nB); q = np.empty(nB); d_ode = np.empty(nB)
        for i, B in enumerate(Bg):
            if B >= Bstar:
                S, qq, VV = right.sol(max(B - Bstar, 1e-7))
            elif which == "mirror":
                S, qq, VV = left.sol(max(Bstar - B, 1e-7))
            else:
                S, VV = bas.sol(max(Bstar - B, 1e-9)); qq = 1.0
            V[i] = VV; q[i] = qq; d_ode[i] = dstar(B, qq)
        return V.reshape(nB, 1), q.reshape(nB, 1), d_ode

    for which in ["mirror", "basin"]:
        V, q, d_ode = on_grid(which)
        lines += ["", f"[{which}] V(0) = {V[0,0]:.6f}, q min {q.min():.5f}, ODE default nodes below B*: {int((d_ode[Bg < Bstar] > 0).sum())}"]
        # HJB and pricing residuals of the discrete scheme, on the whole grid and away from the 50 nodes around B*
        S, c, d, uu, H, wh, S_free, d_free = m.policies(V, q)
        A = m.assemble(S)
        RV = rho * V.ravel() - uu.ravel() - A @ V.ravel()
        Rq = np.array([par.xi(dd) for dd in d.ravel()]) * q.ravel() - (1 - d.ravel()) * cs - A @ q.ravel()
        k = np.argmin(np.abs(Bg - Bstar)); win = slice(max(k - 25, 0), k + 25)
        mask = np.ones(nB, bool); mask[win] = False
        lines.append(f"{which}: max|R_V| {np.abs(RV).max():.2e} (away from B*: {np.abs(RV[mask]).max():.2e}), "
                     f"max|R_q| {np.abs(Rq).max():.2e} (away: {np.abs(Rq[mask]).max():.2e}); "
                     f"nodes with d>0 below B*: {int((d.ravel()[Bg < Bstar] > 0).sum())}, S>0 below B*: {int((S.ravel()[Bg < Bstar] > 0).sum())}/{int((Bg < Bstar).sum())}")

    V, q, _ = on_grid("mirror")
    t0 = time.time()
    res = solve_timestep(m, V.copy(), q.copy(), dt=0.01, maxit=4000, tolV=1e-9, tolq=1e-10, verbose=False, ergodic=False)
    below = Bg < Bstar
    Sr = res.S.ravel(); i0 = np.where(Sr > 1e-10)[0]
    lines += ["", f"time stepping from the mirror candidate: {res.iters} steps, {time.time()-t0:.0f}s, resV {res.resV:.1e} resq {res.resq:.1e}",
              f"  after: nodes with d>0 below B*: {int((res.d.ravel()[below] > 0).sum())}, q min {res.q.min():.5f} at B={Bg[np.argmin(res.q.ravel())]:.4f}, "
              f"max |q - q_mirror| below B*: {np.abs(res.q.ravel()[below] - q.ravel()[below]).max():.2e}, max |V - V_mirror|: {np.abs(res.V.ravel() - V.ravel()).max():.2e}",
              f"  steady state node after stepping: {Bg[i0[-1]] if i0.size else float('nan'):.4f}"]
    return lines


def figure():
    """Price, default intensity, and drift on both paths in a window around B*, with the right branch."""
    import matplotlib
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    BLUE, RED, BLACK, GRAY = "#1c3f6e", "#ba0c2f", "#212325", "#868e92"
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": False, "font.family": "serif",
        "pgf.preamble": r"\usepackage[sc]{mathpazo}\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.0, "legend.frameon": False,
        "figure.dpi": 100, "text.color": BLACK, "axes.labelcolor": BLACK, "axes.edgecolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    W = 6.3
    XL, XR = Bstar - 0.045, Bstar + 0.025        # window around capacity: the default stretch and a little beyond
    Bl = np.linspace(XL, Bstar - 1e-7, 4000); Br = np.linspace(Bstar + 1e-7, XR, 2000)
    qm = left.sol(Bstar - Bl)[1]; Sm = left.sol(Bstar - Bl)[0]; dm = np.array([dstar(B, qq) for B, qq in zip(Bl, qm)])
    Sb = bas.sol(Bstar - Bl)[0]; qb = np.ones_like(Bl); db = np.zeros_like(Bl)
    qr = right.sol(Br - Bstar)[1]; Sr = right.sol(Br - Bstar)[0]; dr = np.array([dstar(B, qq) for B, qq in zip(Br, qr)])

    def brk(x, y):
        """Insert a gap where the default intensity jumps between zero and a positive value."""
        y = y.copy(); j = np.where(np.diff(y > 0) != 0)[0]
        x = x.astype(float); yy = y.astype(float)
        for k in j[::-1]:
            x = np.insert(x, k + 1, np.nan); yy = np.insert(yy, k + 1, np.nan)
        return x, yy

    fig, axes = plt.subplots(1, 3, figsize=(W, 0.3 * W))
    for ax, (ym, yb, yr), title in zip(axes, [(qm, qb, qr), (dm, db, dr), (Sm, Sb, Sr)],
                                       ["bond price $q$", "default intensity $d$", "drift of debt $S$"]):
        if title.startswith("default"):
            ax.plot(*brk(Bl, yb), color=BLUE, ls="-", label="no-default basin")
            ax.plot(*brk(Bl, ym), color=RED, ls="--", label="pessimistic approach")
            ax.plot(*brk(np.concatenate([[Bstar], Br]), np.concatenate([[0.0], yr])), color=BLACK, ls="-", lw=0.8)
        else:
            ax.plot(Bl, yb, color=BLUE, ls="-", label="no-default basin")
            ax.plot(Bl, ym, color=RED, ls="--", label="pessimistic approach")
            ax.plot(Br, yr, color=BLACK, ls="-", lw=0.8)
        ax.axvline(Bstar, color=GRAY, lw=0.5, ls=":")
        ax.set_title(title); ax.set_xlabel("debt $B$"); ax.set_xlim(XL, XR); ax.set_xticks(np.round(np.arange(np.ceil(XL * 100) / 100, XR, 0.02), 2))
    axes[2].axhline(0, color=BLACK, lw=0.5)
    axes[1].legend(loc="center left", handlelength=1.8)      # the middle panel is empty between the two levels
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(OUT, "pessimistic_approach.pdf"))


if __name__ == "__main__":
    if "--solve" in sys.argv:
        for name, lines in (("approach_fig_out.txt", approach_numbers()), ("grid_check_out.txt", grid_check())):
            txt = "\n".join(lines) + "\n"
            open(os.path.join(DATA, name), "w", encoding="utf-8").write(txt)
            print(txt, flush=True)
    else:
        figure()
        print("pessimistic_approach.pdf")
