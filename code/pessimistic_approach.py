"""Pessimistic approach and no-default basin in the deterministic model at the calibrated parameters.

With the calibrated parameters (hard-coded below, floor 0.6) at z = 0, integrates two paths into the
steady state B* = B_pd(0, 1), where default starts at q = 1. The pessimistic approach is the left branch of
the ODE system in (S, q, V), started from the cube-root expansion at B*; it passes through a stretch of partial
default. The no-default basin has q = 1 and starts from the square-root law at B*. Prints the stretch width, the time spent in default, the
travel times from zero debt, and the consumption-equivalent loss. Writes output/pessimistic_approach.pdf with
the price, default intensity, and drift on both paths, and the right branch above B*.
Usage: python code/pessimistic_approach.py
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# calibrated parameters and the static pieces at z = 0
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = (0.07593356, 3.06654097, 0.08152022, -0.28637351,
                                                     0.06176464, 0.37268176, 0.15250675, 0.81922832, 0.06617824)
r = lam = float(np.log(1.04)); nu = 2.0; qfloor = 0.6
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
# steady state B* and the coefficients of the cube-root expansion there
dfirst = ((1 - F0) / ((g1 - 1) * g0 * F0)) ** (1 / g1)
Bstar = Bpd(1.0); cbar = 1 - r * Bstar
Pi0 = (1 - kappa) * cs * dfirst
beta = (3 * a(dfirst) * Bstar * Pi0 * psi(cbar)) ** (1 / 3); alpha = 3 * Pi0 / (2 * beta)
Sig1 = -0.25 * (beta ** 2 * (nu + 1) / cbar + 3 * psi(cbar) * (rho - r + Pi0) / beta); Q1 = -Pi0 * Sig1 / beta ** 2
Vstar = u(cbar) / rho

def rhs(x, y, sgn):
    """ODE in the distance from B* for (S, q, V), with the static default rule."""
    # x = |B - B*|, moving away from B* on the side sgn (+1 right, -1 left); y = (S, q, V)
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
    """Integrate away from B* on side sgn, starting from the cube-root expansion at distance x0."""
    chi = sgn * x0 ** (1 / 3)
    y0 = [-beta * chi - Sig1 * chi ** 2, 1 - alpha * chi ** 2 - Q1 * chi ** 3, Vstar + sgn * x0 * (-up(cbar))]
    return solve_ivp(rhs, (x0, xmax), y0, args=(sgn,), method="LSODA", max_step=5e-4, rtol=1e-10, atol=1e-13, dense_output=True)

def basin(xmax, x0=1e-9):
    """No-default path (q = 1) below B*, from the square-root law of the drift at distance x0."""
    def f(x, y):
        S, V = y; B = Bstar - x; c = 1 - r * B + S
        return [-(r * S - (rho - r) * psi(c)) / S, up(c)]
    return solve_ivp(f, (x0, xmax), [np.sqrt(2 * (rho - r) * psi(cbar) * x0), Vstar + x0 * up(cbar)], method="LSODA", max_step=5e-4, rtol=1e-10, atol=1e-13, dense_output=True)

# right branch, pessimistic approach (left branch), and basin
Bmax = 0.85
right = branch(+1, Bmax - Bstar); left = branch(-1, Bstar); bas = basin(Bstar)
print(f"B* {Bstar:.5f}; left branch reaches B=0 with S {left.y[0,-1]:.4f}; basin S(0) {bas.y[0,-1]:.4f}")


# stretch and travel times
xs = np.linspace(1e-7, Bstar, 200001)
S_m = left.sol(xs)[0]; S_b = bas.sol(xs)[0]
q_m = left.sol(xs)[1]
d_m = np.array([dstar(Bstar - x, qq) for x, qq in zip(xs, q_m)])
x2 = xs[np.argmax(d_m == 0)]                    # start of the no-default part moving down
print(f"B* {Bstar:.4f}, B_2 {Bstar - x2:.4f}, stretch width {x2:.4f} ({100*x2:.2f}% of output)")
print(f"min price {q_m.min():.4f} at B {Bstar - xs[np.argmin(q_m)]:.4f}; spread there {1e4*cs*(1/q_m.min()-1):.0f} bp")
tau_stretch = np.trapezoid(1 / S_m[xs <= x2], xs[xs <= x2])
T_m = np.trapezoid(1 / S_m, xs); T_b = np.trapezoid(1 / S_b, xs)
print(f"time in default on the approach: {tau_stretch:.3f} years ({12*tau_stretch:.1f} months)")
print(f"time from zero debt to capacity: pessimistic {T_m:.2f} years, basin {T_b:.2f} years")
Vm0 = left.sol(Bstar)[2]; Vb0 = bas.sol(Bstar)[1]
c0 = 1 + bas.sol(Bstar)[0]
print(f"V(0): pessimistic {Vm0:.5f}, basin {Vb0:.5f}; consumption-equivalent loss {(Vb0 - Vm0) * rho / up(c0) * 100:.4f}% ")
print(f"d on the stretch: from {d_m[d_m>0].min():.3f} to {d_m[d_m>0].max():.3f}; S at B_2: pessimistic {left.sol(x2)[0]:.4f}, basin {bas.sol(x2)[0]:.4f}")

# figure in the paper's style
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
BLUE, RED, BLACK, GRAY = "#1c3f6e", "#ba0c2f", "#212325", "#868e92"   # navy, OSU scarlet, Gray Dark 80, Gray Dark 20
plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": False, "font.family": "serif",
    "pgf.preamble": r"\usepackage[sc]{mathpazo}\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.0, "legend.frameon": False,
    "figure.dpi": 100, "text.color": BLACK, "axes.labelcolor": BLACK, "axes.edgecolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
W = 6.3
# paths on a window around B*
Bl = np.linspace(0.465, Bstar - 1e-7, 4000); Br = np.linspace(Bstar + 1e-7, 0.505, 2000)
qm = left.sol(Bstar - Bl)[1]; Sm = left.sol(Bstar - Bl)[0]; dm = np.array([dstar(B, qq) for B, qq in zip(Bl, qm)])
Sb = bas.sol(Bstar - Bl)[0]; qb = np.ones_like(Bl); db = np.zeros_like(Bl)
qr = right.sol(Br - Bstar)[1]; Sr = right.sol(Br - Bstar)[0]; dr = np.array([dstar(B, qq) for B, qq in zip(Br, qr)])
def brk(x, y):   # break the line where d jumps
    y = y.copy(); j = np.where(np.diff(y > 0) != 0)[0]
    x = x.astype(float); yy = y.astype(float)
    for k in j[::-1]:
        x = np.insert(x, k + 1, np.nan); yy = np.insert(yy, k + 1, np.nan)
    return x, yy
# three panels: price, default intensity, and drift
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
    ax.set_title(title); ax.set_xlabel("debt $B$"); ax.set_xlim(0.465, 0.505); ax.set_xticks([0.47, 0.48, 0.49, 0.50])
axes[2].axhline(0, color=BLACK, lw=0.5)
axes[0].legend(loc="lower left", handlelength=1.8)
fig.tight_layout(w_pad=1.2)
from paths import OUT
out = os.path.join(OUT, "pessimistic_approach.pdf")
fig.savefig(out); print("saved", out)
