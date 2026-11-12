"""
pinn_solve.py: physics-informed neural-network solution of the diffusion partial default model.
Self-contained: needs numpy and torch (scipy only for the ergodic monitor).

Parameters and the finite-difference reference come from one file, the paper's main solution
(data/ou_main.npz: the parameter vector x, the standard qf, the grid, V, q, S, d).

Representation.  One network maps (B, z), scaled to [-1, 1], to (V, q): V = Vbar + sV * N_0,
q = sigmoid(N_1).  Policies (d, S, c) are the exact maximizers of the Hamiltonian given V_B (autodiff)
and q, by the same kernel as the finite-difference code (pdmodel.best_node): consumption from the
first-order condition clamped to the admissible drift interval (consumption floor, drift cap, the
underwriting standard S <= -a(d) B where q < q_floor, the state constraints at B = 0 and B = Bmax),
and the default intensity by a golden-section search over (0, 1] plus the candidates d = 0, d = 1 and
the closed-form interior intensity.

Scheme.  The stationary residual loss has spurious minimizers for this model,
so the network is trained by the implicit pseudo-time iteration of the paper's finite-difference
scheme with the grid replaced by quasi-random collocation points: at outer step k the policies are
frozen at the current network and the network is retrained (warm-started, Adam) to solve

    V - dt [u(c) + S V_B + L_z V - rho V]              = V_old
    q - dt [(1-d)(lambda+delta) + S q_B + L_z q - xi(d) q] = q_old

in least squares over fresh Sobol points, with Neumann penalties at z = +-zbar.  The pseudo time
step grows geometrically to dt_max.  A final phase keeps the samples fixed and polishes with L-BFGS.
The iteration starts from the static initial guess of the finite-difference code.

Monitors (every --monitor outer steps, on the reference grid): the stationary residuals of the
network solution, the change per unit pseudo-time, rms errors of V and q against the reference,
agreement of the default region and of the drift sign, and the ergodic mean debt and default
probability from the upwind generator assembled with the network's policies.

Optional (off by default): collocation points concentrated near the first-default, floor, and zero-drift loci
(--near, --near_floor, --near_zero); exact reflecting boundaries in z (--zhard); a costate (slope) residual and a
curvature penalty on V_B in the loss (--w_slope, --w_curv); and averages of the network's weights over outer steps
(--ema), which smooth the step-to-step training noise in V_B. The averages are outputs only; each is saved as
<tag>_ema<d>_grid.npz.

Usage:   python code/pinn_solve.py --ref data/ou_main.npz --device cuda --tag base
Outputs: <out>/<tag>_log.txt, <tag>_ckpt.pt (network, resumable with --init), <tag>_grid.npz (the
network solution and its residuals on the reference grid, read by pinn_eval.py).
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="ou_main.npz", help="the main solution (data/ou_main.npz): parameters, grid, reference solution")
ap.add_argument("--outer", type=int, default=2500, help="outer (pseudo-time) steps")
ap.add_argument("--inner", type=int, default=50, help="Adam iterations per outer step")
ap.add_argument("--dt", type=float, default=0.02, help="initial pseudo time step")
ap.add_argument("--dt_max", type=float, default=0.20)
ap.add_argument("--dt_every", type=int, default=100, help="multiply dt by 1.5 every this many outer steps")
ap.add_argument("--n", type=int, default=16384, help="interior collocation points per outer step")
ap.add_argument("--near", type=float, default=0.0, help="share of interior points placed near the default-threshold locus")
ap.add_argument("--near_floor", type=float, default=0.0, help="share of interior points placed near the floor locus q = q_floor")
ap.add_argument("--near_zero", type=float, default=0.0, help="share of interior points placed near the zero-drift locus")
ap.add_argument("--near_width", type=float, default=0.04, help="standard deviation in B of the points placed near a locus")
ap.add_argument("--zhard", action="store_true", help="exact reflecting boundaries: input sin(pi z / (2 zbar)), so V_z = q_z = 0 at +-zbar")
ap.add_argument("--width", type=int, default=128)
ap.add_argument("--depth", type=int, default=3, help="hidden layers")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lr_end", type=float, default=1e-4)
ap.add_argument("--pretrain", type=int, default=3000, help="Adam iterations fitting the static initial guess")
ap.add_argument("--monitor", type=int, default=25)
ap.add_argument("--polish_from", type=float, default=0.85, help="start of the final phase (fraction of outer steps)")
ap.add_argument("--lbfgs", type=int, default=15, help="L-BFGS iterations per outer step in the final phase (0 = none)")
ap.add_argument("--relax", type=float, default=0.5, help="policy relaxation in the final phase")
ap.add_argument("--golden", type=int, default=28, help="golden-section iterations for d")
ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
ap.add_argument("--init", default="", help="checkpoint to warm-start from")
ap.add_argument("--out", default="pinn_out")
ap.add_argument("--tag", default="base")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ema", default="", help="comma-separated decays of weight averages over outer steps, e.g. 0.98,0.995 (empty = none)")
ap.add_argument("--ema_from", type=float, default=0.6, help="start of the weight averaging (fraction of outer steps)")
ap.add_argument("--w_slope", type=float, default=0.0, help="weight of the slope (costate) residual in the loss (0 = off)")
ap.add_argument("--w_curv", type=float, default=0.0, help="weight of the curvature penalty on the slope, mean V_BBB^2 (0 = off)")
ap.add_argument("--dt_force", type=float, default=0.0, help="pseudo time step to use from the start, overriding --dt and a checkpoint's (0 = off)")
ap.add_argument("--slope_mask", type=float, default=0.0, help="leave out of the slope loss the points within this distance in B of the first-default and floor loci")
args = ap.parse_args()

# ----------------------------------------------------------------------------- device, dtype, threads
if args.device == "auto":
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEV = torch.device(args.device)
DT = torch.float64 if args.dtype == "float64" else torch.float32
torch.set_default_dtype(DT)
torch.set_num_threads(args.threads if args.threads > 0 else (os.cpu_count() or 1))
torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
log = open(os.path.join(args.out, f"{args.tag}_log.txt"), "a")


def say(*a):
    """Print and append to the log."""
    s = " ".join(str(v) for v in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()


def T(x):
    """Array to a tensor of the working dtype on the device."""
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=DT, device=DEV)


# ----------------------------------------------------------------------------- parameters from the reference file
# structural parameters, the standard, the continuous-time OU process, and the grid of the reference solution
ref = np.load(args.ref)
x = ref["x"].astype(float)
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = [float(v) for v in x]
QF = float(ref["qf"])                                  # the underwriting standard
r = lam = math.log(1.04); nu = 2.0
mu = -math.log(rho_z); sd = sig_eps / math.sqrt(1 - rho_z ** 2); sigma = sd * math.sqrt(2 * mu)
B_m, z_m = ref["B"].astype(float), ref["z"].astype(float)
BMAX = float(B_m[-1]); ZBAR = float(z_m[-1])
assert abs(z_m[0] + ZBAR) < 1e-12 and abs(B_m[0]) < 1e-12
coupon = delta + lam
ystar = math.exp(zt); F_MIN = 0.10; CMIN_FRAC = 0.05; S_CAP = 1.0
NEG = -1.0e30
V_fd, q_fd, d_fd, S_fd = ref["V"], ref["q"], ref["d"], ref["S"]

say(f"device {DEV} dtype {DT} threads {torch.get_num_threads()} | g0 {g0:.4f} g1 {g1:.4f} g2 {g2:.4f} ztilde {zt:.4f} "
    f"rho {rho:.4f} kappa {kappa:.4f} delta {delta:.4f} mu {mu:.4f} sigma {sigma:.4f} zbar {ZBAR:.4f} Bmax {BMAX} "
    f"q_floor {QF} | args {vars(args)}")


# ----------------------------------------------------------------------------- model pieces (as pdmodel.py, 'level' fixed cost)
def u(c):
    """CRRA utility."""
    return c ** (1 - nu) / (1 - nu)


def F_of(z):
    """Fixed default cost factor: 1 - g2 (y - y*) above y*, floored at F_MIN; 1 below y*."""
    y = torch.exp(z)
    return torch.where(z >= zt, torch.clamp(1 - g2 * (y - ystar), min=F_MIN), torch.ones_like(z))


def phi_of(d, F):
    """Share of output kept at default intensity d (variable and fixed costs)."""
    return torch.where(d > 0, (1 - g0 * d ** g1) * F, torch.ones_like(d))


def a_of(d):
    """Amortization net of recovered defaulted coupons, a(d) = delta - kappa (delta + lambda) d."""
    return delta - kappa * coupon * d


def xi_of(d):
    """Effective discount rate of the bond price, xi(d) = r + delta - kappa (delta + lambda) d."""
    return r + delta - kappa * coupon * d


def best_S(d, VB, q, B, y, F, mode):
    """Maximize u(c(S, d)) + S V_B over the admissible drifts at a given default intensity.
    mode: 1 at B = 0 (S >= 0), -1 at B = Bmax (S <= 0), 2 interior (free).  Returns (H, S, c)."""
    caut = phi_of(d, F) * y - (1 - d) * coupon * B          # consumption with zero issuance
    k = a_of(d) * B                                          # c = caut + q (S + k)
    cmin = CMIN_FRAC * y
    S_lo = torch.clamp((cmin - caut) / q - k, min=-S_CAP)
    S_hi = torch.full_like(B, S_CAP)
    S_hi = torch.where(q < QF, torch.minimum(S_hi, -k), S_hi)   # the standard: no issuance below q_floor
    S_lo = torch.where(mode == 1, torch.clamp(S_lo, min=0.0), S_lo)
    S_hi = torch.where(mode == -1, torch.clamp(S_hi, max=0.0), S_hi)
    feas = S_lo <= S_hi
    cstar = (torch.clamp(-VB, min=1e-12) / q) ** (-1 / nu)
    Sstar = torch.where(VB < 0, (cstar - caut) / q - k, S_hi)
    S = torch.maximum(torch.minimum(Sstar, S_hi), S_lo)
    c = torch.clamp(caut + q * (S + k), min=1e-10)
    H = torch.where(feas, u(c) + S * VB, torch.full_like(B, NEG))
    return H, S, c


INVPHI = (math.sqrt(5.0) - 1.0) / 2.0


def policies(B, z, VB, q, mode):
    """Exact Hamiltonian maximization over d in [0, 1] and S (pdmodel.best_node): the candidates d = 0,
    the golden-section maximizer on (0, 1], d = 1 and the closed-form interior intensity."""
    # start from d = 0
    y = torch.exp(z); F = F_of(z)
    H, S, c = best_S(torch.zeros_like(B), VB, q, B, y, F, mode)
    d = torch.zeros_like(B)
    # golden section on (0, 1]
    a = torch.full_like(B, 1e-9); b = torch.ones_like(B)
    x1 = b - INVPHI * (b - a); x2 = a + INVPHI * (b - a)
    f1 = best_S(x1, VB, q, B, y, F, mode)[0]; f2 = best_S(x2, VB, q, B, y, F, mode)[0]
    for _ in range(args.golden):
        left = f1 <= f2                       # maximum is in [x1, b]
        a = torch.where(left, x1, a); b = torch.where(left, b, x2)
        x1 = b - INVPHI * (b - a); x2 = a + INVPHI * (b - a)
        f1 = best_S(x1, VB, q, B, y, F, mode)[0]; f2 = best_S(x2, VB, q, B, y, F, mode)[0]
    # keep the best of d = 0, the golden-section point, d = 1, and the closed-form interior intensity
    dg = 0.5 * (a + b)
    cands = [dg, torch.ones_like(B)]
    rhs = (1 - kappa * q) * coupon * B / (g0 * g1 * F * y)
    di = torch.clamp(torch.clamp(rhs, min=1e-300) ** (1 / (g1 - 1)), max=1.0)
    cands.append(torch.where((rhs > 0) & (B > 0), di, torch.zeros_like(B)))
    for dc in cands:
        Hc, Sc, cc = best_S(dc, VB, q, B, y, F, mode)
        better = Hc > H
        H = torch.where(better, Hc, H); S = torch.where(better, Sc, S); c = torch.where(better, cc, c); d = torch.where(better, dc, d)
    return d, S, c, H


# ----------------------------------------------------------------------------- network
def mlp(width, depth, nin, nout):
    """Fully connected tanh network with depth hidden layers of the given width."""
    layers = [torch.nn.Linear(nin, width), torch.nn.Tanh()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
    layers += [torch.nn.Linear(width, nout)]
    return torch.nn.Sequential(*layers)


class Net(torch.nn.Module):
    """(B, z) -> (V, q): V = Vbar + sV * output 0, q = sigmoid(output 1), inputs scaled to [-1, 1]."""
    def __init__(self, width, depth, Vbar, sV, zhard=False):
        super().__init__()
        self.body = mlp(width, depth, 2, 2)
        self.Vbar = float(Vbar); self.sV = float(sV); self.zhard = bool(zhard)

    def forward(self, B, z):
        # with zhard the z input is sin(pi z / (2 zbar)), whose derivative vanishes at z = +-zbar, so every network
        # satisfies the reflecting (Neumann) conditions exactly
        zin = torch.sin(0.5 * math.pi * z / ZBAR) if self.zhard else z / ZBAR
        xin = torch.stack([2 * B / BMAX - 1, zin], dim=1)
        o = self.body(xin)
        return self.Vbar + self.sV * o[:, 0], torch.sigmoid(o[:, 1])


def derivs(net, B, z, second=True, third=False, curv=False):
    """V, q, and their derivatives by autodiff: first order always, V_zz and q_zz if second, the derivatives of
    V_B for the costate equation if third, and V_BBB if curv."""
    B = B.detach().requires_grad_(True); z = z.detach().requires_grad_(True)
    V, q = net(B, z)
    gV = torch.autograd.grad(V.sum(), (B, z), create_graph=True)
    gq = torch.autograd.grad(q.sum(), (B, z), create_graph=True)
    out = dict(V=V, q=q, V_B=gV[0], V_z=gV[1], q_B=gq[0], q_z=gq[1])
    if second:
        out["V_zz"] = torch.autograd.grad(gV[1].sum(), z, create_graph=True)[0]
        out["q_zz"] = torch.autograd.grad(gq[1].sum(), z, create_graph=True)[0]
    if third:          # derivatives of the slope p = V_B, for the costate equation
        gB = torch.autograd.grad(gV[0].sum(), (B, z), create_graph=True)
        out["V_BB"] = gB[0]; out["V_Bz"] = gB[1]
        out["V_Bzz"] = torch.autograd.grad(gB[1].sum(), z, create_graph=True)[0]
    if curv:           # curvature of the slope along debt
        V_BB = out["V_BB"] if third else torch.autograd.grad(gV[0].sum(), B, create_graph=True)[0]
        out["V_BBB"] = torch.autograd.grad(V_BB.sum(), B, create_graph=True)[0]
    return out


def costate_source(B, d, S, c, q, q_B, p):
    """Source of the costate equation rho p = source + S p_B + L_z p (envelope theorem, policies held fixed):
    u'(c) [R_B + q_B l], plus (u'(c) q + p) (-a(d)) where the standard pins the drift at -a(d) B."""
    ell = S + a_of(d) * B
    R_B = -(1 - d) * coupon + q * a_of(d)
    mu_c = c ** (-nu)
    bind = (q < QF) & (ell.abs() < 1e-9)
    return mu_c * (R_B + q_B * ell) + torch.where(bind, (mu_c * q + p) * (-a_of(d)), torch.zeros_like(B))


def Lz(f_z, f_zz, z):
    """Generator of the OU endowment process applied to f."""
    return -mu * z * f_z + 0.5 * sigma ** 2 * f_zz


# ----------------------------------------------------------------------------- collocation points
sobol = torch.quasirandom.SobolEngine(2, scramble=True, seed=args.seed)
NT = 41
Z_T = torch.linspace(-ZBAR, ZBAR, NT, device=DEV)
B_T = torch.linspace(0.0, BMAX, 401, device=DEV)


def threshold_table(net):
    """Loci by productivity on a coarse grid, from the network's policies: the first-default debt, the debt at which
    the price falls through q_floor, and the first zero of the drift from above. NaN where a locus does not exist
    (points meant for it are then left uniform). Returns a (3, NT) tensor."""
    BB = B_T.repeat(NT); ZZ = Z_T.repeat_interleave(401)
    o = derivs(net, BB, ZZ, second=False)
    d, S, _, _ = policies(BB, ZZ, o["V_B"].detach(), o["q"].detach(), torch.full_like(BB, 2))
    d = d.view(NT, 401); S = S.view(NT, 401); q = o["q"].detach().view(NT, 401)
    t = torch.full((3, NT), float("nan"), device=DEV)
    for j in range(NT):
        i = torch.nonzero(d[j] > 1e-9)
        if i.numel():
            t[0, j] = B_T[i[0, 0]]
        i = torch.nonzero(q[j] < QF)
        if i.numel():
            t[1, j] = B_T[i[0, 0]]
        i = torch.nonzero((S[j, :-1] > 0) & (S[j, 1:] <= 0))
        if i.numel():
            t[2, j] = B_T[i[0, 0] + 1]
    return t


def sample(n, t_table=None):
    """n interior Sobol points plus n/16 random points on each of B = 0 and B = Bmax; returns B, z, mode.
    With t_table, shares --near, --near_floor, and --near_zero of the interior points are moved to normal draws
    (s.d. --near_width in B) around the first-default, floor, and zero-drift loci."""
    u01 = sobol.draw(n).to(device=DEV, dtype=DT)
    B = u01[:, 0] * BMAX; z = (2 * u01[:, 1] - 1) * ZBAR
    if t_table is not None:
        B = B.clone(); k0 = 0
        for row, share in enumerate((args.near, args.near_floor, args.near_zero)):
            k = int(share * n)
            if k == 0:
                continue
            sl = slice(k0, k0 + k); k0 += k
            pos = (z[sl] + ZBAR) / (2 * ZBAR) * (NT - 1)
            j = torch.clamp(pos.floor().long(), 0, NT - 2); w = pos - j
            tz = (1 - w) * t_table[row, j] + w * t_table[row, j + 1]
            Bn = torch.clamp(tz + args.near_width * torch.randn(k, device=DEV), 0.0, BMAX)
            B[sl] = torch.where(torch.isfinite(Bn), Bn, B[sl])
    nb = n // 16
    zb = (2 * torch.rand(nb, device=DEV) - 1) * ZBAR; ztp = (2 * torch.rand(nb, device=DEV) - 1) * ZBAR
    B = torch.cat([B, torch.zeros(nb, device=DEV), torch.full((nb,), BMAX, device=DEV)])
    z = torch.cat([z, zb, ztp])
    mode = torch.cat([torch.full((n,), 2, device=DEV), torch.full((nb,), 1, device=DEV), torch.full((nb,), -1, device=DEV)])
    return B, z, mode


def z_boundary(n):
    """n random points on the endowment boundaries z = -zbar and z = +zbar."""
    B = torch.rand(n, device=DEV) * BMAX
    z = torch.cat([torch.full((n // 2,), -ZBAR, device=DEV), torch.full((n - n // 2,), ZBAR, device=DEV)])
    return B, z


def static_guess(B, z):
    """The finite-difference code's initial guess (pdmodel.Model.initial_guess, q_init = 'static')."""
    y = torch.exp(z)
    c0 = torch.maximum(y - lam * B, CMIN_FRAC * y)
    V0 = u(c0) / rho
    F = F_of(z)
    dint = torch.clamp((((1 - kappa) * coupon * B) / (g0 * g1 * F * y)).clamp(min=1e-12) ** (1 / (g1 - 1)), max=1.0)
    q0 = torch.clamp((1 - dint) * coupon / xi_of(dint), min=0.05)
    return V0, q0


# ----------------------------------------------------------------------------- monitor on the reference grid
Bg, zg = np.meshgrid(B_m, z_m, indexing="ij")
Bg_t, zg_t = T(Bg.ravel()), T(zg.ravel())
mode_g = torch.full_like(Bg_t, 2); mode_g[Bg_t <= 0] = 1; mode_g[Bg_t >= BMAX] = -1


def ergodic(S_grid, d_grid):
    """Stationary distribution of the upwind generator on the reference grid with the given drift;
    returns E[B], P(d > 0) (NaN without scipy)."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
    except ImportError:
        return float("nan"), float("nan")
    nB, nz = S_grid.shape
    dB = B_m[1] - B_m[0]; dz = z_m[1] - z_m[0]
    idx = np.arange(nB * nz).reshape(nB, nz)
    rows, cols, vals = [], [], []

    def add(src, dst, rate):
        m = rate > 0
        rows.append(src[m]); cols.append(dst[m]); vals.append(rate[m])
        rows.append(src[m]); cols.append(src[m]); vals.append(-rate[m])

    S = S_grid.copy(); S[0] = np.maximum(S[0], 0); S[-1] = np.minimum(S[-1], 0)
    add(idx[:-1].ravel(), idx[1:].ravel(), np.maximum(S[:-1], 0).ravel() / dB)
    add(idx[1:].ravel(), idx[:-1].ravel(), np.maximum(-S[1:], 0).ravel() / dB)
    m_z = -mu * z_m; D = 0.5 * sigma ** 2 / dz ** 2
    up = np.broadcast_to(D + np.maximum(m_z, 0) / dz, (nB, nz)); dn = np.broadcast_to(D + np.maximum(-m_z, 0) / dz, (nB, nz))
    add(idx[:, :-1].ravel(), idx[:, 1:].ravel(), up[:, :-1].ravel())
    add(idx[:, 1:].ravel(), idx[:, :-1].ravel(), dn[:, 1:].ravel())
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nB * nz, nB * nz))
    AT = A.T.tocsr()
    M = AT.tolil(); M[0, :] = 1.0; rhs = np.zeros(nB * nz); rhs[0] = 1.0
    try:
        g = spl.spsolve(M.tocsc(), rhs)
    except Exception:
        g = np.full(nB * nz, np.nan)
    if not np.all(np.isfinite(g)) or np.abs(AT @ g).max() > 1e-6 or g.min() < -1e-8:
        lu = spl.splu((sp.identity(nB * nz, format="csc") - 10.0 * AT.tocsc()))
        g = np.full(nB * nz, 1.0 / (nB * nz))
        for _ in range(400):
            g = lu.solve(g); g = np.maximum(g, 0); g /= g.sum()
    g = g.reshape(nB, nz); g = np.maximum(g, 0); g /= g.sum()
    return float((g * Bg).sum()), float(g[d_grid > 1e-9].sum())


def on_grid(net):
    """The network solution, its policies and the stationary residuals on the reference grid."""
    o = derivs(net, Bg_t, zg_t, second=True, third=True)
    d, S, c, _ = policies(Bg_t, zg_t, o["V_B"].detach(), o["q"].detach(), mode_g)
    rV = rho * o["V"] - u(c) - S * o["V_B"] - Lz(o["V_z"], o["V_zz"], zg_t)
    rP = (rho * o["V_B"] - costate_source(Bg_t, d, S, c, o["q"], o["q_B"], o["V_B"]) - S * o["V_BB"]
          - Lz(o["V_Bz"], o["V_Bzz"], zg_t))
    rq = xi_of(d) * o["q"] - (1 - d) * coupon - S * o["q_B"] - Lz(o["q_z"], o["q_zz"], zg_t)
    f = lambda t: t.detach().cpu().numpy().reshape(Bg.shape)
    return dict(V=f(o["V"]), q=f(o["q"]), V_B=f(o["V_B"]), q_B=f(o["q_B"]), d=f(d), c=f(c), S=f(S), rV=f(rV), rq=f(rq), rP=f(rP), V_BB=f(o["V_BB"]))


# where the reference policies are smooth: no default, six nodes below the threshold, interior z (the ripple measure)
SMOOTH = np.zeros(S_fd.shape, dtype=bool)
for _j in range(3, z_m.size - 3):
    _i = np.where(d_fd[:, _j] > 1e-9)[0]
    _top = (_i[0] - 6) if _i.size else B_m.size - 6
    if _top > 4:
        SMOOTH[2:_top, _j] = True


def ripple(g):
    """RMS drift error against the reference where the reference policies are smooth."""
    return float(np.sqrt(np.mean((g["S"] - S_fd)[SMOOTH] ** 2)))


ref_EB, ref_pdef = ergodic(S_fd, d_fd)
say(f"reference: E[B] {ref_EB:.3f} P(d>0) {ref_pdef:.3f} (upwind generator on the reference grid with its own policies)")


def report_average(k, dcy, g):
    """One log line for a weight-averaged network, in the format of monitor()."""
    EB, pdef = ergodic(g["S"], g["d"])
    errV = float(np.sqrt(np.mean((g["V"] - V_fd) ** 2))); errq = float(np.sqrt(np.mean((g["q"] - q_fd) ** 2)))
    sign_ok = float(np.mean(np.sign(g["S"]) == np.sign(S_fd))); def_ok = float(np.mean((g["d"] > 1e-9) == (d_fd > 1e-9)))
    say(f"      {k:5d} avg {dcy:g} | stat. resid rms V {np.sqrt(np.mean(g['rV'] ** 2)):.2e} q {np.sqrt(np.mean(g['rq'] ** 2)):.2e} p {np.sqrt(np.mean(g['rP'][1:-1] ** 2)):.2e} "
        f"| vs FD: rmsV {errV:.4f} rmsq {errq:.4f} drift-sign {sign_ok:.3f} default-region {def_ok:.3f} ripple {ripple(g):.4f} | E[B] {EB:.3f} P(d>0) {pdef:.3f}")


def monitor(k, net, net_old, dt, phase, Btest, ztest):
    """Log the change per unit pseudo-time, the stationary residuals, errors against the reference, and ergodic
    statistics; return the network solution on the reference grid."""
    g = on_grid(net)
    with torch.no_grad():
        V_new, q_new = net(Btest, ztest); V_old, q_old = net_old(Btest, ztest)
        dV = (V_new - V_old).abs().max().item(); dq = (q_new - q_old).abs().max().item()
    EB, pdef = ergodic(g["S"], g["d"])
    errV = float(np.sqrt(np.mean((g["V"] - V_fd) ** 2))); errq = float(np.sqrt(np.mean((g["q"] - q_fd) ** 2)))
    sign_ok = float(np.mean(np.sign(g["S"]) == np.sign(S_fd))); def_ok = float(np.mean((g["d"] > 1e-9) == (d_fd > 1e-9)))
    say(f"outer {k:5d} {phase} dt {dt:.3f} | change/dt V {dV / dt:.2e} q {dq / dt:.2e} | stat. resid rms V {np.sqrt(np.mean(g['rV'] ** 2)):.2e} "
        f"q {np.sqrt(np.mean(g['rq'] ** 2)):.2e} p {np.sqrt(np.mean(g['rP'][1:-1] ** 2)):.2e} | vs FD: rmsV {errV:.4f} rmsq {errq:.4f} drift-sign {sign_ok:.3f} default-region {def_ok:.3f} ripple {ripple(g):.4f} "
        f"| E[B] {EB:.3f} P(d>0) {pdef:.3f} (FD {ref_EB:.3f} {ref_pdef:.3f}) | {time.time() - t0:.0f}s")
    return g


# ----------------------------------------------------------------------------- pretraining on the static guess
# build the network scaled to the static guess; then either resume from a checkpoint or fit the static guess by Adam
with torch.no_grad():
    V0g, q0g = static_guess(Bg_t, zg_t)
Vbar = 0.5 * (V0g.max() + V0g.min()).item(); sV = max(0.5 * (V0g.max() - V0g.min()).item(), 1.0)
net = Net(args.width, args.depth, Vbar, sV, args.zhard).to(DEV)
say(f"network: 2 -> {args.width} x {args.depth} (tanh) -> 2, {sum(p.numel() for p in net.parameters())} parameters; Vbar {Vbar:.3f} sV {sV:.3f}")
t0 = time.time()
k_start = 1
dt = args.dt
if args.init:
    ck = torch.load(args.init, map_location=DEV)
    ck_args = ck.get("args", {})
    assert bool(ck_args.get("zhard", False)) == args.zhard, "warm start across --zhard"
    assert int(ck_args.get("width", args.width)) == args.width and int(ck_args.get("depth", args.depth)) == args.depth, "warm start across architectures"
    net.load_state_dict(ck["state"])
    k_start = int(ck.get("outer", 0)) + 1; dt = float(ck.get("dt", dt))
    say(f"warm start from {args.init} at outer {k_start}, dt {dt}")
else:
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for it in range(args.pretrain):
        B, z, _ = sample(args.n)
        V0, q0 = static_guess(B, z)
        V, q = net(B, z)
        loss = ((V - V0) / net.sV).pow(2).mean() + 10 * (q - q0).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    say(f"pretraining done in {time.time() - t0:.0f}s, loss {loss.item():.3e}")

# fixed test points for the change per step, and the loci table when sampling or masking uses it
if args.dt_force > 0:
    dt = args.dt_force
    say(f"pseudo time step forced to {dt}")
Btest, ztest, _ = sample(4096)
LOCI = args.near + args.near_floor + args.near_zero > 0
assert args.near + args.near_floor + args.near_zero < 0.9
SLOPE_MASK = (args.w_slope > 0 or args.w_curv > 0) and args.slope_mask > 0
t_table = threshold_table(net) if (LOCI or SLOPE_MASK) else None


def locus_at(row, zq):
    """Debt of locus `row` of the table (0 first default, 1 floor, 2 zero drift) at the endowments zq; NaN if absent."""
    pos = (zq + ZBAR) / (2 * ZBAR) * (NT - 1)
    j = torch.clamp(pos.floor().long(), 0, NT - 2); w = pos - j
    return (1 - w) * t_table[row, j] + w * t_table[row, j + 1]

# ----------------------------------------------------------------------------- implicit pseudo-time iteration
net_old = copy.deepcopy(net)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
nzb = max(256, args.n // 8)
# weight averages over outer steps (outputs only; the iteration does not use them)
EMA_DECAYS = [float(v) for v in args.ema.split(",") if v.strip()]
ema_start = int(args.ema_from * args.outer)
ema_nets, ema_count = {}, {}
if args.init and EMA_DECAYS:
    for key, (state, count) in ck.get("ema", {}).items():
        if float(key) in EMA_DECAYS:
            ema_nets[float(key)] = copy.deepcopy(net); ema_nets[float(key)].load_state_dict(state); ema_count[float(key)] = int(count)
    if ema_nets:
        say(f"weight averages resumed: {sorted(ema_nets)}")
prev = None
fixed = None
polish_start = int(args.polish_from * args.outer)
for k in range(k_start, args.outer + 1):
    # final phase: fixed samples, relaxed policies, and L-BFGS after Adam
    final = k > polish_start
    if final and fixed is None:
        fixed = sample(args.n, t_table)
        say(f"final phase from outer {k}: fixed samples, L-BFGS {args.lbfgs} iterations per step, policy relaxation {args.relax}, dt {dt:.3f}")
    # learning rate decays geometrically from lr to lr_end
    lr = args.lr * (args.lr_end / args.lr) ** (k / args.outer)
    for gparam in opt.param_groups:
        gparam["lr"] = lr
    # policies frozen at the previous network on this step's points
    B, z, mode = fixed if final else sample(args.n, t_table)
    o = derivs(net_old, B, z, second=False)
    d_k, S_k, c_k, _ = policies(B, z, o["V_B"].detach(), o["q"].detach(), mode)
    if final and prev is not None:
        d_k = args.relax * d_k + (1 - args.relax) * prev[0]; S_k = args.relax * S_k + (1 - args.relax) * prev[1]; c_k = args.relax * c_k + (1 - args.relax) * prev[2]
    if final:
        prev = (d_k, S_k, c_k)
    # right-hand sides of the implicit step: old values and frozen flow terms
    V_k, q_k = o["V"].detach(), o["q"].detach()
    uc = u(c_k); src_q = (1 - d_k) * coupon; xid = xi_of(d_k)
    if args.w_slope > 0 or args.w_curv > 0:
        keep_p = mode == 2                                   # interior points only
        if SLOPE_MASK:
            for row in (0, 1):
                keep_p = keep_p & ~((B - locus_at(row, z)).abs() < args.slope_mask)
        keep_p = keep_p.to(DT)
    if args.w_slope > 0:
        p_k = o["V_B"].detach()
        src_p = costate_source(B, d_k, S_k, c_k, q_k, o["q_B"].detach(), p_k)
    Bzb, zzb = z_boundary(nzb)

    def loss_fn():
        """Least-squares residual of the implicit step for V and q, optional slope terms, and the Neumann penalty."""
        o = derivs(net, B, z, second=True, third=args.w_slope > 0, curv=args.w_curv > 0)
        resV = o["V"] - dt * (uc + S_k * o["V_B"] + Lz(o["V_z"], o["V_zz"], z) - rho * o["V"]) - V_k
        resq = o["q"] - dt * (src_q + S_k * o["q_B"] + Lz(o["q_z"], o["q_zz"], z) - xid * o["q"]) - q_k
        L = (resV / (net.sV * dt)).pow(2).mean() + (resq / dt).pow(2).mean()
        if args.w_slope > 0:   # implicit step of the costate equation for p = V_B, same drift, source from the old network
            resp = o["V_B"] - dt * (src_p + S_k * o["V_BB"] + Lz(o["V_Bz"], o["V_Bzz"], z) - rho * o["V_B"]) - p_k
            L = L + args.w_slope * (keep_p * (resp / dt).pow(2)).sum() / keep_p.sum()
        if args.w_curv > 0:    # smoothness of the slope away from the loci
            L = L + args.w_curv * (keep_p * o["V_BBB"].pow(2)).sum() / keep_p.sum()
        if not net.zhard:      # with zhard the Neumann conditions hold exactly
            ob = derivs(net, Bzb, zzb, second=False)
            L = L + 0.1 * (ob["V_z"] / net.sV).pow(2).mean() + 0.1 * ob["q_z"].pow(2).mean()
        return L

    # retrain the network on the implicit step: Adam, then L-BFGS in the final phase
    for it in range(args.inner):
        loss = loss_fn()
        opt.zero_grad(); loss.backward(); opt.step()
    if final and args.lbfgs > 0:
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=args.lbfgs, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(); L = loss_fn(); L.backward(); return L
        lb.step(closure)
    # update the weight averages
    if EMA_DECAYS and k > ema_start:
        with torch.no_grad():
            for dcy in EMA_DECAYS:
                if dcy not in ema_nets:
                    ema_nets[dcy] = copy.deepcopy(net); ema_count[dcy] = 1
                else:
                    t = ema_count[dcy]; w_old = min(dcy, t / (t + 1.0)); ema_count[dcy] = t + 1   # running mean, then exponential
                    for pe, pn in zip(ema_nets[dcy].parameters(), net.parameters()):
                        pe.mul_(w_old).add_(pn, alpha=1.0 - w_old)
    # monitor, and save the averages, the checkpoint, and the solution on the reference grid
    if k % args.monitor == 0 or k == k_start or k == args.outer:
        g = monitor(k, net, net_old, dt, "final" if final else "adam ", Btest, ztest)
        for dcy, enet in ema_nets.items():
            ge = on_grid(enet)
            report_average(k, dcy, ge)
            np.savez(os.path.join(args.out, f"{args.tag}_ema{dcy:g}_grid.npz"), B=B_m, z=z_m, **ge, outer=k, dt=dt,
                     elapsed=time.time() - t0, width=args.width, depth=args.depth, n=args.n, qf=QF, x=x,
                     inner=args.inner, dt0=args.dt, dt_every=args.dt_every, device=str(DEV), ema=dcy, ema_steps=ema_count[dcy])
        torch.save({"state": net.state_dict(), "outer": k, "dt": dt, "args": vars(args), "Vbar": Vbar, "sV": sV,
                    "ema": {f"{dcy:g}": (enet.state_dict(), ema_count[dcy]) for dcy, enet in ema_nets.items()}},
                   os.path.join(args.out, f"{args.tag}_ckpt.pt"))
        np.savez(os.path.join(args.out, f"{args.tag}_grid.npz"), B=B_m, z=z_m, **g, outer=k, dt=dt,
                 elapsed=time.time() - t0, width=args.width, depth=args.depth, n=args.n, qf=QF, x=x,
                 inner=args.inner, dt0=args.dt, dt_every=args.dt_every, device=str(DEV))
    # advance pseudo time: refresh the loci table every 25 steps, grow dt every dt_every steps before the final phase
    net_old = copy.deepcopy(net)
    if (LOCI or SLOPE_MASK) and k % 25 == 0:
        t_table = threshold_table(net)
    if (not final) and k % args.dt_every == 0 and dt < args.dt_max:
        dt = min(args.dt_max, dt * 1.5)
say(f"done {time.time() - t0:.0f}s")
