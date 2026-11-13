"""
pdmodel.py -- the continuous-time sovereign debt model with partial default.

Provides the model's primitives (Params, the endowment process ZProcess, utility, default costs),
the policies (S, c, d) from maximizing the Hamiltonian at every node, the upwind generator of
(B, z) given the drift S (Model), the ergodic density, and diagnostics. The joint time-stepping
solver is in pdtimestep.py.

One model for three endowment processes:
  * 'none'  : z fixed (deterministic)
  * 'jump'  : z on a finite set, continuous-time Markov chain with generator Lambda
  * 'ou'    : dz = -mu z dt + sigma dW with reflecting barriers

Scheme: implicit upwind finite differences (pseudo-transient / policy iteration), with a
proper Hamiltonian maximization at every node:

  for each node and for each one-sided derivative V_B (forward / backward / zero-drift),
      max over d in [0,1] and S in an admissible interval of  u(c(S,d)) + S V_B,
      c(S,d) = c_aut(d) + q (S + delta B - kappa (delta+lambda) d B),
  keep the candidate with the largest Hamiltonian.

  The maximization over S is closed form plus a clamp to the admissible interval (H is concave
  in S). The maximization over d > 0 is a golden-section search (H*(d) is concave). The
  candidates d = 0, d = 1, and the paper's closed-form interior d are also evaluated.

Unknowns V(B,z), q(B,z) are stored as (nB, nz) arrays; the linear systems use the flat
index k = i*nz + j (z fastest) so that the sparse LU is cheap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit, prange

# ----------------------------------------------------------------------------- parameters


@dataclass
class Params:
    """Model parameters. as_array() packs them into the vector p read by the numba kernels."""
    nu: float = 2.0      # CRRA
    rho: float = 0.047     # discount rate
    delta: float = 0.12    # maturity
    kappa: float = 0.7     # arrears per unit of missed payment (haircut 1-kappa)
    r: float = 0.039       # risk free rate
    lam: float | None = None  # coupon; default lam = r (q_rf = 1)
    g0: float = 0.02       # default cost level
    g1: float = 2.0        # default cost curvature (>1)
    g2: float = 3.5        # fixed cost slope in z
    ztilde: float = 0.015  # fixed cost threshold in z
    # regularization (should not matter at the solution)
    cmin_frac: float = 0.05   # consumption floor as a fraction of income
    S_cap: float = 10.0       # cap on |dB/dt|
    F_min: float = 0.10       # floor on the fixed-cost factor 1-(z-ztilde) g2
    # fixed-cost form: "log"  -> 1 - g2 (z - ztilde)            (z = log endowment)
    #                  "level"-> 1 - g2 (exp(z) - exp(ztilde))  (Arellano, Mateos-Planas, and Rios-Rull:
    #                           phi1 (z_level - z*), with z* = exp(ztilde))
    fixed_cost_form: str = "log"
    # underwriting standard: new issuance l = S + a(d) B is capped by L(q), with L = 0 at prices q <= q_floor.
    # q_band = 0: L jumps to no cap above q_floor (the hard standard). q_band > 0: L rises continuously over the
    # band (q_floor, q_floor + q_band), L(q) = l_band w / (1 - w) with w = (q - q_floor) / q_band, and there is
    # no cap above the band.
    q_floor: float = 0.0
    q_band: float = 0.0
    l_band: float = 0.1
    # default intensity: use the closed form wherever it applies (the standard and the consumption floor are slack
    # at it, and it is below one), and the line search elsewhere. False: the line search at every node.
    closed_d: bool = False

    def __post_init__(self):
        if self.lam is None:
            self.lam = self.r

    def as_array(self) -> np.ndarray:
        """Parameter vector p; p[13] is 1 for the level fixed-cost form, p[14] is q_floor."""
        return np.array(
            [self.nu, self.rho, self.delta, self.kappa, self.r, self.lam,
             self.g0, self.g1, self.g2, self.ztilde, self.cmin_frac, self.S_cap, self.F_min,
             1.0 if self.fixed_cost_form == "level" else 0.0, self.q_floor, self.q_band, self.l_band,
             1.0 if self.closed_d else 0.0],
            dtype=np.float64,
        )

    @property
    def a(self) -> float:
        """delta - kappa (delta+lam): decay rate of debt under full default and no issuance."""
        return self.delta - self.kappa * (self.delta + self.lam)

    def xi(self, d):
        """Effective discount rate of the bond price: r + delta - kappa (delta+lam) d."""
        return self.r + self.delta - self.kappa * (self.delta + self.lam) * d


# ----------------------------------------------------------------------------- z process


@dataclass
class ZProcess:
    """Log endowment process on a grid: fixed, a Markov chain, or a reflected OU diffusion."""
    kind: str                     # 'none' | 'jump' | 'ou'
    z: np.ndarray                  # grid (nz,)
    Lambda: np.ndarray | None = None   # (nz,nz) generator for 'jump'
    mu: float = 0.0                # OU mean reversion
    sigma: float = 0.0             # OU volatility

    @staticmethod
    def none(z0: float = 0.0) -> "ZProcess":
        return ZProcess("none", np.array([z0], dtype=np.float64))

    @staticmethod
    def jump2(zL: float, zH: float, eta_LH: float, eta_HL: float | None = None) -> "ZProcess":
        if eta_HL is None:
            eta_HL = eta_LH
        Lam = np.array([[-eta_LH, eta_LH], [eta_HL, -eta_HL]], dtype=np.float64)
        return ZProcess("jump", np.array([zL, zH], dtype=np.float64), Lambda=Lam)

    @staticmethod
    def ou(mu: float, sigma: float, nz: int, zmin: float | None = None, zmax: float | None = None,
           n_sd: float = 2.0) -> "ZProcess":
        sd = sigma / np.sqrt(2.0 * mu)
        if zmin is None:
            zmin = -n_sd * sd
        if zmax is None:
            zmax = n_sd * sd
        return ZProcess("ou", np.linspace(zmin, zmax, nz), mu=mu, sigma=sigma)

    @property
    def nz(self) -> int:
        return self.z.size

    def generator_offdiag(self):
        """Off-diagonal z-generator entries (j, j', value) and the diagonal (nz,)."""
        nz = self.nz
        J, Jp, vals = [], [], []
        diag = np.zeros(nz)
        if self.kind == "none":
            pass
        elif self.kind == "jump":
            for j in range(nz):
                for jp in range(nz):
                    if jp != j and self.Lambda[j, jp] != 0.0:
                        J.append(j); Jp.append(jp); vals.append(self.Lambda[j, jp])
                        diag[j] -= self.Lambda[j, jp]
        elif self.kind == "ou":
            # upwind drift plus central diffusion
            dz = self.z[1] - self.z[0]
            diff = 0.5 * self.sigma ** 2 / dz ** 2
            for j in range(nz):
                m = -self.mu * self.z[j]            # drift of z
                up = max(m, 0.0) / dz + diff        # coefficient on V(j+1)
                dn = max(-m, 0.0) / dz + diff       # coefficient on V(j-1)
                if j < nz - 1:
                    J.append(j); Jp.append(j + 1); vals.append(up); diag[j] -= up
                if j > 0:
                    J.append(j); Jp.append(j - 1); vals.append(dn); diag[j] -= dn
                # reflecting barriers: the ghost node equals the boundary node, so the entries cancel
        else:
            raise ValueError(self.kind)
        return np.array(J, dtype=np.int64), np.array(Jp, dtype=np.int64), np.array(vals), diag

    def stationary(self) -> np.ndarray:
        """Stationary distribution of z alone (probability mass on the grid)."""
        nz = self.nz
        if nz == 1:
            return np.array([1.0])
        J, Jp, vals, diag = self.generator_offdiag()
        # solve L^T g = 0 with the first equation replaced by the normalization g[0] = 1
        L = sp.coo_matrix((np.concatenate([vals, diag]), (np.concatenate([J, np.arange(nz)]),
                                                           np.concatenate([Jp, np.arange(nz)]))),
                          shape=(nz, nz)).tocsc()
        LT = L.T.tolil()
        LT[0, :] = 0.0
        LT[0, 0] = 1.0
        rhs = np.zeros(nz); rhs[0] = 1.0
        g = spla.spsolve(LT.tocsc(), rhs)
        g = np.maximum(g, 0.0)
        return g / g.sum()


# ----------------------------------------------------------------------------- numba kernels

_SQRT5 = np.sqrt(5.0)
_INVPHI = (_SQRT5 - 1.0) / 2.0
_NEG = -1.0e30


@njit(cache=True, fastmath=False)
def u_fn(c, nu):
    """CRRA utility."""
    if nu == 1.0:
        return np.log(c)
    return c ** (1.0 - nu) / (1.0 - nu)


_FIXED_COST_LEVEL = np.array([0.0])  # not read by the kernels; the fixed-cost form is passed as p[13]


@njit(cache=True)
def F_fix(z, g2, ztilde, Fmin, form=0.0):
    """Fixed-cost factor. form = 0: 1 - g2 (z - ztilde) (log form); form = 1: 1 - g2 (e^z - e^ztilde)."""
    if z >= ztilde:
        if form > 0.5:
            v = 1.0 - (np.exp(z) - np.exp(ztilde)) * g2
        else:
            v = 1.0 - (z - ztilde) * g2
        return v if v > Fmin else Fmin
    return 1.0


@njit(cache=True)
def phi_fn(d, z, g0, g1, g2, ztilde, Fmin, form=0.0):
    """Share of the endowment kept at default intensity d: (1 - g0 d^g1) times the fixed-cost
    factor when d > 0, and 1 when d = 0."""
    if d <= 0.0:
        return 1.0
    return (1.0 - g0 * d ** g1) * F_fix(z, g2, ztilde, Fmin, form)


@njit(cache=True)
def issue_cap(q, qf, band, lband):
    """Underwriting standard: (capped, L) with new issuance l <= L when capped. L = 0 for q < qf; with a band,
    L = lband w / (1 - w), w = (q - qf) / band, for qf <= q < qf + band; no cap at higher prices."""
    if q < qf:
        return True, 0.0
    if band > 0.0 and qf > 0.0 and q < qf + band:
        w = (q - qf) / band
        return True, lband * w / (1.0 - w)
    return False, 0.0


@njit(cache=True)
def best_S(d, VB, q, B, z, y, p, mode):
    """Maximize u(c(S,d)) + S*VB over admissible S for a given d.

    mode: 1 -> S >= 0 (forward difference), -1 -> S <= 0 (backward), 0 -> S = 0, 2 -> free.
    Returns (H, S, c). H = -1e30 if infeasible.
    """
    nu = p[0]; delta = p[2]; kappa = p[3]; lam = p[5]
    g0 = p[6]; g1 = p[7]; g2 = p[8]; zt = p[9]; cmf = p[10]; Scap = p[11]; Fmin = p[12]
    ph = phi_fn(d, z, g0, g1, g2, zt, Fmin, p[13])
    caut = ph * y - (1.0 - d) * (delta + lam) * B          # consumption with zero issuance
    k = (delta - kappa * (delta + lam) * d) * B             # c = caut + q (S + k)
    cmin = cmf * y
    capped, Lq = issue_cap(q, p[14], p[15], p[16])           # underwriting standard: l = S + k <= Lq
    # zero drift: S = 0, feasible if the consumption floor and the standard allow it
    if mode == 0:
        if capped and k > Lq:
            return _NEG, 0.0, cmin
        c = caut + q * k
        if c < cmin:
            return _NEG, 0.0, cmin
        return u_fn(c, nu), 0.0, c
    # admissible interval [S_lo, S_hi]: consumption floor, cap on |S|, the standard, and the sign
    # restriction of the one-sided difference
    S_lo = (cmin - caut) / q - k
    if S_lo < -Scap:
        S_lo = -Scap
    S_hi = Scap
    if capped and S_hi > Lq - k:
        S_hi = Lq - k
    if mode == 1:
        if S_lo < 0.0:
            S_lo = 0.0
    elif mode == -1:
        if S_hi > 0.0:
            S_hi = 0.0
    if S_lo > S_hi:
        return _NEG, 0.0, cmin
    # unconstrained optimum from the first-order condition u'(c) = -VB/q, then clamp
    if VB < 0.0:
        cstar = (-VB / q) ** (-1.0 / nu)
        Sstar = (cstar - caut) / q - k
    else:
        Sstar = S_hi
    S = Sstar
    if S < S_lo:
        S = S_lo
    elif S > S_hi:
        S = S_hi
    c = caut + q * (S + k)
    return u_fn(c, nu) + S * VB, S, c


@njit(cache=True)
def best_node(VB, q, B, z, y, p, mode, n_golden):
    """Maximize over d in [0,1] (and S). Returns (H, S, c, d)."""
    # d = 0 candidate
    Hb, Sb, cb = best_S(0.0, VB, q, B, z, y, p, mode)
    if not np.isfinite(Hb):
        Hb = _NEG
    db = 0.0
    # Closed form for d > 0. Given S, d enters the Hamiltonian through consumption alone, so the optimum solves
    # dc/dd = 0 whenever the bounds on S do not move with d. They move with d where the standard caps issuance and
    # where the consumption floor binds. The maximized Hamiltonian is concave in d, so a root in (0, 1) at which
    # both are slack is the maximum over d > 0, and the line search is skipped.
    if p[17] > 0.5 and p[7] > 1.0 and B > 0.0:
        Ff0 = F_fix(z, p[8], p[9], p[12], p[13])
        rhs0 = (1.0 - p[3] * q) * (p[2] + p[5]) * B / (p[6] * p[7] * Ff0 * y)
        if rhs0 > 0.0:
            dc = rhs0 ** (1.0 / (p[7] - 1.0))
            if dc < 1.0:
                H, S, c = best_S(dc, VB, q, B, z, y, p, mode)
                if H > -1.0e29 and c > p[10] * y * (1.0 + 1.0e-10):
                    capped, Lq = issue_cap(q, p[14], p[15], p[16])
                    kd = (p[2] - p[3] * (p[2] + p[5]) * dc) * B
                    if (not capped) or S < Lq - kd - 1.0e-12:
                        if H > Hb:
                            return H, S, c, dc
                        return Hb, Sb, cb, db
    # golden section on (0, 1] for the d > 0 branch (H*(d) concave; the infeasible set is an
    # initial segment because d = 1 is always feasible)
    a = 1.0e-9
    b = 1.0
    x1 = b - _INVPHI * (b - a)
    x2 = a + _INVPHI * (b - a)
    f1 = best_S(x1, VB, q, B, z, y, p, mode)[0]
    f2 = best_S(x2, VB, q, B, z, y, p, mode)[0]
    for _ in range(n_golden):
        if f1 <= f2:
            a = x1
            x1 = x2; f1 = f2
            x2 = a + _INVPHI * (b - a)
            f2 = best_S(x2, VB, q, B, z, y, p, mode)[0]
        else:
            b = x2
            x2 = x1; f2 = f1
            x1 = b - _INVPHI * (b - a)
            f1 = best_S(x1, VB, q, B, z, y, p, mode)[0]
    dg = 0.5 * (a + b)
    H, S, c = best_S(dg, VB, q, B, z, y, p, mode)
    if H > Hb:
        Hb, Sb, cb, db = H, S, c, dg
    # d = 1 candidate
    H, S, c = best_S(1.0, VB, q, B, z, y, p, mode)
    if H > Hb:
        Hb, Sb, cb, db = H, S, c, 1.0
    # paper's closed-form interior d (exact when S is interior)
    g0 = p[6]; g1 = p[7]; g2 = p[8]; zt = p[9]; Fmin = p[12]
    delta = p[2]; kappa = p[3]; lam = p[5]
    if g1 > 1.0 and B > 0.0:
        Ff = F_fix(z, g2, zt, Fmin, p[13])
        rhs = (1.0 - kappa * q) * (delta + lam) * B / (g0 * g1 * Ff * y)
        if rhs > 0.0:
            di = rhs ** (1.0 / (g1 - 1.0))
            if di > 1.0:
                di = 1.0
            H, S, c = best_S(di, VB, q, B, z, y, p, mode)
            if H > Hb:
                Hb, Sb, cb, db = H, S, c, di
    return Hb, Sb, cb, db


@njit(parallel=True, cache=True)
def policies(V, q, Bg, zg, yg, p, hf, hb, n_golden, which_prev, tau):
    """Upwind policy at every node. Returns S, c, d, u, Hval, which, S_free (at Bmax), d_free.
    At each node the forward-difference (S >= 0), backward-difference (S <= 0), and zero-drift
    candidates are compared and the one with the largest Hamiltonian is kept.
    hf[i] = B[i+1] - B[i], hb[i] = B[i] - B[i-1] (per-node spacings; equal on a uniform grid).
    which_prev, tau (arrays over nodes): where tau > 0 a node keeps the branch it had in which_prev (forward,
    backward, or zero drift) unless another branch has a Hamiltonian higher by more than tau. At a node where two
    branches are admissible and tie, the comparison otherwise flips from one step to the next."""
    nB = Bg.size
    nz = zg.size
    N = nB * nz
    S = np.empty((nB, nz)); c = np.empty((nB, nz)); d = np.empty((nB, nz))
    u = np.empty((nB, nz)); Hval = np.empty((nB, nz))
    which = np.empty((nB, nz), dtype=np.int8)   # +1 forward, -1 backward, 0 zero drift
    S_free = np.zeros(nz); d_free = np.zeros(nz)
    nu = p[0]
    for kk in prange(N):
        i = kk // nz
        j = kk - i * nz
        B = Bg[i]; z = zg[j]; y = yg[j]; qq = q[i, j]
        bestH = -2.0e30
        bS = 0.0; bc = 1.0; bd = 0.0; bw = 0
        tau_ = tau[i, j]
        wp = which_prev[i, j] if tau_ > 0.0 else 9
        pH = -2.0e30; pS = 0.0; pc = 1.0; pd = 0.0
        # forward, backward, and zero-drift candidates
        if i < nB - 1:
            VBf = (V[i + 1, j] - V[i, j]) / hf[i]
            H, S_, c_, d_ = best_node(VBf, qq, B, z, y, p, 1, n_golden)
            if H > bestH:
                bestH = H; bS = S_; bc = c_; bd = d_; bw = 1
            if wp == 1:
                pH = H; pS = S_; pc = c_; pd = d_
        if i > 0:
            VBb = (V[i, j] - V[i - 1, j]) / hb[i]
            H, S_, c_, d_ = best_node(VBb, qq, B, z, y, p, -1, n_golden)
            if H > bestH:
                bestH = H; bS = S_; bc = c_; bd = d_; bw = -1
            if wp == -1:
                pH = H; pS = S_; pc = c_; pd = d_
        H, S_, c_, d_ = best_node(0.0, qq, B, z, y, p, 0, n_golden)
        if H > bestH:
            bestH = H; bS = S_; bc = c_; bd = d_; bw = 0
        if wp == 0:
            pH = H; pS = S_; pc = c_; pd = d_
        if wp != bw and pH > -1.0e29 and pH >= bestH - tau_:
            bestH = pH; bS = pS; bc = pc; bd = pd; bw = wp
        if bestH < -1.0e29 or not np.isfinite(bestH):
            # no feasible candidate (should not happen): autarky fallback, flagged by which = 9
            bS = 0.0; bd = 1.0; bc = phi_fn(1.0, z, p[6], p[7], p[8], p[9], p[12], p[13]) * y; bw = 9
            bestH = u_fn(bc, nu)
        S[i, j] = bS; c[i, j] = bc; d[i, j] = bd; u[i, j] = u_fn(bc, nu)
        Hval[i, j] = bestH; which[i, j] = bw
        # at Bmax, the unconstrained policy (used to check whether the debt ceiling binds)
        if i == nB - 1:
            VBb = (V[i, j] - V[i - 1, j]) / hb[i]
            H, S_, c_, d_ = best_node(VBb, qq, B, z, y, p, 2, n_golden)
            S_free[j] = S_; d_free[j] = d_
    return S, c, d, u, Hval, which, S_free, d_free


@njit(parallel=True, cache=True)
def policies_central(V, q, Bg, zg, yg, p, hf, hb, n_golden):
    """Policy from the CENTRAL difference of V (second order); the transport operator is
    still upwind in the resulting drift. Boundary nodes use one-sided differences with the
    state constraint (S >= 0 at Bmin, S <= 0 at Bmax). Same outputs as `policies`."""
    nB = Bg.size
    nz = zg.size
    N = nB * nz
    S = np.empty((nB, nz)); c = np.empty((nB, nz)); d = np.empty((nB, nz))
    u = np.empty((nB, nz)); Hval = np.empty((nB, nz))
    which = np.empty((nB, nz), dtype=np.int8)
    S_free = np.zeros(nz); d_free = np.zeros(nz)
    nu = p[0]
    for kk in prange(N):
        i = kk // nz
        j = kk - i * nz
        B = Bg[i]; z = zg[j]; y = yg[j]; qq = q[i, j]
        if i == 0:
            VB = (V[1, j] - V[0, j]) / hf[0]
            H, S_, c_, d_ = best_node(VB, qq, B, z, y, p, 1, n_golden)
        elif i == nB - 1:
            VB = (V[i, j] - V[i - 1, j]) / hb[i]
            H, S_, c_, d_ = best_node(VB, qq, B, z, y, p, -1, n_golden)
            Hf, Sf_, cf_, df_ = best_node(VB, qq, B, z, y, p, 2, n_golden)
            S_free[j] = Sf_; d_free[j] = df_
        else:
            VB = (V[i + 1, j] - V[i - 1, j]) / (hf[i] + hb[i])
            H, S_, c_, d_ = best_node(VB, qq, B, z, y, p, 2, n_golden)
        if H < -1.0e29 or not np.isfinite(H):
            S_ = 0.0; d_ = 1.0; c_ = phi_fn(1.0, z, p[6], p[7], p[8], p[9], p[12], p[13]) * y
            H = u_fn(c_, nu)
        S[i, j] = S_; c[i, j] = c_; d[i, j] = d_; u[i, j] = u_fn(c_, nu); Hval[i, j] = H
        which[i, j] = 1 if S_ > 0.0 else (-1 if S_ < 0.0 else 0)
    return S, c, d, u, Hval, which, S_free, d_free


@njit(cache=True)
def line_sweep(cc, S, hf, hb, zup, zdn, b, x, tol, maxpass):
    """Solve [diag(cc) - A(S)] x = b by line Gauss-Seidel, in place in x (the starting guess). A(S) is the upwind
    generator of Model.assemble with a nearest-neighbor endowment process: coefficients zup[j] on (i, j+1) and
    zdn[j] on (i, j-1). One pass solves every debt row exactly in the endowment direction (a tridiagonal
    system), first with the rows in increasing order of debt, then in decreasing order, taking the neighboring rows
    from the latest values. Each node is coupled to one neighboring row only, the one its drift points to, so a
    region where the drift has one sign is solved exactly by the sweep that runs against the drift.
    Stops when the largest residual is below tol. Returns (passes, largest residual)."""
    nB, nz = x.shape
    dg = np.empty(nz); rhs = np.empty(nz); cp = np.empty(nz)
    res = 1.0e300
    for it in range(maxpass):
        for sweep in range(2):
            for ii in range(nB):
                i = ii if sweep == 0 else nB - 1 - ii
                for j in range(nz):
                    sp = S[i, j] if S[i, j] > 0.0 else 0.0
                    sm = S[i, j] if S[i, j] < 0.0 else 0.0
                    dg[j] = cc[i, j] + sp / hf[i] - sm / hb[i] + zup[j] + zdn[j]
                    r = b[i, j]
                    if i < nB - 1:
                        r += sp / hf[i] * x[i + 1, j]
                    if i > 0:
                        r -= sm / hb[i] * x[i - 1, j]
                    rhs[j] = r
                # Thomas algorithm: -zdn[j] x[j-1] + dg[j] x[j] - zup[j] x[j+1] = rhs[j]
                cp[0] = -zup[0] / dg[0]
                rhs[0] = rhs[0] / dg[0]
                for j in range(1, nz):
                    den = dg[j] + zdn[j] * cp[j - 1]
                    cp[j] = -zup[j] / den
                    rhs[j] = (rhs[j] + zdn[j] * rhs[j - 1]) / den
                x[i, nz - 1] = rhs[nz - 1]
                for j in range(nz - 2, -1, -1):
                    x[i, j] = rhs[j] - cp[j] * x[i, j + 1]
        res = 0.0
        for i in range(nB):
            for j in range(nz):
                sp = S[i, j] if S[i, j] > 0.0 else 0.0
                sm = S[i, j] if S[i, j] < 0.0 else 0.0
                r = (cc[i, j] + sp / hf[i] - sm / hb[i] + zup[j] + zdn[j]) * x[i, j] - b[i, j]
                if i < nB - 1:
                    r -= sp / hf[i] * x[i + 1, j]
                if i > 0:
                    r += sm / hb[i] * x[i - 1, j]
                if j < nz - 1:
                    r -= zup[j] * x[i, j + 1]
                if j > 0:
                    r -= zdn[j] * x[i, j - 1]
                if abs(r) > res:
                    res = abs(r)
        if res < tol:
            return it + 1, res
    return maxpass, res


# ----------------------------------------------------------------------------- model


class Model:
    """The discretized model: the (B, z) grid, the parameter vector, the constant z part of the
    generator, and methods for the policies, the upwind generator, and an initial guess."""

    def __init__(self, par: Params, zp: ZProcess, Bmin: float, Bmax: float, nB: int, scheme: str = "upwind3",
                 B: np.ndarray | None = None):
        """Uniform grid of nB nodes on [Bmin, Bmax], or the (sorted) node array B if given
        (non-uniform grid; Bmin, Bmax, nB are then taken from it)."""
        self.scheme = scheme
        self.par = par
        self.zp = zp
        # debt grid and its forward, backward, and cell spacings
        if B is None:
            self.B = np.linspace(Bmin, Bmax, nB)
        else:
            self.B = np.asarray(B, dtype=np.float64)
            nB = self.B.size
            assert nB > 2 and np.all(np.diff(self.B) > 0)
        h = np.diff(self.B)
        self.hf = np.append(h, h[-1])          # B[i+1] - B[i]  (last entry unused)
        self.hb = np.insert(h, 0, h[0])        # B[i] - B[i-1]  (first entry unused)
        self.hcell = 0.5 * (self.hf + self.hb)  # cell width for densities
        self.uniform = bool(np.allclose(h, h[0]))
        self.dB = float(h[0]) if self.uniform else float(h.min())   # smallest step on non-uniform grids
        self.z = zp.z
        self.y = np.exp(self.z)
        self.nB = nB
        self.nz = zp.nz
        self.N = nB * self.nz
        self.p = par.as_array()
        # z part of the generator (constant), stored as COO triplets on the flat index k = i*nz + j
        J, Jp, vals, zdiag = zp.generator_offdiag()
        k = np.arange(self.N).reshape(nB, self.nz)
        self._zrows = np.concatenate([k[:, j] for j in J]) if J.size else np.zeros(0, dtype=np.int64)
        self._zcols = np.concatenate([k[:, jp] for jp in Jp]) if J.size else np.zeros(0, dtype=np.int64)
        self._zvals = np.concatenate([np.full(nB, v) for v in vals]) if J.size else np.zeros(0)
        self._zdiag = zdiag      # (nz,)
        self._k = k
        # nearest-neighbor endowment process: coefficients on (j+1) and (j-1), for the line sweeps
        self.zup = np.zeros(self.nz); self.zdn = np.zeros(self.nz)
        self.z_tridiagonal = zp.kind in ("none", "ou")
        if self.z_tridiagonal:
            for jj, jp, v in zip(J, Jp, vals):
                if jp == jj + 1:
                    self.zup[jj] = v
                else:
                    self.zdn[jj] = v

    def assemble(self, S: np.ndarray) -> sp.csc_matrix:
        """Generator A of (B, z) for the drift field S (nB, nz): upwind in B (forward difference
        where S > 0, backward where S < 0) plus the z part."""
        nB, nz, k = self.nB, self.nz, self._k
        hf, hb = self.hf[:, None], self.hb[:, None]
        Sp = np.maximum(S, 0.0)
        Sm = np.minimum(S, 0.0)
        diag = -Sp / hf + Sm / hb + self._zdiag[None, :]
        rows = [k.ravel(), k[:-1, :].ravel(), k[1:, :].ravel(), self._zrows]
        cols = [k.ravel(), k[1:, :].ravel(), k[:-1, :].ravel(), self._zcols]
        vals = [diag.ravel(), (Sp[:-1, :] / hf[:-1]).ravel(), (-Sm[1:, :] / hb[1:]).ravel(), self._zvals]
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.N, self.N))
        return A.tocsc()

    def solve_shifted(self, cc, S, b, x0, A=None, tol=1e-13, maxpass=60):
        """Solve [diag(cc) - A(S)] x = b. Line sweeps from the guess x0 (line_sweep), with the sparse direct solve
        as the fallback when they have not converged (pass the assembled A to avoid assembling it again). cc is a
        scalar or an array over nodes; tol is relative to the largest right-hand side. Returns (x, passes), with
        passes = -1 for the direct solve."""
        cc = np.ascontiguousarray(np.broadcast_to(np.asarray(cc, dtype=np.float64).reshape(-1) if np.ndim(cc) else cc,
                                                  (self.N,)).reshape(self.nB, self.nz))
        b2 = np.ascontiguousarray(np.asarray(b, dtype=np.float64).reshape(self.nB, self.nz))
        if self.z_tridiagonal:
            x = np.array(x0, dtype=np.float64).reshape(self.nB, self.nz).copy()
            n, r = line_sweep(cc, S, self.hf, self.hb, self.zup, self.zdn, b2, x, tol * max(1.0, float(np.abs(b2).max())), maxpass)
            if n < maxpass and np.isfinite(r):
                return x, n
        if A is None:
            A = self.assemble(S)
        x = spla.spsolve(sp.diags(cc.ravel(), format="csc") - A, b2.ravel())
        return x.reshape(self.nB, self.nz), -1

    def policies(self, V, q, n_golden=28, which_prev=None, tau=0.0):
        """Policies given (V, q): upwind by default, central differences if scheme == 'central'.
        which_prev, tau: the branch held at each node and its threshold (see policies)."""
        if self.scheme == "central":
            return policies_central(V, q, self.B, self.z, self.y, self.p, self.hf, self.hb, n_golden)
        if which_prev is None:
            which_prev, tau = np.full((self.nB, self.nz), 9, dtype=np.int8), 0.0
        if np.ndim(tau) == 0:
            tau = np.full((self.nB, self.nz), float(tau))
        return policies(V, q, self.B, self.z, self.y, self.p, self.hf, self.hb, n_golden, which_prev, tau)

    def initial_guess(self, q_init="static"):
        """V from consuming y - lam B (at least the consumption floor) forever; q = 1 ('one') or the steady-state price under the
        interior default rule evaluated at q = 1 ('static')."""
        par = self.par
        BB = self.B[:, None]
        yy = self.y[None, :]
        c0 = np.maximum(yy - par.lam * BB, par.cmin_frac * yy)
        V0 = np.vectorize(u_fn)(c0, par.nu) / par.rho
        if q_init == "one":
            q0 = np.ones((self.nB, self.nz))
        else:
            # steady-state price given the interior default policy at q = 1
            Ff = np.array([F_fix(z, par.g2, par.ztilde, par.F_min, self.p[13]) for z in self.z])[None, :]
            dint = ((1 - par.kappa) * (par.delta + par.lam) * BB / (par.g0 * par.g1 * Ff * yy))
            dint = np.minimum(1.0, dint ** (1.0 / (par.g1 - 1.0)))
            q0 = (1 - dint) * (par.lam + par.delta) / par.xi(dint)
            q0 = np.maximum(q0, 0.05)
        return V0, q0


# ----------------------------------------------------------------------------- solver


@dataclass
class Result:
    """A solution: V, q, policies, generator A, ergodic mass g, and convergence information."""
    model: Model
    V: np.ndarray
    q: np.ndarray
    S: np.ndarray
    c: np.ndarray
    d: np.ndarray
    which: np.ndarray
    S_free: np.ndarray
    d_free: np.ndarray
    A: sp.csc_matrix
    g: np.ndarray | None
    converged: bool
    iters: int
    resV: float
    resq: float
    history: list = field(default_factory=list)
    time: float = 0.0

    @property
    def spread(self):
        """Yield to maturity minus the risk-free rate: (delta+lam)/q - (delta+r)."""
        par = self.model.par
        return (par.delta + par.lam) / self.q - (par.delta + par.r)

    @property
    def issuance(self):
        """l = S + delta B - kappa (delta+lam) d B"""
        par = self.model.par
        B = self.model.B[:, None]
        return self.S + par.delta * B - par.kappa * (par.delta + par.lam) * self.d * B


def ergodic_density(model: Model, A: sp.csc_matrix) -> np.ndarray:
    """Stationary distribution: A^T g = 0, returned as probability MASS on nodes (nB,nz)."""
    N = model.N
    # direct solve with the first equation replaced by the normalization g[0] = 1
    AT = A.T.tolil()
    AT[0, :] = 0.0
    AT[0, 0] = 1.0
    rhs = np.zeros(N); rhs[0] = 1.0
    try:
        g = spla.spsolve(AT.tocsc(), rhs)
    except Exception:
        g = np.full(N, np.nan)
    g = np.maximum(np.nan_to_num(g), 0.0)
    s = g.sum()
    if s > 0:
        g /= s
    # The pinned node (B = 0 in the lowest endowment state) may carry no stationary mass. Then the
    # pinned system is inconsistent and the solve returns a vector outside the null space. If the
    # residual is not small, iterate the forward equation implicitly (dt = 200) to the stationary density.
    res = float(np.abs(A.T @ g).max()) if s > 0 else np.inf
    if not np.isfinite(res) or res > 1e-8:
        g = np.ones(N) / N
        lu = spla.splu((sp.identity(N, format="csc") - 200.0 * A.T).tocsc())
        for _ in range(2000):
            g = lu.solve(g)
            g = np.maximum(g, 0.0)
            g /= g.sum()
            if float(np.abs(A.T @ g).max()) < 1e-10:
                break
    return g.reshape(model.nB, model.nz)


# ----------------------------------------------------------------------------- diagnostics


def diagnostics(res: Result, verbose=True) -> dict:
    """Shape checks of a solution: top of the positive-drift region, whether the debt ceiling binds,
    zero-drift nodes, monotonicity of V, q, d, and ergodic summaries."""
    m = res.model
    B, z = m.B, m.z
    out = {}
    # positive-drift upper boundary per z
    Bbar = np.full(m.nz, np.nan)
    for j in range(m.nz):
        pos = np.where(res.S[:, j] > 1e-10)[0]
        if pos.size:
            Bbar[j] = B[pos.max()]
    out["Bbar"] = Bbar
    out["ceiling_binds"] = res.S_free > 1e-10
    out["S_free_at_Bmax"] = res.S_free
    # steady states / zero-drift nodes per z
    out["zero_drift_nodes"] = [B[np.where(res.which[:, j] == 0)[0]] for j in range(m.nz)]
    # monotonicity checks (allow tiny violations)
    out["V_decreasing_in_B"] = bool(np.all(np.diff(res.V, axis=0) <= 1e-9))
    out["q_decreasing_in_B"] = bool(np.all(np.diff(res.q, axis=0) <= 1e-7))
    out["d_increasing_in_B"] = bool(np.all(np.diff(res.d, axis=0) >= -1e-6))
    if m.nz > 1:
        out["V_increasing_in_z"] = bool(np.all(np.diff(res.V, axis=1) >= -1e-9))
        out["q_increasing_in_z"] = bool(np.all(np.diff(res.q, axis=1) >= -1e-7))
    if res.g is not None:
        gB = res.g.sum(axis=1)
        out["mass_top3"] = float(gB[-3:].sum())
        out["mass_d1"] = float(res.g[res.d >= 1 - 1e-9].sum())
        out["mean_B"] = float((gB * B).sum())
        out["mean_d"] = float((res.g * res.d).sum())
        out["frac_default"] = float(res.g[res.d > 1e-9].sum())
        out["mean_q"] = float((res.g * res.q).sum())
        out["mean_spread"] = float((res.g * res.spread).sum())
    if verbose:
        print("--- diagnostics")
        print(f"  Bmax = {B[-1]:.3f};  ceiling binds at z = {z[out['ceiling_binds']]}")
        print(f"  S_free at Bmax by z: {np.array2string(res.S_free, precision=4)}")
        print(f"  Bbar(z) (top of positive-drift region): {np.array2string(Bbar, precision=3)}")
        for j in range(m.nz):
            zd = out["zero_drift_nodes"][j]
            print(f"  z={z[j]:+.3f}: zero-drift nodes at B = {np.array2string(zd, precision=3)}")
        for key in ["V_decreasing_in_B", "q_decreasing_in_B", "d_increasing_in_B",
                    "V_increasing_in_z", "q_increasing_in_z"]:
            if key in out:
                print(f"  {key}: {out[key]}")
        if res.g is not None:
            print(f"  ergodic: mean B {out['mean_B']:.3f}, mean d {out['mean_d']:.3f}, "
                  f"P(d>0) {out['frac_default']:.3f}, mass in top 3 B nodes {out['mass_top3']:.4f}, "
                  f"mass at d=1 {out['mass_d1']:.4f}, mean spread {out['mean_spread']:.4f}")
    return out
