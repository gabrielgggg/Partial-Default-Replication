r"""Self-fulfilling partial default with a risk-neutral sovereign and an OU endowment.

Class of equilibria: the block construction of Proposition A.9 with shocks. Lenders expect the sovereign to settle
at a debt level B° (independent of z) and stay there forever; from any B < B° it issues a block to B° at once.

  At B°: S = 0, d°(z) = d*(B°, z, q°(z)) (static rule), and the price solves the ODE in z
         xi(d°) q° - (1-d°)(lam+delta) - L_z q° = 0;
         the value solves (rho - L_z) V° = R*(B°, z, q°).
  Below: q(B,z) = q°(z), V(B,z) = V°(z) + q°(z)(B° - B); the HJB variational inequality requires standing still
         not to be better, D(B,z) = rho V - R*(B,z,q°) - L_z V >= 0. D is concave in B with D(B°,z) = 0, so the
         condition is D(0,z) = R*(B°,z,q°) + rho q° B° - y(z) - B° L_z q° >= 0 for every z.
  Above: V_B = -q (indifference); differentiating rho V = R* + L_z V in B gives the price equation
         a(d) B q_B = (1-d)(lam+delta) - (rho + a(d)) q + L_z q,   d = d*(B,z,q),
         parabolic in B (march upward, implicit in L_z), and the drift from the pricing equation,
         S = [xi(d) q - (1-d)(lam+delta) - L_z q] / q_B.  Attraction: S < 0 and q_B < 0 above B°.
  Standard: issuance at B° is a(d°)B° > 0, so q°(z) >= q_floor for every z.
At sigma = 0 the conditions are those of Proposition A.9: Theta(d) <= rho - r < Omega(d), q_r(d) >= q_floor.
Library module (illustrative parameters by default), imported by rn_scan_free.py and region.py.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class RN:
    """Risk-neutral model with an OU endowment on a z grid; checks members of the block-construction family."""

    def __init__(self, rho, sigma, mu=0.225, nz=201, n_sd=3.0, r=0.039, delta=0.12, kappa=0.7,
                 g0=0.075, g1=1.5, g2=0.001, ztilde=-1.0, q_floor=0.7):
        self.rho, self.sigma, self.mu = rho, sigma, mu
        self.r, self.delta, self.kappa, self.lam = r, delta, kappa, r
        self.cs = delta + r
        self.g0, self.g1, self.g2, self.zt, self.qf = g0, g1, g2, ztilde, q_floor
        # upwind generator L_z of the OU process on +/- n_sd unconditional s.d.
        if sigma == 0:
            self.z = np.zeros(1)
            self.L = sp.csr_matrix((1, 1))
        else:
            sd = sigma / np.sqrt(2 * mu)
            self.z = np.linspace(-n_sd * sd, n_sd * sd, nz)
            dz = self.z[1] - self.z[0]
            dif = 0.5 * sigma ** 2 / dz ** 2
            m = -mu * self.z
            up = np.maximum(m, 0) / dz + dif
            dn = np.maximum(-m, 0) / dz + dif
            up[-1] = 0.0; dn[0] = 0.0                      # reflecting
            self.L = sp.diags([-(up + dn), up[:-1], dn[1:]], [0, 1, -1], format="csr")
        self.y = np.exp(self.z)
        self.F = 1 - self.g2 * (self.z - self.zt)
        self.nz = self.z.size
        # stationary weights of z (for reporting)
        if self.nz > 1:
            LT = self.L.T.tolil(); LT[0, :] = 1.0
            rhs = np.zeros(self.nz); rhs[0] = 1.0
            g = spla.spsolve(LT.tocsc(), rhs); self.gz = np.maximum(g, 0) / np.maximum(g, 0).sum()
        else:
            self.gz = np.ones(1)

    # ------------------------------------------------------------------ static pieces
    def a(self, d):
        """Net debt decay a(d) = delta - kappa (delta + lam) d."""
        return self.delta - self.kappa * self.cs * d

    def xi(self, d):
        return self.r + self.a(d)

    def dstar(self, B, q):
        """static default rule at every z (vectorized in z; B scalar, q (nz,))"""
        m = (1 - self.kappa * q) * self.cs * B
        di = np.minimum(1.0, (np.maximum(m, 0) / (self.g0 * self.g1 * self.F * self.y)) ** (1 / (self.g1 - 1)))
        gain = self.y * (self.F - 1 - self.F * self.g0 * di ** self.g1) + m * di
        return np.where(gain > 0, di, 0.0)

    def phi(self, d):
        return np.where(d > 0, self.F * (1 - self.g0 * d ** self.g1), 1.0)

    def Rstar(self, B, q, d=None):
        """Consumption at zero drift, R*(B, z, q), at every z."""
        d = self.dstar(B, q) if d is None else d
        return self.phi(d) * self.y - (1 - d) * self.cs * B + q * self.a(d) * B

    def d_first(self):
        """Default intensity at the default threshold (vector in z)."""
        return ((1 - self.F) / (self.g0 * self.F * (self.g1 - 1))) ** (1 / self.g1)

    def s_first(self):
        """Spread at the first default intensity."""
        d = self.d_first()
        return (1 - self.kappa) * self.cs * d / (1 - d)

    # ------------------------------------------------------------------ the steady locus
    def price_at(self, B0, q_start=1.0, maxit=500, tol=1e-13):
        """q°(z) at B0: fixed point of q -> [xi(d*(q)) - L]^{-1} (1-d*(q)) cs. The map is monotone (a higher price
        lowers default and raises the price), so iterating from q = 1 gives the largest fixed point."""
        q = np.full(self.nz, q_start, dtype=float)
        for it in range(maxit):
            d = self.dstar(B0, q)
            A = sp.diags(self.xi(d)) - self.L
            qn = spla.spsolve(A.tocsc(), (1 - d) * self.cs) if self.nz > 1 else (1 - d) * self.cs / self.xi(d)
            if np.max(np.abs(qn - q)) < tol:
                q = qn; break
            q = qn
        d = self.dstar(B0, q)
        return q, d, it

    def price_newton(self, B0, q_init, maxit=60, tol=1e-12):
        """q°(z) at B0 by Newton on G(q) = xi(d*(q)) q - (1-d*(q)) cs - L q from q_init (reaches unstable roots of
        the monotone map too). d* depends on q at the same node only, so the Jacobian is diag(g'(q)) - L."""
        q = np.array(q_init, dtype=float) * np.ones(self.nz)

        def g(qq):
            d = self.dstar(B0, qq)
            return self.xi(d) * qq - (1 - d) * self.cs

        for it in range(maxit):
            G = g(q) - (self.L @ q if self.nz > 1 else 0.0)
            if np.max(np.abs(G)) < tol:
                break
            h = 1e-7
            gp = (g(q + h) - g(q - h)) / (2 * h)
            J = sp.diags(gp) - self.L
            dq = spla.spsolve(J.tocsc(), -G) if self.nz > 1 else -G / gp
            lam = 1.0
            while lam > 1e-6:
                qn = np.clip(q + lam * dq, 1e-6, 1.0)
                Gn = g(qn) - (self.L @ qn if self.nz > 1 else 0.0)
                if np.max(np.abs(Gn)) < (1 - 1e-4 * lam) * np.max(np.abs(G)):
                    break
                lam *= 0.5
            q = qn
        d = self.dstar(B0, q)
        G = g(q) - (self.L @ q if self.nz > 1 else 0.0)
        return q, d, float(np.max(np.abs(G)))

    def value_at(self, B0, q0, d0):
        """V°(z) at B°: solves (rho - L_z) V° = R*(B°, z, q°)."""
        R = self.Rstar(B0, q0, d0)
        if self.nz == 1:
            return R / self.rho
        return spla.spsolve((self.rho * sp.identity(self.nz) - self.L).tocsc(), R)

    def D0(self, B0, q0, d0):
        """D(0, z): rho V(0,z) - y - L_z V(0,z), with V(0,z) = V°(z) + q°(z) B0. Nonnegative = the block is optimal."""
        Lq = self.L @ q0 if self.nz > 1 else np.zeros(1)
        return self.Rstar(B0, q0, d0) + self.rho * q0 * B0 - self.y - B0 * Lq

    def P(self, B, q):
        """xi(d) q - (1-d)(lam+delta) at the static default rule."""
        d = self.dstar(B, q)
        return self.xi(d) * q - (1 - d) * self.cs

    def S_slope(self, B0, q0):
        """Exact slope of the drift at B°+ (h -> 0): S = N/q_B with N = P(B,q) - L q, N(B°) = 0, so
        S_B = N_B / q_B = (P_q q_B + P_B - L q_B) / q_B, with q_B(B°, z) from the price equation."""
        d0 = self.dstar(B0, q0)
        Lq = self.L @ q0 if self.nz > 1 else np.zeros(self.nz)
        qB = ((1 - d0) * self.cs - (self.rho + self.a(d0)) * q0 + Lq) / (self.a(d0) * B0)
        e = 1e-7
        PB = (self.P(B0 + e, q0) - self.P(B0 - e, q0)) / (2 * e)
        Pq = (self.P(B0, q0 + e) - self.P(B0, q0 - e)) / (2 * e)
        LqB = self.L @ qB if self.nz > 1 else np.zeros(self.nz)
        return (Pq * qB + PB - LqB) / qB, qB

    def march_up(self, B0, q0, Bmax=1.7, h=5e-4, skip=0.01):
        """Implicit march of a(d) B q_B = (1-d)cs - (rho+a(d)) q + L q from (B0, q0) upward; returns the drift sign
        record: first B where S >= 0 or q_B >= 0 at some z (violation), or where d = 1 at every z (full default)."""
        q = q0.copy(); B = B0
        I = sp.identity(self.nz, format="csc")
        first = True
        S_near = None
        while B < Bmax - 1e-12:
            d = self.dstar(B, q)
            aB = self.a(d) * B
            # (aB/h)(qn - q) = (1-d)cs - (rho+a) qn + L qn
            A = sp.diags(aB / h + self.rho + self.a(d)) - self.L
            qn = spla.spsolve(A.tocsc(), aB / h * q + (1 - d) * self.cs) if self.nz > 1 else \
                (aB / h * q + (1 - d) * self.cs) / (aB / h + self.rho + self.a(d))
            Bn = B + h
            qB = (qn - q) / h
            dn = self.dstar(Bn, qn)
            Lq = self.L @ qn if self.nz > 1 else np.zeros(1)
            S = (self.xi(dn) * qn - (1 - dn) * self.cs - Lq) / qB
            if first:
                S_near = S.copy(); first = False
            if np.any((qB[1:-1] if self.nz > 2 else qB) >= 0):
                return dict(ok=False, why="q_B>=0", B=Bn, S_near=S_near)
            Si = S[1:-1] if self.nz > 2 else S
            if Bn > B0 + skip and np.any(Si >= 0):
                return dict(ok=False, why="S>=0", B=Bn, S_near=S_near, zbad=self.z[1:-1][Si >= 0] if self.nz > 2 else self.z)
            if np.all(dn >= 1.0):
                return dict(ok=True, why="full default", B=Bn, S_near=S_near)
            q, B = qn, Bn
        return dict(ok=True, why="Bmax", B=B, S_near=S_near)

    # ------------------------------------------------------------------ free boundary: waiting at zero debt
    def stopping_at_zero(self, B0, q0, V0, maxit=200):
        """Optimal timing of the block from zero debt: min{rho V - L V - y, V - G} = 0 with G = V° + q° B°
        (Howard iteration on the complementarity problem). Returns V(0,z), the jump set J0 (bool) and the price of a
        marginal bond at zero debt, q(0,z): (r + delta) q - L q = lam + delta on W0, q = q° on J0."""
        nz = self.nz
        G = V0 + q0 * B0
        if nz == 1:
            Vw = self.y / self.rho
            J = np.array([G[0] >= Vw[0]])
            return np.maximum(G, Vw), J, np.where(J, q0, self.cs / (self.r + self.delta))
        A = (self.rho * sp.identity(nz) - self.L).tocsr()
        J = np.ones(nz, dtype=bool)
        I = sp.identity(nz, format="csr")
        for it in range(maxit):
            M = sp.vstack([I[k] if J[k] else A[k] for k in range(nz)]).tocsc()
            rhs = np.where(J, G, self.y)
            V = spla.spsolve(M, rhs)
            Jn = (V - G) <= (A @ V - self.y)        # obstacle row is the smaller one
            if np.array_equal(Jn, J):
                break
            J = Jn
        Aq = ((self.r + self.delta) * sp.identity(nz) - self.L).tocsr()
        Mq = sp.vstack([I[k] if J[k] else Aq[k] for k in range(nz)]).tocsc()
        qz = spla.spsolve(Mq, np.where(J, q0, self.cs))
        return V, J, qz

    def check_free(self, d_ref, nB=60, march=True):
        """The member indexed by d_ref with optimal waiting at zero debt. Conditions:
           (i)   no buyback-and-wait at B°: V°(z) >= V(0,z) - q(0,z) B°;
           (ii)  standing still not better on (0, B°): D(B,z) = rho V - R*(B,z,q) - L V >= 0, with
                 V = max(V(0) - q(0) B, V° + q° (B° - B)) and q the price of the chosen destination;
           (iii) attraction, continuation and floor at B° as in check()."""
        B0 = float(self.B_of_d(d_ref))
        q0, d0, res = self.price_newton(B0, self.qr(d_ref))
        if res > 1e-8:
            qa, da, _ = self.price_at(B0, 1.0)
            Ga = self.xi(da) * qa - (1 - da) * self.cs - (self.L @ qa if self.nz > 1 else 0.0)
            if da.max() > 0 and qa.min() > 0.05 and np.max(np.abs(Ga)) < 1e-9:
                q0, d0, res = qa, da, float(np.max(np.abs(Ga)))
            elif qa.min() <= 0.05:
                return dict(B0=B0, d_ref=d_ref, collapsed=True, price_res=np.inf)
        out = dict(B0=B0, d_ref=d_ref, q0=q0, d0=d0, price_res=res, collapsed=False,
                   floor=bool(q0.min() >= self.qf), qmin=float(q0.min()), dmean=float(self.gz @ d0),
                   spread_mean=float(self.gz @ (self.cs * (1 / q0 - 1))))
        V0 = self.value_at(B0, q0, d0)
        Vz, J, qz = self.stopping_at_zero(B0, q0, V0)
        inner = slice(1, -1) if self.nz > 2 else slice(None)
        # (i) no buyback-and-wait at B°
        out["J0"] = J; out["frac_wait"] = float(self.gz @ (~J))
        out["zstar"] = float(self.z[np.argmax(J)]) if J.any() else np.inf
        out["any_jump"] = bool(J[inner].any())
        c1 = V0 - (Vz - qz * B0)
        out["buyback_min"] = float(c1[inner].min())
        # (ii) standing still not better on a grid of (0, B°)
        Dmin = np.inf
        for B in np.linspace(0.0, B0, nB + 1)[1:-1]:
            Vd, Vu = Vz - qz * B, V0 + q0 * (B0 - B)
            up = Vu >= Vd
            V = np.where(up, Vu, Vd); q = np.where(up, q0, qz)
            LV = self.L @ V if self.nz > 1 else 0.0
            D = self.rho * V - self.Rstar(B, q) - LV
            Dmin = min(Dmin, float(D[inner].min()))
        out["Dmin_free"] = Dmin
        # (iii) attraction and floor; if all hold, march upward for continuation
        SB, qB = self.S_slope(B0, q0)
        out["SBmax"] = float(SB[inner].max())
        ok_static = out["floor"] and out["any_jump"] and out["buyback_min"] >= -1e-12 and Dmin >= -1e-10 and out["SBmax"] < 0
        if march and ok_static:
            out["march"] = self.march_up(B0, q0)
        return out

    def B_of_d(self, d):
        """deterministic steady-state debt with intensity d at z = 0 (Proposition A.9)"""
        F = 1 - self.g2 * (0.0 - self.zt)
        return self.g0 * self.g1 * F * d ** (self.g1 - 1) * self.xi(d) / ((1 - self.kappa) * self.cs ** 2)

    def qr(self, d):
        """Lenders' stationary price at a constant default intensity, deterministic case."""
        return (1 - d) * self.cs / self.xi(d)

    def check_d(self, d_ref, march=True):
        """the member of the family indexed by the deterministic intensity d_ref: B° = B(d_ref), price root
        continued from q_r(d_ref)"""
        B0 = float(self.B_of_d(d_ref))
        q0, d0, res = self.price_newton(B0, self.qr(d_ref))
        if res > 1e-8:
            # Newton stalls where d* jumps from 0 to d_first across z; the largest fixed point of the monotone map is
            # exact. Accept it if it is a partial-default locus (d > 0 somewhere, price not collapsed).
            qa, da, _ = self.price_at(B0, 1.0)
            Ga = self.xi(da) * qa - (1 - da) * self.cs - (self.L @ qa if self.nz > 1 else 0.0)
            if da.max() > 0 and qa.min() > 0.05 and np.max(np.abs(Ga)) < 1e-9:
                q0, d0, res = qa, da, float(np.max(np.abs(Ga)))
            elif qa.min() <= 0.05:
                out = dict(B0=B0, d_ref=d_ref, price_res=np.inf, collapsed=True, Dmin=np.nan, floor=False,
                           SBmax=np.nan, qmin=float(qa.min()), dmean=1.0, zD=np.nan, spread_mean=np.nan)
                return out
        out = self.check(B0, march, q0=q0, d0=d0)
        out["d_ref"], out["price_res"] = d_ref, res
        return out

    def check(self, B0, march=True, q0=None, d0=None):
        """Immediate block from zero debt to B0: D(0, z) >= 0, floor, attraction, and continuation above B0."""
        if q0 is None:
            q0, d0, it = self.price_at(B0)
        D = self.D0(B0, q0, d0)
        out = dict(B0=B0, q0=q0, d0=d0, Dmin=float(D.min()), zD=float(self.z[np.argmin(D)]),
                   floor=bool(q0.min() >= self.qf), qmin=float(q0.min()), dmean=float(self.gz @ d0),
                   spread_mean=float(self.gz @ (self.cs * (1 / q0 - 1))))
        SB, qB = self.S_slope(B0, q0)
        inner = slice(1, -1) if self.nz > 2 else slice(None)   # the reflecting nodes are a truncation artifact
        out["SBmax"] = float(SB[inner].max()); out["zSB"] = float(self.z[inner][np.argmax(SB[inner])])
        if march and out["Dmin"] >= 0 and out["floor"] and out["SBmax"] < 0:
            out["march"] = self.march_up(B0, q0)
        return out
