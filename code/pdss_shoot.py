"""Deterministic model of the self-fulfilling example, as a planar system in debt.

The deterministic equilibrium (z = 0, y = 1) is the planar system in B
    q'(B) = P(B,q) / S,                 P = xi(d) q - (1-d)(delta+lam)
    c'(B) = -(c/nu) [ (rho - r) + l P / (q S) ] / S,   l = S + a(d) B   (Euler with the dilution term)
    S = (c - R*(B,q)) / q,  d = d*(B,q) static.
A candidate steady state B0 has S = 0, P = 0 (q0 = q_r(d0)), and the price slope q1 = -q0 (rho-r)/(a0 B0).
Its local branch is the analytic trajectory S ~ s1 x, q ~ q0 + q1 x, c ~ c0 + c1 x (x = B - B0), with
    s1 = P_q + P_B / q1,   c1 = R_B + R_q q1 + q0 s1     (total derivatives through d*).
The class Det holds the static pieces, the candidates, their local expansion, and the right-hand side of the
system; illus returns it at the parameters of the example.

Reads and writes no file. Imported by pdss_fwd2.py (through its callers), pdss_sweep.py, and pdss_threshold.py.
"""
import numpy as np
from scipy.optimize import brentq


class Det:
    """Deterministic model at z = 0: static default choice, candidate steady states, and the ODE in B."""

    def __init__(self, rho, nu=2.0, r=0.039, delta=0.12, kappa=0.7, g0=0.075, g1=1.5, F=0.999, q_floor=0.7, lam=None):
        self.rho, self.nu, self.r, self.delta, self.kappa = rho, nu, r, delta, kappa
        self.lam = r if lam is None else lam
        self.g0, self.g1, self.F, self.qf = g0, g1, F, q_floor
        self.cs = self.delta + self.lam                         # debt service per unit

    # ---------------------------------------------------------------- static pieces
    def a(self, d):
        """Net debt decay a(d) = delta - kappa (delta + lam) d."""
        return self.delta - self.kappa * self.cs * d

    def xi(self, d):
        return self.r + self.a(d)

    def qr(self, d):
        """lenders' stationary price at a constant default intensity"""
        return (1 - d) * self.cs / self.xi(d)

    def spread(self, q):
        return self.cs * (1 / q - 1)

    def d_first(self):
        return ((1 - self.F) / (self.g0 * self.F * (self.g1 - 1))) ** (1 / self.g1)

    def dstar(self, B, q):
        """Static optimal default intensity; zero unless defaulting beats full repayment."""
        m = (1 - self.kappa * q) * self.cs * B
        if m <= 0:
            return 0.0
        di = min(1.0, (m / (self.g0 * self.g1 * self.F)) ** (1 / (self.g1 - 1)))
        gain = self.F - 1 - self.F * self.g0 * di ** self.g1 + m * di
        return di if gain > 0 else 0.0

    def phi(self, d):
        """Output net of the default cost."""
        return 1.0 if d <= 0 else (1 - self.g0 * d ** self.g1) * self.F

    def Rstar(self, B, q, d=None):
        """Consumption at zero drift: output net of the default cost, minus service paid, plus the issuance that keeps B constant."""
        d = self.dstar(B, q) if d is None else d
        return self.phi(d) - (1 - d) * self.cs * B + q * self.a(d) * B

    def P(self, B, q, d=None):
        """Price-equation residual at zero drift, xi(d) q - (1-d)(delta+lam)."""
        d = self.dstar(B, q) if d is None else d
        return self.xi(d) * q - (1 - d) * self.cs

    # ---------------------------------------------------------------- candidates
    def candidates(self, B0, nq=4000):
        """roots q of q = q_r(d*(B0,q)) in the partial-default region"""
        qs = np.linspace(self.qf, 0.99999, nq)
        G = np.array([q - self.qr(self.dstar(B0, q)) for q in qs])
        out = []
        for i in range(nq - 1):
            if G[i] * G[i + 1] < 0:
                d_lo, d_hi = self.dstar(B0, qs[i]), self.dstar(B0, qs[i + 1])
                if d_lo > 0 and d_hi > 0:                       # skip the jump at the threshold
                    out.append(brentq(lambda q: q - self.qr(self.dstar(B0, q)), qs[i], qs[i + 1], xtol=1e-14))
        return out

    def local(self, B0, q0):
        """Linear expansion of the branch at a candidate (B0, q0): slopes q1, s1, c1 and the level c0."""
        d0 = self.dstar(B0, q0)
        a0 = self.a(d0)
        Pd = (1 - self.kappa * q0) * self.cs
        dB = d0 / ((self.g1 - 1) * B0)
        dq = -d0 / (self.g1 - 1) * self.kappa / (1 - self.kappa * q0)
        if d0 >= 1:
            dB = dq = 0.0
        PB, Pq = Pd * dB, self.xi(d0) + Pd * dq
        q1 = -q0 * (self.rho - self.r) / (a0 * B0)
        s1 = Pq + PB / q1
        c0 = self.Rstar(B0, q0, d0)
        RB, Rq = -(1 - d0) * self.cs + q0 * a0, a0 * B0
        c1 = RB + Rq * q1 + q0 * s1
        return dict(B0=B0, q0=q0, d0=d0, a0=a0, q1=q1, s1=s1, c0=c0, c1=c1, spread=self.spread(q0))

    # ---------------------------------------------------------------- the ODE in B
    def rhs(self, B, y, d):
        """Right-hand side (q', c') of the ODE in B, at a fixed default intensity d."""
        q, c = y
        S = (c - self.Rstar(B, q, d)) / q
        P = self.P(B, q, d)
        qp = P / S
        l = S + self.a(d) * B
        cp = -(c / self.nu) * ((self.rho - self.r) + l * qp / q) / S
        return [qp, cp]

    def S_of(self, B, y):
        """Drift implied by (q, c) at B."""
        q, c = y
        return (c - self.Rstar(B, q)) / q


def illus(rho, nu=2.0):
    """The model at the parameters of the self-fulfilling example, with discount rate rho and CRRA curvature nu."""
    return Det(rho, nu=nu)
