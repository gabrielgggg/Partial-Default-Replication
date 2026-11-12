"""Self-fulfilling partial-default steady states of the deterministic model as a shooting problem.

The deterministic equilibrium (z = 0, y = 1) is the planar system in B
    q'(B) = P(B,q) / S,                 P = xi(d) q - (1-d)(delta+lam)
    c'(B) = -(c/nu) [ (rho - r) + l P / (q S) ] / S,   l = S + a(d) B   (Euler with the dilution term)
    S = (c - R*(B,q)) / q,  d = d*(B,q) static.
A candidate steady state B0 has S = 0, P = 0 (q0 = q_r(d0)), and the price slope q1 = -q0 (rho-r)/(a0 B0).
Its local branch is the analytic trajectory S ~ s1 x, q ~ q0 + q1 x, c ~ c0 + c1 x (x = B - B0), with
    s1 = P_q + P_B / q1,   c1 = R_B + R_q q1 + q0 s1     (total derivatives through d*).
Each branch starts at B0 -/+ eps and is integrated down to B = 0 (left branch) or up (right branch). It stops
when the drift reaches zero, consumption hits the floor, or the price leaves (q_floor, 1].
The class Det is also imported by the other pdss_*.py scripts. Run as a script, it prints the candidates on a
grid of B0 and how their left branches end.
Usage: python code/pdss_shoot.py [illus|calib] [rho ...]
"""
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp
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

    def gain(self, B, q):
        """Gain from the interior default intensity over full repayment (zero at the threshold)."""
        m = (1 - self.kappa * q) * self.cs * B
        di = min(1.0, (max(m, 0) / (self.g0 * self.g1 * self.F)) ** (1 / (self.g1 - 1)))
        return self.F - 1 - self.F * self.g0 * di ** self.g1 + m * di

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

    def branch(self, loc, direction=-1, eps=1e-4, Bend=None, cmin=0.05, rtol=1e-10):
        """integrate from B0 + direction*eps to Bend (0 for the left branch); returns a dict with the path and the stop reason"""
        B0 = loc["B0"]
        Bend = (0.0 if direction < 0 else 3.0) if Bend is None else Bend
        x = direction * eps
        B, y = B0 + x, np.array([loc["q0"] + loc["q1"] * x, loc["c0"] + loc["c1"] * x])
        path = [(B0, loc["q0"], loc["c0"], 0.0, loc["d0"])]
        reason = "reached end"
        nudges = 0
        # integrate piecewise, with d frozen within each regime, until a terminal event
        while (B - Bend) * direction < 0:
            d = self.dstar(B, y[0])

            def ev_S(Bv, yv, d=d):
                return (yv[1] - self.Rstar(Bv, yv[0], d)) / yv[0] * (-direction) - 1e-9   # S keeps its sign
            ev_S.terminal = True

            def ev_reg(Bv, yv, d=d):                         # regime change in d (threshold, or d hits 1)
                g = self.gain(Bv, yv[0])
                return g if d > 0 else -g if d == 0 else g
            ev_reg.terminal = True

            def ev_c(Bv, yv):
                return yv[1] - cmin
            ev_c.terminal = True

            qlow = self.qf if direction > 0 else 1e-3        # the right branch stops where the standard can bind

            def ev_q(Bv, yv, qlow=qlow):
                return min(1.0 + 1e-9 - yv[0], yv[0] - qlow)
            ev_q.terminal = True

            try:
                sol = solve_ivp(lambda Bv, yv: self.rhs(Bv, yv, d), (B, Bend), y, method="LSODA", rtol=rtol, atol=1e-12,
                                events=[ev_S, ev_reg, ev_c, ev_q], dense_output=False, max_step=0.002)
            except ValueError:                               # an event started at its root: nudge and retry
                B = B + direction * 1e-7
                nudges += 1
                if nudges > 50:
                    reason = "stuck at a regime boundary"
                    break
                continue
            for Bv, qv, cv in zip(sol.t, sol.y[0], sol.y[1]):
                path.append((Bv, qv, cv, self.S_of(Bv, (qv, cv)), self.dstar(Bv, qv)))
            if sol.status == 1:
                hit = [k for k in range(4) if sol.t_events[k].size]
                k = hit[0]
                B, y = sol.t_events[k][0], sol.y_events[k][0]
                if k == 1:                                   # regime switch: step across the threshold and continue
                    B = B + direction * 1e-9
                    continue
                reason = {0: "drift reached zero", 2: "consumption floor", 3: ("reached the standard" if direction > 0 else "price left (0,1]")}[k]
                break
            elif sol.status < 0:
                reason = "integrator failure: " + sol.message
                break
            else:
                B, y = sol.t[-1], sol.y[:, -1]
        p = np.array(path)
        return dict(path=p, reason=reason, Bstop=p[-1, 0], q_end=p[-1, 1], c_end=p[-1, 2], S_end=p[-1, 3],
                    Smin=p[1:, 3].min() if len(p) > 1 else np.nan, Smax=p[1:, 3].max() if len(p) > 1 else np.nan)


def illus(rho):
    """Illustrative parameterization at discount rate rho."""
    import os
    nu = float(os.environ.get("PDSS_NU", "2.0"))             # CRRA curvature, from PDSS_NU (default 2)
    return Det(rho, nu=nu)                                   # g0 0.075, g1 1.5, fixed cost 0.001 (log form at z=0)


def calib(rho):
    """Calibrated parameters at z = 0, with discount rate rho."""
    g0, g1, g2, zt = 0.05869479, 3.21745438, 0.05715232, -0.29477503
    F = 1 - g2 * (1 - math.exp(zt))                          # level form at z = 0
    return Det(rho, r=math.log(1.04), delta=0.1692636, kappa=0.56210513, g0=g0, g1=g1, F=F, q_floor=0.7)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "illus"
    rhos = [float(v) for v in sys.argv[2:]] or [0.06]
    for rho in rhos:
        m = illus(rho) if which == "illus" else calib(rho)
        print(f"== {which} rho {rho} wedge {1e4 * (rho - m.r):.0f} bp, d_first {m.d_first():.3f}")
        # candidates on a grid of B0; integrate the left branch of each attracting one
        for B0 in np.arange(0.62, 1.40, 0.02):
            for q0 in m.candidates(B0):
                loc = m.local(B0, q0)
                if loc["s1"] >= 0:
                    print(f"  B0 {B0:.2f} q0 {q0:.4f} d0 {loc['d0']:.3f} spread {1e4 * loc['spread']:.0f}bp s1 {loc['s1']:+.4f} repelling")
                    continue
                L = m.branch(loc, -1)
                print(f"  B0 {B0:.2f} q0 {q0:.4f} d0 {loc['d0']:.3f} spread {1e4 * loc['spread']:.0f}bp s1 {loc['s1']:+.4f} | left: {L['reason']:20s} "
                      f"at B {L['Bstop']:.3f} q {L['q_end']:.4f} S {L['S_end']:+.4f} Smax {L['Smax']:.3f}")
