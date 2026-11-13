"""Forward integration in B of the deterministic ODE from a starting point (q, c).

Uses explicit RK4 with a step that resolves the fast consumption mode, whose rate is c (rho-r) / (nu q S^2).
The model m is a pdss_shoot.Det. fwd classifies the start:
  CAPT  the drift falls to Smin with the price slope P/S at the slope condition and d > 0: captured by a
        partial-default steady state (the path approaches the singular curve along a leaf)
  BRK   the drift falls to Smin with the price slope off the slope condition (breakdown)
  OVER  the drift stays positive up to Bend
  QOUT  the price leaves (0, 1]
  NEG   the drift is not positive at the start
right follows the right branch of a candidate steady state.

Reads and writes no file. Imported by pdss_sweep.py and pdss_threshold.py.
"""
import numpy as np


def fwd(m, q0, c0, B0=0.0, Bend=1.5, Smin=1e-4, hmax=2e-3):
    """Integrate from (B0, q0, c0) until the drift falls to Smin, B reaches Bend, or q leaves (0, 1]; classify the end."""
    B, y = B0, np.array([q0, c0], dtype=float)
    S = m.S_of(B, y)
    if S <= Smin:
        return dict(cls="NEG", B=B, q=q0, c=c0, S=S, d=0.0, ratio=np.nan, slope=np.nan)
    nudge = 0
    while True:
        q, c = y
        S = m.S_of(B, y)
        d = m.dstar(B, q)
        if S <= Smin or B >= Bend or q > 1 + 1e-9 or q < 1e-3:
            break
        # step bounded by the inverse rate of the fast mode
        lam_f = c * (m.rho - m.r) / (m.nu * q * S * S)
        h = min(hmax, 1.0 / lam_f, Bend - B)

        def f(Bv, yv):
            return np.array(m.rhs(Bv, yv, d))            # regime frozen within a step
        k1 = f(B, y); k2 = f(B + h / 2, y + h / 2 * k1); k3 = f(B + h / 2, y + h / 2 * k2); k4 = f(B + h, y + h * k3)
        ynew = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if m.dstar(B + h, ynew[0]) > 0 and d == 0 or m.dstar(B + h, ynew[0]) == 0 and d > 0:
            if h > 1e-9 and nudge < 60:                      # refine the step across a regime switch
                hmax = h / 4; nudge += 1; continue
        hmax = 2e-3 if nudge else hmax
        nudge = 0
        B, y = B + h, ynew
    # compare the price slope P/S with the slope condition at the end point
    q, c = y; d = m.dstar(B, q); S = m.S_of(B, y)
    ratio = m.P(B, q, d) / S if S > 0 else np.nan
    slope = -(m.rho - m.r) * q / (m.a(d) * B) if B > 0 else np.nan
    if S <= Smin:
        cls = "CAPT" if (d > 0 and abs(ratio - slope) < 0.1 * abs(slope)) else "BRK"
    elif B >= Bend - 1e-12:
        cls = "OVER"
    else:
        cls = "QOUT"
    return dict(cls=cls, B=B, q=q, c=c, S=S, d=d, ratio=ratio, slope=slope)


def right(m, loc, eps=2e-3, Bend=2.0, hmax=2e-3):
    """the right branch from a candidate: start at B0 + eps on the linear expansion (the fast mode decays in this
    direction), integrate up with RK4 resolving the fast mode. Stops when the drift returns to zero (breakdown),
    the price reaches the standard (where the constrained regime takes over), d reaches one, or Bend."""
    x = eps
    B, y = loc["B0"] + x, np.array([loc["q0"] + loc["q1"] * x, loc["c0"] + loc["c1"] * x])
    Smax = -np.inf
    while True:
        q, c = y; S = m.S_of(B, y); d = m.dstar(B, q)
        if S >= 0:
            return dict(cls="BRK", B=B, q=q, d=d, Smax=Smax)
        if q < m.qf:
            return dict(cls="STD", B=B, q=q, d=d, Smax=Smax)
        if d >= 1:
            return dict(cls="FULL", B=B, q=q, d=d, Smax=Smax)
        if B >= Bend:
            return dict(cls="END", B=B, q=q, d=d, Smax=Smax)
        Smax = max(Smax, S)
        lam_f = c * (m.rho - m.r) / (m.nu * q * S * S)
        h = min(hmax, 1.0 / lam_f)

        def f(Bv, yv):
            return np.array(m.rhs(Bv, yv, d))
        k1 = f(B, y); k2 = f(B + h / 2, y + h / 2 * k1); k3 = f(B + h / 2, y + h / 2 * k2); k4 = f(B + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); B += h
