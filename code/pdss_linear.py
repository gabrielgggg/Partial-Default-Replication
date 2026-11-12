"""Risk-neutral (linear utility) limit of the deterministic model, where the slow manifold of the CRRA model is exact.

 On a borrowing arc the sovereign is indifferent (V' = -q) and the price solves the
regular scalar ODE
    q'(B) = -(rho + a(d)) (q - q_rho(d)) / (a(d) B),   q_rho(d) = (1-d)(delta+lam)/(rho + a(d)),
with drift S = a B (r+a) (q_r(d) - q) / ((rho+a) (q - q_rho(d))), positive iff q_rho(d) < q < q_r(d).
Through each candidate steady state (q = q_r(d), a point of the same curve as in the CRRA model) passes one arc,
with slope equal to the slope condition. A partial-default equilibrium needs an attracting candidate whose arc,
followed down, stays in the borrowing band and leaves the default region at a price above q_rho(0) =
(delta+lam)/(rho+delta), that is, at a spread below the wedge; below that the sovereign lands from a jump.
Also tabulates W_att(B0) = P_B a B / (q P_q), the largest wedge at which the candidate attracts (CRRA and linear).
Prints one summary line per rho; -v adds a sample of candidates.
Usage: python code/pdss_linear.py [illus|calib] rho [rho ...] [-v]
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
from pdss_shoot import illus, calib


def qrho(m, d):
    """Price q_rho(d) at which the drift on the arc becomes infinite."""
    return (1 - d) * m.cs / (m.rho + m.a(d))


def W_att(m, B0, q0):
    """Largest wedge at which the candidate (B0, q0) attracts."""
    d0 = m.dstar(B0, q0); a0 = m.a(d0)
    Pd = (1 - m.kappa * q0) * m.cs
    dB = d0 / ((m.g1 - 1) * B0); dq = -d0 / (m.g1 - 1) * m.kappa / (1 - m.kappa * q0)
    PB, Pq = Pd * dB, m.xi(d0) + Pd * dq
    return PB * a0 * B0 / (q0 * Pq) if Pq > 0 else np.inf


def arc_down(m, B0, q0):
    """follow the arc from the candidate down to the default threshold; report how it leaves"""
    def f(B, y):
        d = m.dstar(B, y[0]); a = m.a(d)
        return [-(m.rho + a) * (y[0] - qrho(m, d)) / (a * B)]

    def ev_lo(B, y):   # q reaches q_rho(d): the drift becomes infinite (jump region starts)
        return y[0] - qrho(m, m.dstar(B, y[0]))
    ev_lo.terminal = True

    def ev_hi(B, y):   # q reaches q_r(d): the drift falls to zero before the threshold
        return m.qr(m.dstar(B, y[0])) - y[0] - 1e-12
    ev_hi.terminal = True

    def ev_thr(B, y):  # the default threshold
        return m.gain(B, y[0])
    ev_thr.terminal = True
    h = 1e-6
    sol = solve_ivp(f, (B0 - h, 1e-6), [q0 + h * (m.rho - m.r) * q0 / (m.a(m.dstar(B0, q0)) * B0)], method="LSODA",
                    rtol=1e-11, atol=1e-13, events=[ev_lo, ev_hi, ev_thr], max_step=0.002)
    k = [j for j in range(3) if sol.t_events[j].size]
    if not k:
        return dict(how="no threshold", B=sol.t[-1], q=sol.y[0, -1])
    k = k[0]; B, q = sol.t_events[k][0], sol.y_events[k][0][0]
    return dict(how=["hits q_rho (jump)", "drift zero", "threshold"][k], B=B, q=q, spread=m.spread(q),
                ok=(k == 2 and q > (m.cs / (m.rho + m.delta)) and q < 1))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "illus"
    for rho in [float(v) for v in sys.argv[2:]]:
        m = illus(rho) if which == "illus" else calib(rho)
        sf = m.spread(m.qr(m.d_first()))
        # attracting candidates and how their arcs leave the default region
        rows = []
        for B0 in np.arange(0.2, 1.6, 0.005):
            for q0 in m.candidates(B0):
                W = W_att(m, B0, q0)
                if W <= rho - m.r:
                    continue
                A = arc_down(m, B0, q0)
                rows.append((B0, q0, m.dstar(B0, q0), W, A))
        ok = [r for r in rows if r[4].get("ok")]
        print(f"{which} rho {rho:.4f} wedge {1e4 * (rho - m.r):5.0f} bp | s_first {1e4 * sf:4.0f} bp | attracting {len(rows):3d}"
              + (f" B0 [{min(r[0] for r in rows):.3f},{max(r[0] for r in rows):.3f}]" if rows else "")
              + f" | arc reaches threshold below the wedge: {len(ok):3d}"
              + (f" B0 [{min(r[0] for r in ok):.3f},{max(r[0] for r in ok):.3f}], entry spread {1e4 * min(r[4]['spread'] for r in ok):.0f}-{1e4 * max(r[4]['spread'] for r in ok):.0f} bp" if ok else ""))
        if "-v" in sys.argv:
            for r in rows[::4]:
                A = r[4]
                print(f"     B0 {r[0]:.3f} d0 {r[2]:.3f} W_att {1e4 * r[3]:.0f}bp  arc: {A['how']} at B {A['B']:.3f} q {A['q']:.4f} spread {1e4 * A.get('spread', np.nan):.0f}bp")
