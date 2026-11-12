"""Second-order coefficients of the cube-root expansion at the deterministic steady state
(Theorem "cube-root law", Step 4 of its proof):
    -S(B*+x) = beta x^{1/3} + Sigma1 x^{2/3} + O(x),   1 - q(B*+x) = alpha x^{2/3} + Q1 x + O(x^{4/3}),
    Sigma1 = -[2 a(d_first) B* alpha + beta^2 phi'(cbar)/phi(cbar) + 3 phi(cbar) (R_B + rho)/beta] / 4,
    Q1 = -Pi0 Sigma1 / beta^2,
with phi = u'/|u''| (= c/nu for CRRA, so phi'/phi = 1/cbar), R_B = -(1-d_first)(delta+lambda) + a(d_first),
Pi0 = (1-kappa)(delta+lambda) d_first, a(d) = delta - kappa (delta+lambda) d.
"""
import numpy as np


def second_order(consts, x, lam=float(np.log(1.04)), nu=2.0):
    """consts: npz/dict with Bpd, dfirst, cbar, beta, alpha (from illustrations.py);
    x: the 9-vector of calibrated parameters (g0, g1, g2, ztilde, rho, kappa, delta, rho_z, sig_eps)."""
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = [float(v) for v in x]
    Bpd, df, cbar, beta, alpha = [float(consts[k]) for k in ("Bpd", "dfirst", "cbar", "beta", "alpha")]
    a_df = delta - kappa * (delta + lam) * df
    Pi0 = (1 - kappa) * (delta + lam) * df
    RB = -(1 - df) * (delta + lam) + a_df
    Sig1 = -(2 * a_df * Bpd * alpha + beta ** 2 / cbar + 3 * (cbar / nu) * (RB + rho) / beta) / 4
    Q1 = -Pi0 * Sig1 / beta ** 2
    return Sig1, Q1
