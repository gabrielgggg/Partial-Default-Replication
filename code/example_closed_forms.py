"""Closed-form numbers of the self-fulfilling example (Figure det_multiplicity), at its own parameters.

Computes d_first, s_first, (g1-1) xi(d_first), Omega(d_first), and the intensities d whose risk-neutral steady
states pass Theta(d) <= rho - r < Omega(d) with q_r(d) at or above the underwriting standard, mapped to debt B(d).
The continuation condition above the steady state is not checked here.

Reads and writes no file; prints a summary. Imported by text_numbers.py for the values.
Usage: python code/example_closed_forms.py
"""
import numpy as np

# parameters of the example and the closed forms
r = 0.039; lam = r; delta = 0.12; kappa = 0.7; g0 = 0.075; g1 = 1.5; F = 0.999; qf = 0.7; rho = 0.06
cs = delta + lam
a = lambda d: delta - kappa * cs * d
xi = lambda d: r + a(d)
qr = lambda d: (1 - d) * cs / xi(d)
s = lambda d: (1 - kappa) * cs * d / (1 - d)
dfirst = ((1 - F) / ((g1 - 1) * g0 * F)) ** (1 / g1)
Theta = lambda d: s(d) * (1 + (g1 - 1) * (dfirst / d) ** g1) / g1
def Omega(d):
    den = (g1 - 1) * xi(d) - kappa * cs * d
    return np.inf if den <= 0 else s(d) * a(d) / den
Bd = lambda d: g0 * g1 * F * d ** (g1 - 1) * xi(d) / ((1 - kappa) * cs ** 2)

w = rho - r
print(f"d_first {dfirst:.4f}; s_first {1e4 * s(dfirst):.1f} bp; (g1-1) xi(d_first) {(g1 - 1) * xi(dfirst):.4f} vs delta {delta}; "
      f"Omega(d_first) {1e4 * Omega(dfirst):.1f} bp; q_r(d_first) {qr(dfirst):.4f}; wedge {1e4 * w:.1f} bp")
# intensities on a fine grid that pass the conditions
ds = np.linspace(dfirst, 0.999, 200001)
ok = np.array([Theta(d) <= w < Omega(d) and qr(d) >= qf for d in ds])
if ok.any():
    sel = ds[ok]
    print(f"passing intensities d in [{sel.min():.4f}, {sel.max():.4f}] -> B(d) in [{Bd(sel.min()):.4f}, {Bd(sel.max()):.4f}]; "
          f"contiguous: {np.all(np.diff(np.where(ok)[0]) == 1)}")
else:
    print("no intensity passes")
