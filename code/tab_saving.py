"""tab_saving.py: Table C.2, model moments without and with saving, written to output/tab_saving_rows.tex.
Inputs: data/ou_main.npz and data/ou_saving.npz. Usage: python code/tab_saving.py"""
import os

import numpy as np

import moments_sav as MS
from paths import DATA, OUT
from pdmodel_sav import Params

# parameters of the main solution; moments of both solutions under the same parameters
a = np.load(os.path.join(DATA, "ou_main.npz"))
s = np.load(os.path.join(DATA, "ou_saving.npz"))
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = map(float, a["x"])
mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
par = Params(nu=2.0, rho=rho, r=float(np.log(1.04)), delta=delta, kappa=kappa, g0=g0, g1=g1, g2=g2,
             ztilde=zt, fixed_cost_form="level", S_cap=1.0, q_floor=float(a["qf"]))
Mb, _ = MS.combined_sav(a, par, mu, sigma)
Ms, _ = MS.combined_sav(s, par, mu, sigma)

# rows: moment key, label, decimals; None marks the rule between targeted and untargeted moments
rows = [("pd_freq", "Partial default: frequency", 2), ("pd_mean_cond", "Partial default: mean $\\mid d>0$", 2),
        ("pd_sd_cond", "Partial default: s.d. $\\mid d>0$", 2), ("debt_y_mean", "Debt to output: mean", 2),
        ("debt_y_sd", "Debt to output: s.d.", 2), ("ds_y_mean", "Debt service to output: mean", 3),
        ("ds_y_sd", "Debt service to output: s.d.", 3), ("dd_y_mean", "Debt due to output: mean", 3),
        ("spread_sd", "Spread: s.d.", 3), ("y_persist", "Output: persistence", 2), ("y_sd", "Output: s.d.", 2),
        None,
        ("dc_y_mean_cond", "Defaulted coupons to output: mean $\\mid d>0$", 3),
        ("dc_y_sd_cond", "Defaulted coupons to output: s.d. $\\mid d>0$", 3), ("spread_mean", "Spread: mean", 3),
        ("corr_spread_y", "Spread: correlation with output", 2), ("corr_spread_debt", "Spread: correlation with debt", 2),
        ("c_sd_rel", "Consumption s.d.\\ relative to output s.d.", 2)]


def fmt(x, k):
    """Number with k decimals, in math mode when negative."""
    s_ = f"{x:.{k}f}"
    return f"${s_}$" if x < 0 else s_


tex = []
for r in rows:
    if r is None:
        tex.append(r"\midrule")
        continue
    key, lab, k = r
    tex.append(f"{lab} & {fmt(Mb[key], k)} & {fmt(Ms[key], k)} \\\\")
open(os.path.join(OUT, "tab_saving_rows.tex"), "w", newline="\n").write("\n".join(tex) + "\n")
