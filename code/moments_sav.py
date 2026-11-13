"""Moments of a solution that may hold assets (B < 0), with the definitions of pdmoments.combined_moments.

Output is net of default costs, as in pdmoments. Two differences from pdmoments, both inactive when B >= 0:
- debt, debt due, debt service, and defaulted coupons use gross debt max(B, 0): a sovereign with assets has
  no debt and no debt service;
- the spread in a state with B < 0 is the spread at which the sovereign would issue at zero debt, q(0, z),
  which is how the solution without saving treats its atom at B = 0.
Everything else (16 paths of 25,000 years, annual accounting, seeds, the split between ergodic and simulated
moments) is as in pdmoments.

Reads and writes no file. Imported by tab_saving.py: combined_sav(solution, par, mu, sigma).
"""
import numpy as np
from numba import njit, prange
import pdmoments as PM
from pdmodel import phi_fn


@njit(cache=True)
def simulate_sav(S, d, c, q, Bg, zg, mu, sigma, T_years, steps_per_year, B0, z0, seed, delta_lam, r, delta,
                 g0, g1, g2, zt, Fmin, form, net):
    """pdmoments.simulate for the OU process, with gross debt and the zero-debt price in savings states."""
    np.random.seed(seed)
    dt = 1.0 / steps_per_year
    out = np.zeros((11, T_years))
    B = B0; z = z0
    zmin = zg[0]; zmax = zg[-1]; Bmin = Bg[0]; Bmax = Bg[-1]
    for t in range(T_years):
        y_acc = 0.0; c_acc = 0.0; dd_acc = 0.0; ds_acc = 0.0; dc_acc = 0.0
        for k in range(steps_per_year):
            Sv = PM._interp2(S, Bg, zg, B, z)
            dv = PM._interp2(d, Bg, zg, B, z)
            cv = PM._interp2(c, Bg, zg, B, z)
            y = np.exp(z)
            if net > 0.5:
                y = phi_fn(dv, z, g0, g1, g2, zt, Fmin, form) * y    # output net of default costs
            due = delta_lam * max(B, 0.0)                       # gross debt
            y_acc += y * dt; c_acc += cv * dt; dd_acc += due * dt
            ds_acc += (1.0 - dv) * due * dt; dc_acc += dv * due * dt
            B = B + Sv * dt
            if B < Bmin: B = Bmin
            if B > Bmax: B = Bmax
            z = z - mu * z * dt + sigma * np.sqrt(dt) * np.random.randn()
            if z < zmin: z = 2 * zmin - z
            if z > zmax: z = 2 * zmax - z
        qv = PM._interp2(q, Bg, zg, max(B, 0.0), z)             # price at zero debt when saving
        dv = PM._interp2(d, Bg, zg, B, z)
        out[0, t] = y_acc; out[1, t] = c_acc; out[2, t] = dd_acc; out[3, t] = ds_acc; out[4, t] = dc_acc
        out[5, t] = dc_acc / dd_acc if dd_acc > 1e-12 else 0.0
        out[6, t] = max(B, 0.0); out[7, t] = delta_lam / qv - (delta + r); out[8, t] = dv; out[9, t] = B; out[10, t] = z
    return out


@njit(parallel=True, cache=True)
def _paths(S, d, c, q, Bg, zg, mu, sigma, T_years, steps_per_year, B0, z0, seed0, delta_lam, r, delta, n_paths,
           g0, g1, g2, zt, Fmin, form, net):
    """n_paths calls of simulate_sav in parallel, path k with seed seed0 + 1000 k."""
    out = np.zeros((n_paths, 11, T_years))
    for k in prange(n_paths):
        out[k] = simulate_sav(S, d, c, q, Bg, zg, mu, sigma, T_years, steps_per_year, B0, z0, seed0 + 1000 * k,
                              delta_lam, r, delta, g0, g1, g2, zt, Fmin, form, net)
    return out


def ergodic_sav(B, z, q, d, c, g, par):
    """pdmoments.ergodic_moments with gross debt and the zero-debt price in savings states."""
    w = g / g.sum()
    Bp = np.maximum(B, 0.0)[:, None] * np.ones_like(w)
    y = np.exp(z)[None, :] * np.ones_like(w)
    if PM.NET_OUTPUT:                                             # output net of default costs
        form = 1.0 if par.fixed_cost_form == "level" else 0.0
        y = y * np.array([[phi_fn(d[i, j], z[j], par.g0, par.g1, par.g2, par.ztilde, par.F_min, form)
                           for j in range(z.size)] for i in range(B.size)])
    # price at zero debt in savings states
    i0 = int(np.argmin(np.abs(B)))
    qe = q.copy(); qe[B < -1e-12, :] = q[i0, :][None, :]
    coupon = par.delta + par.lam
    debt_y = Bp / y; dd_y = coupon * Bp / y; ds_y = (1 - d) * dd_y
    sp = coupon / qe - (par.delta + par.r)
    def mean(x): return float((w * x).sum())
    def sd(x): mu = mean(x); return float(np.sqrt((w * (x - mu) ** 2).sum()))
    E = dict(debt_y_mean=mean(debt_y), debt_y_sd=sd(debt_y), ds_y_mean=mean(ds_y), ds_y_sd=sd(ds_y),
             dd_y_mean=mean(dd_y), spread_mean=mean(sp), spread_sd=sd(sp))
    ly = np.log(y)
    E["corr_spread_y"] = float((w * (sp - E["spread_mean"]) * (ly - mean(ly))).sum() / (E["spread_sd"] * sd(ly) + 1e-300))
    E["corr_spread_debt"] = float((w * (sp - E["spread_mean"]) * (debt_y - E["debt_y_mean"])).sum()
                                  / (E["spread_sd"] * E["debt_y_sd"] + 1e-300))
    return E


def combined_sav(sol, par, mu, sigma, n_paths=16, T_years=25000, steps_per_year=100, seed=1, burn=1000):
    """Moments of a saved solution with or without saving (dict with B, z, S, c, d, q, g): ergodic for
    pdmoments.ERGODIC_KEYS, simulated for pdmoments.SIM_KEYS, the loss, and the ergodic mass on B < 0.
    Returns (M, across-path standard deviations)."""
    B, z = sol["B"], sol["z"]
    E = ergodic_sav(B, z, sol["q"], sol["d"], sol["c"], sol["g"], par)
    B0 = 0.5 * (B[0] + B[-1]) * 0.5 if B[0] >= 0 else 0.25        # the start of pdmoments on grids with B >= 0
    out = _paths(sol["S"], sol["d"], sol["c"], sol["q"], B, z, mu, sigma, T_years + burn, steps_per_year,
                 B0, 0.0, seed, par.delta + par.lam, par.r, par.delta, n_paths, *PM._phi_args(par))
    names = ["y", "cons", "debt_due", "debt_service", "def_coupons", "pd", "B", "spread", "d_eoy", "B_eoy", "z_eoy"]
    sims = [{n: out[k, i, burn:] for i, n in enumerate(names)} for k in range(n_paths)]
    # combine ergodic and simulated moments as in pdmoments.combined_moments
    Sm, Ss = PM.moments_paths(sims, 0.0)
    M = {k: E[k] for k in PM.ERGODIC_KEYS}
    M.update({k: Sm[k] for k in PM.SIM_KEYS})
    M["loss"] = PM.loss(M)
    neg = B < -1e-12
    gn = sol["g"] / sol["g"].sum()
    M["mass_assets"] = float(gn[neg].sum())
    return M, Ss
