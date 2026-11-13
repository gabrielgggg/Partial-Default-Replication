"""
pdmoments.py -- simulated and ergodic moments with the paper's definitions, and the calibration loss.

Moments follow the accounting of Arellano, Mateos-Planas and Rios-Rull (2023, JPE), Table 4.
State moments come from the exact ergodic distribution of the discretized model, path moments
from long simulations with annual accounting (combined_moments). loss() is their weighted
squared relative distance to the targets.

Mapping of the continuous-time model to their annual accounting (their delta_AMR = 0.88 is the
survival rate, ours delta = 0.12 is the maturity rate; R - delta_AMR = 0.16 ~ delta + lam = 0.159):

    debt due (flow, per year)      = (delta + lam) B
    debt service                   = (1 - d) (delta + lam) B
    defaulted coupons              = d (delta + lam) B
    partial default (annual)       = defaulted coupons / debt due   (AMR eq. 4)
    debt level                     = PV of promised coupons at r = B   (q_rf = 1)
    spread                         = (delta + lam)/q - (delta + r)   (yield to maturity minus r)
    output                         = y = phi(d, z) exp(z), net of default costs, when NET_OUTPUT (the paper);
                                     the endowment exp(z) otherwise. Log output is linearly detrended.
    consumption                    = c

Flows are integrated within the year; stocks (debt level, spread) are year-end values.
A year is "in partial default" if its annual partial default exceeds `pd_thresh`.

The simulation is Euler-Maruyama on (B, z) with the OU reflected at the z-grid bounds and
policies (S, d, c, q) bilinearly interpolated on the (B, z) grid; B is kept inside the grid.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from pdmodel import Model, Result, phi_fn
from amr_values import AMR_DATA, AMR_MODEL, TARGETS, WEIGHTS

# Output in the moments: net of default costs, phi(d, z) e^z (True, the paper), or the endowment e^z (False)
NET_OUTPUT = True


def _phi_args(par):
    """Default-cost parameters for simulate(): (g0, g1, g2, ztilde, F_min, form, net)."""
    return (par.g0, par.g1, par.g2, par.ztilde, par.F_min, 1.0 if par.fixed_cost_form == "level" else 0.0,
            1.0 if NET_OUTPUT else 0.0)


def output_grid(res) -> np.ndarray:
    """Output on the (B, z) grid: phi(d, z) e^z when NET_OUTPUT, else e^z."""
    m = res.model; par = m.par; y = np.exp(m.z)[None, :] * np.ones_like(res.d)
    if not NET_OUTPUT:
        return y
    g0, g1, g2, zt, Fm, form, _ = _phi_args(par)
    ph = np.array([[phi_fn(res.d[i, j], m.z[j], g0, g1, g2, zt, Fm, form) for j in range(m.z.size)]
                   for i in range(m.B.size)])
    return ph * y


@njit(cache=True)
def _interp2(F, Bg, zg, B, z):
    """Bilinear interpolation of F on the (B, z) grid, clamped to the grid; z uniform."""
    nB = Bg.size; nz = zg.size
    i = np.searchsorted(Bg, B) - 1          # Bg[i] <= B < Bg[i+1]  (works on non-uniform grids)
    if i < 0: i = 0
    if i > nB - 2: i = nB - 2
    wB = (B - Bg[i]) / (Bg[i + 1] - Bg[i])
    if wB < 0.0: wB = 0.0
    if wB > 1.0: wB = 1.0
    if nz == 1:
        return (1 - wB) * F[i, 0] + wB * F[i + 1, 0]
    dz = zg[1] - zg[0]
    j = int((z - zg[0]) / dz)
    if j < 0: j = 0
    if j > nz - 2: j = nz - 2
    wz = (z - zg[j]) / dz
    if wz < 0.0: wz = 0.0
    if wz > 1.0: wz = 1.0
    return ((1 - wB) * (1 - wz) * F[i, j] + wB * (1 - wz) * F[i + 1, j]
            + (1 - wB) * wz * F[i, j + 1] + wB * wz * F[i + 1, j + 1])


@njit(cache=True)
def simulate(S, d, c, q, Bg, zg, mu, sigma, kind_ou, Lam, T_years, steps_per_year, B0, z0, seed,
             delta_lam, r, delta, g0=0.0, g1=1.0, g2=0.0, zt=0.0, Fmin=0.0, form=0.0, net=0.0):
    """Returns annual arrays: y, cons, debt_due, debt_service, def_coupons, pd, debt_level,
    spread, d_eoy, B_eoy, z_eoy.  kind_ou = 1 for OU, 0 for a chain with generator Lam."""
    np.random.seed(seed)
    dt = 1.0 / steps_per_year
    nT = T_years
    out = np.zeros((11, nT))
    B = B0; z = z0
    jz = 0
    nz = zg.size
    if kind_ou == 0:
        # nearest state
        best = 1e9
        for j in range(nz):
            if abs(zg[j] - z0) < best:
                best = abs(zg[j] - z0); jz = j
        z = zg[jz]
    zmin = zg[0]; zmax = zg[-1]
    Bmin = Bg[0]; Bmax = Bg[-1]
    for t in range(nT):
        # accumulate the flows over the year
        y_acc = 0.0; c_acc = 0.0; dd_acc = 0.0; ds_acc = 0.0; dc_acc = 0.0
        for k in range(steps_per_year):
            Sv = _interp2(S, Bg, zg, B, z)
            dv = _interp2(d, Bg, zg, B, z)
            cv = _interp2(c, Bg, zg, B, z)
            y = np.exp(z)
            if net > 0.5:
                y = phi_fn(dv, z, g0, g1, g2, zt, Fmin, form) * y
            due = delta_lam * B
            y_acc += y * dt; c_acc += cv * dt; dd_acc += due * dt
            ds_acc += (1.0 - dv) * due * dt; dc_acc += dv * due * dt
            # state update: Euler step for B (kept in the grid), OU step reflected at the z bounds
            # or a chain jump
            B = B + Sv * dt
            if B < Bmin: B = Bmin
            if B > Bmax: B = Bmax
            if kind_ou == 1:
                z = z - mu * z * dt + sigma * np.sqrt(dt) * np.random.randn()
                if z < zmin: z = 2 * zmin - z
                if z > zmax: z = 2 * zmax - z
            else:
                u = np.random.rand()
                rate = -Lam[jz, jz]
                if u < rate * dt:
                    # jump to another state proportional to Lam[jz, :]
                    v = np.random.rand() * rate
                    acc = 0.0
                    for jp in range(nz):
                        if jp == jz:
                            continue
                        acc += Lam[jz, jp]
                        if v <= acc:
                            jz = jp; break
                    z = zg[jz]
        # year-end stocks and the annual partial default rate
        qv = _interp2(q, Bg, zg, B, z)
        dv = _interp2(d, Bg, zg, B, z)
        out[0, t] = y_acc; out[1, t] = c_acc; out[2, t] = dd_acc; out[3, t] = ds_acc; out[4, t] = dc_acc
        out[5, t] = dc_acc / dd_acc if dd_acc > 1e-12 else 0.0
        out[6, t] = B; out[7, t] = delta_lam / qv - (delta + r); out[8, t] = dv; out[9, t] = B; out[10, t] = z
    return out


def moments(sim: dict, pd_thresh=0.0) -> dict:
    """Annual-accounting moments of one simulated path; output and consumption linearly detrended in logs."""
    y = sim["y"]; c = sim["cons"]
    ly = np.log(y); lc = np.log(c)
    t = np.arange(ly.size)
    ly_d = ly - np.polyval(np.polyfit(t, ly, 1), t)
    lc_d = lc - np.polyval(np.polyfit(t, lc, 1), t)
    pdv = sim["pd"]
    inpd = pdv > pd_thresh
    debt_y = sim["B"] / y
    ds_y = sim["debt_service"] / y
    dd_y = sim["debt_due"] / y
    dc_y = sim["def_coupons"] / y
    sp = sim["spread"]
    M = {}
    M["pd_freq"] = float(inpd.mean())
    M["pd_mean_cond"] = float(pdv[inpd].mean()) if inpd.any() else 0.0
    M["pd_sd_cond"] = float(pdv[inpd].std()) if inpd.any() else 0.0
    M["debt_y_mean"] = float(debt_y.mean()); M["debt_y_sd"] = float(debt_y.std())
    M["ds_y_mean"] = float(ds_y.mean()); M["ds_y_sd"] = float(ds_y.std())
    M["dd_y_mean"] = float(dd_y.mean())
    M["spread_sd"] = float(sp.std()); M["spread_mean"] = float(sp.mean())
    M["y_persist"] = float(np.corrcoef(ly_d[1:], ly_d[:-1])[0, 1])
    M["y_sd"] = float(ly_d.std())
    M["dc_y_mean_cond"] = float(dc_y[inpd].mean()) if inpd.any() else 0.0
    M["dc_y_sd_cond"] = float(dc_y[inpd].std()) if inpd.any() else 0.0
    M["corr_spread_y"] = float(np.corrcoef(sp, ly_d)[0, 1])
    M["corr_spread_debt"] = float(np.corrcoef(sp, debt_y)[0, 1])
    M["c_sd_rel"] = float(lc_d.std() / ly_d.std())
    return M


# the targets and weights of AMR (2023) are in amr_values.py


def loss(M: dict, target=AMR_DATA) -> float:
    """Weighted sum of squared relative deviations from the targets over TARGETS."""
    s = 0.0
    for k in TARGETS:
        w = WEIGHTS.get(k, 1.0)
        s += w * ((M[k] - target[k]) / target[k]) ** 2
    return s


@njit(parallel=True, cache=True)
def _simulate_paths(S, d, c, q, Bg, zg, mu, sigma, kind_ou, Lam, T_years, steps_per_year, B0, z0, seed0,
                    delta_lam, r, delta, n_paths, g0=0.0, g1=1.0, g2=0.0, zt=0.0, Fmin=0.0, form=0.0, net=0.0):
    """n_paths calls of simulate in parallel, path k with seed seed0 + 1000 k."""
    out = np.zeros((n_paths, 11, T_years))
    for k in prange(n_paths):
        out[k] = simulate(S, d, c, q, Bg, zg, mu, sigma, kind_ou, Lam, T_years, steps_per_year, B0, z0,
                          seed0 + 1000 * k, delta_lam, r, delta, g0, g1, g2, zt, Fmin, form, net)
    return out


def simulate_paths(res: Result, n_paths=16, T_years=25000, steps_per_year=100, seed=1, B0=None, z0=0.0, burn=1000):
    """n_paths independent long simulations (parallel).  Returns a list of sim dicts."""
    m = res.model; par = m.par; zp = m.zp
    kind_ou = 1 if zp.kind == "ou" else 0
    Lam = zp.Lambda if zp.kind == "jump" else np.zeros((m.nz, m.nz))
    if B0 is None:
        B0 = 0.5 * (m.B[0] + m.B[-1]) * 0.5
    out = _simulate_paths(res.S, res.d, res.c, res.q, m.B, m.z, zp.mu, zp.sigma, kind_ou, Lam, T_years + burn,
                          steps_per_year, B0, z0, seed, par.delta + par.lam, par.r, par.delta, n_paths, *_phi_args(par))
    names = ["y", "cons", "debt_due", "debt_service", "def_coupons", "pd", "B", "spread", "d_eoy", "B_eoy", "z_eoy"]
    return [{n: out[k, i, burn:] for i, n in enumerate(names)} for k in range(n_paths)]


def moments_paths(sims: list, pd_thresh=0.0):
    """Mean across paths of the annual-accounting moments, and the across-path standard deviation
    (a direct estimate of the simulation noise)."""
    Ms = [moments(s, pd_thresh) for s in sims]
    keys = Ms[0].keys()
    mean = {k: float(np.mean([M[k] for M in Ms])) for k in keys}
    sd = {k: float(np.std([M[k] for M in Ms])) for k in keys}
    return mean, sd


def ergodic_moments(res: Result, top_frac=0.1) -> dict:
    """Exact stationary moments of the discretized chain (instantaneous accounting): no sampling
    noise, and the tail (the explosive full-default region near the debt ceiling) is weighted exactly.
    Also reports the mass and the spread-variance share of the top `top_frac` of the B grid."""
    m = res.model; par = m.par
    # ergodic weights and the state functions on the grid
    w = res.g / res.g.sum()
    B = m.B[:, None]; y = output_grid(res)
    coupon = par.delta + par.lam
    debt_y = B / y; dd_y = coupon * B / y; ds_y = (1 - res.d) * dd_y; dc_y = res.d * dd_y
    sp = coupon / res.q - (par.delta + par.r)
    def mean(x): return float((w * x).sum())
    def sd(x): mu = mean(x); return float(np.sqrt((w * (x - mu) ** 2).sum()))
    # weighted means, standard deviations, and correlations
    E = {}
    E["debt_y_mean"] = mean(debt_y); E["debt_y_sd"] = sd(debt_y)
    E["ds_y_mean"] = mean(ds_y); E["ds_y_sd"] = sd(ds_y)
    E["dd_y_mean"] = mean(dd_y); E["dd_y_sd"] = sd(dd_y)
    E["spread_mean"] = mean(sp); E["spread_sd"] = sd(sp)
    inpd = res.d > 1e-10
    E["pd_point_freq"] = float(w[inpd].sum())
    E["d_mean_cond"] = float((w * res.d)[inpd].sum() / max(w[inpd].sum(), 1e-300))
    E["dc_y_mean_cond"] = float((w * dc_y)[inpd].sum() / max(w[inpd].sum(), 1e-300))
    E["c_sd_rel"] = sd(np.log(res.c)) / sd(np.log(y) * np.ones_like(res.c))
    E["corr_spread_y"] = float((w * (sp - E["spread_mean"]) * (np.log(y) - mean(np.log(y) * np.ones_like(sp)))).sum()
                               / (E["spread_sd"] * sd(np.log(y) * np.ones_like(sp)) + 1e-300))
    E["corr_spread_debt"] = float((w * (sp - E["spread_mean"]) * (debt_y - E["debt_y_mean"])).sum()
                                  / (E["spread_sd"] * E["debt_y_sd"] + 1e-300))
    # tail diagnostics: mass and spread-variance share of the top of the B grid, mass in full default
    top = np.broadcast_to(m.B[:, None] > m.B[-1] - top_frac * (m.B[-1] - m.B[0]), w.shape)
    E["mass_top"] = float(w[top].sum())
    E["spread_var_share_top"] = float((w * (sp - E["spread_mean"]) ** 2)[top].sum() / (E["spread_sd"] ** 2 + 1e-300))
    E["mass_full_default"] = float(w[res.d > 0.999].sum())
    return E


# which moments are taken from the exact stationary distribution (state functions, tail-sensitive)
# and which from the annual-accounting simulation (path functionals)
ERGODIC_KEYS = ["debt_y_mean", "debt_y_sd", "ds_y_mean", "ds_y_sd", "dd_y_mean", "spread_sd", "spread_mean",
                "corr_spread_y", "corr_spread_debt"]
SIM_KEYS = ["pd_freq", "pd_mean_cond", "pd_sd_cond", "y_persist", "y_sd", "dc_y_mean_cond", "dc_y_sd_cond", "c_sd_rel"]


def combined_moments(res: Result, n_paths=16, T_years=25000, steps_per_year=100, seed=1, pd_thresh=0.0):
    """Moments used in the calibration: exact ergodic for ERGODIC_KEYS, multi-path simulation
    (annual accounting) for SIM_KEYS.  Returns (M, info) with info holding both full sets and the
    across-path noise."""
    E = ergodic_moments(res)
    sims = simulate_paths(res, n_paths=n_paths, T_years=T_years, steps_per_year=steps_per_year, seed=seed)
    Smean, Ssd = moments_paths(sims, pd_thresh)
    M = {k: E[k] for k in ERGODIC_KEYS}
    M.update({k: Smean[k] for k in SIM_KEYS})
    return M, dict(ergodic=E, sim_mean=Smean, sim_sd=Ssd)


def table_combined(M: dict, info: dict, target=AMR_DATA, model=AMR_MODEL) -> str:
    """Markdown table of the moments used, their ergodic and simulated versions, the targets, and the loss."""
    rows = ["| moment | data (AMR) | AMR model | CT model (used) | ergodic | simulation (noise) |", "|---|---|---|---|---|---|"]
    E, Sm, Ss = info["ergodic"], info["sim_mean"], info["sim_sd"]
    for k in TARGETS + ["dc_y_mean_cond", "dc_y_sd_cond", "spread_mean", "corr_spread_y", "corr_spread_debt", "c_sd_rel"]:
        tag = "" if k in TARGETS else " (untargeted)"
        e = f"{E[k]:.3f}" if k in E else "-"
        s = f"{Sm[k]:.3f} ({Ss[k]:.3f})" if k in Sm else "-"
        rows.append(f"| {k}{tag} | {target[k]:.3f} | {model[k]:.3f} | {M[k]:.3f} | {e} | {s} |")
    rows.append(f"| loss (AMR weights) | | | {loss(M, target):.4f} | | |")
    rows.append(f"| mass top 10% of B grid / in full default | | | {E['mass_top']:.4f} / {E['mass_full_default']:.4f} | | |")
    rows.append(f"| share of spread variance from top 10% of B grid | | | {E['spread_var_share_top']:.3f} | | |")
    return "\n".join(rows)


def table(M: dict, target=AMR_DATA, model=AMR_MODEL) -> str:
    """Markdown table of the moments M against the data and the AMR model."""
    rows = ["| moment | data (AMR) | AMR model | CT model |", "|---|---|---|---|"]
    for k in TARGETS + ["dc_y_mean_cond", "dc_y_sd_cond", "spread_mean", "corr_spread_y", "corr_spread_debt", "c_sd_rel"]:
        tag = "" if k in TARGETS else " (untargeted)"
        rows.append(f"| {k}{tag} | {target[k]:.3f} | {model[k]:.3f} | {M[k]:.3f} |")
    rows.append(f"| loss (AMR weights) | | | {loss(M, target):.4f} |")
    return "\n".join(rows)
