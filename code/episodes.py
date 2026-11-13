"""episodes.py: default episodes in simulations of the main diffusion solution, for the figures episode_one and
episode_typical.

Simulates 3,000 paths of 160 years from the ergodic distribution, finds the default episodes (default with no
interruption of a year or more, lasting at least a year, after at least three years without default), and keeps the
window from 5 years before to 20 years after the first default of each. Stores the series of the one episode shown
and, across all episodes, the medians, quartiles, means, and shares that the figure of the typical episode draws,
with the summary statistics it prints.

Reads:  data/ou_main.npz
Writes: data/episodes.npz
Usage:  python code/episodes.py
"""
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d

from paths import DATA

DT, KEEP, T, NP = 0.02, 5, 160.0, 3000          # step, store every KEEP steps (0.1 year), years, paths
PRE, POST = 5.0, 20.0                            # event window, years

a = np.load(os.path.join(DATA, "ou_main.npz"))
g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = a["x"].astype(float)
lam = np.log(1.04)
mu = -np.log(rho_z); sigma = sig_eps / np.sqrt(1 - rho_z ** 2) * np.sqrt(2 * mu)
B, z, d, S, g = a["B"], a["z"], a["d"], a["S"], a["g"]
coupon = delta + lam
# default threshold by endowment node: the first debt node with default
thr = np.array([B[np.argmax(d[:, j] > 1e-9)] if (d[:, j] > 1e-9).any() else B[-1] for j in range(z.size)])


def simulate():
    """Paths of debt, log endowment, default intensity, new issues, and arrears, every KEEP steps."""
    fS = RegularGridInterpolator((B, z), S, bounds_error=False, fill_value=None)
    rng = np.random.default_rng(11)
    w = (g / g.sum()).ravel()
    iB, jz = np.unravel_index(rng.choice(w.size, size=NP, p=w), g.shape)
    Bt, zt_ = B[iB].copy(), z[jz].copy()
    n = int(round(T / DT)); m = n // KEEP + 1
    out = {k: np.zeros((NP, m), dtype=np.float32) for k in ("B", "z", "d", "issue", "arr")}
    dB, dz = B[1] - B[0], z[1] - z[0]
    for t in range(n + 1):
        i = np.clip(np.rint((Bt - B[0]) / dB).astype(int), 0, B.size - 1)
        j = np.clip(np.rint((zt_ - z[0]) / dz).astype(int), 0, z.size - 1)
        dv = d[i, j]                                   # the default policy jumps at the threshold: nearest node
        Sv = fS(np.stack([Bt, zt_], 1))
        if t % KEEP == 0:
            k = t // KEEP
            out["B"][:, k] = Bt; out["z"][:, k] = zt_; out["d"][:, k] = dv
            out["arr"][:, k] = kappa * coupon * dv * Bt
            out["issue"][:, k] = Sv + delta * Bt - kappa * coupon * dv * Bt
        Bt = np.clip(Bt + Sv * DT, B[0], B[-1])
        zt_ = zt_ - mu * zt_ * DT + rng.standard_normal(NP) * np.sqrt(DT) * sigma
        zt_ = np.where(zt_ < z[0], 2 * z[0] - zt_, zt_); zt_ = np.where(zt_ > z[-1], 2 * z[-1] - zt_, zt_)
    return out


def runs(x):
    """Starts and ends of the runs of True in a boolean series."""
    e = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]


sim = simulate()
h = DT * KEEP
npre, npost = int(round(PRE / h)), int(round(POST / h))
ybar = float((np.exp(z)[None, :] * g).sum() / g.sum())
quiet = int(round(3.0 / h))
YEAR = int(round(1.0 / h))

# default spells: fill interruptions shorter than a year, then drop spells shorter than a year
ind = sim["d"] > 1e-9
for i in range(NP):
    x = ind[i].copy()
    s0, s1 = runs(~x)
    for u, v in zip(s0, s1):
        if v - u < YEAR and u > 0 and v < x.size:
            x[u:v] = True
    s0, s1 = runs(x)
    for u, v in zip(s0, s1):
        if v - u < YEAR:
            x[u:v] = False
    ind[i] = x
# episodes: a first default after at least three years without default, with the whole window inside the sample
events = []
for i in range(NP):
    x = ind[i]
    on = np.nonzero(x[1:] & ~x[:-1])[0] + 1
    for k in on:
        if k - npre < 20 or k + npost >= x.size or x[k - quiet:k].any():
            continue
        off = np.nonzero(~x[k:])[0]
        events.append((i, k, (off[0] if off.size else x.size - k) * h))
dur = np.array([e[2] for e in events])
print("episodes:", len(events), " median duration %.1f years" % np.median(dur))
win = np.arange(-npre, npost + 1)
tw = win * h


def stack(key):
    """The window of one simulated series around the first default of every episode."""
    return np.array([sim[key][i, k + win] for i, k, _ in events])


EB, EZ, ED = stack("B"), stack("z"), stack("d")
EIN = np.array([ind[i, k + win] for i, k, _ in events])
ETHR = np.interp(EZ, z, thr)
EISS, EARR = stack("issue"), stack("arr")
rise = EB[:, npre:].max(1) - EB[:, npre]
over = [(dur < y).mean() for y in (5, 10, 20)]
print("median rise of debt during the window: %.3f; share of episodes over by 5, 10, 20 years: %.2f %.2f %.2f"
      % (np.median(rise), *over))

# the episode shown: one default spell in the window, no default before or after it, few interruptions,
# duration and rise of debt close to the medians
score = np.full(len(events), np.inf)
for e, (i, k, du) in enumerate(events):
    x = EIN[e]
    ke = npre + int(round(du / h))
    if np.count_nonzero(x[1:] & ~x[:-1]) != 1 or (ED[e, :npre] > 0).any() or du > POST - 6 or du < 6:
        continue
    if (ED[e, ke + 2:] > 0).any():
        continue
    score[e] = abs(du - np.median(dur[dur >= 4])) / 5 + abs(rise[e] - np.median(rise)) / 0.05 \
        + 30 * (ED[e, npre:ke] <= 0).mean()
e = int(np.argmin(score))
i0, k0, du0 = events[e]
kend = npre + int(round(du0 / h))
xb, xd = EB[e].astype(float), ED[e].astype(float)
zs = uniform_filter1d(EZ[e].astype(float), YEAR, mode="nearest")          # one-year moving average, for display
xthr = np.interp(zs, z, thr)
print("episode shown: path %d, duration %.1f years, debt %.3f at entry, %.3f at the peak, %.3f at exit; share withheld at entry %.2f"
      % (i0, du0, xb[npre], xb.max(), xb[kend], xd[npre + 1]))

# the typical episode: quartiles of output and debt, the median threshold, the mean share withheld among the
# episodes in default (shown where more than 50 are), the share still in default, and the mean flows
output = np.percentile(100 * (np.exp(EZ) / ybar - 1), [25, 50, 75], axis=0)
debt = np.percentile(EB, [25, 50, 75], axis=0)
thr_md = np.median(ETHR, 0)
raw = ED > 1e-9
cond = np.where(raw.sum(0) > 50, np.where(raw, ED, 0).sum(0) / np.maximum(raw.sum(0), 1), np.nan)
cond[:npre] = np.nan
issues, arrears = 100 * EISS.mean(0) / ybar, 100 * EARR.mean(0) / ybar
md, om = debt[1], output[1]
print("median debt %.3f at entry, peak %.3f at %.1f years; median output at entry %.1f, trough %.1f; share withheld in default %.2f at entry, %.2f at 10 years; "
      "median duration %.1f; issues vs arrears at 1 year %.2f %.2f, at 10 years %.2f %.2f; threshold median crosses back at %.1f years"
      % (md[npre], md.max(), tw[md.argmax()], om[npre], om.min(), cond[npre + 1], cond[npre + int(10 / h)], np.median(dur),
         issues[npre + YEAR], arrears[npre + YEAR], issues[npre + 10 * YEAR], arrears[npre + 10 * YEAR],
         tw[npre + 5 + np.argmax((thr_md > md)[npre + 5:])]))

np.savez(os.path.join(DATA, "episodes.npz"), t=tw, h=h, ybar=ybar, z=z, thr=thr,
         one_B=xb, one_d=xd, one_z=zs, one_thr=xthr, one_duration=du0, one_path=i0,
         typ_output=output, typ_B=debt, typ_thr=thr_md, typ_withheld=cond, typ_in_default=EIN.mean(0),
         typ_issues=issues, typ_arrears=arrears,
         n_episodes=len(events), duration_median=np.median(dur), rise_median=np.median(rise), share_over=np.array(over))
