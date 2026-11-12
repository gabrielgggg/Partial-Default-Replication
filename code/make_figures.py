"""
make_figures.py: Figures 1 to 9, B.1 to B.4, and C.1 of the paper (Figure A.1 is drawn by pessimistic_approach.py).
Each figure is a PDF in output/, rendered through pdflatex (pgf backend) in the mathpazo font of the text.
Inputs: the solutions in data/ (see the docstring of each figure function).
Usage:  python code/make_figures.py [names...]   (names are keys of ALL at the end; default: all figures)
"""
import os
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt
from cycler import cycler

from paths import DATA, OUT as FIG
os.makedirs(FIG, exist_ok=True)

# Palette: Ohio State scarlet and grays plus a navy halfway in lightness between scarlet and the near-black,
# so the three lines separate in grayscale print and under red-green colorblindness.
# Lines differ by color and line style; markers only for computed points. BLUE, RED, BLACK, GRAY are aliases.
SCARLET, NAVY, INK, MIDGRAY = "#ba0c2f", "#1c3f6e", "#212325", "#868e92"   # Scarlet, navy, Gray Dark 80, Gray Dark 20
LIGHT_80, LIGHT_60, LIGHT_20 = "#eff1f2", "#dfe3e5", "#bfc6cb"           # OSU Gray Light 80, 60, 20 (fills)
BLUE, RED, BLACK, GRAY = NAVY, SCARLET, INK, MIDGRAY
STYLES = [dict(color=BLUE, ls="-"), dict(color=RED, ls="--"),
          dict(color=BLACK, ls="-."), dict(color=GRAY, ls=":"),
          dict(color=BLUE, ls=":")]
plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": False, "font.family": "serif",
    "pgf.preamble": r"\usepackage[sc]{mathpazo}\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.0, "lines.markersize": 3.2,
    "legend.frameon": False, "axes.prop_cycle": cycler(color=[BLUE, RED, BLACK, GRAY]),
    "figure.dpi": 100, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 6.3   # text width in inches (letterpaper, 1in margins)
IRF_T, IRF_XMAX = 21.0, 20.0   # impulse responses computed to 21 years and drawn to 20, so every path reaches the frame


def line(ax, x, y, k, label=None, n_markers=None, **kw):
    """Line k of a panel: color and line style from STYLES. The n_markers argument has no effect."""
    st = dict(STYLES[k % len(STYLES)])
    st.update(kw)
    ax.plot(x, y, label=label, **st)


def zero_line(ax):
    """Thin horizontal line at zero."""
    ax.axhline(0, color=BLACK, lw=0.5)


def with_break(x, y, tol=1e-9):
    """Insert a NaN between consecutive samples where y jumps from zero to positive, so that the plotted
    line breaks at the default threshold instead of drawing a vertical segment."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    jumps = np.where((y[:-1] <= tol) & (y[1:] > tol))[0]
    if jumps.size == 0:
        return x, y
    xs, ys = [], []
    last = 0
    for i in jumps:
        xs.append(x[last:i + 1]); ys.append(y[last:i + 1])
        xs.append([0.5 * (x[i] + x[i + 1])]); ys.append([np.nan])
        last = i + 1
    xs.append(x[last:]); ys.append(y[last:])
    return np.concatenate(xs), np.concatenate(ys)


def save(fig, name):
    """Write output/<name>.pdf and close the figure."""
    t = time.time()
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print(f"  {name}.pdf  ({time.time() - t:.0f}s)", flush=True)


# ============================================================ deterministic case
def fig_det():
    """Figures 3 (det_policies) and 4 (det_value): the deterministic model. Figure 3 plots the bond price,
    default intensity, drift of debt, and issuance against debt; Figure 4 the value and consumption."""
    f = os.path.join(DATA, "det_fine15.npz")          # 500 nodes per unit on [0, 1.5]
    if not os.path.exists(f):
        f = os.path.join(DATA, "det_fine.npz")
    if not os.path.exists(f):
        print("  (skip fig_det: no data)"); return
    a = np.load(f)
    B, q, d, S, c, l = a["B"], a["q"], a["d"], a["S"], a["c"], a["issuance"]
    Bs = float(a["Bpd"])          # the closed-form steady state B* = B_pd(0, q_rf)
    # debt at which the price crosses the floor, by linear interpolation
    qf = float(load_ou()[0]["qf"])
    i = np.where(q < qf)[0]
    Bf = (B[i[0] - 1] + (q[i[0] - 1] - qf) / (q[i[0] - 1] - q[i[0]]) * (B[i[0]] - B[i[0] - 1])) if i.size else np.nan
    print(f"  floor {qf}: q crosses it at B = {Bf:.4f}")

    def floor_line(ax):
        if np.isfinite(Bf):
            ax.axvline(Bf, color=GRAY, lw=0.8, ls="--")
    # Figure 3: four policy panels, with B* (dotted) and the floor crossing (dashed)
    fig, axes = plt.subplots(2, 2, figsize=(W, 0.62 * W))
    panels = [("bond price $q(B)$", q), ("default intensity $d(B)$", d), ("drift of debt $\\dot B = S(B)$", S),
              ("issuance $\\ell(B)$", l)]
    for ax, (t, y) in zip(axes.ravel(), panels):
        xx, yy = with_break(B, y) if t.startswith("default") else (B, y)
        ax.plot(xx, yy, color=BLACK, ls="-")
        ax.axvline(Bs, color=GRAY, lw=0.6, ls=":")
        floor_line(ax)
        ax.set_title(t)
        if t.startswith("drift") or t.startswith("issuance"):
            zero_line(ax)
    for ax in axes[1]:
        ax.set_xlabel("debt $B$")
    axes[0, 0].annotate("$B^*$", xy=(Bs, 1.0), xytext=(Bs + 0.04, 0.93), fontsize=8)
    if np.isfinite(Bf):
        axes[0, 0].annotate(r"$q=\underline{q}$", xy=(Bf, 1.0), xytext=(Bf + 0.03, 0.93), fontsize=8)
    fig.tight_layout(w_pad=1.5)
    save(fig, "det_policies")

    # Figure 4: value function and consumption
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.36 * W))
    axes[0].plot(B, a["V"], color=BLACK); axes[0].set_title("value $V(B)$")
    axes[1].plot(B, c, color=BLACK); axes[1].set_title("consumption $c(B)$")
    for ax in axes:
        ax.axvline(Bs, color=GRAY, lw=0.6, ls=":"); floor_line(ax); ax.set_xlabel("debt $B$")
    fig.tight_layout(w_pad=1.5)
    save(fig, "det_value")


# ============================================================ cube-root law at B*
def fig_cuberoot():
    """Figure B.2 (det_cuberoot): local laws at the steady state B* on log-log axes. The drift and the price
    discount above B* against the cube-root expansions, and the drift below B* against the square-root law,
    with the computed points of two grids (500 and 200 nodes per unit)."""
    f5 = os.path.join(DATA, "det_fine.npz"); f2 = os.path.join(DATA, "det_floor.npz")
    if not (os.path.exists(f5) and os.path.exists(f2)):
        print("  (skip fig_cuberoot: no data)"); return
    from cuberoot_consts import second_order
    a5 = np.load(f5)
    Bpd, beta, alpha, sq = float(a5["Bpd"]), float(a5["beta"]), float(a5["alpha"]), float(a5["sq"])
    Sig1, Q1 = second_order(a5, load_ou()[0]["x"])
    sols = [("500 nodes per unit", a5["B"], a5["S"], a5["q"])]
    a2 = np.load(f2)
    sols.append(("200 nodes per unit", a2["B"], a2["S"], a2["q"]))
    # analytic curves: two-term and leading-term expansions, and the square-root law
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.36 * W))
    xx = np.logspace(-3.3, -0.6, 60); ss = xx ** (1 / 3)
    axes[0].loglog(xx, beta * ss + Sig1 * ss ** 2, color=BLACK, ls="-", label="two terms")
    axes[0].loglog(xx, beta * ss, color=GRAY, ls="--", label=r"$\beta x^{1/3}$")
    axes[1].loglog(xx, alpha * ss ** 2 + Q1 * ss ** 3, color=BLACK, ls="-", label="two terms")
    axes[1].loglog(xx, alpha * ss ** 2, color=GRAY, ls="--", label=r"$\alpha x^{2/3}$")
    xl = np.logspace(-3.3, -1.1, 40)
    axes[2].loglog(xl, sq * xl ** 0.5, color=BLACK, ls="-", label=r"$\sqrt{2(\rho-\lambda)\varphi(\bar c)}\,x^{1/2}$")
    # computed points near B*: above it where the drift is negative, below it where it is positive
    for k, (lab, B, S, q) in enumerate(sols):
        x = B - Bpd
        selr = (x > 0) & (x < 0.25) & (S < 0)
        sell = (x < 0) & (x > -0.08) & (S > 0)
        st = dict(color=[BLUE, RED][k], marker=["o", "s"][k], ls="none", markersize=2.6, markerfacecolor="none")
        axes[0].loglog(x[selr], -S[selr], label=lab, **st)
        axes[1].loglog(x[selr], 1 - q[selr], label=lab, **st)
        axes[2].loglog(-x[sell], S[sell], label=lab, **st)
    axes[0].set_title(r"$-S(B^*+x)$"); axes[1].set_title(r"$1-q(B^*+x)$"); axes[2].set_title(r"$S(B^*-x)$")
    for ax in axes:
        ax.set_xlabel("$x$")
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h0[:2] + h2[:1] + h0[2:], ["two terms (left, center)", "leading term", "square-root law (right)"] + l0[2:],
               loc="lower center", ncol=5, fontsize=7, handlelength=1.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    save(fig, "det_cuberoot")


# ============================================================ two-state jump case
def fig_jump():
    """Figure B.1 (jump_policies): the chain model with three states. Bond price, default intensity, and drift of debt
    by state, and the ergodic density of debt by state with atoms drawn as labeled stems."""
    f = os.path.join(DATA, "jump_floor.npz")
    if not os.path.exists(f):
        print("  (skip fig_jump: no data)"); return
    a = np.load(f)
    B, z, q, d, S, g = a["B"], a["z"], a["q"], a["d"], a["S"], a["g"]
    nz = z.size
    labs = [f"$z={zz:+.2f}$" for zz in z]
    fig, axes = plt.subplots(2, 2, figsize=(W, 0.62 * W))
    for k in range(nz):
        line(axes[0, 0], B, q[:, k], k, labs[k])
        line(axes[0, 1], *with_break(B, d[:, k]), k, labs[k])
        line(axes[1, 0], B, S[:, k], k, labs[k])
    zero_line(axes[1, 0])
    # ergodic density by state: nodes with mass above 0.05 are atoms, the rest a density
    w = g / g.sum()
    h = B[1] - B[0]
    ax = axes[1, 1]
    atoms = []
    for k in range(nz):
        gk = w[:, k] / h
        big = w[:, k] > 0.05                      # atoms: drawn as marked stems, not as part of the density
        gk = np.where(big, np.nan, gk)
        ax.plot(B, gk, color=STYLES[k]["color"], ls=STYLES[k]["ls"], label=labs[k])
        for i in np.where(big)[0]:
            atoms.append((B[i], w[i, k], k))
    ymax = np.nanmax([np.nanmax(np.where(w[:, k] > 0.05, np.nan, w[:, k] / h)) for k in range(nz)])
    seen = {}
    for Bi, mass, k in atoms:
        n_here = seen.get(round(Bi, 6), 0); seen[round(Bi, 6)] = n_here + 1   # stack labels of atoms at the same node
        ax.vlines(Bi, 0, 1.15 * ymax, color=STYLES[k]["color"], ls=STYLES[k]["ls"], lw=1.0)
        ax.annotate(f"atom {mass:.2f}", xy=(Bi, 1.15 * ymax), xytext=(0, 2 + 9 * n_here), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=STYLES[k]["color"])
    ax.set_ylim(0, 1.45 * ymax)
    lo, hi = B[np.where(w.sum(1) > 1e-4)[0][[0, -1]]]
    ax.set_xlim(max(0.0, lo - 0.05), min(1.0, hi + 0.05))
    axes[0, 0].set_title("bond price $q(B,z)$"); axes[0, 1].set_title("default intensity $d(B,z)$")
    axes[1, 0].set_title("drift of debt $S(B,z)$"); axes[1, 1].set_title("ergodic density of debt, by state")
    for ax in axes[1]:
        ax.set_xlabel("debt $B$")
    axes[0, 0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "jump_policies")


# ============================================================ OU diffusion case (calibrated)
def load_ou():
    """Main diffusion solution and its parameters (structural ones from x, continuous-time OU from rho_z, sig_eps)."""
    f = os.path.join(DATA, "ou_main.npz")      # the main diffusion solution; else the calibration-grid one
    if not os.path.exists(f):
        f = os.path.join(DATA, "ou_best.npz")
    a = np.load(f)
    x = a["x"].astype(float)
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = x
    r = lam = np.log(1.04)
    mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
    par = dict(rho=rho, kappa=kappa, delta=delta, r=r, lam=lam, mu=mu, sigma=sigma, sd=sd, g0=g0, g1=g1, g2=g2, zt=zt)
    return a, par


def load_density():
    """Ergodic density for the figures: grid and density of the main solution (load_ou)."""
    a = load_ou()[0]
    return a["B"], a["z"], a["g"], int(a["z"].size)


def fig_ou_slices():
    """Figure 6 (ou_slices): the calibrated diffusion model. Value, bond price, consumption, default intensity,
    and drift of debt against debt at four endowment levels, and the marginal ergodic density of debt."""
    a, p = load_ou()
    B, z, V, q, c, d, S, g = a["B"], a["z"], a["V"], a["q"], a["c"], a["d"], a["S"], a["g"]
    nz = z.size
    js = [0, nz // 4, nz // 2, nz - 1]
    B, V, q, c, d, S = (v[:-1] for v in (B, V, q, c, d, S))   # drop the top node, where the state constraint shapes the policies
    labs = [f"$z={z[j]:+.2f}$" for j in js]
    fig, axes = plt.subplots(2, 3, figsize=(W, 0.6 * W))
    for k, j in enumerate(js):
        line(axes[0, 0], B, V[:, j], k, labs[k])
        line(axes[0, 1], B, q[:, j], k, labs[k])
        line(axes[0, 2], B, c[:, j], k, labs[k])
        line(axes[1, 0], *with_break(B, d[:, j]), k, labs[k])
        line(axes[1, 1], B, S[:, j], k, labs[k])
    zero_line(axes[1, 1])
    # marginal density of debt, with the mass at B = 0 reported as a number
    Bg, zg, gg, nzg = load_density()
    w = gg / gg.sum(); gB = w.sum(1); h = Bg[1] - Bg[0]
    from scipy.ndimage import gaussian_filter1d
    gB_s = gaussian_filter1d(gB[1:] / h, sigma=3, mode="nearest")   # display only: the grid marginal has spikes at the thresholds of the z nodes
    axes[1, 2].plot(Bg[1:], gB_s, color=BLACK)
    axes[1, 2].annotate(f"mass at $B=0$: {gB[0]:.2f}", xy=(0.3, 0.9), xycoords="axes fraction", fontsize=7.5)
    titles = ["value $V(B,z)$", "bond price $q(B,z)$", "consumption $c(B,z)$",
              "default intensity $d(B,z)$", "drift of debt $S(B,z)$", "ergodic density of debt"]
    for ax, t in zip(axes.ravel(), titles):
        ax.set_title(t)
    for ax in axes[1]:
        ax.set_xlabel("debt $B$")
    handles, labels_ = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, handlelength=2.4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=1.2, h_pad=1.0, rect=(0, 0.04, 1, 1))
    save(fig, "ou_slices")


def fig_ou_contours():
    """Figure 7 (ou_contours): contours in the (B, z) plane of the bond price, the default intensity, and the
    ergodic density, each with the zero-drift locus (red, dashed)."""
    a, p = load_ou()
    B, z, q, d, S, g = a["B"], a["z"], a["q"], a["d"], a["S"], a["g"]
    from scipy.ndimage import gaussian_filter
    BB, ZZ = np.meshgrid(B, z, indexing="ij")
    h = B[1] - B[0]; dz = z[1] - z[0]
    Bg, zg, gg, nzg = load_density()
    dzg = zg[1] - zg[0]
    dens = gg / gg.sum() / (h * dzg)
    dens_s = gaussian_filter(dens, sigma=(1.5, 0.8 * nzg / 25))      # light display smoothing
    BBg, ZZg = np.meshgrid(Bg, zg, indexing="ij")
    d_s = gaussian_filter(d, sigma=(1.5, 0.6))            # removes the grid staircase of the default policy
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.36 * W))
    cs = axes[0].contour(BB, ZZ, q, levels=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], colors=[BLACK], linewidths=0.7)
    axes[0].clabel(cs, fmt="%.2f", fontsize=6.5, inline=True)
    cs = axes[1].contour(BB, ZZ, d_s, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors=[BLACK], linewidths=0.7)
    axes[1].clabel(cs, levels=[0.5, 0.7, 0.9], fmt="%.1f", fontsize=6.5, inline=True)   # 0.1 and 0.3 sit on the jump
    lev = np.quantile(dens_s[1:][dens_s[1:] > 1e-3], [0.5, 0.7, 0.85, 0.95, 0.99])
    axes[2].contour(BBg[1:], ZZg[1:], dens_s[1:], levels=lev, colors=[BLACK], linewidths=0.7)
    for ax in axes:
        ax.contour(BB, ZZ, S, levels=[0.0], colors=[RED], linewidths=1.2, linestyles="--")
        ax.set_xlabel("debt $B$")
    axes[0].set_ylabel("log endowment $z$")
    axes[0].set_title("bond price $q$"); axes[1].set_title("default intensity $d$"); axes[2].set_title("ergodic density")
    axes[2].set_xlim(0, 0.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ou_contours")


def fig_saving_density():
    """Figure C.1 (ou_saving_density): ergodic density contours and the zero-drift locus without saving
    (ou_main.npz) and with saving (ou_saving.npz, from run_saving.py). Panel widths are proportional to the
    range of debt each spans."""
    f = os.path.join(DATA, "ou_saving.npz")
    if not os.path.exists(f):
        print("  (skip fig_saving_density: no data)"); return
    from scipy.ndimage import gaussian_filter
    a = load_ou()[0]; s = np.load(f)
    ranges = [(0.0, 0.8), (-0.6, 0.8)]
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.36 * W), sharey=True,
                             gridspec_kw=dict(width_ratios=[r[1] - r[0] for r in ranges]))
    for ax, sol, skip, (lo, hi), title in zip(axes, (a, s), (1, 0), ranges, ("borrowing only", "with saving")):
        B, z, g, S = sol["B"], sol["z"], sol["g"], sol["S"]
        BB, ZZ = np.meshgrid(B, z, indexing="ij")
        dens = gaussian_filter(g / g.sum() / ((B[1] - B[0]) * (z[1] - z[0])), sigma=(1.5, 0.8 * z.size / 25))
        sl = slice(skip, None)
        lev = np.quantile(dens[sl][dens[sl] > 1e-3], [0.5, 0.7, 0.85, 0.95, 0.99])
        ax.contour(BB[sl], ZZ[sl], dens[sl], levels=lev, colors=[BLACK], linewidths=0.7)
        ax.contour(BB, ZZ, S, levels=[0.0], colors=[RED], linewidths=1.2, linestyles="--")
        ax.axvline(0.0, color=GRAY, lw=0.6, ls=":")
        ax.set_xlim(lo, hi); ax.set_xlabel("debt $B$"); ax.set_title(title)
    axes[0].set_ylabel("log endowment $z$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "ou_saving_density")


# ============================================================ impulse responses (OU)
def fig_irf():
    """Figure 8 (ou_irf): responses to a fall in the endowment, from the ergodic distribution. Output,
    consumption, default intensity, debt, spread, and the flows of new issues and arrears (shock minus control)."""
    a, p = load_ou()
    # expected paths: read data/irf.npz if it covers the horizon, else compute and store them
    f = os.path.join(DATA, "irf.npz")
    if os.path.exists(f) and np.load(f)["t"][-1] >= IRF_T - 1e-9 and "shock_yn" in np.load(f):
        d = np.load(f); tt = d["t"]; out = {s: {k: d[f"{s}_{k}"] for k in ["y", "yn", "c", "d", "B", "spread", "issue", "defc"]} for s in ("shock", "control")}
    else:
        t0 = time.time()
        from generator import expected_paths      # exact expectations under the forward equation (no simulation)
        tt, out = expected_paths(a, T=IRF_T)
        np.savez(f, t=tt, **{f"{s}_{k}": v for s in out for k, v in out[s].items()})
        print(f"  IRF by the forward equation {time.time() - t0:.0f}s")
    sh, co = out["shock"], out["control"]
    ybar = co["yn"].mean()                      # mean output, net of default costs
    # five single-line panels and one panel with the two flows
    fig, axes = plt.subplots(2, 3, figsize=(W, 0.52 * W))
    panels = [("output, \\% of mean", 100 * (sh["yn"] - co["yn"]) / ybar),
              ("consumption, \\% of mean output", 100 * (sh["c"] - co["c"]) / ybar),
              ("default intensity $d$", sh["d"] - co["d"]),
              ("debt $B$, \\% of mean output", 100 * (sh["B"] - co["B"]) / ybar),
              ("spread, basis points", 1e4 * (sh["spread"] - co["spread"]))]
    for ax, (t, y) in zip(axes.ravel(), panels):
        ax.plot(tt, y, color=BLACK); zero_line(ax); ax.set_title(t)
    ax = axes[1, 2]
    line(ax, tt, 100 * (sh["issue"] - co["issue"]) / ybar, 0, "new issues (face value)", n_markers=6)
    line(ax, tt, 100 * p["kappa"] * (sh["defc"] - co["defc"]) / ybar, 1, "arrears", n_markers=6)
    zero_line(ax); ax.set_title("flows, \\% of mean output"); ax.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    for ax in axes[1]:
        ax.set_xlabel("years")
    for ax in axes.ravel():
        ax.set_xlim(0, IRF_XMAX); ax.set_xticks([0, 5, 10, 15, 20])   # paths run past the right edge
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "ou_irf")


def fig_irf_states():
    """Figure 9 (ou_irf_states): state-contingent impulse responses: the shock of fig_irf applied at the mean endowment with debt
    at its ergodic mean, 25 percent below it, and 25 percent above it. Expected paths under the forward
    equation from a point mass at each state, minus the paths from the same state without the shock."""
    a, p = load_ou()
    g = a["g"] / a["g"].sum()
    EB = float((g.sum(1) * a["B"]).sum())
    mults = [1.0, 0.75, 1.25]          # style 0 (solid) for the ergodic mean, 1 (dashed) below, 2 (dash-dot) above
    keys = ["y", "yn", "c", "d", "B", "spread", "issue", "defc"]
    # expected paths: read data/irf_states.npz if it matches the horizon and E[B], else compute and store them
    f = os.path.join(DATA, "irf_states.npz")
    ok = os.path.exists(f)
    if ok:
        d = np.load(f); ok = d["t"][-1] >= IRF_T - 1e-9 and abs(float(d["EB"]) - EB) < 1e-9 and "0_shock_yn" in d
    if ok:
        tt = d["t"]; ybar = float(d["ybar"])
        res = [{s: {k: d[f"{i}_{s}_{k}"] for k in keys} for s in ("shock", "control")} for i in range(len(mults))]
    else:
        t0 = time.time()
        from generator import expected_paths_from_states
        tt, res, ybar = expected_paths_from_states(a, [(m * EB, 0.0) for m in mults], T=IRF_T)
        np.savez(f, t=tt, EB=EB, ybar=ybar, **{f"{i}_{s}_{k}": v for i, r in enumerate(res) for s in r for k, v in r[s].items()})
        print(f"  state-contingent IRFs by the forward equation {time.time() - t0:.0f}s  (E[B] = {EB:.3f})")
    labels = ["$B_0=\\mathbb{E}[B]$", "$B_0=0.75\\,\\mathbb{E}[B]$", "$B_0=1.25\\,\\mathbb{E}[B]$"]

    def dev(r, k, scale=1.0):
        return scale * (r["shock"][k] - r["control"][k])

    # output (the same for all states), then four panels with one line per initial debt
    fig, axes = plt.subplots(2, 3, figsize=(W, 0.52 * W))
    ax = axes[0, 0]
    ax.plot(tt, 100 * dev(res[0], "yn") / ybar, color=BLACK); zero_line(ax); ax.set_title("output, \\% of mean")
    panels = [(axes[0, 1], "consumption, \\% of mean output", lambda r: 100 * dev(r, "c") / ybar),
              (axes[0, 2], "default intensity $d$", lambda r: dev(r, "d")),
              (axes[1, 0], "debt $B$, \\% of mean output", lambda r: 100 * dev(r, "B") / ybar),
              (axes[1, 1], "spread, basis points", lambda r: 1e4 * dev(r, "spread"))]
    for ax, t, fn in panels:
        for k, r in enumerate(res):
            line(ax, tt, fn(r), k, labels[k], n_markers=6)
        zero_line(ax); ax.set_title(t)
    # last panel: arrears as a share of new debt (positive new issues plus arrears) on the shocked path
    ax = axes[1, 2]
    for k, r in enumerate(res):
        arrears = p["kappa"] * r["shock"]["defc"]
        gross = np.maximum(r["shock"]["issue"], 0.0) + arrears
        share = np.where(gross > 1e-12, 100 * arrears / np.maximum(gross, 1e-12), np.nan)
        line(ax, tt, share, k, labels[k], n_markers=6)
    ax.set_title("arrears, \\% of new debt"); ax.set_ylim(bottom=0)
    axes[0, 2].legend(loc="upper right", fontsize=6.5, handlelength=1.8)
    for ax in axes[1]:
        ax.set_xlabel("years")
    for ax in axes.ravel():
        ax.set_xlim(0, IRF_XMAX); ax.set_xticks([0, 5, 10, 15, 20])
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "ou_irf_states")


def fig_multiplicity():
    """Figure 5 (det_multiplicity): two deterministic equilibria at an illustrative parameterization (small
    first default) and an impatience wedge of about 200 bp. Bond price, drift, and default intensity of the
    no-default steady state and of a self-fulfilling partial-default steady state found by the branching
    march (data/wedge/rho0.060_illus.npz, from wedge_search.py 0.06 200 1.7 illus)."""
    f = os.path.join(DATA, "wedge", "rho0.060_illus.npz")
    if not os.path.exists(f):
        print("  (skip fig_multiplicity: no data)"); return
    a = np.load(f)
    B = a["B"]
    # select by the solution's steady state, not by index: the no-default one (q* = 1, d* = 0) and the
    # partial-default one with the deepest default (the rule text_numbers.py uses for the macros)
    metas = {int(k[3:-5]): a[k] for k in a.files if k.endswith("_meta")}
    j_nd = next(j for j, v in sorted(metas.items()) if v[1] > 0.999 and v[2] < 1e-6)
    j_pd = max((j for j, v in metas.items() if v[2] > 0.15), key=lambda j: metas[j][2])
    sols = [(j_nd, "no-default steady state"), (j_pd, "partial-default steady state")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.36 * W))
    for k, (j, lab) in enumerate(sols):
        q, S, d = a[f"sol{j}_q"], a[f"sol{j}_S"], a[f"sol{j}_d"]
        Bs = float(a[f"sol{j}_meta"][0])
        sel = B <= 1.4
        line(axes[0], B[sel], q[sel], k, lab, n_markers=7)
        line(axes[1], B[sel], S[sel], k, lab, n_markers=7)
        line(axes[2], *with_break(B[sel], d[sel]), k, lab, n_markers=7)
        for ax in axes:
            ax.axvline(Bs, color=STYLES[k]["color"], lw=0.6, ls=":")
    zero_line(axes[1])
    axes[0].set_title("bond price $q(B)$"); axes[1].set_title("drift of debt $S(B)$"); axes[2].set_title("default intensity $d(B)$")
    for ax in axes:
        ax.set_xlabel("debt $B$")
    axes[1].legend(loc="upper right", fontsize=7, handlelength=1.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "det_multiplicity")


# ============================================================ introduction: the anatomy of a crisis in the calibrated model
def fig_anatomy():
    """Figure 1 (anatomy): three panels from the calibrated diffusion model. The no-default region and the
    zero-drift locus in the (B, z) plane, and the responses of debt and of the default intensity to a fall in
    the endowment (data/irf.npz, written by fig_irf)."""
    a, p = load_ou()
    B, z, d, S = a["B"], a["z"], a["d"], a["S"]
    # no-default region: debt below the first default node, endowment by endowment
    thr = np.array([B[np.argmax(d[:, j] > 1e-9)] if (d[:, j] > 1e-9).any() else np.nan for j in range(z.size)])
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.34 * W))
    ax = axes[0]
    ax.fill_betweenx(z, 0, thr, color=LIGHT_60, lw=0)
    ax.plot(thr, z, color=BLACK, lw=1.0)
    BB, ZZ = np.meshgrid(B, z, indexing="ij")
    ax.contour(BB, ZZ, S, levels=[0.0], colors=[RED], linewidths=1.2, linestyles="--")
    ax.text(0.06, 0.15, "no default", fontsize=7.5)
    ax.text(0.70, 0.0, "partial\ndefault", fontsize=7.5)
    ax.text(0.66, -0.25, "stationary\ndebt", fontsize=7.5, color=RED)
    ax.set_xlim(0, 1); ax.set_ylim(z[0], z[-1])
    ax.set_xlabel("debt $B$"); ax.set_ylabel("log endowment $z$")
    ax.set_title("default regions and stationary debt")
    # impulse responses of debt and default intensity
    ir = np.load(os.path.join(DATA, "irf.npz"))
    tt = ir["t"]; yb = ir["control_yn"].mean()
    axes[1].plot(tt, 100 * (ir["shock_B"] - ir["control_B"]) / yb, color=BLACK)
    axes[1].set_title("debt, \\% of mean output")
    axes[2].plot(tt, ir["shock_d"] - ir["control_d"], color=BLACK)
    axes[2].set_title("default intensity $d$")
    for ax in axes[1:]:
        zero_line(ax); ax.set_xlabel("years after the shock")
        ax.set_xlim(0, IRF_XMAX); ax.set_xticks([0, 5, 10, 15, 20])
    fig.tight_layout(w_pad=1.2)
    save(fig, "anatomy")


# ============================================================ Section 3.1: the default decision at the mean endowment
def fig_regions():
    """Figure 2 (regions): the default decision at z = 0 and the calibrated parameters, from the closed forms of Proposition 2:
    the gain from defaulting R(d) - R(0) at three debt levels at the risk-free price, and the regions of
    no, partial and full default in the (B, q) plane."""
    a, p = load_ou()
    g0, g1, g2, zt, kap, cs = p["g0"], p["g1"], p["g2"], p["zt"], p["kappa"], p["delta"] + p["lam"]
    F = 1 - g2 * (1 - np.exp(zt))                      # z = 0: output 1, above y*
    dfirst = ((1 - F) / ((g1 - 1) * g0 * F)) ** (1 / g1)
    G = g0 * g1 * F
    K = lambda q: (1 - kap * q) * cs
    Bpd = lambda q: (G * dfirst ** (g1 - 1)) / K(q)    # Proposition 2 when d_first < 1, written through d_first
    Bfd = lambda q: G / K(q)
    Bs = Bpd(1.0)
    # left: gain from defaulting against d at 0.7 B*, B*, and 1.3 B*
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.38 * W))
    ax = axes[0]
    dd = np.linspace(1e-4, 1, 600)
    for k, (fac, lab) in enumerate([(0.7, "$B=0.7\\,B^*$"), (1.0, "$B=B^*$"), (1.3, "$B=1.3\\,B^*$")]):
        b = fac * Bs
        gain = K(1.0) * b * dd - g0 * F * dd ** g1 - (1 - F)
        line(ax, dd, gain, k, lab, n_markers=6)
    ax.axhline(0, color=BLACK, lw=0.5)
    ax.plot([0], [0], "o", color=BLACK, ms=3.5, mfc="white", clip_on=False)   # d = 0: no gain, no fixed cost
    ax.axvline(dfirst, color=GRAY, lw=0.6, ls=":")
    ax.annotate("$d_{\\mathrm{first}}$", xy=(dfirst, -0.045), xytext=(dfirst + 0.03, -0.045), fontsize=7.5)
    ax.set_xlabel("default intensity $d$"); ax.set_title("gain from defaulting, $\\mathcal{R}(d)-\\mathcal{R}(0)$")
    ax.set_ylim(-0.06, 0.06)
    ax.legend(loc="upper left", fontsize=7, handlelength=2.0)
    # right: no, partial, and full default regions bounded by B_pd(0, q) and B_fd(0, q)
    ax = axes[1]
    qq = np.linspace(0, 1, 400)
    bp, bf = Bpd(qq), Bfd(qq)
    Bmax = 2.0
    ax.fill_betweenx(qq, 0, bp, color=LIGHT_80, lw=0)
    ax.fill_betweenx(qq, bf, Bmax, color=LIGHT_20, lw=0)
    ax.plot(bp, qq, color=BLACK, lw=1.0, label="$B_{\\mathrm{pd}}(0,q)$")
    ax.plot(bf, qq, color=BLACK, lw=1.0, ls="--", label="$B_{\\mathrm{fd}}(0,q)$")
    ax.plot([Bs], [1.0], "o", color=BLACK, ms=3.5, clip_on=False)
    ax.annotate("$B^*$", xy=(Bs, 1.0), xytext=(Bs + 0.04, 0.93), fontsize=7.5)
    ax.text(0.215, 0.865, "no", fontsize=7.5, ha="center")          # two centered lines (pgf ignores multialignment)
    ax.text(0.215, 0.80, "default", fontsize=7.5, ha="center")
    ax.text(0.62, 0.45, "partial default", fontsize=7.5)
    ax.text(1.6, 0.30, "full\ndefault", fontsize=7.5)
    ax.set_xlim(0, Bmax); ax.set_ylim(0, 1)
    ax.set_xlabel("debt $B$"); ax.set_ylabel("bond price $q$"); ax.set_title("default regions")
    fig.tight_layout(w_pad=1.6)
    save(fig, "regions")


# ============================================================ the debt limits of the calibrated diffusion model
def bpd_closed(z, q, p, Fmin=0.10):
    """No-default threshold B_pd(z, q) of Proposition 2, level form of the fixed cost."""
    y = np.exp(z)
    ystar = np.exp(p["zt"])
    F = 1.0 - p["g2"] * (y - ystar) if y >= ystar else 1.0
    F = max(F, Fmin)
    if F >= 1.0:
        return 0.0
    g0, g1 = p["g0"], p["g1"]
    K = (1 - p["kappa"] * q) * (p["delta"] + p["lam"])
    dfirst = ((1 - F) / ((g1 - 1) * g0 * F)) ** (1 / g1)
    if dfirst >= 1:
        return (1 - (1 - g0) * F) * y / K
    G = g0 * g1 * F * y
    return (g1 / (g1 - 1) * (1 - F) * y * G ** (1 / (g1 - 1)) / K ** (g1 / (g1 - 1))) ** ((g1 - 1) / g1)


def fig_debt_limits():
    """Not in the paper. The four debt limits of the calibrated diffusion model, by endowment, over the ergodic density:
    the risk-free capacity B_pd(z, q_rf) (closed form, preference-free), the equilibrium no-default threshold
    (first node with d > 0), the stationary debt (zero-drift locus), and the issuance limit (price at the floor)."""
    a, p = load_ou()
    B, z, q, d, S, g = a["B"], a["z"], a["q"], a["d"], a["S"], a["g"]
    qf = float(a["qf"])
    # each limit as a function of z
    cap = np.array([bpd_closed(zz, 1.0, p) for zz in z])
    thr = np.array([B[np.argmax(d[:, j] > 1e-9)] if (d[:, j] > 1e-9).any() else np.nan for j in range(z.size)])
    tgt = []
    for j in range(z.size):
        k = np.nonzero((S[:-1, j] > 0) & (S[1:, j] <= 0))[0]
        tgt.append(B[k[0] + 1] if k.size else (0.0 if S[0, j] <= 0 else np.nan))
    tgt = np.array(tgt)
    flo = np.array([B[np.argmax(q[:, j] < qf)] if (q[:, j] < qf).any() else np.nan for j in range(z.size)])
    # density contours, then the four limits
    from scipy.ndimage import gaussian_filter
    fig, ax = plt.subplots(figsize=(0.66 * W, 0.52 * W))
    BB, ZZ = np.meshgrid(B, z, indexing="ij")
    dens = np.where(BB > 0, g, 0.0)                   # leave out the atom at zero debt
    dens = gaussian_filter(dens, sigma=(2.0, 1.0))    # display only: the grid density has spikes at the thresholds
    dens = dens / max(dens.max(), 1e-300)
    ax.contour(ZZ, BB, dens, levels=[0.1, 0.3, 0.6, 0.9], colors=[GRAY], linewidths=0.6)
    ax.plot(z, cap, color=BLACK, lw=1.4, ls="-", label=r"risk-free capacity $B_{\mathrm{pd}}(z,q^{\mathrm{rf}})$")
    ax.plot(z, thr, color=BLUE, lw=1.2, ls="--", label=r"no-default threshold $B_{\mathrm{pd}}(z,q(B,z))$")
    ax.plot(z, tgt, color=RED, lw=1.2, ls="-.", label="stationary debt (zero drift)")
    ax.plot(z, flo, color=GRAY, lw=1.3, ls=":", label=r"issuance limit ($q=\underline{q}$)")
    ax.set_xlim(z[0], z[-1]); ax.set_ylim(0, 1.0)
    ax.set_xlabel("log endowment $z$"); ax.set_ylabel("debt $B$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7, handlelength=2.6, frameon=False)
    fig.tight_layout()
    save(fig, "debt_limits")
    # printed: the limits at five endowments and the mean debt relative to the risk-free capacity
    for j in (0, z.size // 4, z.size // 2, 3 * z.size // 4, z.size - 1):
        print(f"  z={z[j]:+.3f}: capacity {cap[j]:.3f}  threshold {thr[j]:.3f}  target {tgt[j]:.3f}  floor {flo[j]:.3f}")
    gm = g / g.sum()
    ok = cap[None, :] > 0
    util = np.where(ok, BB / np.where(ok, np.repeat(cap[None, :], B.size, 0), 1.0), np.nan)
    print(f"  ergodic mean of B / risk-free capacity (states with a fixed cost): {np.nansum(gm * util) / gm[np.isfinite(util)].sum():.3f}")


# ============================================================ self-fulfilling partial default: curvature and uncertainty
def fig_sf_region():
    """Figure B.4 (sf_region): where self-fulfilling partial-default equilibria exist, against the impatience wedge. Left: CRRA curvature in
    the deterministic model (smallest wedge at which a partial-default steady state is reached from zero debt).
    Right: the s.d. of the log endowment in the risk-neutral diffusion model (block construction with a steady debt
    that does not depend on the endowment; lower and upper boundaries). Data: data/sfregion.npz
    (collect_sf.py)."""
    f = os.path.join(DATA, "sfregion.npz")
    if not os.path.exists(f):
        print("  (skip fig_sf_region: no data)"); return
    a = np.load(f)
    YMAX, FILL = 300.0, LIGHT_60
    # left panel: threshold wedge by curvature nu, starting from the risk-neutral value at nu = 0
    sf = float(a["s_first"])
    nu = np.concatenate([[0.0], a["nu"]]); thr = np.concatenate([[sf], 0.5 * (a["nu_lo"] + a["nu_hi"])])
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.36 * W))
    ax = axes[0]
    YL = 180.0                                   # the left panel has its own, narrower range
    xx = np.linspace(0.0, 2.0, 400); yy = np.interp(xx, nu, thr)
    ax.fill_between(xx, yy, YL, color=FILL, lw=0)
    ax.plot(xx, yy, color=BLACK, lw=1.1)
    ax.set_xlim(0, 2.0); ax.set_ylim(0, YL)
    ax.set_xlabel(r"relative risk aversion $\nu$"); ax.set_ylabel(r"impatience wedge $\rho-r$ (bp)")
    ax.set_title("curvature, no shocks")
    ax.text(0.75, 150, "pessimism is self-fulfilling", fontsize=8, ha="center")
    ax.text(1.3, 45, "too patient to\nborrow into default", fontsize=7.5, ha="center")
    # right panel: lower and upper wedge boundaries by shock s.d., closed at the s.d. where they meet
    ax = axes[1]
    sd, lo, hi = a["sd"], a["w_lo"], a["w_hi"]
    ok = np.isfinite(lo)
    tip_sd = 0.5 * float(a["sd_close"][0] + a["sd_close"][1])
    k = np.nonzero(ok)[0][-1]
    tip_w = 0.5 * (lo[k] + hi[k])
    xs = np.concatenate([100 * sd[ok], [100 * tip_sd]])
    ylo = np.concatenate([lo[ok], [tip_w]]); yhi = np.concatenate([np.minimum(hi[ok], YMAX + 50), [tip_w]])
    ax.fill_between(xs, ylo, np.minimum(yhi, YMAX), color=FILL, lw=0)
    ax.plot(xs, ylo, color=BLACK, lw=1.1)
    fin = np.concatenate([np.isfinite(hi[ok]), [True]])
    ax.plot(xs[fin], yhi[fin], color=BLACK, lw=1.1, ls="--")
    ax.set_xlim(0, 8.5); ax.set_ylim(0, YMAX); ax.set_xlabel(r"s.d.\ of log endowment (\%)")
    ax.set_title("shocks, risk-neutral")
    ax.text(1.2, 175, "pessimism is\nself-fulfilling", fontsize=8)
    ax.text(3.6, 18, "too patient to borrow into default", fontsize=7.5, ha="center")
    ax.text(5.75, 245, "too impatient to\nstay at the\nsteady state", fontsize=7.5, va="center")
    fig.tight_layout(w_pad=1.6)
    save(fig, "sf_region")



# ============================================================ neural-network solution against finite differences
def fig_pinn():
    """Figure B.3 (ou_pinn): value, price, consumption, default intensity, drift, and the ergodic debt density of
    the finite-difference solution (solid / dash-dotted) and the neural-network solution (dashed / dotted) at two
    endowment levels (data/pinn_main.npz, from pinn_eval.py)."""
    f = os.path.join(DATA, "pinn_main.npz")
    if not os.path.exists(f):
        print("  (skip fig_pinn: no data)"); return
    a, p = load_ou()
    n = np.load(f)
    B, z = a["B"], a["z"]
    nz = z.size
    js = [nz // 2, nz // 4]
    B_, = (B[:-1],)
    fig, axes = plt.subplots(2, 3, figsize=(W, 0.6 * W))
    panels = [("value $V(B,z)$", "V"), ("bond price $q(B,z)$", "q"), ("consumption $c(B,z)$", "c"),
              ("default intensity $d(B,z)$", "d"), ("drift of debt $S(B,z)$", "S")]
    # policy panels: for each endowment, the FD line then the network line
    k = 0
    for j in js:
        for src, name in [(a, "FD"), (n, "network")]:
            lab = f"{name}, $z={z[j]:+.2f}$"
            for ax, (t, key) in zip(axes.ravel(), panels):
                y = src[key][:-1, j]
                xx, yy = with_break(B_, y) if key == "d" else (B_, y)
                line(ax, xx, yy, k, lab)
            k += 1
    zero_line(axes[1, 1])
    for ax, (t, key) in zip(axes.ravel(), panels):
        ax.set_title(t)
    # ergodic density of debt (marginal), both solutions, display smoothing as in fig_ou_slices
    from scipy.ndimage import gaussian_filter1d
    h = B[1] - B[0]
    ax = axes[1, 2]
    for kk, (gg, name) in enumerate([(n["g_fd"], "FD"), (n["g_nn"], "network")]):
        w = gg / gg.sum(); gB = w.sum(1)
        line(ax, B[1:], gaussian_filter1d(gB[1:] / h, sigma=3, mode="nearest"), kk)
        ax.annotate(f"mass at $B=0$: {gB[0]:.2f} ({name})", xy=(0.42, 0.88 - 0.1 * kk), xycoords="axes fraction", fontsize=7.5)
    ax.set_title("ergodic density of debt")
    for ax in axes[1]:
        ax.set_xlabel("debt $B$")
    handles, labels_ = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, handlelength=2.4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=1.2, h_pad=1.0, rect=(0, 0.04, 1, 1))
    save(fig, "ou_pinn")


# figure name on the command line -> function
ALL = dict(det=fig_det, cuberoot=fig_cuberoot, jump=fig_jump, ou_slices=fig_ou_slices, ou_contours=fig_ou_contours, irf=fig_irf, irf_states=fig_irf_states, saving_density=fig_saving_density,
           multiplicity=fig_multiplicity, anatomy=fig_anatomy, regions=fig_regions, sf_region=fig_sf_region, pinn=fig_pinn)
if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print(n, flush=True)
        ALL[n]()
