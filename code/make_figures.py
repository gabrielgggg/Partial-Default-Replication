"""
make_figures.py: the figures of the paper drawn from stored results (pessimistic_approach.pdf is drawn by
pessimistic_approach.py). Each figure is a PDF in output/, rendered through pdflatex (pgf backend) in the mathpazo
font of the text.

Reads (in data/):
  episode_one, episode_typical      episodes.npz
  regions                           ou_main.npz
  det_policies                      det_fine15.npz, ou_main.npz
  det_multiplicity                  wedge/rho0.060_illus.npz
  jump_policies                     jump_floor.npz
  ou_slices, ou_contours            ou_main.npz
  ou_irf                            irf.npz, ou_main.npz
  ou_irf_states                     irf_states.npz, ou_main.npz
  det_cuberoot                      det_fine.npz, det_floor.npz, ou_main.npz
  sf_region                         sfregion.npz
  ou_saving_density                 ou_main.npz, ou_saving.npz
Writes: output/<name>.pdf for each name above.

Usage:  python code/make_figures.py [names...]   (names as above; default: all figures)
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
IRF_XMAX = 20.0   # impulse responses are computed to 21 years and drawn to 20, so every path reaches the frame


def line(ax, x, y, k, label=None, **kw):
    """Line k of a panel: color and line style from STYLES."""
    st = dict(STYLES[k % len(STYLES)])
    st.update(kw)
    ax.plot(x, y, label=label, **st)


def zero_line(ax):
    """Thin horizontal line at zero."""
    ax.axhline(0, color=BLACK, lw=0.5)


def clean(ax):
    """Remove the top and right spines."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


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


def load_ou():
    """The main diffusion solution (data/ou_main.npz) and its parameters as a dict."""
    a = np.load(os.path.join(DATA, "ou_main.npz"))
    x = a["x"].astype(float)
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = x
    r = lam = np.log(1.04)
    mu = -np.log(rho_z); sd = sig_eps / np.sqrt(1 - rho_z ** 2); sigma = sd * np.sqrt(2 * mu)
    par = dict(rho=rho, kappa=kappa, delta=delta, r=r, lam=lam, mu=mu, sigma=sigma, sd=sd, g0=g0, g1=g1, g2=g2, zt=zt)
    return a, par


# ============================================================ default episodes in the diffusion model
def fig_episode_one():
    """One simulated default episode: debt against its default threshold over time, with output, the share of debt
    service withheld, and the path in the plane of output and debt (data/episodes.npz, from episodes.py)."""
    e = np.load(os.path.join(DATA, "episodes.npz"))
    tw, xb, xd, zs, xthr, z, thr = (e[k] for k in ("t", "one_B", "one_d", "one_z", "one_thr", "z", "thr"))
    du0, ybar, h = float(e["one_duration"]), float(e["ybar"]), float(e["h"])
    PRE, POST = -tw[0], tw[-1]
    npre = int(round(PRE / h))
    kend = npre + int(round(du0 / h))
    fig = plt.figure(figsize=(W, 0.72 * W))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.62, wspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(0, du0, color=LIGHT_80, lw=0)
    ax.plot(tw, xthr, color=BLACK, lw=1.0, ls="--", label="default threshold")
    ax.plot(tw, xb, color=NAVY, lw=1.4, label="debt")
    top = 1.5 * max(xthr.max(), xb.max())
    ax.set_xlim(-PRE, POST); ax.set_ylim(0, top)
    ax.legend(loc="upper left", fontsize=7, handlelength=2.0, ncol=2, borderaxespad=0.2, columnspacing=1.5)
    note = dict(fontsize=7.2, ha="center", va="center", linespacing=1.15)
    arrow = dict(arrowstyle="-", color=MIDGRAY, lw=0.6, shrinkA=1, shrinkB=2)
    ax.annotate("the sovereign keeps\nits debt below the threshold", xy=(-2.8, xb[int(2.2 / h)]), xytext=(-2.4, 0.13 * top), arrowprops=arrow, **note)
    ax.annotate("output falls, the threshold drops\nbelow the debt: default begins", xy=(0, xb[npre]), xytext=(3.6, 0.72 * top), arrowprops=arrow, **note)
    km = npre + int(0.55 * du0 / h)
    ax.annotate("in default the sovereign\nkeeps borrowing", xy=(tw[km], xb[km]), xytext=(tw[km] + 0.6, 0.13 * top), arrowprops=arrow, **note)
    ax.annotate("output recovers, the threshold\nrises above the debt: default ends", xy=(du0, xb[kend]), xytext=(min(du0 + 4.2, POST - 4.2), 0.80 * top), arrowprops=arrow, **note)
    ax.set_xlabel("years from the first default")
    ax.set_title("one default episode: debt and its default threshold")
    clean(ax)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axvspan(0, du0, color=LIGHT_80, lw=0)
    ax1.plot(tw, 100 * (np.exp(zs) / ybar - 1), color=BLACK, lw=1.0)
    ax1.axhline(0, color=BLACK, lw=0.5)
    ax1.set_title("output, \\% from mean")
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axvspan(0, du0, color=LIGHT_80, lw=0)
    ax2.plot(tw, xd, color=SCARLET, lw=1.1)
    ax2.set_ylim(0, 1)
    ax2.set_title("share withheld")
    for x in (ax1, ax2):
        x.set_xlim(-PRE, POST); x.set_xticks([-5, 0, 5, 10, 15, 20]); x.set_xlabel("years"); clean(x)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.fill_between(z, 0, thr, color=LIGHT_60, lw=0)
    ax3.plot(z, thr, color=BLACK, lw=0.9, ls="--")
    ax3.plot(zs[:npre + 1], xb[:npre + 1], color=NAVY, lw=1.1)
    ax3.plot(zs[npre:kend + 1], xb[npre:kend + 1], color=SCARLET, lw=1.1)
    ax3.plot(zs[kend:], xb[kend:], color=NAVY, lw=1.1)
    ax3.plot([zs[0]], [xb[0]], "o", ms=3, color=NAVY)
    ax3.text(0.97, 0.06, "no default", fontsize=7, transform=ax3.transAxes, ha="right")
    ax3.text(0.04, 0.90, "default", fontsize=7, transform=ax3.transAxes)
    zlo, zhi = min(zs.min() - 0.04, -0.12), max(zs.max() + 0.04, 0.12)
    ax3.set_xlim(zlo, zhi); ax3.set_ylim(0, 1.6 * xb.max())
    ax3.set_xlabel("log output"); ax3.set_title("debt against output")
    clean(ax3)
    save(fig, "episode_one")


def fig_episode_typical():
    """The typical default episode: medians and quartiles of output and debt, the median default threshold, the
    share withheld and the share of episodes still in default, and the mean flows that add to debt, around the
    first default, across all simulated episodes (data/episodes.npz, from episodes.py)."""
    e = np.load(os.path.join(DATA, "episodes.npz"))
    tw = e["t"]
    PRE, POST = -tw[0], tw[-1]

    def band(ax, Q, color, **kw):
        """Median line over the band between the quartiles; Q holds the 25th, 50th, and 75th percentiles."""
        lo, md, hi = Q
        ax.fill_between(tw, lo, hi, color=color, alpha=0.13, lw=0)
        ax.plot(tw, md, color=color, **kw)

    fig, axes = plt.subplots(2, 2, figsize=(W, 0.62 * W))
    ax = axes[0, 0]
    band(ax, e["typ_output"], BLACK, lw=1.1)
    ax.axhline(0, color=BLACK, lw=0.5)
    ax.set_title("output, \\% from mean")
    ax = axes[0, 1]
    ax.plot(tw, e["typ_thr"], color=BLACK, ls="--", label="default threshold")
    band(ax, e["typ_B"], NAVY, lw=1.3, label="debt")
    ax.set_ylim(0, None)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.8)
    ax.set_title("debt and its default threshold")
    ax = axes[1, 0]
    ax.plot(tw, e["typ_withheld"], color=SCARLET, lw=1.3, label="share of debt service withheld, in default")
    ax.plot(tw, e["typ_in_default"], color=MIDGRAY, ls=":", lw=1.4, label="share of episodes still in default")
    ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper right", fontsize=7, handlelength=1.8)
    ax.set_title("default")
    ax = axes[1, 1]
    ax.plot(tw, e["typ_issues"], color=NAVY, lw=1.3, label="new issues")
    ax.plot(tw, e["typ_arrears"], color=SCARLET, ls="--", lw=1.3, label="arrears")
    ax.axhline(0, color=BLACK, lw=0.5)
    ax.legend(loc="upper right", fontsize=7, handlelength=1.8)
    ax.set_title("additions to debt, \\% of mean output a year")
    for ax in axes.ravel():
        ax.axvline(0, color=MIDGRAY, lw=0.5)
        ax.set_xlim(-PRE, POST); ax.set_xticks([-5, 0, 5, 10, 15, 20])
        clean(ax)
    for ax in axes[1]:
        ax.set_xlabel("years from the first default")
    fig.tight_layout(w_pad=1.4, h_pad=1.0)
    save(fig, "episode_typical")


# ============================================================ the default decision at the mean endowment
def fig_regions():
    """The default decision at z = 0 and the calibrated parameters, from the closed forms: the gain from
    defaulting R(d) - R(0) at three debt levels at the risk-free price, and the regions of no, partial, and full
    default in the (B, q) plane."""
    a, p = load_ou()
    g0, g1, g2, zt, kap, cs = p["g0"], p["g1"], p["g2"], p["zt"], p["kappa"], p["delta"] + p["lam"]
    F = 1 - g2 * (1 - np.exp(zt))                      # z = 0: output 1, above y*
    dfirst = ((1 - F) / ((g1 - 1) * g0 * F)) ** (1 / g1)
    G = g0 * g1 * F
    K = lambda q: (1 - kap * q) * cs
    Bpd = lambda q: (G * dfirst ** (g1 - 1)) / K(q)    # the no-default threshold when d_first < 1, written through d_first
    Bfd = lambda q: G / K(q)
    Bs = Bpd(1.0)
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.38 * W))
    ax = axes[0]
    dd = np.linspace(1e-4, 1, 600)
    for k, (fac, lab) in enumerate([(0.7, "$B=0.7\\,B^*$"), (1.0, "$B=B^*$"), (1.3, "$B=1.3\\,B^*$")]):
        b = fac * Bs
        gain = K(1.0) * b * dd - g0 * F * dd ** g1 - (1 - F)
        line(ax, dd, gain, k, lab)
    ax.axhline(0, color=BLACK, lw=0.5)
    ax.plot([0], [0], "o", color=BLACK, ms=3.5, mfc="white", clip_on=False)   # d = 0: no gain, no fixed cost
    ax.axvline(dfirst, color=GRAY, lw=0.6, ls=":")
    ax.annotate("$d_{\\mathrm{first}}$", xy=(dfirst, -0.027), xytext=(dfirst + 0.03, -0.027), fontsize=7.5)
    ax.set_xlabel("default intensity $d$"); ax.set_title("gain from defaulting, $\\mathcal{R}(d)-\\mathcal{R}(0)$")
    ax.set_ylim(-0.03, 0.015)
    ax.legend(loc="lower left", fontsize=7, handlelength=2.0)
    ax = axes[1]
    qq = np.linspace(0, 1, 400)
    bp, bf = Bpd(qq), Bfd(qq)
    Bmax = 1.25
    ax.fill_betweenx(qq, 0, bp, color=LIGHT_80, lw=0)
    ax.fill_betweenx(qq, bf, Bmax, color=LIGHT_20, lw=0)
    ax.plot(bp, qq, color=BLACK, lw=1.0, label="$B_{\\mathrm{pd}}(0,q)$")
    ax.plot(bf, qq, color=BLACK, lw=1.0, ls="--", label="$B_{\\mathrm{fd}}(0,q)$")
    ax.plot([Bs], [1.0], "o", color=BLACK, ms=3.5, clip_on=False)
    ax.annotate("$B^*$", xy=(Bs, 1.0), xytext=(Bs + 0.025, 0.93), fontsize=7.5)
    ax.text(0.155, 0.865, "no", fontsize=7.5, ha="center")          # two centered lines (pgf ignores multialignment)
    ax.text(0.155, 0.80, "default", fontsize=7.5, ha="center")
    ax.text(0.47, 0.55, "partial default", fontsize=7.5)
    ax.text(0.98, 0.22, "full\ndefault", fontsize=7.5)
    ax.set_xlim(0, Bmax); ax.set_ylim(0, 1)
    ax.set_xlabel("debt $B$"); ax.set_ylabel("bond price $q$"); ax.set_title("default regions")
    fig.tight_layout(w_pad=1.6)
    save(fig, "regions")


# ============================================================ deterministic case
def fig_det():
    """The deterministic equilibrium at the calibrated parameters, 500 nodes per unit of debt on [0, 1.5]: price,
    default intensity, drift, issuance, value, and consumption (data/det_fine15.npz)."""
    a = np.load(os.path.join(DATA, "det_fine15.npz"))
    B, q, d, S, c, l = a["B"], a["q"], a["d"], a["S"], a["c"], a["issuance"]
    Bs = float(a["Bpd"])          # the closed-form steady state B* = B_pd(0, q_rf)
    qf = float(load_ou()[0]["qf"])
    i = np.where(q < qf)[0]
    Bf = (B[i[0] - 1] + (q[i[0] - 1] - qf) / (q[i[0] - 1] - q[i[0]]) * (B[i[0]] - B[i[0] - 1])) if i.size else np.nan
    print(f"  underwriting standard {qf}: q crosses it at B = {Bf:.4f}")

    def floor_line(ax):
        """Dashed vertical line at the debt where the price crosses the underwriting standard."""
        if np.isfinite(Bf):
            ax.axvline(Bf, color=GRAY, lw=0.8, ls="--")
    fig, axes = plt.subplots(2, 3, figsize=(W, 0.6 * W))
    panels = [("bond price $q(B)$", q), ("default intensity $d(B)$", d), ("drift of debt $\\dot B = S(B)$", S),
              ("issuance $\\ell(B)$", l), ("value $V(B)$", a["V"]), ("consumption $c(B)$", c)]
    for ax, (t, y) in zip(axes.ravel(), panels):
        xx, yy = with_break(B, y) if t.startswith("default") else (B, y)
        ax.plot(xx, yy, color=BLACK, ls="-")
        ax.axvline(Bs, color=GRAY, lw=0.6, ls=":")
        floor_line(ax)
        ax.set_title(t)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
        if t.startswith("drift") or t.startswith("issuance"):
            zero_line(ax)
    for ax in axes[1]:
        ax.set_xlabel("debt $B$")
    axes[0, 0].text(Bs - 0.04, 0.04, "$B^*$", fontsize=8, ha="right", va="bottom", transform=axes[0, 0].get_xaxis_transform())
    if np.isfinite(Bf):
        axes[0, 0].annotate(r"$q=\underline{q}$", xy=(Bf, 1.0), xytext=(Bf + 0.05, 0.96), fontsize=8, va="top", textcoords=axes[0, 0].get_xaxis_transform())
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "det_policies")


def fig_multiplicity():
    """Two deterministic equilibria at an illustrative parameterization (small first default) and an
    impatience wedge of about 200 bp: the no-default steady state and a self-fulfilling partial-default
    steady state (data/wedge/rho0.060_illus.npz)."""
    a = np.load(os.path.join(DATA, "wedge", "rho0.060_illus.npz"))
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
        line(axes[0], B[sel], q[sel], k, lab)
        line(axes[1], B[sel], S[sel], k, lab)
        line(axes[2], *with_break(B[sel], d[sel]), k, lab)
        for ax in axes:
            ax.axvline(Bs, color=STYLES[k]["color"], lw=0.6, ls=":")
    zero_line(axes[1])
    axes[0].set_title("bond price $q(B)$"); axes[1].set_title("drift of debt $S(B)$"); axes[2].set_title("default intensity $d(B)$")
    for ax in axes:
        ax.set_xlabel("debt $B$")
    axes[1].legend(loc="upper right", fontsize=7, handlelength=1.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "det_multiplicity")


def fig_cuberoot():
    """The cube-root law at the steady state B*: the computed drift and price gap to the right of B* and the drift
    to its left, on two grids, against the two-term expansions and the square-root law
    (data/det_fine.npz, data/det_floor.npz)."""
    from cuberoot_consts import second_order
    a5 = np.load(os.path.join(DATA, "det_fine.npz"))
    Bpd, beta, alpha, sq = float(a5["Bpd"]), float(a5["beta"]), float(a5["alpha"]), float(a5["sq"])
    Sig1, Q1 = second_order(a5, load_ou()[0]["x"])
    sols = [("500 nodes per unit", a5["B"], a5["S"], a5["q"])]
    a2 = np.load(os.path.join(DATA, "det_floor.npz"))
    sols.append(("200 nodes per unit", a2["B"], a2["S"], a2["q"]))
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.36 * W))
    xx = np.logspace(-3.3, -0.6, 60); ss = xx ** (1 / 3)
    axes[0].loglog(xx, beta * ss + Sig1 * ss ** 2, color=BLACK, ls="-", label="two terms")
    axes[0].loglog(xx, beta * ss, color=GRAY, ls="--", label=r"$\beta x^{1/3}$")
    axes[1].loglog(xx, alpha * ss ** 2 + Q1 * ss ** 3, color=BLACK, ls="-", label="two terms")
    axes[1].loglog(xx, alpha * ss ** 2, color=GRAY, ls="--", label=r"$\alpha x^{2/3}$")
    xl = np.logspace(-3.3, -1.1, 40)
    axes[2].loglog(xl, sq * xl ** 0.5, color=BLACK, ls="-", label=r"$\sqrt{2(\rho-\lambda)\varphi(\bar c)}\,x^{1/2}$")
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


# ============================================================ two-state chain
def fig_jump():
    """The chain with two endowment states: price, default intensity, and drift by state, over the whole debt
    range, with the debt levels that carry the ergodic mass (data/jump_floor.npz)."""
    a = np.load(os.path.join(DATA, "jump_floor.npz"))
    B, z, q, d, S, g = a["B"], a["z"], a["q"], a["d"], a["S"], a["g"]
    nz = z.size
    labs = [f"$z={zz:+.2f}$" for zz in z]
    w = g / g.sum()
    fig, (axq, axd, axS) = plt.subplots(1, 3, figsize=(W, 0.34 * W))
    for k in range(nz):
        line(axq, B, q[:, k], k, labs[k])
        line(axd, *with_break(B, d[:, k]), k, labs[k])
        line(axS, B, S[:, k], k, labs[k])
    zero_line(axS)
    rest = sorted({float(B[i]) for k in range(nz) for i in np.where(w[:, k] > 0.05)[0]})
    for ax in (axq, axd, axS):
        for Br in rest:
            ax.axvline(Br, color=GRAY, lw=0.6, ls=":")
        ax.set_xlabel("debt $B$")
    axq.set_title("bond price $q(B,z)$"); axd.set_title("default intensity $d(B,z)$"); axS.set_title("drift of debt $S(B,z)$")
    axq.legend(loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "jump_policies")


# ============================================================ diffusion model at the calibrated parameters
def fig_ou_slices():
    """Value, price, consumption, default intensity, and drift against debt at four endowment levels, and the
    ergodic density of debt (data/ou_main.npz)."""
    a, p = load_ou()
    B, z, V, q, c, d, S, g = a["B"], a["z"], a["V"], a["q"], a["c"], a["d"], a["S"], a["g"]
    # endowment levels shown: one and two stationary standard deviations below and above the mean. The lowest is
    # the highest node below the fixed-cost threshold ln y* (about 2.1 standard deviations below the mean), so that
    # one line shows the region without a fixed cost.
    sd = p["sd"]
    js = [int(np.where(z < p["zt"])[0].max())] + [int(np.argmin(np.abs(z - k * sd))) for k in (-1, 1, 2)]
    w = g / g.sum(); gB = w.sum(1); h = B[1] - B[0]
    Bg = B
    B, V, q, c, d, S = (v[:-1] for v in (B, V, q, c, d, S))   # drop the top node, where the state constraint shapes the policies
    labs = [f"${k:+d}$ s.d." for k in (-2, -1, 1, 2)]      # in stationary standard deviations of the log endowment
    fig, axes = plt.subplots(2, 3, figsize=(W, 0.6 * W))
    for k, j in enumerate(js):
        line(axes[0, 0], B, V[:, j], k, labs[k])
        line(axes[0, 1], B, q[:, j], k, labs[k])
        line(axes[0, 2], B, c[:, j], k, labs[k])
        line(axes[1, 0], *with_break(B, d[:, j]), k, labs[k])
        line(axes[1, 1], B, S[:, j], k, labs[k])
    zero_line(axes[1, 1])
    from scipy.ndimage import gaussian_filter1d
    gB_s = gaussian_filter1d(gB[1:] / h, sigma=3, mode="nearest")   # display only: the grid marginal has spikes at the thresholds of the z nodes
    axes[1, 2].plot(Bg[1:], gB_s, color=BLACK)
    axes[1, 2].annotate(f"mass at $B=0$: {gB[0]:.2f}", xy=(0.40, 0.84), xycoords="axes fraction", fontsize=7.5)
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
    """Contours of the price, the default intensity, and the ergodic density in the (B, z) plane, with the
    zero-drift locus (data/ou_main.npz)."""
    a, p = load_ou()
    B, z, q, d, S, g = a["B"], a["z"], a["q"], a["d"], a["S"], a["g"]
    from scipy.ndimage import gaussian_filter
    BB, ZZ = np.meshgrid(B, z, indexing="ij")
    h = B[1] - B[0]; dz = z[1] - z[0]
    dens = g / g.sum() / (h * dz)
    # display smoothing: the grid density has ridges along the zero-drift locus, one per endowment node. The atom at
    # zero debt (row 0) is left out of the smoothing and of the contours.
    dens_s = dens.copy()
    dens_s[1:] = gaussian_filter(dens[1:], sigma=(4.0, 3.0), mode="nearest")
    d_s = gaussian_filter(d, sigma=(1.5, 0.6))            # removes the grid staircase of the default policy
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.36 * W))
    cs = axes[0].contour(BB, ZZ, q, levels=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], colors=[BLACK], linewidths=0.7)
    axes[0].clabel(cs, fmt="%.2f", fontsize=6.5, inline=True)
    cs = axes[1].contour(BB, ZZ, d_s, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors=[BLACK], linewidths=0.7)
    axes[1].clabel(cs, levels=[0.5, 0.7, 0.9], fmt="%.1f", fontsize=6.5, inline=True)   # 0.1 and 0.3 sit on the jump
    # contours of highest density: each encloses the given share of the mass on positive debt
    v = np.sort(dens_s[1:].ravel())[::-1]
    cum = np.cumsum(v) / v.sum()
    lev = sorted(float(v[np.searchsorted(cum, share)]) for share in (0.50, 0.75, 0.90, 0.99))
    axes[2].contour(BB[1:], ZZ[1:], dens_s[1:], levels=lev, colors=[BLACK], linewidths=0.7)
    for ax in axes:
        ax.contour(BB, ZZ, S, levels=[0.0], colors=[RED], linewidths=1.6, linestyles=":")
        ax.set_xlabel("debt $B$")
    axes[0].set_ylabel("log endowment $z$")
    axes[0].set_title("bond price $q$"); axes[1].set_title("default intensity $d$"); axes[2].set_title("ergodic density")
    axes[2].set_xlim(0, 0.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ou_contours")


def fig_saving_density():
    """The ergodic density without saving (the main solution) and with saving (data/ou_saving.npz), panel widths
    proportional to the range of debt each spans."""
    from scipy.ndimage import gaussian_filter
    a = load_ou()[0]; s = np.load(os.path.join(DATA, "ou_saving.npz"))
    ranges = [(0.0, 0.8), (-0.6, 0.8)]
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.36 * W), sharey=True,
                             gridspec_kw=dict(width_ratios=[r[1] - r[0] for r in ranges]))
    for ax, sol, skip, (lo, hi), title in zip(axes, (a, s), (1, 0), ranges, ("borrowing only", "with saving")):
        B, z, g, S = sol["B"], sol["z"], sol["g"], sol["S"]
        BB, ZZ = np.meshgrid(B, z, indexing="ij")
        # as in fig_ou_contours: the density smoothed for display (without the atom at zero debt when the sovereign
        # only borrows), and contours that enclose 50, 75, 90, and 99 percent of the mass shown
        sl = slice(skip, None)
        dens = g / g.sum() / ((B[1] - B[0]) * (z[1] - z[0]))
        dens[sl] = gaussian_filter(dens[sl], sigma=(4.0, 3.0), mode="nearest")
        v = np.sort(dens[sl].ravel())[::-1]
        cum = np.cumsum(v) / v.sum()
        lev = sorted(float(v[np.searchsorted(cum, share)]) for share in (0.50, 0.75, 0.90, 0.99))
        ax.contour(BB[sl], ZZ[sl], dens[sl], levels=lev, colors=[BLACK], linewidths=0.7)
        ax.contour(BB, ZZ, S, levels=[0.0], colors=[RED], linewidths=1.6, linestyles=":")
        ax.axvline(0.0, color=GRAY, lw=0.6, ls=":")
        ax.set_xlim(lo, hi); ax.set_xlabel("debt $B$"); ax.set_title(title)
    axes[0].set_ylabel("log endowment $z$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "ou_saving_density")


# ============================================================ impulse responses in the diffusion model
IRF_KEYS = ["y", "yn", "c", "d", "B", "spread", "issue", "defc"]


def fig_irf():
    """Responses to a fall in the endowment from the ergodic distribution: expected paths with the shock minus
    expected paths without it (data/irf.npz, from irf.py)."""
    a, p = load_ou()
    d = np.load(os.path.join(DATA, "irf.npz"))
    tt = d["t"]; out = {s: {k: d[f"{s}_{k}"] for k in IRF_KEYS} for s in ("shock", "control")}
    sh, co = out["shock"], out["control"]
    ybar = co["yn"].mean()                      # mean output, net of default costs
    fig, axes = plt.subplots(2, 3, figsize=(W, 0.52 * W))
    panels = [("output, \\% of mean", 100 * (sh["yn"] - co["yn"]) / ybar),
              ("consumption, \\% of mean output", 100 * (sh["c"] - co["c"]) / ybar),
              ("default intensity $d$", sh["d"] - co["d"]),
              ("debt $B$, \\% of mean output", 100 * (sh["B"] - co["B"]) / ybar),
              ("spread, basis points", 1e4 * (sh["spread"] - co["spread"]))]
    for ax, (t, y) in zip(axes.ravel(), panels):
        ax.plot(tt, y, color=BLACK); zero_line(ax); ax.set_title(t)
    ax = axes[1, 2]
    line(ax, tt, 100 * (sh["issue"] - co["issue"]) / ybar, 0, "new issues (face value)")
    line(ax, tt, 100 * p["kappa"] * (sh["defc"] - co["defc"]) / ybar, 1, "arrears")
    zero_line(ax); ax.set_title("flows, \\% of mean output"); ax.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    for ax in axes[1]:
        ax.set_xlabel("years")
    for ax in axes.ravel():
        ax.set_xlim(0, IRF_XMAX); ax.set_xticks([0, 5, 10, 15, 20])   # paths run past the edge: the responses continue
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "ou_irf")


def fig_irf_states():
    """State-contingent impulse responses: the shock of fig_irf applied at the mean endowment with debt
    at its ergodic mean, 25 percent below it, and 25 percent above it. Expected paths from a point mass at each
    state, minus the paths from the same state without the shock (data/irf_states.npz, from irf.py)."""
    a, p = load_ou()
    d = np.load(os.path.join(DATA, "irf_states.npz"))
    tt = d["t"]; ybar = float(d["ybar"])
    # state 0 (solid line) is the ergodic mean of debt, 1 (dashed) is below it, 2 (dash-dot) above it
    res = [{s: {k: d[f"{i}_{s}_{k}"] for k in IRF_KEYS} for s in ("shock", "control")} for i in range(3)]
    labels = ["$B_0=\\mathbb{E}[B]$", "$B_0=0.75\\,\\mathbb{E}[B]$", "$B_0=1.25\\,\\mathbb{E}[B]$"]

    def dev(r, k, scale=1.0):
        """Path of variable k with the shock minus the path without it."""
        return scale * (r["shock"][k] - r["control"][k])

    fig, axes = plt.subplots(2, 3, figsize=(W, 0.52 * W))
    ax = axes[0, 0]
    ax.plot(tt, 100 * dev(res[0], "yn") / ybar, color=BLACK); zero_line(ax); ax.set_title("output, \\% of mean")
    panels = [(axes[0, 1], "consumption, \\% of mean output", lambda r: 100 * dev(r, "c") / ybar),
              (axes[0, 2], "default intensity $d$", lambda r: dev(r, "d")),
              (axes[1, 0], "debt $B$, \\% of mean output", lambda r: 100 * dev(r, "B") / ybar),
              (axes[1, 1], "spread, basis points", lambda r: 1e4 * dev(r, "spread"))]
    for ax, t, fn in panels:
        for k, r in enumerate(res):
            line(ax, tt, fn(r), k, labels[k])
        zero_line(ax); ax.set_title(t)
    ax = axes[1, 2]
    for k, r in enumerate(res):
        arrears = p["kappa"] * r["shock"]["defc"]
        gross = np.maximum(r["shock"]["issue"], 0.0) + arrears
        share = np.where(gross > 1e-12, 100 * arrears / np.maximum(gross, 1e-12), np.nan)
        line(ax, tt, share, k, labels[k])
    ax.set_title("arrears, \\% of new debt"); ax.set_ylim(bottom=0)
    axes[0, 2].legend(loc="upper right", fontsize=6.5, handlelength=1.8)
    for ax in axes[1]:
        ax.set_xlabel("years")
    for ax in axes.ravel():
        ax.set_xlim(0, IRF_XMAX); ax.set_xticks([0, 5, 10, 15, 20])
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    save(fig, "ou_irf_states")


# ============================================================ self-fulfilling partial default: curvature and uncertainty
def fig_sf_region():
    """Where self-fulfilling partial-default equilibria exist, against the impatience wedge. Left: CRRA curvature in
    the deterministic model (smallest wedge at which a partial-default steady state is reached from zero debt).
    Right: the s.d. of the log endowment in the risk-neutral diffusion model (steady debt that does not depend on
    the endowment; lower and upper boundaries). Data: data/sfregion.npz (collect_sf.py)."""
    a = np.load(os.path.join(DATA, "sfregion.npz"))
    YMAX, FILL = 300.0, LIGHT_60
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


ALL = dict(episode_one=fig_episode_one, regions=fig_regions, det_policies=fig_det, det_multiplicity=fig_multiplicity,
           jump_policies=fig_jump, ou_slices=fig_ou_slices, ou_contours=fig_ou_contours, ou_irf=fig_irf,
           episode_typical=fig_episode_typical, det_cuberoot=fig_cuberoot, sf_region=fig_sf_region,
           ou_irf_states=fig_irf_states, ou_saving_density=fig_saving_density)
if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print(n, flush=True)
        ALL[n]()
