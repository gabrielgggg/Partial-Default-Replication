"""Generator of the main solution and expected paths under the forward equation.

model_from_solution rebuilds the discretized model and its generator matrix from a saved solution
(data/ou_main.npz: V, q, S, c, d, B, z, x, qf, band, lband). expected_paths and expected_paths_from_states
integrate the forward equation dg/dt = A^T g, which gives impulse responses without simulation.
Imported by irf.py and text_numbers.py.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from calutil import R_FIX


def model_from_solution(a):
    """Model, parameters, generator A, and the saved policies (S, c, d) of a solution.

    The saved policies are used as they are: the solver holds the upwind branch of a node where two branches
    tie, so policies recomputed from (V, q) alone can differ from the converged ones at such nodes."""
    from pdmodel import Model, Params, ZProcess
    g0, g1, g2, zt, rho, kappa, delta, rho_z, sig_eps = a["x"].astype(float)
    par = Params(nu=2.0, rho=float(rho), r=R_FIX, delta=float(delta), kappa=float(kappa), g0=float(g0),
                 g1=float(g1), g2=float(g2), ztilde=float(zt), fixed_cost_form="level", S_cap=1.0,
                 q_floor=float(a["qf"]), q_band=float(a["band"]), l_band=float(a["lband"]), closed_d=True)
    z = a["z"]
    mu = -np.log(rho_z); sigma = sig_eps / np.sqrt(1 - rho_z ** 2) * np.sqrt(2 * mu)
    zp = ZProcess.ou(mu, sigma, z.size, zmin=float(z[0]), zmax=float(z[-1]))
    assert np.allclose(zp.z, z)
    m = Model(par, zp, 0.0, float(a["B"][-1]), a["B"].size)
    S, c, d = a["S"], a["c"], a["d"]
    A = m.assemble(S)
    return m, par, A, S, c, d


def shift_in_z(g, z, dz_shift):
    """Push the mass of g(B, z) to z + dz_shift, split linearly between the two neighbouring nodes;
    mass pushed beyond the ends of the grid is placed on the end node (reflecting boundary)."""
    nB, nz = g.shape
    out = np.zeros_like(g)
    h = z[1] - z[0]
    for k in range(nz):
        pos = (z[k] + dz_shift - z[0]) / h
        if pos <= 0:
            out[:, 0] += g[:, k]
        elif pos >= nz - 1:
            out[:, nz - 1] += g[:, k]
        else:
            j = int(np.floor(pos)); w = pos - j
            out[:, j] += (1 - w) * g[:, k]
            out[:, j + 1] += w * g[:, k]
    return out


def net_output(par, d, z):
    """Output net of default costs, phi(d, z) e^z, on the (B, z) grid."""
    from pdmodel import phi_fn
    form = 1.0 if par.fixed_cost_form == "level" else 0.0
    ph = np.array([[phi_fn(d[i, j], z[j], par.g0, par.g1, par.g2, par.ztilde, par.F_min, form)
                    for j in range(z.size)] for i in range(d.shape[0])])
    return ph * np.exp(z)[None, :]


def expected_paths(a, T=12.0, dt=0.01, shock=-0.10):
    """Expected paths of output, consumption, default intensity, debt, spread, issuance and defaulted
    coupons under the forward equation dg/dt = A^T g (Crank-Nicolson in time), starting from the ergodic
    distribution with the endowment shifted by `shock`; and the same expectations under the ergodic
    distribution itself (the control, constant in time)."""
    m, par, A, S, c, d = model_from_solution(a)
    B, z = m.B, m.z
    g0 = a["g"] / a["g"].sum()
    coupon = par.delta + par.lam
    BB = B[:, None] * np.ones((1, z.size))
    fields = dict(y=np.exp(z)[None, :] * np.ones((B.size, 1)), yn=net_output(par, d, z), c=c, d=d, B=BB,
                  spread=coupon / a["q"] - (par.delta + par.r),
                  issue=S + par.delta * BB - par.kappa * coupon * d * BB, defc=d * coupon * BB)
    nT = int(round(T / dt))
    N = m.N
    AT = A.T.tocsc()
    I = sp.identity(N, format="csc")
    # Crank-Nicolson step, factorized once
    lu = spla.splu((I - 0.5 * dt * AT).tocsc())
    Mr = (I + 0.5 * dt * AT).tocsr()
    g = shift_in_z(g0, z, shock).ravel()
    out = {"shock": {k: np.zeros(nT + 1) for k in fields}, "control": {k: np.zeros(nT + 1) for k in fields}}
    ctrl = {k: float((g0 * v).sum()) for k, v in fields.items()}
    for t in range(nT + 1):
        G = g.reshape(B.size, z.size)
        for k, v in fields.items():
            out["shock"][k][t] = float((G * v).sum())
            out["control"][k][t] = ctrl[k]
        g = lu.solve(Mr @ g)
        g = np.maximum(g, 0.0); g /= g.sum()
    tt = np.arange(nT + 1) * dt
    return tt, out


def point_mass(B, z, B0, z0):
    """Unit mass at (B0, z0), split bilinearly between the neighbouring grid nodes."""
    g = np.zeros((B.size, z.size))
    iB = int(np.clip(np.searchsorted(B, B0) - 1, 0, B.size - 2))
    jz = int(np.clip(np.searchsorted(z, z0) - 1, 0, z.size - 2))
    wB = float(np.clip((B0 - B[iB]) / (B[iB + 1] - B[iB]), 0.0, 1.0))
    wz = float(np.clip((z0 - z[jz]) / (z[jz + 1] - z[jz]), 0.0, 1.0))
    g[iB, jz] += (1 - wB) * (1 - wz); g[iB + 1, jz] += wB * (1 - wz)
    g[iB, jz + 1] += (1 - wB) * wz; g[iB + 1, jz + 1] += wB * wz
    return g


def expected_paths_from_states(a, states, T=12.0, dt=0.01, shock=-0.10):
    """State-contingent impulse responses. For each (B0, z0) in `states`, the expected paths under the
    forward equation from a point mass at (B0, z0 + shock) and, as the control, from (B0, z0) without
    the shock. Same fields and time stepping as expected_paths. Returns (t, [out_1, out_2, ...]) with
    out_k = {"shock": {...}, "control": {...}}, plus the ergodic mean output for normalization."""
    m, par, A, S, c, d = model_from_solution(a)
    B, z = m.B, m.z
    coupon = par.delta + par.lam
    BB = B[:, None] * np.ones((1, z.size))
    fields = dict(y=np.exp(z)[None, :] * np.ones((B.size, 1)), yn=net_output(par, d, z), c=c, d=d, B=BB,
                  spread=coupon / a["q"] - (par.delta + par.r),
                  issue=S + par.delta * BB - par.kappa * coupon * d * BB, defc=d * coupon * BB)
    nT = int(round(T / dt))
    N = m.N
    AT = A.T.tocsc()
    I = sp.identity(N, format="csc")
    # Crank-Nicolson step, factorized once
    lu = spla.splu((I - 0.5 * dt * AT).tocsc())
    Mr = (I + 0.5 * dt * AT).tocsr()

    def run(g):
        """Expectations of the fields along the forward equation from the distribution g."""
        g = g.ravel().copy()
        out = {k: np.zeros(nT + 1) for k in fields}
        for t in range(nT + 1):
            G = g.reshape(B.size, z.size)
            for k, v in fields.items():
                out[k][t] = float((G * v).sum())
            g = lu.solve(Mr @ g)
            g = np.maximum(g, 0.0); g /= g.sum()
        return out

    results = []
    for (B0, z0) in states:
        g = point_mass(B, z, B0, z0)
        results.append({"shock": run(shift_in_z(g, z, shock)), "control": run(g)})
    g0 = a["g"] / a["g"].sum()
    ybar = float((g0 * fields["yn"]).sum())
    tt = np.arange(nT + 1) * dt
    return tt, results, ybar
