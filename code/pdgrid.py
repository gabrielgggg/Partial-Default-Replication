"""pdgrid.py -- non-uniform debt grid from the resolution rule (pddiag.py):
spacing h(B) = clip(h0 * qmin(B)^2, hmin, h0), qmin the minimum over income states of a reference
price on a coarse grid, smoothed so that neighbouring spacings differ by at most 10%. Also a warm start that interpolates
a solution onto a new grid."""
import numpy as np


def build_grid(B_ref, qmin_ref, Bmax, h0=0.005, hmin=2.5e-4, growth=1.1):
    """Nodes on [0, Bmax] with spacing h(B) from the reference minimum price qmin_ref on B_ref."""
    # reference price on a fine auxiliary grid, extended log-linearly beyond B_ref
    ref = np.arange(0.0, Bmax + 1e-9, 1e-4)
    qmin = np.interp(ref, B_ref, qmin_ref)
    if Bmax > B_ref[-1]:
        sel = B_ref >= B_ref[-1] - 0.1
        slope = np.polyfit(B_ref[sel], np.log(qmin_ref[sel]), 1)[0]
        ext = ref > B_ref[-1]
        qmin[ext] = np.maximum(qmin_ref[-1] * np.exp(slope * (ref[ext] - B_ref[-1])), 0.05)
    # spacing rule, then smoothing so that neighbouring spacings grow by at most the factor `growth`
    h = np.clip(h0 * qmin ** 2, hmin, h0)
    for _ in range(3):
        for i in range(1, h.size):
            h[i] = min(h[i], growth * h[i - 1])
        for i in range(h.size - 2, -1, -1):
            h[i] = min(h[i], growth * h[i + 1])
    # place the nodes, ending exactly at Bmax
    nodes = [0.0]
    while nodes[-1] < Bmax:
        nodes.append(nodes[-1] + np.interp(nodes[-1], ref, h))
    nodes = np.array(nodes)
    if nodes[-1] > Bmax:
        nodes = nodes[:-1]
        if Bmax - nodes[-1] < 0.5 * np.interp(nodes[-1], ref, h):
            nodes = nodes[:-1]
        nodes = np.append(nodes, Bmax)
    return nodes


def warm_start(B_old, z_old, V_old, q_old, B_new, z_new):
    """Interpolate (V, q) onto a new grid; beyond the old Bmax extend V linearly and q geometrically."""
    from scipy.interpolate import RegularGridInterpolator
    BB, ZZ = np.meshgrid(np.clip(B_new, B_old[0], B_old[-1]), np.clip(z_new, z_old[0], z_old[-1]), indexing="ij")
    pts = np.stack([BB.ravel(), ZZ.ravel()], axis=1)
    V0 = RegularGridInterpolator((B_old, z_old), V_old)(pts).reshape(B_new.size, z_new.size)
    q0 = RegularGridInterpolator((B_old, z_old), q_old)(pts).reshape(B_new.size, z_new.size)
    ext = B_new > B_old[-1]
    if ext.any():
        slope = (V_old[-1] - V_old[-2]) / (B_old[-1] - B_old[-2])
        V0[ext] = np.interp(z_new, z_old, V_old[-1])[None, :] + np.interp(z_new, z_old, slope)[None, :] * (B_new[ext] - B_old[-1])[:, None]
        ratio = np.clip(np.interp(z_new, z_old, q_old[-1] / q_old[-2]), 0.5, 1.0)
        q0[ext] = np.interp(z_new, z_old, q_old[-1])[None, :] * ratio[None, :] ** ((B_new[ext] - B_old[-1])[:, None] / (B_old[-1] - B_old[-2]))
    return V0, np.clip(q0, 1e-6, 1.0)
