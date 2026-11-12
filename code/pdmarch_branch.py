"""Branching march for the deterministic case.

At the default threshold the node equations have several roots (a no-default branch, a partial
default branch, and so on). Following only the root nearest in q to the previous node, as
pdmarch.march does, can select the wrong branch, after which every consistency check fails.
Here, at every next node, a root is kept if the previous node keeps its own candidate once the
next node value is known (the alternative one-sided candidate at the previous node, evaluated
with that V, must not beat the candidate chosen there). All consistent roots are followed
(branching, with a cap). If no root is consistent, the nearest one is followed and the
violation is recorded; the first such violation above the steady state is `Bhat`, the
resolution limit of the first-order upwind scheme (see pddiag.py).
Library module, imported by wedge_search.py.
"""

from __future__ import annotations

import numpy as np

from pdmodel import Model, best_node
from pdmarch import node_solutions, zero_drift_fixed_points, to_result  # noqa: F401


def _H(m: Model, VB, q, B, mode, n_golden):
    """Hamiltonian of the one-sided candidate `mode` at a node."""
    return best_node(VB, q, B, m.z[0], m.y[0], m.p, mode, n_golden)[0]


def consistency(m: Model, V, q, which, n_golden=40):
    """Nodes where the chosen candidate is beaten by another one: list of (node, chosen, best, gain)."""
    B = m.B; z = m.z[0]; y = m.y[0]; p = m.p; dB = m.dB
    nB = m.nB
    viol = []
    for k in range(nB):
        if not np.isfinite(V[k]):
            continue
        cands = {}
        if k < nB - 1 and np.isfinite(V[k + 1]):
            cands[1] = best_node((V[k + 1] - V[k]) / dB, q[k], B[k], z, y, p, 1, n_golden)[0]
        if k > 0 and np.isfinite(V[k - 1]):
            cands[-1] = best_node((V[k] - V[k - 1]) / dB, q[k], B[k], z, y, p, -1, n_golden)[0]
        cands[0] = best_node(0.0, q[k], B[k], z, y, p, 0, n_golden)[0]
        best = max(cands, key=cands.get)
        if cands[best] > cands[int(which[k])] + 1e-9:
            viol.append((k, int(which[k]), best, cands[best] - cands[int(which[k])]))
    return viol


def _copy_branch(br):
    """Deep copy of a branch."""
    return dict(V=br["V"].copy(), q=br["q"].copy(), S=br["S"].copy(), c=br["c"].copy(),
                d=br["d"].copy(), which=br["which"].copy(), nsol=br["nsol"].copy(),
                viol=list(br["viol"]), stop=None)


def march_branching(m: Model, istar: int, fp_index: int, nq=200, n_golden=40, max_branches=8,
                    verbose=False, max_nodes=None):
    """March up and down from steady-state node istar (fixed point fp_index), branching on consistent roots.

    Combines up to four upward and two downward branches into full solutions with their global violations."""
    nB = m.nB
    B = m.B; z = m.z[0]; y = m.y[0]; p = m.p; dB = m.dB
    nfp, qf, Vf, cf, df, Hf = zero_drift_fixed_points(B[istar], z, y, p, 400, n_golden)
    f = fp_index
    base = dict(V=np.full(nB, np.nan), q=np.full(nB, np.nan), S=np.zeros(nB), c=np.full(nB, np.nan),
                d=np.full(nB, np.nan), which=np.zeros(nB, dtype=np.int8), nsol=np.zeros(nB, dtype=int),
                viol=[], stop=None)
    base["V"][istar] = Vf[f]; base["q"][istar] = qf[f]; base["c"][istar] = cf[f]; base["d"][istar] = df[f]
    base["nsol"][istar] = nfp

    def march_dir(direction):
        mode = -1 if direction == 1 else 1          # candidate used at the marched nodes
        alt_mode = 1 if direction == 1 else -1      # alternative candidate at the previous node
        branches = [_copy_branch(base)]
        done = []
        while branches:
            br = branches.pop()
            i = istar + direction
            while 0 <= i < nB and np.isfinite(br["V"][i]):
                i += direction
            while 0 <= i < nB and (max_nodes is None or abs(i - istar) <= max_nodes):
                prev = i - direction
                n, qr, Vr, Sr, cr, dr = node_solutions(br["V"][prev], br["q"][prev], mode, B[i], z, y, p, dB, nq, n_golden)
                br["nsol"][i] = n
                if n == 0:
                    br["stop"] = i
                    break
                # keep the roots under which the previous node keeps its candidate
                keep = []
                for k in range(n):
                    if direction == 1:
                        VB_alt = (Vr[k] - br["V"][prev]) / dB
                    else:
                        VB_alt = (br["V"][prev] - Vr[k]) / dB
                    H_alt = _H(m, VB_alt, br["q"][prev], B[prev], alt_mode, n_golden)
                    if prev == istar:
                        H_cur = Hf[f]
                    else:
                        if direction == 1:
                            VB_cur = (br["V"][prev] - br["V"][prev - 1]) / dB
                        else:
                            VB_cur = (br["V"][prev + 1] - br["V"][prev]) / dB
                        H_cur = _H(m, VB_cur, br["q"][prev], B[prev], mode, n_golden)
                    if H_alt <= H_cur + 1e-9:
                        keep.append(k)
                # none: follow the nearest root and record the violation
                if not keep:
                    keep = [int(np.argmin(np.abs(qr[:n] - br["q"][prev])))]
                    br["viol"].append((prev, i))
                # nearest root continues this branch; the others open new branches up to the cap
                keep.sort(key=lambda kk: abs(qr[kk] - br["q"][prev]))
                for k in keep[1:]:
                    if len(branches) + len(done) < max_branches:
                        nb = _copy_branch(br)
                        nb["V"][i] = Vr[k]; nb["q"][i] = qr[k]; nb["S"][i] = Sr[k]; nb["c"][i] = cr[k]
                        nb["d"][i] = dr[k]; nb["which"][i] = mode
                        branches.append(nb)
                k = keep[0]
                br["V"][i] = Vr[k]; br["q"][i] = qr[k]; br["S"][i] = Sr[k]; br["c"][i] = cr[k]
                br["d"][i] = dr[k]; br["which"][i] = mode
                i += direction
            done.append(br)
        return done

    # combine the best upward and downward branches and check each combination globally
    ups = march_dir(+1)
    downs = march_dir(-1)
    downs = sorted(downs, key=lambda b: (b["stop"] is not None, len(b["viol"])))[:2]
    ups = sorted(ups, key=lambda b: (b["stop"] is not None, len(b["viol"])))[:4]
    results = []
    for up in ups:
        for dn in downs:
            sol = dict(istar=istar, Bstar=float(B[istar]), qstar=float(qf[f]), dstar=float(df[f]), cstar=float(cf[f]))
            for key in ("V", "q", "S", "c", "d", "which", "nsol"):
                arr = up[key].copy()
                arr[:istar] = dn[key][:istar]
                sol[key] = arr
            sol["stop_up"] = up["stop"]; sol["stop_dn"] = dn["stop"]
            sol["viol"] = consistency(m, sol["V"], sol["q"], sol["which"], n_golden)
            sol["consistent"] = (not sol["viol"]) and up["stop"] is None and dn["stop"] is None
            results.append(sol)
            if verbose:
                print(f"  istar={istar} (B={B[istar]:.3f}) q*={qf[f]:.4f} d*={df[f]:.3f}: stop_up={up['stop']} "
                      f"stop_dn={dn['stop']} global violations={len(sol['viol'])} "
                      f"multi-root nodes={int(np.sum(sol['nsol'] > 1))}")
    return results


def solve_deterministic(m: Model, verbose=True, n_golden=40, nq=200, prefilter=25):
    """Branching march from every zero-drift node that is locally consistent for some root of
    its neighbours. Usable = no missing root and no violation at or below the steady state;
    violations above are reported through Bhat (resolution limit)."""
    B = m.B; z = m.z[0]; y = m.y[0]; p = m.p; dB = m.dB
    sols = []
    for istar in range(m.nB):
        nfp, qf, Vf, cf, df, Hf = zero_drift_fixed_points(B[istar], z, y, p, 400, n_golden)
        for f in range(nfp):
            # local test: some root at each neighbour leaves zero drift optimal at istar
            ok_up = istar == m.nB - 1
            if not ok_up:
                n, qr, Vr, Sr, cr, dr = node_solutions(Vf[f], qf[f], -1, B[istar + 1], z, y, p, dB, 100, n_golden)
                ok_up = any(_H(m, (Vr[k] - Vf[f]) / dB, qf[f], B[istar], 1, n_golden) <= Hf[f] + 1e-9 for k in range(n))
            ok_dn = istar == 0
            if ok_up and not ok_dn:
                n, qr, Vr, Sr, cr, dr = node_solutions(Vf[f], qf[f], +1, B[istar - 1], z, y, p, dB, 100, n_golden)
                ok_dn = any(_H(m, (Vf[f] - Vr[k]) / dB, qf[f], B[istar], -1, n_golden) <= Hf[f] + 1e-9 for k in range(n))
            if ok_up and ok_dn:
                # short march of `prefilter` nodes each way to discard candidates early
                if prefilter:
                    short = march_branching(m, istar, f, nq=nq, n_golden=n_golden, verbose=False, max_nodes=prefilter)
                    if not any((s["stop_up"] is None and s["stop_dn"] is None and
                                not [v for v in s["viol"] if abs(v[0] - istar) < prefilter]) for s in short):
                        continue
                if verbose:
                    print(f"candidate steady state B={B[istar]:.4f} q*={qf[f]:.4f} d*={df[f]:.4f}", flush=True)
                sols.extend(march_branching(m, istar, f, nq=nq, n_golden=n_golden, verbose=verbose))

    def usable(s):
        """Complete march with no violation at or below the steady state."""
        below = [v for v in s["viol"] if v[0] <= s["istar"]]
        return s["stop_up"] is None and s["stop_dn"] is None and not below

    good = []
    for s in sols:
        if usable(s):
            above = [v[0] for v in s["viol"] if v[0] > s["istar"]]
            s["Bhat"] = float(B[min(above)]) if above else None
            s["ihat"] = int(min(above)) if above else None
            good.append(s)
    good.sort(key=lambda s: len(s["viol"]))
    if verbose:
        print(f"{len(good)} usable solution(s) out of {len(sols)} marched candidates")
        for s in good:
            print(f"   B*={s['Bstar']:.4f} q*={s['qstar']:.4f} d*={s['dstar']:.4f}  violations {len(s['viol'])}  "
                  f"first violation above the steady state: B = {s['Bhat']}")
    return good, sols
