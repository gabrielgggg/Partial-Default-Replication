"""Solver modules for the model with saving, built from pdmodel.py and pdtimestep.py.

build() copies the two solver modules and applies the changes below, each marked SAVING in the copies, so that
the model with saving always runs on the same solver as the main solution:
  pdmodel:    at B <= 0 there is nothing to default on (d = 0); a backward step at B = 0 moves into savings at the
              risk-free price; the interior default intensity uses max(B, 0).
  pdtimestep: the pricing equation lives on B >= 0 only: its generator has no drift below zero debt and no backward
              drift at zero debt, its residual is taken on B >= 0, and q = 1 below zero debt.
Each change replaces a code fragment that must occur exactly once in the source module.

Reads code/pdmodel.py and code/pdtimestep.py. Writes code/pdmodel_sav.py and code/pdtimestep_sav.py.
Usage: imported by run_saving.py (saving_solver.build()), or python code/saving_solver.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PDMODEL = [
    ("    Hb, Sb, cb = best_S(0.0, VB, q, B, z, y, p, mode)\n    if not np.isfinite(Hb):\n        Hb = _NEG\n    db = 0.0\n",
     "    Hb, Sb, cb = best_S(0.0, VB, q, B, z, y, p, mode)\n"
     "    if B <= 1e-12:                       # SAVING: no debt, nothing to default on\n"
     "        if not np.isfinite(Hb):\n"
     "            Hb = _NEG\n"
     "        return Hb, Sb, cb, 0.0\n"
     "    if not np.isfinite(Hb):\n        Hb = _NEG\n    db = 0.0\n"),
    ("            H, S_, c_, d_ = best_node(VBb, qq, B, z, y, p, -1, n_golden)\n",
     "            qb = 1.0 if B <= 1e-12 else qq   # SAVING: moving into or within savings at the risk-free price\n"
     "            H, S_, c_, d_ = best_node(VBb, qb, B, z, y, p, -1, n_golden)\n"),
    ("            dint = ((1 - par.kappa) * (par.delta + par.lam) * BB / (par.g0 * par.g1 * Ff * yy))",
     "            dint = ((1 - par.kappa) * (par.delta + par.lam) * np.maximum(BB, 0.0) / (par.g0 * par.g1 * Ff * yy))"),
]
PDTIMESTEP = [
    ("from pdmodel import ", "from pdmodel_sav import "),
    ("        rq = float(np.max(np.abs(xi * qf - (1.0 - df) * coupon - A @ qf)))\n",
     "        # SAVING: pricing generator on B >= 0 only; at B = 0 the bond stays outstanding\n"
     "        neg = model.B < -1e-12\n"
     "        i0 = int(np.argmin(np.abs(model.B)))\n"
     "        Sq = S.copy(); Sq[neg, :] = 0.0; Sq[i0, :] = np.maximum(S[i0, :], 0.0)\n"
     "        Aq = model.assemble(Sq)\n"
     "        rq_all = np.abs(xi * qf - (1.0 - df) * coupon - Aq @ qf).reshape(model.nB, model.nz)\n"
     "        rq = float(np.max(rq_all[~neg, :]))\n"),
    ("            qn, n2 = model.solve_shifted(xi + 1.0 / dt, S, (1.0 - df) * coupon + qf / dt, q, A)\n",
     "            qn, n2 = model.solve_shifted(xi + 1.0 / dt, Sq, (1.0 - df) * coupon + qf / dt, q, Aq)\n"),
    ("            qn = spla.spsolve(sp.diags(xi + 1.0 / dt, format=\"csc\") - A, (1.0 - df) * coupon + qf / dt)\n"
     "        q = np.clip(qn, 1e-12, 1.0).reshape(model.nB, model.nz)\n",
     "            qn = spla.spsolve(sp.diags(xi + 1.0 / dt, format=\"csc\") - Aq, (1.0 - df) * coupon + qf / dt)\n"
     "        q = np.clip(qn, 1e-12, 1.0).reshape(model.nB, model.nz)\n"
     "        q[neg, :] = 1.0\n"),
]


def patch(src, dst, pairs):
    """Write code/<dst> as code/<src> with each (fragment, replacement) pair applied; unchanged files are kept."""
    s = open(os.path.join(HERE, src), encoding="utf-8").read()
    for a, b in pairs:
        if s.count(a) != 1:
            raise RuntimeError(f"{src}: fragment found {s.count(a)} times, expected once:\n{a}")
        s = s.replace(a, b)
    s = f"# Built by saving_solver.py from {src}; do not edit.\n" + s
    f = os.path.join(HERE, dst)
    if not os.path.exists(f) or open(f, encoding="utf-8").read() != s:      # keep the compiled cache when unchanged
        open(f, "w", encoding="utf-8", newline="\n").write(s)


def build():
    """Write pdmodel_sav.py and pdtimestep_sav.py next to this file."""
    patch("pdmodel.py", "pdmodel_sav.py", PDMODEL)
    patch("pdtimestep.py", "pdtimestep_sav.py", PDTIMESTEP)


if __name__ == "__main__":
    build()
    print("wrote pdmodel_sav.py and pdtimestep_sav.py")
