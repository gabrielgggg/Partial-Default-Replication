"""cal_cma.py -- population search of the calibration loss (CMA-ES), with the evaluations of a generation in parallel.

Searches the seven calibrated parameters (calutil.FREE_NAMES), in units of cal_core.SCALE, from the parameters of
data/calibration/start.npz. The recovery factor and the maturity are fixed, and the standard is chosen inside every evaluation
by the rule --rule (cal_core.evaluate). Each generation draws one point per worker, solves them in parallel, and
updates the mean, the step size, and the covariance of the search distribution (Hansen's (mu/mu_w, lambda)-CMA-ES).
Every evaluation warm-starts from the solution at the best point so far, at that point's standard.

Stops a run when the search distribution has collapsed (--xtol) or the best value has stopped improving (--ftol
over --stall generations), then restarts from the best point with a wider step. Stops for good at --deadline
(epoch seconds): no generation starts that would not finish in time.

Resumes from data/calibration/<tag>_state.npz when it exists (pass --fresh to ignore it).
Reads data/calibration/start.npz. Writes, in data/calibration/: <tag>_evals.md (one line per evaluation),
<tag>_gens.md (one line per generation), <tag>_best.{npz,md,json} (the best point), and <tag>_state.npz.

Usage:
  python cal_cma.py --tag cma_a --workers 24 --seed 1 --sigma 0.5 --deadline <epoch seconds>
  python cal_cma.py --selftest          checks the optimizer on a test function, without the model
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np


class CMA:
    """(mu/mu_w, lambda)-CMA-ES for minimization, ask and tell."""

    def __init__(self, n, lam, sigma, mean=None):
        self.n, self.lam = n, lam
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
        me = 1.0 / (w ** 2).sum()
        self.mu, self.w, self.mueff = mu, w, me
        self.cs = (me + 2) / (n + me + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.reset(np.zeros(n) if mean is None else mean, sigma)

    def reset(self, mean, sigma):
        """Start a run at `mean` with step `sigma` and the identity covariance."""
        n = self.n
        self.m, self.sigma, self.C = np.array(mean, float), float(sigma), np.eye(n)
        self.pc, self.ps, self.g = np.zeros(n), np.zeros(n), 0
        self._eig()

    def _eig(self):
        """Eigendecomposition of the covariance, and its inverse square root."""
        self.C = (self.C + self.C.T) / 2
        d, B = np.linalg.eigh(self.C)
        self.D, self.B = np.sqrt(np.maximum(d, 1e-20)), B
        self.invsqrt = (B / self.D) @ B.T

    def ask(self, rng, k=None):
        """Draw a generation (or k points) from the search distribution."""
        Z = rng.standard_normal((self.lam if k is None else k, self.n))
        return self.m + self.sigma * (Z * self.D) @ self.B.T

    def tell(self, Y, f):
        """Update the mean, the evolution paths, the covariance, and the step from the values f of the points Y."""
        idx = np.argsort(f, kind="stable")[: self.mu]
        m_old, Ysel = self.m, Y[idx]
        self.m = self.w @ Ysel
        dm = (self.m - m_old) / self.sigma
        cs, cc, me = self.cs, self.cc, self.mueff
        self.ps = (1 - cs) * self.ps + np.sqrt(cs * (2 - cs) * me) * (self.invsqrt @ dm)
        self.g += 1
        hs = float(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - cs) ** (2 * self.g)) < (1.4 + 2 / (self.n + 1)) * self.chi)
        self.pc = (1 - cc) * self.pc + hs * np.sqrt(cc * (2 - cc) * me) * dm
        art = (Ysel - m_old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * cc * (2 - cc) * self.C)
                  + self.cmu * (art.T * self.w) @ art)
        self.sigma = float(np.clip(self.sigma * np.exp((cs / self.ds) * (np.linalg.norm(self.ps) / self.chi - 1)), 1e-8, 20.0))
        self._eig()

    def spread(self):
        """Largest standard deviation of the search distribution along a coordinate."""
        return float(self.sigma * np.sqrt(np.diag(self.C).max()))

    def state(self):
        """The search distribution, for the state file."""
        return dict(m=self.m, sigma=self.sigma, C=self.C, pc=self.pc, ps=self.ps, g=self.g)

    def load(self, s):
        """Restore the search distribution from a state file."""
        self.m, self.sigma, self.C = s["m"].astype(float), float(s["sigma"]), s["C"].astype(float)
        self.pc, self.ps, self.g = s["pc"].astype(float), s["ps"].astype(float), int(s["g"])
        self._eig()


def selftest():
    """The optimizer on a rotated ill-conditioned quadratic in nine dimensions, with noise."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((9, 9)))
    target = rng.uniform(-3, 3, 9)
    f = lambda y: float(((Q @ (y - target)) ** 2 * np.logspace(0, 3, 9)).sum()) + 1e-4 * rng.standard_normal()
    es = CMA(9, 22, 0.5)
    for g in range(400):
        Y = es.ask(rng)
        es.tell(Y, np.array([f(y) for y in Y]))
        if es.spread() < 1e-3:
            break
    err = float(np.abs(es.m - target).max())
    print(f"selftest: {g + 1} generations, distance to the minimum {err:.1e}, step {es.sigma:.1e}")
    assert err < 0.05, "the optimizer did not find the minimum of the test function"
    print("selftest ok")


def main():
    """Run or resume one search."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cma")
    ap.add_argument("--rule", default="top_nofd", choices=["top", "top_nofd"])
    ap.add_argument("--workers", type=int, default=24, help="parallel evaluations = points per generation")
    ap.add_argument("--sigma", type=float, default=0.5, help="initial step, in search units")
    ap.add_argument("--xtol", type=float, default=0.02, help="converged when the widest coordinate s.d. is below this")
    ap.add_argument("--ftol", type=float, default=5e-4)
    ap.add_argument("--stall", type=int, default=20)
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none")
    ap.add_argument("--gen-guess", type=float, default=240.0, help="seconds assumed for a generation before one is timed")
    ap.add_argument("--maxgen", type=int, default=10 ** 6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nz", type=int, default=75)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--years", type=int, default=25000)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from cal_core import SCALE, BAD, CAL, START, FREE, FREE_NAMES, Book, evaluate, feasible, full_x
    tag, lam, n = args.tag, args.workers, len(SCALE)
    a0 = np.load(START)
    x0 = a0["x"].astype(float)[FREE]
    es = CMA(n, lam, args.sigma)
    f_state, f_gens = os.path.join(CAL, f"{tag}_state.npz"), os.path.join(CAL, f"{tag}_gens.md")
    gen, restarts, hist = 0, 0, []
    resumed = (not args.fresh) and os.path.exists(f_state)
    if resumed:
        s = np.load(f_state)
        assert np.allclose(s["x0"], x0) and int(s["lam"]) == lam, "the saved state belongs to another start or population size"
        es.load(s); gen, restarts, hist = int(s["gen"]), int(s["restarts"]), list(s["hist"])
    book = Book(tag, f"rule '{args.rule}'; nz {args.nz}; population {lam}; calibrated: {FREE_NAMES}; x0 = {x0.tolist()}; "
                f"scale = {SCALE.tolist()}; fixed: {dict(zip(('kappa', 'delta'), full_x(x0)[[5, 6]].round(6).tolist()))}", fresh=args.fresh)
    if book.warm is None:                            # nothing found yet: start from the start file, at its standard
        book.warm, book.level = (a0["V"], a0["q"], a0["z"]), float(a0["qf"])

    def say(s):
        """Print a line and append it to the log of generations."""
        print(s, flush=True)
        with open(f_gens, "a", encoding="utf-8") as fh:
            fh.write(s + "\n")

    say(f"\n# {tag}: {'resumed at generation ' + str(gen) if resumed else 'new search'}; rule {args.rule}; "
        f"population {lam}; step {es.sigma:.3f}; deadline {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(args.deadline)) if args.deadline else 'none'}\n")
    say("| gen | evals | best of generation | median | failed | best so far | its standard | step | widest s.d. | axis ratio | seconds |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    base = dict(rule=args.rule, nz=args.nz, paths=args.paths, years=args.years)
    make_pool = lambda: ProcessPoolExecutor(max_workers=lam, mp_context=get_context("spawn"))
    pool, times, why = make_pool(), [], "maxgen"
    try:
        while gen < args.maxgen:
            need = 1.15 * max(times[-3:]) + 30 if times else args.gen_guess
            if args.deadline and time.time() + need > args.deadline:
                why = "deadline"; break
            t0 = time.time()
            rng = np.random.default_rng([args.seed, restarts, gen])
            Y = es.ask(rng)
            if es.g == 0 and gen == 0:
                Y[0] = es.m                                  # the first generation includes the starting point itself
            for k in range(lam):                             # redraw points outside the parameter bounds
                for _ in range(200):
                    if feasible(full_x(x0 + SCALE * Y[k])):
                        break
                    Y[k] = es.ask(rng, 1)[0]
            tasks = [dict(base, x=full_x(x0 + SCALE * y), warm=book.warm, level=book.level) for y in Y]
            try:
                results = list(pool.map(evaluate, tasks))
            except Exception as e:                           # a dead worker breaks the pool: rebuild it and repeat once
                say(f"generation {gen}: pool failed ({e!r}); rebuilding and repeating")
                pool.shutdown(wait=False, cancel_futures=True); pool = make_pool()
                results = list(pool.map(evaluate, tasks))
            f = np.array([book.record(r, gen) for r in results])
            es.tell(Y, f)
            gen += 1; times.append(time.time() - t0); hist.append(float(f.min()))
            ok = f[f < BAD]
            say(f"| {gen} | {book.n} | {f.min():.4f} | {np.median(ok) if ok.size else np.nan:.4f} | {int((f >= BAD).sum())} | "
                f"{book.best[0]:.4f} | {book.level:.3f} | {es.sigma:.3f} | {es.spread():.3f} | "
                f"{es.D.max() / es.D.min():.1f} | {times[-1]:.0f} |")
            # convergence of this run: the distribution has collapsed, or the best value has stalled
            reason = ""
            if es.spread() < args.xtol:
                reason = f"widest s.d. {es.spread():.3f} below {args.xtol}"
            elif es.g >= 2 * args.stall and np.median(hist[-args.stall:]) > np.median(hist[-2 * args.stall: -args.stall]) - args.ftol:
                reason = f"best value did not improve by {args.ftol} over {args.stall} generations"
            elif es.D.max() / es.D.min() > 1e7:
                reason = "covariance ill-conditioned"
            if reason:
                restarts += 1
                step = min(2.0, args.sigma * 2 ** restarts)
                say(f"\nconverged after {es.g} generations of this run ({reason}); restart {restarts} from the best point, step {step}\n")
                es.reset(np.zeros(n) if book.x is None else (book.x[FREE] - x0) / SCALE, step); hist = []
            np.savez(f_state, x0=x0, lam=lam, gen=gen, restarts=restarts, hist=np.array(hist), **es.state())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    say(f"\nstopped ({why}) after {gen} generations, {book.n} evaluations, {restarts} restarts; best loss {book.best[0]:.4f} "
        f"(evaluation {book.best[1]}), standard {book.level}")


if __name__ == "__main__":
    main()
