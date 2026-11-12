"""Replicate the exhibits.

    python run_all.py                 exhibits from the solutions in data/        -> output/
    python run_all.py --solve         recompute the solutions in data/, then the exhibits
    python run_all.py --solve --jobs N    run up to N solver jobs at once (default: number of cores, at most 8)

Logs of every step go to logs/.
"""
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)
os.makedirs(os.path.join(DATA, "pdss"), exist_ok=True)

SOLVE = "--solve" in sys.argv
JOBS = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else min(8, os.cpu_count() or 1)
THREADS = max(1, (os.cpu_count() or 1) // JOBS)


def run(name, args, out=None, env=None):
    """Run code/<args[0]> with the remaining arguments; stdout to logs/<name>.txt (and to out, if given)."""
    t0 = time.time()
    e = dict(os.environ, NUMBA_NUM_THREADS=str(THREADS), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
             OPENBLAS_NUM_THREADS="1", **(env or {}))
    log = os.path.join(LOGS, name + ".txt")
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, os.path.join(CODE, args[0])] + args[1:], cwd=CODE, env=e,
                           stdout=fh, stderr=subprocess.STDOUT)
    if out:
        shutil.copy(log, os.path.join(DATA, out))
    print(f"  {name:28s} {'ok' if p.returncode == 0 else 'FAILED (see logs/' + name + '.txt)'}  {time.time() - t0:7.0f}s",
          flush=True)
    if p.returncode:
        raise SystemExit(1)


def parallel(steps):
    """Run steps (argument tuples of run) on JOBS threads; stop at the first failure."""
    with ThreadPoolExecutor(JOBS) as ex:
        for f in [ex.submit(run, *s) for s in steps]:
            f.result()


# wedge bracket and bisection steps for the low-curvature threshold searches; s.d. lists for region.py jobs
LO = {"THR_LO": "0.0045", "THR_HI": "0.008", "THR_IT": "6"}
REGION = ["0 0.03 0.07 0.11", "0.0025 0.035 0.075 0.115", "0.005 0.04 0.08 0.12", "0.0075 0.045 0.085",
          "0.01 0.05 0.09", "0.015 0.055 0.095", "0.02 0.06 0.1", "0.025", "0.065"]

T0 = time.time()
# solutions, in three waves: each wave uses files written by the waves before it
if SOLVE:
    print(f"solutions ({JOBS} jobs, {THREADS} threads each)", flush=True)
    # wave 1: main diffusion solution, self-fulfilling regions and thresholds, illustrations, comparative statics
    parallel([("main75", ["main75.py", "9000"])]
             + [("region_" + r.split()[0], ["region.py"] + r.split()) for r in REGION] + [
        ("threshold_wide_nu0.05", ["pdss_threshold.py", "0.05", "--wide"], "pdss/threshold_wide_nu0.05.txt", LO),
        ("threshold_wide_nu0.02", ["pdss_threshold.py", "0.02", "--wide"], "pdss/threshold_wide_nu0.02.txt", LO),
        ("threshold_wide_nu0.25", ["pdss_threshold.py", "0.25", "--wide"], "pdss/threshold_wide_nu0.25.txt"),
        ("threshold_wide_nu0.1", ["pdss_threshold.py", "0.1", "--wide"], "pdss/threshold_wide_nu0.1.txt"),
        ("threshold_wide_nu0.5", ["pdss_threshold.py", "0.5", "--wide"], "pdss/threshold_wide_nu0.5.txt"),
        ("threshold_nu1.0", ["pdss_threshold.py", "1.0"], "pdss/threshold_nu1.0.txt"),
        ("threshold_nu2.0", ["pdss_threshold.py", "2.0"], "pdss/threshold_nu2.0.txt"),
        ("threshold_nu4.0", ["pdss_threshold.py", "4.0"], "pdss/threshold_nu4.0.txt"),
        ("wedge_search", ["wedge_search.py", "0.06", "200", "1.7", "illus"]),
        ("pdss_sweep", ["pdss_sweep.py", "illus", "0.06"], "pdss/sweep_0.06.txt"),
        ("illustrations_det", ["illustrations.py", "det"]),
        ("illustrations_fine", ["illustrations.py", "fine"]),
        ("illustrations_fine15", ["illustrations.py", "fine15"]),
        ("illustrations_jump", ["illustrations.py", "jump"]),
        ("compstat", ["compstat.py"]),
    ])
    # wave 2: robustness grids, no standard, saving on a grid from -0.4, impulse responses, region data
    parallel([
        ("robustness_nonuni", ["robustness.py", "nonuni"]),
        ("robustness_fine", ["robustness.py", "fine"]),
        ("robustness_wide", ["robustness.py", "wide"]),
        ("robustness_z30", ["robustness.py", "z30", "9000"]),
        ("robustness_z35", ["robustness.py", "z35", "9000"]),
        ("floorscan_0.0", ["floorscan.py", "0.0"]),
        ("saving_m04", ["run_saving.py", "-0.4", "0.005", "6000", "m04"]),
        ("irf", ["irf.py"]),
        ("collect_sf", ["collect_sf.py"]),
    ])
    # wave 3: no standard on the wider grid, saving on a grid from -1, zero-drift locus and ergodic set of the main solution
    parallel([
        ("nofloor_wide", ["nofloor_wide.py"]),
        ("saving_m10", ["run_saving.py", "-1.0", "0.005", "5000", "m10", os.path.join(DATA, "saving", "saving_m04.npz")]),
        ("cal_check", ["cal_check.py", os.path.join(DATA, "ou_main.npz"), "--md", os.path.join(DATA, "main_check.md"),
                       "--irf", os.path.join(DATA, "irf.npz")]),
    ])
    shutil.copy(os.path.join(DATA, "saving", "saving_m10.npz"), os.path.join(DATA, "ou_saving.npz"))

# exhibits from the solutions in data/, in order: later scripts read files written by earlier ones
print("exhibits", flush=True)
run("numbers_main", ["numbers_main.py"])
run("pinn_eval", ["pinn_eval.py", os.path.join(DATA, "pinn", "polish_grid.npz"), "--label", "polish",
                  "--alt", os.path.join(DATA, "pinn", "base_grid.npz")])
run("make_figures", ["make_figures.py"])
run("pessimistic_approach", ["pessimistic_approach.py"])
run("text_numbers", ["text_numbers.py"])
run("tab_compstat", ["tab_compstat.py"])
run("tab_saving", ["tab_saving.py"])
print(f"done in {time.time() - T0:.0f}s; exhibits in output/", flush=True)
