"""Replicate the exhibits of "Debt Capacity and the Anatomy of Sovereign Crises".

    python run_all.py               exhibits from the results stored in data/            -> output/
    python run_all.py --solve       recompute the results in data/ at the reported calibration, then the exhibits
    python run_all.py --calibrate   run the calibration search, adopt its best point, then as --solve

Options:
    --jobs N      solver jobs run at once in --solve (default: the number of cores, at most 8)
    --hours H     length of the calibration search in hours (default 5.5, as reported)
    --workers W   points per generation of each of the two searches (default 24, as reported)

The console output of every step goes to logs/<step>.txt. --solve and --calibrate overwrite the files in data/.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
CAL = os.path.join(DATA, "calibration")
CORES = os.cpu_count() or 1

ap = argparse.ArgumentParser(description="Replicate the exhibits of the paper.")
ap.add_argument("--solve", action="store_true", help="recompute the results in data/ first")
ap.add_argument("--calibrate", action="store_true", help="run the calibration search first, then as --solve")
ap.add_argument("--jobs", type=int, default=min(8, CORES), help="solver jobs run at once")
ap.add_argument("--hours", type=float, default=5.5, help="length of the calibration search")
ap.add_argument("--workers", type=int, default=24, help="points per generation of each search")
args = ap.parse_args()
os.makedirs(LOGS, exist_ok=True)


def run(name, script, *script_args, threads=None):
    """Run code/<script> with its arguments, in code/, with stdout and stderr in logs/<name>.txt. Stop on failure."""
    t0 = time.time()
    threads = threads or max(1, CORES // args.jobs)
    env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    with open(os.path.join(LOGS, name + ".txt"), "w") as log:
        code = subprocess.run([sys.executable, os.path.join(CODE, script), *map(str, script_args)], cwd=CODE, env=env,
                              stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"  {name:26s} {'ok' if code == 0 else 'FAILED, see logs/' + name + '.txt'}  {time.time() - t0:7.0f}s", flush=True)
    if code:
        raise SystemExit(1)


def parallel(steps, jobs=None):
    """Run the steps (argument tuples of run) at most `jobs` at a time. Stop at the first failure."""
    with ThreadPoolExecutor(jobs or args.jobs) as pool:
        for future in [pool.submit(run, *step) for step in steps]:
            future.result()


def calibrate():
    """Two population searches from data/calibration/start.npz until the deadline, the strict re-solves of their
    best points, the summary, and the adoption of the best confirmed point as data/ou_main.npz."""
    threads = max(1, min(2, CORES // (2 * args.workers)))
    print(f"calibration: 2 searches x {args.workers} evaluations x {threads} threads, {args.hours} hours", flush=True)
    for f in glob.glob(os.path.join(CAL, "*")):          # results of an earlier search; the starting point stays
        if os.path.basename(f) != "start.npz":
            os.remove(f)
    deadline = int(time.time() + 3600 * args.hours)
    search = lambda tag, seed, sigma: (tag, "cal_cma.py", "--tag", tag, "--workers", args.workers, "--deadline",
                                       deadline, "--seed", seed, "--sigma", sigma, "--fresh")
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(run, *search("cma_a", 1, 0.5), threads=threads),
                       pool.submit(run, *search("cma_b", 2, 1.0), threads=threads)]:
            future.result()
    run("cal_confirm", "cal_confirm.py", "--workers", 3, threads=max(1, CORES // 3))
    run("cal_report", "cal_report.py")
    run("cal_adopt", "cal_adopt.py")


# the nine jobs of the self-fulfilling region with shocks: lists of standard deviations of the log endowment
REGION = ["0 0.03 0.07 0.11", "0.0025 0.035 0.075 0.115", "0.005 0.04 0.08 0.12", "0.0075 0.045 0.085",
          "0.01 0.05 0.09", "0.015 0.055 0.095", "0.02 0.06 0.1", "0.025", "0.065"]


def solve(evaluate_only):
    """Every result in data/. The main solution and the self-fulfilling example (which has its own parameters) come
    first; the steps of the second group read the main solution."""
    print(f"solutions ({args.jobs} jobs at a time)", flush=True)
    main = ("solve_main", "solve_main.py", "--evaluate") if evaluate_only else ("solve_main", "solve_main.py")
    parallel([main,
              ("wedge_search", "wedge_search.py"),
              ("pdss_sweep", "pdss_sweep.py")]
             + [(f"threshold_nu{nu}", "pdss_threshold.py", nu) for nu in ("1.0", "2.0", "4.0")]
             + [(f"threshold_wide_nu{nu}", "pdss_threshold.py", nu, "--wide") for nu in ("0.5", "0.25", "0.1", "0.05", "0.02")]
             + [("region_" + sds.split()[0], "region.py", *sds.split()) for sds in REGION])
    run("collect_sf", "collect_sf.py")
    parallel([(f"robustness_{case}", "robustness.py", case, maxit) for case, maxit in
              [("fine", 8000), ("wide", 8000), ("nz51", 8000), ("nz101", 8000), ("z30", 6000), ("z35", 6000),
               ("nonuni", 6000), ("nostd", 8000), ("nostd2", 8000)]]
             + [(f"illustrations_{mode}", "illustrations.py", mode) for mode in ("det", "fine", "fine15", "jump", "jumpbuf", "jump3s")]
             + [("irf", "irf.py"),
                ("run_saving", "run_saving.py"),
                ("approach_solve", "pessimistic_approach.py", "--solve"),
                ("episodes", "episodes.py")])


def exhibits():
    """Figures, tables, and number macros from the results in data/, into output/."""
    print("exhibits", flush=True)
    run("numbers_main", "numbers_main.py")
    run("make_figures", "make_figures.py")
    run("pessimistic_approach", "pessimistic_approach.py")
    run("text_numbers", "text_numbers.py")
    run("tab_saving", "tab_saving.py")


T0 = time.time()
if args.calibrate:
    calibrate()
if args.solve or args.calibrate:
    solve(evaluate_only=args.calibrate)      # after a search the adopted point is already a strict solve
exhibits()
print(f"done in {time.time() - T0:.0f}s; exhibits in output/", flush=True)
