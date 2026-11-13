"""cal_confirm.py -- re-solve the best points of the searches at the tolerances of the paper.

Candidates: the start point (data/calibration/start.npz), and data/calibration/<tag>_best.{npz,json} of every search.
Each is evaluated again with the strict tolerances (cal_core.STRICT), starting from its own saved solution and
standard, so the smallest admissible standard is looked for again as well. Writes
data/calibration/confirm_<name>.{npz,md,json} as the points finish.

Usage:
  python cal_confirm.py --count          number of candidates
  python cal_confirm.py --workers 4
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from calutil import CAL, NAMES, save_json


def candidates():
    """(name, npz path, rule) of the points to confirm."""
    from cal_core import START, RULE
    out = [("start", START, RULE)]
    for f in sorted(glob.glob(os.path.join(CAL, "*_best.npz"))):
        name = os.path.basename(f)[:-4]
        if name.startswith("confirm_"):
            continue
        out.append((name, f, json.load(open(f[:-4] + ".json", encoding="utf-8"))["rule"]))
    return out


def main():
    """Evaluate the candidates in parallel and write each result as it finishes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nz", type=int, default=75)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--years", type=int, default=25000)
    ap.add_argument("--count", action="store_true")
    args = ap.parse_args()
    cand = candidates()
    if args.count:
        print(len(cand)); return
    from cal_core import evaluate, trail_text, BAND, LBAND
    tasks = {}
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(cand))), mp_context=get_context("spawn")) as pool:
        for name, f, rule in cand:
            a = np.load(f)
            t = dict(x=a["x"].astype(float), rule=rule, nz=args.nz, paths=args.paths, years=args.years, strict=True,
                     level=float(a["qf"]), warm=(a["V"], a["q"], a["z"]))
            tasks[pool.submit(evaluate, t)] = (name, f, rule)
        for fut in as_completed(tasks):
            name, f, rule = tasks[fut]
            r = fut.result()
            if not r["ok"]:
                print(f"{name}: failed: {r['err']}", flush=True); continue
            base = os.path.join(CAL, "confirm_" + name)
            np.savez(base + ".npz", **r["sol"], x=r["x"], names=np.array(NAMES), qf=r["level"], band=BAND, lband=LBAND,
                     resV=r["resV"], resq=r["resq"], loss=r["loss"])
            with open(base + ".md", "w", encoding="utf-8") as fh:
                fh.write(f"# {name}: strict solve, loss {r['loss']:.4f}, standard {r['level']:.3f} (rule '{rule}'), residuals "
                         f"{r['resV']:.1e}, {r['resq']:.1e}\n\n| " + " | ".join(NAMES) + " |\n|" + "---|" * len(NAMES) + "\n| "
                         + " | ".join(f"{v:.4f}" for v in r["x"]) + " |\n\n" + r["table"] + "\n\n" + r["diag"]
                         + f"\n\nlevels tried (steps): {trail_text(r['trail'])}\n")
            j = json.load(open(f[:-4] + ".json", encoding="utf-8")) if os.path.exists(f[:-4] + ".json") else {}
            save_json(base + ".json", dict(name=name, source=os.path.basename(f), rule=rule, loss=r["loss"], level=r["level"],
                                           search_loss=j.get("loss"), search_level=j.get("level"), n=j.get("n"),
                                           x=list(map(float, r["x"])), M=r["M"], resV=r["resV"], resq=r["resq"],
                                           diag=r["D"], trail=r["trail"]))
            print(f"{name}: loss {r['loss']:.4f}, standard {r['level']:.3f}"
                  + (f" (search: {j['loss']:.4f}, {j['level']:.3f})" if j else "") + f", {r['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
