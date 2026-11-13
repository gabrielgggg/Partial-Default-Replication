"""cal_adopt.py -- install the best confirmed point of the calibration as the main solution.

Reads the strict re-solves of cal_confirm.py (data/calibration/confirm_*.json) and takes the one with the smallest
strict loss among those that converged. Copies its solution, data/calibration/confirm_<name>.npz, to
data/ou_main.npz. Run solve_main.py afterwards to write the report of the main solution.

Usage: python cal_adopt.py
"""
import glob
import json
import os
import shutil

from calutil import CAL
from paths import DATA

points = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(CAL, "confirm_*.json")))]
points = [j for j in points if j["diag"]["converged"]]
if not points:
    raise SystemExit("no confirmed point that converged in data/calibration/")
best = min(points, key=lambda j: j["loss"])
shutil.copy(os.path.join(CAL, f"confirm_{best['name']}.npz"), os.path.join(DATA, "ou_main.npz"))
print(f"main solution set from confirm_{best['name']}.npz: strict loss {best['loss']:.4f}, standard {best['level']:.3f}")
