"""cal_report.py -- one table of everything the calibration found: data/calibration/summary.md.

Reads the best point of each search (data/calibration/<tag>_best.json), the strict re-solves
(data/calibration/confirm_*.json), and the last line of each search log (<tag>_gens.md). Reports, for each point,
the standard the rule selects, the loss in the search and in the strict solve, the diagnostics, the moments next to
the data and to the model of Arellano, Mateos-Planas, and Rios-Rull (2023), and the parameters. Works on whatever
files exist, so it can run while the searches are still going.

Usage: python cal_report.py
"""
import glob
import json
import os

from amr_values import AMR_DATA, AMR_MODEL, TARGETS, AMR_LOSS
from cal_core import UNTARGETED, closer_than_amr, trail_text
from calutil import CAL, NAMES, KAPPA_FIX, DELTA_FIX

load = lambda f: json.load(open(f, encoding="utf-8"))
rows = {}
for f in sorted(glob.glob(os.path.join(CAL, "*_best.json"))):
    name = os.path.basename(f)[:-5]
    if not name.startswith("confirm_"):
        j = load(f)
        rows[name] = dict(name=name, rule=j["rule"], search=j, conf=None)
for f in sorted(glob.glob(os.path.join(CAL, "confirm_*.json"))):
    j = load(f)
    rows.setdefault(j["name"], dict(name=j["name"], rule=j["rule"], search=None, conf=None))["conf"] = j
use = lambda r: r["conf"] or r["search"]
order = sorted(rows.values(), key=lambda r: use(r)["loss"])
L = ["# Calibration search: summary\n",
     f"Recovery factor fixed at {KAPPA_FIX}, maturity at delta = {DELTA_FIX:.4f}. The standard is the smallest level on the menu "
     "that is admissible under the rule: 'top' = the solve converges and the last two debt nodes carry no mass; 'top_nofd' = the "
     "same, and no mass issues in full default. 'Search' is the loss at the tolerances of the search, 'strict' the loss of the "
     f"re-solve at the tolerances of the paper. Loss of the model column of AMR against their data: {AMR_LOSS:.4f}. 'Closer than "
     "AMR' counts the moments at least as close to the data as their model, among the 11 targets and the 6 untargeted moments.\n",
     "| point | rule | standard (search) | loss (search) | standard (strict) | loss (strict) | residuals | standard binds | "
     "issuing in full default | max B with mass | tie nodes (mass) | closer than AMR (targets) | (untargeted) | found at evaluation |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in order:
    s, c, j = r["search"], r["conf"], use(r)
    D, M = j["diag"], j["M"]
    L.append(f"| {r['name']} | {r['rule']} | " + (f"{s['level']:.3f} | {s['loss']:.4f}" if s else " | ") + " | "
             + (f"{c['level']:.3f} | {c['loss']:.4f}" if c else " | ") + f" | {j['resV']:.0e}, {j['resq']:.0e} | {D['binding_mass']:.4f} | "
             + f"{D['full_issuing_mass']:.1e} | {D['max_B_with_mass']:.3f} | {D['n_frozen']} ({D['frozen_mass']:.1e}) | "
             + f"{closer_than_amr(M, TARGETS)} of {len(TARGETS)} | {closer_than_amr(M, UNTARGETED)} of {len(UNTARGETED)} | "
             + (str(s["n"]) if s else "") + " |")
L += ["\n## Moments\n", "| moment | data | AMR model | " + " | ".join(r["name"] for r in order) + " |", "|---|---|---|" + "---|" * len(order)]
for k in TARGETS + UNTARGETED:
    L.append(f"| {k}{'' if k in TARGETS else ' (untargeted)'} | {AMR_DATA[k]:.3f} | {AMR_MODEL[k]:.3f} | "
             + " | ".join(f"{use(r)['M'][k]:.3f}" for r in order) + " |")
L += ["\n## Parameters\n", "| point | " + " | ".join(NAMES) + " | standard |", "|---|" + "---|" * (len(NAMES) + 1)]
for r in order:
    L.append(f"| {r['name']} | " + " | ".join(f"{v:.4f}" for v in use(r)["x"]) + f" | {use(r)['level']:.3f} |")
L.append("\n## Levels tried at each point (steps of each solve)\n")
for r in order:
    L.append(f"- {r['name']}: {trail_text(use(r)['trail'])}")
L.append("\n## How each search ended\n")
for f in sorted(glob.glob(os.path.join(CAL, "*_gens.md"))):
    last = [s for s in open(f, encoding="utf-8").read().splitlines() if s.startswith("stopped")]
    L.append(f"- {os.path.basename(f)}: {last[-1] if last else 'no final line (still running, or cut off)'}")
open(os.path.join(CAL, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
