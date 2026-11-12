"""tab_compstat.py: Table B.1, comparative statics of the deterministic steady state, written to
output/compstat_table.tex. Input: data/compstat.npz (from compstat.py).
Usage: python code/tab_compstat.py"""
import os

import numpy as np

from compstat import LABELS
from paths import DATA, OUT

rows = np.load(os.path.join(DATA, "compstat.npz"), allow_pickle=True)["rows"]


def cell(v, k):
    """Number with k decimals, or a dash when not finite."""
    return f"{v:.{k}f}" if np.isfinite(v) else "--"


# header, then one row per parameter variation
tex = [r"\begin{tabular}{lccccccc}", r"\toprule",
       r" & \multicolumn{2}{c}{Steady state} & \multicolumn{2}{c}{First default} & & Standard & Full default \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r"Variation & $\Bpd(0,\qrf)$ & computed & $\dfirst$ & computed & $\Bfd(0,\qrf)$ & binds from & from \\",
       r"\midrule"]
for r in rows:
    tex.append(f"{LABELS.get(r['label'], r['label'])} & {r['Bpd']:.3f} & {r['Bstar']:.3f} & {r['dfirst']:.3f} & "
               f"{r['dfirst_num']:.3f} & {r['Bfd1']:.2f} & {cell(r['Bfloor'], 3)} & {cell(r['Bfull'], 2)} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "compstat_table.tex"), "w", newline="\n").write("\n".join(tex) + "\n")
