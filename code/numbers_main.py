"""numbers_main.py: LaTeX macros for the numbers of the deterministic, three-state, and main diffusion solutions
quoted in the text, written to output/numbers.tex.
Inputs: data/det_fine.npz, det_fine15.npz, jump_floor.npz, illustrations_jump.log, ou_best.npz,
ou_main.npz, main_moments.md.
Usage: python code/numbers_main.py"""
import os
import re

import numpy as np

from cuberoot_consts import second_order
from paths import DATA, OUT
from pdutil import AMR_LOSS


def read(f):
    """Contents of a text file in data/."""
    return open(os.path.join(DATA, f), encoding="utf-8").read()


def trim(s):
    """Drop trailing zeros of a plain decimal (0.100 -> 0.1)."""
    return s.rstrip("0").rstrip(".") if re.fullmatch(r"-?\d+\.\d+", s) else s


def sci(v):
    """Scientific notation in LaTeX, one significant digit."""
    e = int(np.floor(np.log10(v)))
    return f"{v / 10 ** e:.0f}\\times10^{{{e}}}"


# deterministic model: cube-root constants, last node with positive drift, first default node
fine = np.load(os.path.join(DATA, "det_fine.npz"))
fine15 = np.load(os.path.join(DATA, "det_fine15.npz"))
ob = np.load(os.path.join(DATA, "ou_best.npz"))
QF = float(ob["qf"])
Sig1, Q1 = second_order(fine, ob["x"])
B, d, S = fine["B"], fine["d"], fine["S"]
i1 = np.where(d > 1e-9)[0][0]
i0 = np.where(S > 1e-10)[0][-1]
B15, q15, d15 = fine15["B"], fine15["q"].ravel(), fine15["d"].ravel()

# three-state chain: states L, M, H and the node of largest ergodic mass in L and M
jmp = np.load(os.path.join(DATA, "jump_floor.npz"))
Bj, zj, dj, gj = jmp["B"], jmp["z"], jmp["d"], jmp["g"]
w = gj / gj.sum()
nzj = zj.size
jL, jM, jH = 0, (1 if nzj == 3 else 0), nzj - 1
kL, kM = int(np.argmax(w[:, jL])), int(np.argmax(w[:, jM]))


def first_def(j):
    """First default node of state j and the default intensity there."""
    i = np.where(dj[:, j] > 1e-9)[0]
    return (Bj[i[0]], dj[i[0], j]) if i.size else (np.nan, np.nan)


# residuals of the chain solution, main diffusion solution, and calibration loss
resq_j = float(re.search(r"residuals [0-9.e+-]+ / ([0-9.e+-]+)", read("illustrations_jump.log")).group(1))
am = np.load(os.path.join(DATA, "ou_main.npz"))
loss = float(re.search(r"^\| loss \(AMR weights\) \|[^|]*\|[^|]*\| ([0-9.]+)", read("main_moments.md"), re.M).group(1))

# macro name -> value
vals = {
    "detBpd": f"{float(fine['Bpd']):.3f}", "detdfirst": f"{float(fine['dfirst']):.2f}",
    "detbeta": f"{float(fine['beta']):.3f}", "detalpha": f"{float(fine['alpha']):.2f}",
    "detSigmaOne": f"{Sig1:.3f}", "detQOne": f"{Q1:.3f}",
    "detBstarGrid": f"{B[i0]:.3f}", "detFirstNode": f"{B[i1]:.3f}", "detdFirstNode": f"{d[i1]:.3f}",
    "detFloorB": f"{B15[np.where(q15 < QF)[0][0]]:.2f}", "detFullB": f"{B15[np.where(d15 >= 1 - 1e-9)[0][0]]:.2f}",
    "jumpSd": f"{abs(zj[0]):.2f}", "jumpEta": f"{float(jmp['eta']):.3f}", "jumpEps": f"{float(jmp['eps']):.2f}",
    "jumpFirstL": f"{first_def(jL)[0]:.3f}", "jumpFirstM": f"{first_def(jM)[0]:.3f}", "jumpFirstH": f"{first_def(jH)[0]:.3f}",
    "jumpdFirstL": f"{first_def(jL)[1]:.2f}",
    "jumpStatL": f"{Bj[kL]:.3f}", "jumpMassL": f"{w[kL, jL]:.2f}", "jumpStatM": f"{Bj[kM]:.3f}", "jumpMassM": f"{w[kM, jM]:.2f}",
    "jumpEB": f"{(w * Bj[:, None]).sum():.2f}", "jumpPdef": f"{w[dj > 1e-9].sum():.2f}", "jumpResq": sci(resq_j),
    "mainNz": str(int(am["z"].size)), "mainResV": sci(float(am["resV"])), "mainResq": sci(float(am["resq"])),
    "calLoss": f"{loss:.3f}", "calLossAMR": f"{AMR_LOSS:.3f}",
}
with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8", newline="\n") as fh:
    for k, v in vals.items():
        fh.write("\\newcommand{\\" + k + "}{" + trim(v) + "}\n")
