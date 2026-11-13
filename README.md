# Debt Capacity and the Anatomy of Sovereign Crises

Replication package for Sangdong Kim and Gabriel Mihalache, *Debt Capacity and the Anatomy of Sovereign Crises*.

Corresponding author: Gabriel Mihalache, [gmihalache.com](https://gmihalache.com/),
[mihalache@gmail.com](mailto:mihalache@gmail.com)

The package reproduces every figure and table of the paper and the computed numbers quoted in its text, which it
writes as LaTeX macros. It can do so at three depths:

| Command | What it does | Time |
|---|---|---|
| `python run_all.py` | builds the exhibits from the results stored in `data/` | 1 minute |
| `python run_all.py --solve` | recomputes the results at the reported calibration, then builds the exhibits | 10 hours of single-job time, of which 8 are the self-fulfilling example; jobs run in parallel |
| `python run_all.py --calibrate` | runs the calibration search, adopts its best point, then does everything `--solve` does | the search takes 6 hours on a 96-core node |

## Requirements

- Python 3.10 or later, with the packages in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- A LaTeX distribution with `pdflatex` on the `PATH` and the packages `mathpazo` (psnfss), `amsmath`, and `amssymb`.
  Matplotlib renders the figures through `pdflatex` (pgf backend), so that they use the font of the paper.

Tested with Python 3.14, numpy 2.4, scipy 1.18, numba 0.65, matplotlib 3.11, and MiKTeX on Windows 11. The
calibration search ran on Linux with Python 3.10.

## Layout

| Path | Contents |
|---|---|
| `run_all.py` | the driver |
| `code/` | all scripts, in one flat folder; `paths.py` sets the folders |
| `data/` | the reported calibration and the stored results |
| `output/` | the exhibits (created by `run_all.py`) |
| `logs/` | console output of each step (created by `run_all.py`) |

Every script starts with a description of what it computes, reads, and writes. To run one by hand:
`python code/<script>.py`.

## 1. Exhibits from the stored results

```bash
python run_all.py
```

All exhibits are written to `output/`. The `.tex` files hold table bodies, which the paper reads with `\input`, or
macros for the numbers quoted in the text.

| Exhibit | File in `output/` | Script | Main inputs in `data/` |
|---|---|---|---|
| Figure 1 | `episode_one.pdf` | `make_figures.py` | `episodes.npz` |
| Figure 2 | `regions.pdf` | `make_figures.py` | `ou_main.npz` (parameters; closed forms) |
| Figure 3 | `det_policies.pdf` | `make_figures.py` | `det_fine15.npz` |
| Figure 4 | `det_multiplicity.pdf` | `make_figures.py` | `wedge/rho0.060_illus.npz` |
| Figure 5 | `jump_policies.pdf` | `make_figures.py` | `jump_floor.npz` |
| Figure 6 | `ou_slices.pdf` | `make_figures.py` | `ou_main.npz` |
| Figure 7 | `ou_contours.pdf` | `make_figures.py` | `ou_main.npz` |
| Figure 8 | `ou_irf.pdf` | `make_figures.py` | `irf.npz` |
| Figure 9 | `episode_typical.pdf` | `make_figures.py` | `episodes.npz` |
| Figure D.1 | `pessimistic_approach.pdf` | `pessimistic_approach.py` | `ou_main.npz` (parameters; ODEs) |
| Figure E.1 | `det_cuberoot.pdf` | `make_figures.py` | `det_fine.npz`, `det_floor.npz` |
| Figure E.2 | `sf_region.pdf` | `make_figures.py` | `sfregion.npz` |
| Figure F.1 | `ou_irf_states.pdf` | `make_figures.py` | `irf_states.npz` |
| Figure F.2 | `ou_saving_density.pdf` | `make_figures.py` | `ou_saving.npz`, `ou_main.npz` |
| Figure F.3 | `ou_phase_saving.pdf` | `make_figures.py` | `ou_saving.npz`, `ou_main.npz` |
| Table 1 | `tab_calibration.tex`, `tab_calibration_fixed.tex` | `text_numbers.py` | `ou_main.npz` |
| Table 2 | `tab_fit.tex` | `text_numbers.py` | `main_moments.md` |
| Table F.1 | `tab_robfloor_rows.tex` | `text_numbers.py` | `main_row.tex`, `robustness/` |
| Table F.2 | `tab_saving_rows.tex` | `tab_saving.py` | `ou_main.npz`, `ou_saving.npz` |
| Text | `numbers.tex` | `numbers_main.py` | `ou_main.npz`, `det_fine.npz`, `jump_floor.npz` |
| Text | `numbers_text.tex` | `text_numbers.py` | most files in `data/` |

`text_numbers.py` also writes `logs/numbers_text_report.md`: every number with its source, and a check of each
statement the text makes in words (for example, that a moment changes little across grids).

## 2. Results at the reported calibration

```bash
python run_all.py --solve [--jobs N]
```

recomputes every file in `data/` and then builds the exhibits. It runs up to `N` solver jobs at once (default: the
number of cores, at most 8). The files in `data/` are overwritten.

`data/ou_main.npz` holds the reported calibration (the parameter vector `x`, the level `qf` of the underwriting
standard, and the band over which the standard takes effect) together with the main solution. The times below are
for one job on a 4-core laptop (Intel i5-1145G7).

**First group.** The main solution, and the numerical example of self-fulfilling partial default, which has its own
parameters and does not depend on the calibration.

| Command | Computes | Writes to `data/` | Time |
|---|---|---|---|
| `solve_main.py` | the main solution, 201 x 75 grid, and its moments | `ou_main.npz`, `main_moments.md`, `main_row.tex`, `main_check.md` | 1 min |
| `wedge_search.py` | the two equilibria of the example (Figure 4) | `wedge/rho0.060_illus.npz` | 10 min |
| `pdss_sweep.py` | the steady states of the example reached from zero debt | `pdss/sweep_0.06.txt` | 5 min |
| `pdss_threshold.py <nu> [--wide]` | the smallest impatience wedge with a self-fulfilling path, for eight values of risk aversion | `pdss/threshold_*.txt` | 3-35 min each |
| `region.py <sd ...>` | the self-fulfilling region with shocks, in nine jobs | `sf_region/sd*.npz` | 5.5 hours in total |
| `collect_sf.py` | collects the two panels of Figure E.2 | `sfregion.npz` | <1 min |

**Second group.** These read the main solution.

| Command | Computes | Writes to `data/` | Time |
|---|---|---|---|
| `robustness.py fine\|wide\|nz51\|nz101\|z30\|z35\|nonuni` | the main solution on other grids | `robustness/` | 1-15 min each |
| `robustness.py nostd\|nostd2` | no underwriting standard, borrowing limit at 1 and at 2 | `robustness/` | 5 and 20 min |
| `illustrations.py det\|fine\|fine15\|jump\|jumpbuf\|jump3s` | the deterministic model and the chain models | `det_*.npz`, `jump*_floor.npz`, `illustrations_*.log` | <1 min each |
| `irf.py` | impulse responses | `irf.npz`, `irf_states.npz` | <1 min |
| `episodes.py` | default episodes in a simulation | `episodes.npz` | <1 min |
| `run_saving.py` | the model with saving | `ou_saving.npz` | 2 min |
| `pessimistic_approach.py --solve` | the second equilibrium along the path from zero debt | `approach_fig_out.txt`, `grid_check_out.txt` | 2 min |

Two remarks on what to expect from a recomputation:

- The main solution is reproduced up to the solver tolerance, and the moments of Table 2 to the digits shown. The
  solver holds the upwind branch of a few nodes where two branches tie, and which branch a node ends on depends on
  the path of the iteration. `solve_main.py` starts from `data/calibration/start.npz`, not from the stored solution,
  so the bond price can differ from the stored one by up to 0.003 at individual nodes.
- Without the underwriting standard the iteration does not settle. Those two runs take a fixed number of steps, as
  the paper describes. On the wide and the non-uniform grids the iteration stops at its step limit short of the
  tolerance, at nodes that carry no mass.

## 3. Calibration

```bash
python run_all.py --calibrate [--hours H] [--workers W] [--jobs N]
```

Seven parameters are calibrated to the eleven targets of Arellano, Mateos-Planas, and Rios-Rull (2023), with their
weights. The recovery factor and the maturity are fixed at their values. The level of the underwriting standard is
not a free parameter: every evaluation picks the lowest level, on a menu with steps of 0.025, at which the sovereign
does not issue in full default on the ergodic set and no mass reaches the borrowing limit (`cal_core.py`).

The steps, each of which can also be run by hand from `code/`:

| Command | What it does | Writes to `data/calibration/` |
|---|---|---|
| `cal_cma.py --tag cma_a --seed 1 --sigma 0.5 --workers W --deadline <epoch seconds> --fresh` | population search (CMA-ES), `W` evaluations in parallel per generation | `cma_a_*` |
| `cal_cma.py --tag cma_b --seed 2 --sigma 1.0 ...` | a second search from the same start, as a check on the first; the two run side by side | `cma_b_*` |
| `cal_confirm.py` | re-solves the starting point and the best point of each search at the strict tolerances | `confirm_*` |
| `cal_report.py` | one table of everything the search found | `summary.md` |
| `cal_adopt.py` | installs the confirmed point with the smallest loss as `data/ou_main.npz` | |

Both searches start from `data/calibration/start.npz` and stop `H` hours after the start (default 5.5). The reported
search used `W = 24` evaluations per generation and two threads per evaluation, on one 96-core node, and made about
5,000 evaluations. `W` is the population size of the search, so a different `W` gives a different search. With
fewer than `4 W` cores the evaluations share cores and the search makes fewer generations in the same time.

A new search does not reproduce the reported point exactly. The search stops on a wall-clock deadline, and each
evaluation starts from the best solution found so far, so the path of a search depends on timing. It should end at
parameters and a loss close to the reported ones. `--calibrate` deletes the stored search files in
`data/calibration/` (it keeps `start.npz`) and overwrites `data/ou_main.npz`.

## Data

The package contains no empirical data. The calibration targets are the moments published by Arellano,
Mateos-Planas, and Rios-Rull (2023, *Journal of Political Economy*), copied from their Table 4 into
`code/amr_values.py` together with the weights of their loss and the parameter values of their Table 3. Every other
script imports these values from that file.

## License

Everything in this repository is dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). See `LICENSE`.
