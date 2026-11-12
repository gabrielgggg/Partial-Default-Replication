Replication package for
Sangdong Kim and Gabriel Mihalache, "Debt Capacity and the Anatomy of Sovereign Crises"

Corresponding author: Gabriel Mihalache, https://gmihalache.com/, mihalache@gmail.com

The package reproduces every figure and table in the paper, and the computed numbers quoted in its text (written as
LaTeX macros). All results use the final calibration, stored in data/ou_best.npz. The calibration search itself is
not part of the package.


REQUIREMENTS

- Python 3.10 or later, with the packages in requirements.txt:
      pip install -r requirements.txt
- A LaTeX distribution with pdflatex on the PATH and the packages mathpazo (psnfss), amsmath, and amssymb. Matplotlib
  renders the figures through pdflatex (pgf backend) so that they use the paper's font.
- Optional, only for retraining the neural network (see below): PyTorch and a CUDA GPU.

Tested with Python 3.14, numpy 2.4, scipy 1.18, numba 0.65, matplotlib 3.11, and MiKTeX on Windows 11.


RUNNING

    python run_all.py

builds all exhibits from the solutions stored in data/ and writes them to output/. It takes about 2 minutes.

    python run_all.py --solve [--jobs N]

first recomputes every solution in data/ from the calibrated parameters, then builds the exhibits. It runs up to N
solver jobs at once (default: the number of cores, at most 8). It takes about 4 hours on a 4-core laptop, and about
1 hour on a machine with 16 or more cores. The stored files in data/ are overwritten.

Each step writes its console output to logs/<step>.txt.


LAYOUT

    run_all.py      the driver
    code/           all scripts; paths.py sets the folders
    data/           calibrated parameters and stored solutions
    output/         the exhibits (created by run_all.py)
    logs/           console output of each step (created by run_all.py)

The package is dedicated to the public domain (CC0 1.0); see LICENSE.


EXHIBITS

All figures and tables are written to output/. The .tex files hold table bodies (\input by the paper) or LaTeX
macros for the numbers quoted in the text.

    Exhibit      File                       Script                   Main inputs (data/)
    Figure 1     anatomy.pdf                make_figures.py          ou_main.npz, irf.npz (computed)
    Figure 2     regions.pdf                make_figures.py          ou_main.npz (parameters; closed forms)
    Figure 3     det_policies.pdf           make_figures.py          det_fine15.npz
    Figure 4     det_value.pdf              make_figures.py          det_fine15.npz
    Figure 5     det_multiplicity.pdf       make_figures.py          wedge/rho0.060_illus.npz
    Figure 6     ou_slices.pdf              make_figures.py          ou_main.npz
    Figure 7     ou_contours.pdf            make_figures.py          ou_main.npz
    Figure 8     ou_irf.pdf                 make_figures.py          ou_main.npz
    Figure 9     ou_irf_states.pdf          make_figures.py          ou_main.npz
    Figure A.1   pessimistic_approach.pdf   pessimistic_approach.py  (closed forms and ODEs)
    Figure B.1   jump_policies.pdf          make_figures.py          jump_floor.npz
    Figure B.2   det_cuberoot.pdf           make_figures.py          det_fine.npz, det_floor.npz
    Figure B.3   ou_pinn.pdf                make_figures.py          pinn_main.npz (from pinn_eval.py)
    Figure B.4   sf_region.pdf              make_figures.py          sfregion.npz
    Figure C.1   ou_saving_density.pdf      make_figures.py          ou_saving.npz, ou_main.npz
    Table 1      tab_calibration.tex        text_numbers.py          ou_best.npz, main_moments.md
    Table 2      tab_fit.tex                text_numbers.py          main_moments.md
    Table B.1    compstat_table.tex         tab_compstat.py          compstat.npz
    Table B.2    tab_pinn_rows.tex          pinn_eval.py             pinn/polish_grid.npz, pinn/base_grid.npz
    Table C.1    tab_robfloor_rows.tex      text_numbers.py          main_row.tex, rob75/
    Table C.2    tab_saving_rows.tex        tab_saving.py            ou_main.npz, ou_saving.npz
    Text         numbers.tex                numbers_main.py          det_*.npz, jump_floor.npz, ou_main.npz
    Text         numbers_text.tex           text_numbers.py          most files in data/
    Text         numbers_pinn.tex           pinn_eval.py             pinn/

To run one script by hand: python code/<script>.py. The exhibit scripts need only the files in data/, except
text_numbers.py, which also reads output/numbers.tex and data/irf.npz (written by numbers_main.py and
make_figures.py).


SOLUTIONS (run_all.py --solve)

The approximate times are for one job on a 4-core laptop (Intel i5-1145G7). Steps in the same group run in
parallel.

Group 1, from the calibrated parameters (data/ou_best.npz)
    main75.py 9000                  main solution, 201 x 75 grid          ou_main.npz, main_moments.md, main_row.tex  25 min
    illustrations.py det|fine|fine15|jump
                                    deterministic and three-state models  det_floor, det_fine, det_fine15,
                                                                          jump_floor (.npz), illustrations_*.log      1-2 min each
    compstat.py                     comparative statics                   compstat.npz, compstat.log                  5 min
    wedge_search.py 0.06 200 1.7 illus
                                    self-fulfilling example (Figure 5)    wedge/rho0.060_illus.npz                    10 min
    pdss_sweep.py illus 0.06        forward sweep of the same example     pdss/sweep_0.06.txt                         3 min
    pdss_threshold.py <nu> [--wide] smallest wedge with a self-fulfilling
                                    path, for nu in 0.02 ... 4            pdss/threshold_*.txt                        3-35 min each
    region.py <sd ...>              self-fulfilling region with shocks    sf_region/sd*.npz                           5.5 hours in total,
                                                                                                                      in 9 jobs

Group 2
    robustness.py fine|wide|nonuni|z30|z35
                                    main solution on other grids          rob75/                                      20-35 min each
    floorscan.py 0.0                no underwriting standard              floorcf/qf0.00.*                            20 min
    run_saving.py -0.4 0.005 6000 m04
                                    saving allowed, debt grid from -0.4   saving/saving_m04.npz                       17 min
    irf.py                          impulse responses                     irf.npz                                     <1 min
    collect_sf.py                   collects the self-fulfilling region   sfregion.npz                                <1 min

Group 3
    nofloor_wide.py                 no standard, debt grid to 2           floorcf/qf0.00_wide2.*                      35 min
    run_saving.py -1.0 0.005 5000 m10 data/saving/saving_m04.npz
                                    saving allowed, debt grid from -1     saving/saving_m10.npz -> ou_saving.npz      17 min
    cal_check.py                    zero-drift locus and ergodic set      main_check.md                               <1 min

The package contains no empirical data. The calibration targets are the moments published by Arellano, Mateos-Planas,
and Rios-Rull (2023), copied from their Table 4 into code/amr_values.py together with the weights of their loss. The
other scripts import these values from that file.


NEURAL NETWORK (Figure B.3, Table B.2)

run_all.py evaluates the stored network solutions data/pinn/polish_grid.npz and data/pinn/base_grid.npz. To retrain
them on a GPU (each run about 30-45 minutes on one A100), from the folder code/:

    COMMON="--ref ../data/ou_main.npz --inner 50 --dt 0.02 --dt_max 0.2 --threads 8 --out pinn_out --device cuda:0 --width 128 --depth 3"
    python pinn_solve.py $COMMON --outer 2500 --n 16384 --seed 0 --tag base
    python pinn_solve.py $COMMON --outer 2500 --n 16384 --seed 0 --near 0.25 --tag near
    python pinn_solve.py $COMMON --outer 4500 --n 32768 --seed 2 --near 0.25 --init pinn_out/near_ckpt.pt --lr 1e-3 --lr_end 3e-5 --polish_from 0.75 --tag polish

then copy pinn_out/base_grid.npz and pinn_out/polish_grid.npz to data/pinn/. Training on a GPU is not bit-for-bit
reproducible, so a retrained network gives numbers close to, but not identical to, those in the paper.
