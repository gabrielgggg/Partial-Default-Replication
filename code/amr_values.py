"""amr_values.py -- the values taken from Arellano, Mateos-Planas, and Rios-Rull (2023, JPE).

Every number below is copied from their published paper: the data column and the model column of their Table 4,
the weights of their footnote 22, and the number of digits they print. Nothing in this file is computed from data.
Scripts that need one of these values import it from here.
"""
# Table 4, data column, in the units of pdmoments.moments
AMR_DATA = dict(pd_freq=0.36, pd_mean_cond=0.38, pd_sd_cond=0.22, debt_y_mean=0.32, debt_y_sd=0.18,
                ds_y_mean=0.036, ds_y_sd=0.021, dd_y_mean=0.049, spread_sd=0.041, y_persist=0.89, y_sd=0.10,
                dc_y_mean_cond=0.052, dc_y_sd_cond=0.064, spread_mean=0.053, corr_spread_y=-0.17,
                corr_spread_debt=0.24, c_sd_rel=1.0)
# Table 4, model column
AMR_MODEL = dict(pd_freq=0.37, pd_mean_cond=0.39, pd_sd_cond=0.19, debt_y_mean=0.32, debt_y_sd=0.25,
                 ds_y_mean=0.035, ds_y_sd=0.022, dd_y_mean=0.05, spread_sd=0.037, y_persist=0.88, y_sd=0.12,
                 dc_y_mean_cond=0.040, dc_y_sd_cond=0.036, spread_mean=0.016, corr_spread_y=-0.38,
                 corr_spread_debt=0.56, c_sd_rel=0.91)
# the eleven targeted moments, and the weights that differ from one (footnote 22)
TARGETS = ["pd_freq", "pd_mean_cond", "pd_sd_cond", "debt_y_mean", "debt_y_sd", "ds_y_mean", "ds_y_sd",
           "dd_y_mean", "spread_sd", "y_persist", "y_sd"]
WEIGHTS = dict(dd_y_mean=3.0, spread_sd=1.25)
# one unit of the last digit they print for each target
REPORTED_PRECISION = dict(pd_freq=0.01, pd_mean_cond=0.01, pd_sd_cond=0.01, debt_y_mean=0.01, debt_y_sd=0.01,
                          ds_y_mean=0.001, ds_y_sd=0.001, dd_y_mean=0.001, spread_sd=0.001, y_persist=0.01, y_sd=0.01)
# benchmark: the loss of their model column against their data column under their weights (0.2276)
AMR_LOSS = sum(WEIGHTS.get(k, 1.0) * ((AMR_MODEL[k] - AMR_DATA[k]) / AMR_DATA[k]) ** 2 for k in TARGETS)
