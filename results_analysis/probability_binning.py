'''Analyse results of the random forest binning experiment'''
# %%
# Notes:
# config is in experiments/RF/rf_binned_config.R
# Results are in experiments/RF/rf_binned_results.csv

import pandas as pd
import pylab as plt
%matplotlib inline
RESULTS_FILE = "../experiments/RF/binned_rf_results.csv"
results_df = pd.read_csv(RESULTS_FILE)

# %%
HYP_NAMES = [
    'min_samples_split',
    "bootstrap",
    "min_samples_split",
    "min_samples_leaf",
    "n_estimators"
    "max_depth",
    "n_probability_bins"
]
# %%
x_col = 'target_AUC'
y_col = 'mia_AUC'
