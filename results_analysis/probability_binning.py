'''Analyse results of the random forest binning experiment'''
# %%
# Notes:
# config is in experiments/RF/rf_binned_config.R
# Results are in experiments/RF/rf_binned_results.csv

import pandas as pd
import pylab as plt
%matplotlib inline

font = {
    'size': 18
}
plt.rc('font', **font)

# %%
RESULTS_FILE = "../experiments/RF/binned_rf_results.csv"
results_df = pd.read_csv(RESULTS_FILE)

# %%
HYP_NAMES = [
    'min_samples_split',
    "bootstrap",
    "min_samples_leaf",
    "n_estimators",
    "max_depth"
]

P_NAME = ['n_probability_bins']

MET_NAMES = [
    "mia_FDIF"
]

temp_df = results_df.loc[:, HYP_NAMES + P_NAME + MET_NAMES].copy()
a = temp_df.pivot(index = HYP_NAMES, columns = P_NAME[0], values = MET_NAMES[0])

# %%
plot_vals = [3, 5, 10, 20]
pos = 1
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 8))
fig.tight_layout()
for p in plot_vals:
    plt.subplot(1, 4, pos)
    plt.scatter(a[0], a[p])
    xl = plt.xlim()
    yl = plt.ylim()
    xr = [min(xl[0], yl[0]), max(xl[1], yl[1])]
    plt.plot(xr, xr, 'k--')
    plt.xlabel('Standard Probs')
    plt.ylabel(f'{p} bins')
    plt.title(MET_NAMES[0])
    pos += 1
plt.savefig(f'binned_{MET_NAMES[0]}.png')
# %%
