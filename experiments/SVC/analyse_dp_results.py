# %%
from configparser import Interpolation
import sys
import os
from tkinter import Image
import pandas as pd
import pylab as plt
import numpy as np

%matplotlib inline

ROOT_PROJECT_FOLDER = os.path.dirname(
        os.path.dirname(__file__)
    )


sys.path.append(ROOT_PROJECT_FOLDER)

from results_analysis.results_loader import load_results_csv
# %%
CONFIG_FILE = os.path.join(
    ROOT_PROJECT_FOLDER,
    "experiments", "SVC", "svc_rbf_dp_config.json"
)
results_csv = load_results_csv(CONFIG_FILE, one_hot_encode=False, impute_missing=False, project_root_folder=ROOT_PROJECT_FOLDER)
# %%
worst_case = results_csv[results_csv['scenario'] == "WorstCase"].copy()

# %%

def make_image(results, filter, x_col, y_col, col_col):
    sub_results = results.copy()
    for filter_col, filter_val in filter.items():
        sub_results = sub_results[sub_results[filter_col] == filter_val].copy()

    sub_results = sub_results[[x_col, y_col, col_col]]
    x_vals = sorted(sub_results[x_col].unique())
    y_vals = sorted(sub_results[y_col].unique())
    col_mat = np.zeros((len(y_vals), len(x_vals)), float)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            col_mat[j, i] = sub_results[
                (
                    sub_results[x_col] == x
                ) * (
                    sub_results[y_col] == y
                )
            ][col_col]
    return col_mat, x_vals, y_vals


filter = {
    "scenario": "WorstCase",
    "target_classifier": "SVC"
}
col_mat_svc = []
for dataset in results_csv['dataset'].unique():
    filter['dataset'] = dataset
    col_mat_svc.append((dataset, make_image(results_csv, filter, 'gamma', 'C', 'mia_AUC')))

col_mat_dp = []
filter['target_classifier'] = "DPSVC"
filter['eps'] = 500
filter['dhat'] = 100
for dataset in results_csv['dataset'].unique():
    filter['dataset'] = dataset
    col_mat_dp.append((dataset, make_image(results_csv, filter, 'gamma', 'C', 'mia_AUC')))


# %%
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cm
fig = plt.figure(figsize=(20, 14))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.3)
c_map = cm.get_cmap('hot')
for i, (dataset, (col_mat, x_val, y_val)) in enumerate(col_mat_svc):
    ax = grid[i]
    ax.imshow(col_mat, aspect='auto', interpolation='nearest', cmap=c_map, vmin=0, vmax=1)
    ax.set_title(dataset)
    ax.set_xticks(range(len(x_val)), x_val)
    ax.set_yticks(range(len(y_val)), y_val)
    ax.set_ylabel('C')

for i, (dataset, (col_mat, x_val, y_val)) in enumerate(col_mat_dp):
    ax = grid[i + 3]
    ax.imshow(col_mat, aspect='auto', interpolation='nearest', cmap=c_map, vmin=0, vmax=1)
    ax.set_xticks(range(len(x_val)), x_val)
    ax.set_yticks(range(len(y_val)), y_val)
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')

# Colorbar
ax.cax.colorbar(col_mat)
ax.cax.toggle_label(True)
# %%
