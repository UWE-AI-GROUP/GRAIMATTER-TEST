'''
Decimal place comparison for RF

Compares the MIA AUC metric (easy to try others) for the rounded and unrounded dataset versions
'''

# %%
import os
import pandas as pd
import pylab as plt
from sklearn.neighbors import KernelDensity
import numpy as np
%matplotlib inline

plt.rcParams.update({'font.size': 24})


WORKING_FOLDER = os.path.dirname(__file__)
print(WORKING_FOLDER)

unrounded_results  = pd.read_csv(
    os.path.join(WORKING_FOLDER, "Random_Forest_loop_results.csv")
)
rounded_results = pd.read_csv(
    os.path.join(WORKING_FOLDER, "round_rf_results.csv")
)
# %%
datasets = [
    'mimic2-iaccd',
    'in-hospital-mortality',
    'indian liver'
]
scenario = "WorstCase"
metric = 'mia_AUC'
# %%

def filter_df(df, conditions):
    return_df = df.copy()
    for condition_col, condition_val in conditions.items():
        return_df = return_df[return_df[condition_col] == condition_val]
    return return_df


result_df = pd.DataFrame()
for dataset in datasets:
    conditions = {
        'dataset': dataset,
        'scenario': scenario,
        'min_samples_leaf': 1,
        'bootstrap': False
    }
    unrounded = filter_df(unrounded_results, conditions)[['param_id', metric]].copy()
    r_dataset = f'round {dataset}'
    conditions['dataset'] = r_dataset
    rounded = filter_df(rounded_results, conditions)[['param_id', metric]].copy()
    rounded['mia_AUC_rounded'] = rounded['mia_AUC']
    rounded.drop('mia_AUC', axis=1, inplace=True)
    temp = pd.merge(unrounded, rounded, how = 'outer', on = 'param_id')

    result_df[dataset] = temp['mia_AUC_rounded'] - temp['mia_AUC']
# %%

for dataset in datasets:
    x_dat = result_df[dataset].values
    x_dat.sort()
    x_dat = x_dat[:, None]
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=0.02, kernel='gaussian')
    kde.fit(x_dat)

    # score_samples returns the log of the probability density
    x = np.arange(-0.3, 0.3, 0.001)
    logprob = kde.score_samples(x[:, None])
    plt.figure(figsize=(16, 10))
    plt.fill_between(x, np.exp(logprob), alpha=0.5)
    plt.plot(x_dat.flatten(), np.full_like(x_dat, -0.02), '|k', markeredgewidth=1)
    plt.ylim(-0.1, np.exp(logprob).max()*1.1)
    plt.xlabel('mia AUC rounded minus mia AUC unrounded')
    plt.title(f'{dataset}, mean = {x_dat[:, 0].mean():.4f}')
    plt.savefig(f"{dataset.replace(' ','_')}_round.png")
# %%

# %%

# %%
