'''
Do a comparison between scenarios present in a results file
'''
# %%
import os
import logging
import pandas as pd
import pylab as plt

%matplotlib inline

logging.basicConfig(level = logging.INFO)
ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
logging.info(ROOT_FOLDER)

# %%
RESULTS_FILE = os.path.join(
    ROOT_FOLDER,
    'experiments', 'RF', 'Random_Forest_loop_results.csv'
)

results_df = pd.read_csv(RESULTS_FILE)
# %%
def filter_df(df, conditions):
    return_df = df.copy()
    for condition_col, condition_val in conditions.items():
        return_df = return_df[return_df[condition_col] == condition_val]
    return return_df

# %%
conditions = {
    'dataset': 'mimic2-iaccd',
}
metric = 'mia_AUC'
# %%
scenario_specifics = {}


# %%
datasets = ['mimic2-iaccd', 'in-hospital-mortality', 'indian liver']
nrows = len(datasets)
n_scenario_pairs = int((3 * (3 - 1)) / 2)
ncols = n_scenario_pairs
plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
index = 0
for row, dataset in enumerate(datasets):
    conditions['dataset'] = dataset
    for scenario in results_df['scenario'].unique():
        conditions['scenario'] = scenario
        scenario_specifics[scenario] = filter_df(results_df, conditions)[['param_id', metric]]
        scenario_specifics[scenario][f'{scenario}_{metric}'] = scenario_specifics[scenario][metric]
        scenario_specifics[scenario].drop(metric, axis=1, inplace=True)
    for s1 in list(scenario_specifics.keys())[:-1]:
        for s2 in list(scenario_specifics.keys())[1:]:
            if s1 == s2:
                continue
            index += 1
            diff = pd.merge(
                scenario_specifics[s1],
                scenario_specifics[s2],
                how='inner',
                on='param_id'
            )
            col1 = f'{s1}_{metric}'
            col2 = f'{s2}_{metric}'
            diff_vals = diff[col1] - diff[col2]
            plt.subplot(nrows, ncols, index)
            plt.hist(diff_vals)
            plt.xlabel(f'{s1} {metric} minus {s2} {metric}')
            plt.title(f"{conditions['dataset']} {metric} (mean = {diff_vals.mean():.3f})")
            plt.plot([0, 0], plt.ylim(), 'k--')
plt.savefig(f'{metric}_grid.png')
# %%
