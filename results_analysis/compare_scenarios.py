'''
Do a comparison between scenarios present in a results file to attempt to answer
the question of whether WorstCase is systematically worse
'''
# %% Imports
import os
import logging
import glob
import pandas as pd
import pylab as plt
import numpy as np
from scipy.stats import ttest_rel

%matplotlib inline
font = {'size': 14}
plt.rc('font', **font)

# %%
logging.basicConfig(level = logging.INFO)
logging.info(os.getcwd())
os.chdir("../")
logging.info(os.getcwd())


# %% Load all results files
result_file_names = glob.glob("results/from_aws/*_corrected.csv")

results = pd.DataFrame()
for f in result_file_names:
    results = pd.concat([results, pd.read_csv(f)], ignore_index=True)
    
results['target_classifier'] = [" ".join(x) for x in zip(list(results.target_classifier), list(results.kernel.fillna('')))]

# %% Define the filter function (useful below)
def filter_df(df, conditions):
    return_df = df.copy()
    for condition_col, condition_val in conditions.items():
        return_df = return_df[return_df[condition_col] == condition_val]
    return return_df

# %% Where should plots be stored?
plot_folder = 'results/from_aws/plots'

# %% Risk isn't generalisable
# Look at the range of a metric for the same params across datasets
# Makes plots for each classifier that show, for each set of hyper-params, the
# min versus max value of the metric.
# Set metric to be the metric wanted
metric = 'mia_Advantage'
classifiers = results['target_classifier'].unique()
for classifier in classifiers:
    sub = filter_df(results, {'target_classifier': classifier, 'scenario': 'WorstCase'})
    pid = sub['param_id'].unique()
    temp = sub.groupby(['param_id', 'dataset']).agg(
        mean_metric = pd.NamedAgg(column=metric, aggfunc=np.mean)
    ).reset_index().groupby('param_id').agg(
        max_metric = pd.NamedAgg(column='mean_metric', aggfunc=max),
        min_metric = pd.NamedAgg(column='mean_metric', aggfunc=min)
    )
    temp['diff'] = temp['max_metric'] - temp['min_metric']
    plt.figure()
    plt.scatter(temp['min_metric'], temp['max_metric'], color=[0.3, 0.3, 0.3, 0.3])
    plt.xlabel(f'Min {metric}')
    plt.ylabel(f'Max {metric}')
    plt.title(classifier)
    plot_name = os.path.join(plot_folder, f'{classifier}_{metric}_range.png')
    plt.savefig(plot_name)

# %% Comparing performance across scenarios
# plot the difference for paired samples (same hyp, different scenarios)
# %% Useful method to plot a comparison of scenarios
def comparison(results_df, metric, conditions, scenario_1, scenario_2, plot=True, save_plot=None):
    if len(conditions) > 0:
        sub_df = filter_df(results_df, conditions)[['param_id', 'repetition', metric, 'scenario']]
    else:
        sub_df = results_df
    pivoted = pd.pivot_table(sub_df, index=['param_id', 'repetition'], values=[metric], columns=['scenario'])
    pivoted.reset_index(inplace=True)
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(pivoted[metric][scenario_1] - pivoted[metric][scenario_2])
        plt.title(' '.join(conditions.values()))
        plt.xlabel(f'{metric}: {scenario_1} - {scenario_2}')
        plt.subplot(1, 2, 2)
        plt.scatter(pivoted[metric][scenario_1], pivoted[metric][scenario_2], alpha=0.2)
        xl = plt.xlim()
        yl = plt.ylim()
        mi = min([xl[0], yl[0]])
        ma = max([xl[1], yl[1]])
        plt.plot([mi, ma], [mi, ma], 'k--')
        plt.xlabel(f'{scenario_1} {metric}')
        plt.ylabel(f'{scenario_2} {metric}')
        plt.title(' '.join(conditions.values()))
        if save_plot is not None:
            plt.savefig(save_plot)
    _, p_1_gt_2 = ttest_rel(pivoted[metric][scenario_1], pivoted[metric][scenario_2], alternative='greater')
    _, p_2_gt_1 = ttest_rel(pivoted[metric][scenario_2], pivoted[metric][scenario_1], alternative='greater')
    

# Filter out some of the meaningless results
filtered_res = results[results['target_train_pred_prob_var'] > 1e-2]

# Plot each metric for each dataset
plot_metrics = ['mia_AUC', 'mia_F1score', 'mia_Advantage']
for metric in plot_metrics:
    # conditions = {'dataset': 'synth-ae', 'target_classifier': 'XGBClassifier '}
    # conditions = {'dataset': 'mimic2-iaccd'}
    conditions = {}
    for dataset in results.dataset.unique():
        if 'mimic2-iaccd' in dataset:
            continue
        conditions['dataset'] = dataset
        plot_file = os.path.join(plot_folder, f'{dataset}_{metric}.png')
        p = comparison(filtered_res, metric, conditions, 'WorstCase', 'Salem1', save_plot=plot_file)
# %% The same plot, but for all datasets together
conditions = {}
for metric in plot_metrics:
    plot_file = os.path.join(plot_folder, f'{metric}.png')
    p = comparison(results, metric, conditions, 'WorstCase', 'Salem1', save_plot=plot_file)


# %%
