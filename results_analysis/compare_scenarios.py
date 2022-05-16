'''
Do a comparison between scenarios present in a results file
'''
# %%
import os
import logging
import glob
import pandas as pd
import pylab as plt

#%matplotlib inline

logging.basicConfig(level = logging.INFO)
#ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
ROOT_FOLDER = "~/studies/GRAIMatter/"
logging.info(ROOT_FOLDER)

# %%
result_file_names = glob.glob("../results/from_aws/*_corrected.csv")

results = pd.DataFrame()
for f in result_file_names:
    results = pd.concat([results, pd.read_csv(f)], ignore_index=True)
    
results['target_classifier'] = [" ".join(x) for x in zip(list(results.target_classifier), list(results.kernel.fillna('')))]

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

metrics = ['mia_TPR', 'mia_FPR', 'mia_FAR',
           'mia_TNR', 'mia_PPV', 'mia_NPV',
           'mia_FNR', 'mia_ACC', 'mia_F1score',
           'mia_Advantage', 'mia_AUC', 'mia_FDIF']
# %%
scenario_specifics = {}


# %%
classifiers = results.target_classifier.unique()


# %%
%matplotlib inline
from scipy.stats import ttest_rel
fig_folder = "../results/from_aws/plots"
font = {'size': 14}
plt.rc('font', **font)
stat_df = pd.DataFrame()
for classifier in classifiers:
    results_df = results[results.target_classifier==classifier]
    datasets = results_df['dataset'].unique()
    nrows = len(datasets)
    n_scenario_pairs = int((3 * (3 - 1)) / 2)
    ncols = n_scenario_pairs
    for metric in metrics:
        # plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 45))
        index = 0
        for row, dataset in enumerate(datasets):
            conditions['dataset'] = dataset
            for scenario in results_df['scenario'].unique():
                conditions['scenario'] = scenario
                scenario_specifics[scenario] = filter_df(results_df, conditions)[['param_id', 'repetition', metric]]
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
                        on=['param_id', 'repetition']
                    )
                    col1 = f'{s1}_{metric}'
                    col2 = f'{s2}_{metric}'
                    diff_vals = diff[col1] - diff[col2]
                    t1, p1 = ttest_rel(diff[col1], diff[col2], alternative='greater')
                    t2, p2 = ttest_rel(diff[col1], diff[col2], alternative='less')
                    # print(f'{col1} > {col2}: {p1}')
                    # print(f'{col1} < {col2}: {p2}')
                    
                    new_df = {
                        'classifier': [classifier],
                        'dataset': [dataset],
                        'col1': [col1],
                        'col2': [col2],
                        'p_col1_gt_col2': [p1],
                        'p_col2_gt_col1': [p2],
                        't_col1_gt_col2': [t1],
                        't_col2_gt_col1': [t2]
                    }

                    stat_df = pd.concat((stat_df, pd.DataFrame(new_df)))


                    # plt.subplot(nrows, ncols, index)
                    # plt.hist(diff_vals)
                    # plt.xlabel(f'{s1} {metric} minus {s2} {metric}')
                    # plt.title(f"{conditions['dataset']} {metric} (mean = {diff_vals.mean():.3f})")
                    # plt.plot([0, 0], plt.ylim(), 'k--')
        # plt.tight_layout()
        # plt.savefig(os.path.join(
            # fig_folder,
            # f'{classifier}_{metric}_hist_grid.png'
            # )
        # )
        
    # break
stat_df.to_csv(
    os.path.join(
        fig_folder,
        'stat_df.csv'
    ),
    index=False
)

        # %%
        nrows = len(datasets)
        n_scenario_pairs = int((3 * (3 - 1)) / 2)
        ncols = n_scenario_pairs
        plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 45))
        index = 0
        for row, dataset in enumerate(datasets):
            conditions['dataset'] = dataset
            for scenario in results_df['scenario'].unique():
                conditions['scenario'] = scenario
                scenario_specifics[scenario] = filter_df(results_df, conditions)[['param_id', 'repetition', metric]]
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
                            on=['param_id', 'repetition']
                        )
                    col1 = f'{s1}_{metric}'
                    col2 = f'{s2}_{metric}'
                    plt.subplot(nrows, ncols, index)
                    plt.plot(diff[col1], diff[col2], 'o', color=[0, 0, 0.7, 0.5])
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.title(f"{conditions['dataset']}")
                    mi = min(plt.xlim()[0], plt.ylim()[0])
                    ma = max(plt.xlim()[1], plt.ylim()[1])

                    plt.plot([mi, ma], [mi, ma], 'k')
        plt.tight_layout()
        plt.savefig(f'{classifier}_{metric}_scatter_scatter_grid.png')
        # %%

# %%
# Analyse stat_df -- is worst case always worse

scenario_1 = 'WorstCase'
scenario_2 = 'Salem2'
a = []
b = []

for metric in metrics:
    temp = stat_df[
        (stat_df['col1'].str.startswith(scenario_1)) &
        (stat_df['col2'].str.startswith(scenario_2))
    ]

    temp = temp[temp['col1'].str.contains(metric)]
    # print(len(temp))
    p_thresh = 0.05
    temp2 = temp[temp['p_col1_gt_col2'] < p_thresh]
    n_1_gt_2 = len(temp2)
    temp2 = temp[temp['p_col2_gt_col1'] < p_thresh]
    n_2_gt_1 = len(temp2)
    print(f'{metric}: {scenario_1} > {scenario_2}: {n_1_gt_2}. {scenario_2} > {scenario_1}: {n_2_gt_1}')
    a.append(n_1_gt_2)
    b.append(n_2_gt_1)
q = pd.DataFrame({'metric': metrics, f'N {scenario_1} > {scenario_2}': a, f'N {scenario_2} > {scenario_1}': b
})
# %%


# Look at the range of a metric for the same params across datasets
classifiers = results['target_classifier'].unique()
classifier = 'RandomForestClassifier '
sub = filter_df(results, {'target_classifier': classifier, 'scenario': 'WorstCase'})
pid = sub['param_id'].unique()
metric = 'mia_AUC'
import numpy as np
temp = sub.groupby(['param_id', 'repetition']).agg(
    mean_metric = pd.NamedAgg(column=metric, aggfunc=np.mean)
).ungroup().groupby('param_id').agg(
    max_metric = pd.NamedAgg(column=metric, aggfunc=max),
    min_metric = pd.NamedAgg(column=metric, aggfunc=min)
)
temp['diff'] = temp['max_metric'] - temp['min_metric']
plt.scatter(temp['min_metric'], temp['max_metric'], color=[0.3, 0.3, 0.3, 0.3])
plt.xlabel(f'Min {metric}')
plt.ylabel(f'Max {metric}')

# %%
