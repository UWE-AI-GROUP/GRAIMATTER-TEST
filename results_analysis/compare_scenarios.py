'''
Do a comparison between scenarios present in a results file
'''
# %%
import os
import logging
import pandas as pd
import pylab as plt

#%matplotlib inline

logging.basicConfig(level = logging.INFO)
#ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
ROOT_FOLDER = "~/studies/GRAIMatter/"
logging.info(ROOT_FOLDER)

# %%
file_names = ['Random_Forest_loop_results.csv',
    "AdaBoost_results.csv",
    "DecisionTree_results.csv", 
    "round_rf_results.csv",
    "SVC_poly_results.csv",
    "SVC_rbf_results.csv",
    "SVC_rbf_dp_results.csv",
    "xgboost_results.csv", 
    "AdaBoost_results_minmax_round.csv",
    "DecisionTreeClassifier_minmax_round_results.csv",
    "round_minmax_rf_results.csv",
    "round_rf_results.csv"]

#RESULTS_FILE = os.path.join(
#    ROOT_FOLDER,
#    'experiments',
#)

#names = ['RandomForest', 'AdaBoost', 'DecisionTree', ]
results = pd.DataFrame()
for f in file_names:
    results = pd.concat([results, pd.read_csv(os.path.join(ROOT_FOLDER, 'experiments', f))], ignore_index=True)
    
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
metric = 'mia_Advantage'

metrics = ['mia_TPR', 'mia_FPR', 'mia_FAR',
           'mia_TNR', 'mia_PPV', 'mia_NPV',
           'mia_FNR', 'mia_ACC', 'mia_F1score',
           'mia_Advantage', 'mia_AUC', 'mia_pred_prob_var']
# %%
scenario_specifics = {}


# %%
#datasets = ['mimic2-iaccd', 'in-hospital-mortality', 'indian liver']
datasets = results.dataset.unique() #list(set(list(results_df.dataset)))
classifiers = results.target_classifier.unique() #list(set(list(results_df.target_classifier)))


for classifier in classifiers:
    results_df = results[results.target_classifier==classifier]
    nrows = len(datasets)
    n_scenario_pairs = int((3 * (3 - 1)) / 2)
    ncols = n_scenario_pairs
    for metric in metrics:
        plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 45))
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
        plt.tight_layout()
        plt.savefig(f'{classifier}_{metric}_hist_grid.png')

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
