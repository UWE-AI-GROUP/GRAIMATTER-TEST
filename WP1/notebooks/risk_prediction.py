'''
Code to experiment with risk prediction
'''

# %%
import os
import logging
from random import Random
from tkinter import ON
import pylab as plt
import pandas as pd
import seaborn as sbs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
logger.info(PROJECT_ROOT_FOLDER)

%matplotlib inline
# %%
RESULTS_CSV_FILE = os.path.join(
    PROJECT_ROOT_FOLDER,
    'experiments/RF/Random_Forest_loop_results.csv'
)
# %%
# Load the results into a dataframe
all_results = pd.read_csv(RESULTS_CSV_FILE)
all_results = all_results[all_results.scenario == 'WorstCase'].copy().reset_index()
# %%
# Some exploratory plots
# 1. Plot box plots of MIA AUC for the bootstrap = True, bootstrap = False
x = all_results[all_results.bootstrap==True].mia_AUC
y = all_results[all_results.bootstrap==False].mia_AUC
plt.boxplot([x, y], labels=['Boostrap True', 'Bootsrap False'])

# %%
# Try some prediction
mia_metrics = all_results[[col for col in all_results.columns if col.startswith('mia')] + ['full_id', 'dataset']]
target_metrics = all_results[[col for col in all_results.columns if col.startswith('target')] + ['full_id', 'dataset']].drop('target_classifier', axis=1)
hyp_params = all_results[
    [
        'bootstrap', 'min_samples_split', 'min_samples_leaf', 'n_estimators',
        'criterion', 'max_depth', 'class_weight', 'full_id', 'dataset'
    ]
]
# %%
# Need to one-hot encode the string ones: bootstrap, criterion, class_weight
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
cat_cols = ['bootstrap', 'criterion', 'class_weight']
temp = oh.fit_transform(hyp_params[cat_cols]).toarray()
temp = pd.DataFrame(temp, columns=oh.get_feature_names_out())
temp['full_id'] = hyp_params.full_id.copy()
hyp_params_oh = hyp_params.drop(columns=cat_cols)
hyp_params_oh = hyp_params_oh.merge(temp, how='left', on='full_id')
# %%
features = target_metrics.drop('dataset', axis=1)\
    .merge(hyp_params_oh, how='left', on='full_id')\
    .drop([ 'full_id'], axis=1)

features.max_depth[features.max_depth.isna()] = 200
target = mia_metrics[['mia_AUC', 'dataset']].copy()

# %%
# Random train test split
plt.rc('font', size=24) 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(
    features.drop('dataset', axis=1),
    target.drop('dataset', axis=1),
    train_size = 0.4
)
rfr = RandomForestRegressor()
rfr.fit(train_x, train_y)
predictions = rfr.predict(test_x)
fig, ax = plt.subplots(figsize=(20, 12))
f_col = 'min_samples_split'
import numpy as np
scatter = plt.scatter(predictions, test_y, c = test_x[f_col].values, cmap="Set1")
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title=f'{f_col}')
ax.add_artist(legend1)

plt.xlabel('Predicted MI AUC')
plt.ylabel('Actual MIA AUC')

# %% 
train_proportion = 0.40
datasets = list(features.dataset.unique())
for i, test_dataset in enumerate(datasets):
    train_datasets = datasets.copy()
    del train_datasets[i]
    train_x = features[features.dataset.isin(train_datasets)].copy()
    test_x = features[features.dataset == test_dataset].copy()
    train_y = target[target.dataset.isin(train_datasets)].copy()
    train_x, _, train_y, _ = train_test_split(train_x, train_y, train_size = train_proportion)
    test_y = target[target.dataset == test_dataset].copy()
    rfr.fit(train_x.drop('dataset', axis=1), train_y.drop('dataset', axis=1))
    predictions = rfr.predict(test_x.drop('dataset', axis=1))
    plt.figure(figsize=(20, 12))
    plt.plot(predictions, test_y.drop('dataset', axis=1), 'ko', color=[0, 0, 0])
    plt.xlabel('Predicted MI AUC')
    plt.ylabel('Actual MIA AUC')
    plt.title(f'{test_dataset} held out')
# %%
temp = list(zip(rfr.feature_names_in_, rfr.feature_importances_))
temp.sort(key = lambda x: x[1], reverse=True)
for f, i in temp:
    print(f, i)

# %%
