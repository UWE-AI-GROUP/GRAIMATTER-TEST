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
    .drop(['target_FAR', 'target_PPV', 'full_id'], axis=1)

features.max_depth[features.max_depth.isna()] = 200
target = mia_metrics.mia_AUC

# %%
# 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(features.drop('dataset', axis=1), target, train_size = 0.4)
rfr = RandomForestRegressor()
rfr.fit(train_x, train_y)
predictions = rfr.predict(test_x)
plt.plot(predictions, test_y, 'ko', color=[0, 0, 0, 0.2])
plt.xlabel('Predicted MI AUC')
plt.ylabel('Actual MIA AUC')
# %%
