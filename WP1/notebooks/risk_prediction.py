'''
Code to experiment with risk prediction
'''

# %%
from json import load
import os
import sys
import logging
import pylab as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROJECT_ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
logger.info(PROJECT_ROOT_FOLDER)
sys.path.append(PROJECT_ROOT_FOLDER)

from results_analysis.results_loader import load_results_csv

%matplotlib inline

# %%
CONFIG_FILE = os.path.join(
    PROJECT_ROOT_FOLDER,
    'experiments', 'RF', 'randomForest_config.json'
)

all_results = load_results_csv(CONFIG_FILE, project_root_folder=PROJECT_ROOT_FOLDER)
# %%
# Keep only the chosen scenario
# Remove results from image datasets as they seem to behave completely differently
all_results = all_results[all_results.scenario == 'WorstCase'].copy()
all_results = all_results[all_results.dataset != "medical-mnist-ab-v-br-100"].copy()
all_results = all_results[all_results.dataset != "medical-mnist-ab-v-br-500"].copy()
all_results.reset_index(inplace=True)


# %%
# Tease apart the different bits of the table
mia_cols = [col for col in all_results.columns if col.startswith('mia')]
id_cols = [col for col in all_results.columns if '_id' in col]
shadow_cols = [col for col in all_results.columns if col.startswith('shadow')]
cols_to_remove = [
    'scenario',
    'target_classifier',
    'repetition',
    'attack_classifier'
]

target = all_results[['mia_AUC', 'dataset']].copy()
features = all_results.drop(
    mia_cols + cols_to_remove + id_cols + shadow_cols,
    axis=1
)

# %%
# Code to train and test with random stratification across all rows
def train_predict_plot(train_x, train_y, test_x, test_y, f_col=None, title=None):
    rfr = RandomForestRegressor()
    rfr.fit(train_x, train_y)
    predictions = rfr.predict(test_x)
    fig, ax = plt.subplots(figsize=(20, 12))
    if f_col is None:
        scatter = plt.scatter(predictions, test_y)
    else:
        scatter = plt.scatter(predictions, test_y, c = test_x[f_col].values, cmap="Set1")
        legend1 = ax.legend(*scatter.legend_elements(),
            loc="lower left", title=f'{f_col}')
        ax.add_artist(legend1)

    plt.xlabel('Predicted MI AUC')
    plt.ylabel('Actual MIA AUC')
    if title is not None:
        plt.title(title)

def random_stratify_fit_plot(features, target, f_col=None, train_size=0.4):
    # Stratify randomly across all rows
    plt.rc('font', size=24) 
    train_x, test_x, train_y, test_y = train_test_split(
        features.drop('dataset', axis=1),
        target.drop('dataset', axis=1),
        train_size = train_size
    )
    train_predict_plot(train_x, train_y, test_x, test_y, f_col=f_col)
    

random_stratify_fit_plot(features, target, f_col="min_samples_leaf")

# %% Code to loop over datasets, holding out one at a time

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
    train_predict_plot(
        train_x.drop('dataset', axis=1),
        train_y.drop('dataset', axis=1).values.flatten(),
        test_x.drop('dataset', axis=1),
        test_y.drop('dataset', axis=1).values.flatten(),
        f_col="min_samples_leaf",
        title=f"{test_dataset} held out"
    )


# %%
# Treat as a classification problem, using a defined threshold on AUC
auc_threshold = 0.7
target['label'] = 1*(target.mia_AUC > auc_threshold)

# %%
def random_stratify_fit_plot_classification(features, target, train_size=0.4):
    # Stratify randomly across all rows
    plt.rc('font', size=24) 
    train_x, test_x, train_y, test_y = train_test_split(
        features.drop('dataset', axis=1),
        target.label.values.flatten(),
        train_size = train_size
    )
    train_predict_plot_classification(train_x, train_y, test_x, test_y)

def train_predict_plot_classification(train_x, train_y, test_x, test_y, title=""):
    rf = RandomForestClassifier()
    rf.fit(train_x, train_y)
    pred_probs = rf.predict_proba(test_x)[:, 1]
    auc = roc_auc_score(test_y, pred_probs)
    fpr, tpr, _ = roc_curve(test_y, pred_probs)
    plt.figure(figsize=(20, 12))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    title = f"{title} (AUC = {auc}"
    plt.title("AUC")



random_stratify_fit_plot_classification(features, target)    
# %%
