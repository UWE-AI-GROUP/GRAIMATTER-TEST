'''
Code to experiment with risk prediction
'''

# %%
import os
import logging
from random import Random
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

%matplotlib inline

# %% Load the results file
RESULTS_CSV_FILE = os.path.join(
    PROJECT_ROOT_FOLDER,
    'experiments/RF/Random_Forest_loop_results.csv'
)
# %%
# Load the results into a dataframe
# Keep only the chosen scenario
# Remove results from image datasets as they seem to behave completely differently
all_results = pd.read_csv(RESULTS_CSV_FILE)
all_results = all_results[all_results.scenario == 'WorstCase'].copy()
all_results = all_results[all_results.dataset != "medical-mnist-ab-v-br-100"].copy()
all_results = all_results[all_results.dataset != "medical-mnist-ab-v-br-500"].copy()
all_results.reset_index(inplace=True)


# %%
# Tease apart the different bits of the table
mia_metrics = all_results[[col for col in all_results.columns if col.startswith('mia')] + ['full_id', 'dataset']]
target_metrics = all_results[[col for col in all_results.columns if col.startswith('target')] + ['full_id', 'dataset']].drop('target_classifier', axis=1)
hyp_params = all_results[
    [
        'bootstrap', 'min_samples_split', 'min_samples_leaf', 'n_estimators',
        'criterion', 'max_depth', 'class_weight', 'full_id', 'dataset'
    ]
]
# %%
# Need to one-hot encode any object columns
# Find the non-numeric columns (skipping some we know we don't want to encode)
column_names = hyp_params.columns
column_dtype = hyp_params.dtypes
cat_cols = []
for i, col_name in enumerate(column_names):
    if "_id" in col_name:
        continue # don't encode IDs
    if "dataset" in col_name:
        continue # don't encode dataset
    d_type = column_dtype[col_name]
    if d_type == "int64" or d_type == "float64":
        continue
    cat_cols.append(col_name)

oh = OneHotEncoder()
temp = oh.fit_transform(hyp_params[cat_cols]).toarray()
temp = pd.DataFrame(temp, columns=oh.get_feature_names_out())
temp['full_id'] = hyp_params.full_id.copy()
hyp_params_oh = hyp_params.drop(columns=cat_cols)
hyp_params_oh = hyp_params_oh.merge(temp, how='left', on='full_id')
# %%
# Merge the target metrics with the hyper-params
features = target_metrics.drop('dataset', axis=1)\
    .merge(hyp_params_oh, how='left', on='full_id')\
    .drop([ 'full_id'], axis=1)

# Impute for the feature with NAs
features.max_depth[features.max_depth.isna()] = 200

# Define the target for prediction
target = mia_metrics[['mia_AUC', 'dataset']].copy()

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
