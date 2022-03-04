'''
Code to experiment with risk prediction
'''

# %%
import os
import logging
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
# %%
# Some exploratory plots
# 1. Plot box plots of MIA AUC for the bootstrap = True, bootstrap = False
v1 = all_results[all_results.bootstrap == True].mia_AUC
v2 = all_results[all_results.bootstrap == False].mia_AUC
# %%
