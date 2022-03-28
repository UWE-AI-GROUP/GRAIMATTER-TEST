'''
Generate model
Generate the dummy model for the ppie example
'''

# %% Imports
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import pylab as plt

%matplotlib inline

# %% Constants
N_TRAIN = 200 # Number training points
N_VAL = 50 # Number validation points
N_DATA = N_TRAIN + N_VAL

# %% Generate data
target = np.random.binomial(1, 0.5, (N_DATA, ))

AGE_POS_MEAN = 70
AGE_NEG_MEAN = 50

N_POS = (target == 1).sum()
N_NEG = (target == 0).sum()

age = np.zeros((N_DATA, ), np.int)
age[target == 1] = np.random.poisson(AGE_POS_MEAN, (N_POS, ))
age[target == 0] = np.random.poisson(AGE_NEG_MEAN, (N_NEG, ))

BP_POS_MEAN = 120
BP_NEG_MEAN = 90

blood_pressure = np.zeros((N_DATA, ), np.int)
blood_pressure[target == 1] = np.random.poisson(BP_POS_MEAN, (N_POS, ))
blood_pressure[target == 0] = np.random.poisson(BP_NEG_MEAN, (N_NEG, ))

SMOKE_POS_PROB = 0.3
SMOKE_NEG_PROB = 0.1

smoker = np.zeros((N_DATA, ), np.int)
smoker[target == 1] = np.random.binomial(1, SMOKE_POS_PROB, (N_POS, ))
smoker[target == 0] = np.random.binomial(1, SMOKE_NEG_PROB, (N_NEG, ))

CHOLEST_POS_PROB = 6
CHOLEST_NEG_PROB = 4

cholest = np.zeros((N_DATA, ), np.int)
cholest[target == 1] = np.random.poisson(CHOLEST_POS_PROB, (N_POS, ))
cholest[target == 0] = np.random.poisson(CHOLEST_NEG_PROB, (N_NEG, ))

features = pd.DataFrame(
    {
        'age': age,
        'blood_pressure': blood_pressure,
        'smoker': smoker,
        'total_cholestorol': cholest
    }
)

# %% Data Split
train_x, val_x, train_y, val_y = train_test_split(features, target, train_size=N_TRAIN)

# %% Train model
random_forest = RandomForestClassifier()
random_forest.fit(train_x, train_y)
# %% Validate
pred_probs = random_forest.predict_proba(val_x)
auc = roc_auc_score(val_y, pred_probs[:, 1])
fpr, tpr, _ = roc_curve(val_y, pred_probs[:, 1])
plt.plot(fpr, tpr, 'k')
plt.plot([0, 1], [0, 1], 'k--')
# %% Save the model
dump(random_forest, 'ppie_rf.joblib')
