import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from metricPlots import *


def create_mia_data(clf, xtrain, xtest, sort_probs=False, keep_top=-1):
    """
    This function predict the probability of train and test data
    (from building the model split) belonging to their
    corresponding train or test respectively
    and it also creates labels.

    clf: fitted classifier (original model)
    xtrain: training data (original model)
    xtest: test data (original model)
    """    
    miX = np.concatenate(
        (
            clf.predict_proba(xtrain),
            clf.predict_proba(xtest)
        )
    )
    
    if sort_probs:
        miX = -np.sort(-miX, axis=1)
        if keep_top > -1:
            miX = miX[:, :keep_top]

    miY = np.concatenate(
        (
        np.ones((len(xtrain), 1), int),
        np.zeros((len(xtest), 1), int)
        )
    ).flatten()
    return(miX,miY)


def run_membership_inference_attack(clf_name, model, xtrain, xtest, MIA_classifier, MIA_name):
    """
    This function performs MIA (membership inference attack)
    and plots metrics.

    clf_name: name of the original model in which MIA is performed.
    model: original model in which MIA is performed.
    xtrain: train data of the original model.
    xtest: test data of the original model.
    MIA_classifier: unfitted classifier for MIA.
    MIA_name: name of the MIA classifier.
    """

    miX, miY = create_mia_data(model, xtrain, xtest, sort_probs=True)
    mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(miX, miY, test_size=0.2, stratify=miY)

    mi_rf = MIA_classifier
    mi_rf.fit(mi_train_x, mi_train_y)

    plotROC_classifier(mi_rf, mi_test_x, mi_test_y, f'{clf_name} - MIA {MIA_name}')

    pred_y = model.predict_proba(xtest)
    pred_train_y = model.predict_proba(xtrain)

    _, n_classes = pred_y.shape
    for cl in range(n_classes):
        plot_prob_test_train(pred_y, pred_train_y, f'{clf_name} - MIA {MIA_name}', plot_class = cl)
    
        plot_detection_error_tradeoff(mi_rf, mi_test_x, mi_test_y, clf_name+" Detection Error Tradeoff (DET) curve", clf_name)
    
        print('Attacker advantage', attacker_advantage(mi_rf, mi_test_x, mi_test_y)) 
