#import itertools, os
from random import Random
import numpy as np
#import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense, Dropout
#import matplotlib.pyplot as plt
#from sklearn.metrics import auc, RocCurveDisplay, DetCurveDisplay, det_curve, confusion_matrix,  classification_report
#import scikitplot as skplt
from typing import Any, Iterable, Optional


def split_target_data(X:Iterable[float],
                      y:Iterable,
                     i:Optional[float]=0.67,
                     j:Optional[float]=0.33,
                     r_state:Optional[int]=5):
    """
    This functions splits the data into 3 sets: target, train and test. The default split is 1/3 of the data for each of the 3 sets. The random state default is set to 5.
    
    X: matrix of feature data
    y: labels of the data
    i: size of the target vs train+test data
    j: size of train vs test data
    r_state: variable to split the data randomly but in a predictable way.
    """
    X_target, X_test_tmp, y_target, y_test_tmp = train_test_split(X, y, shuffle=True, test_size=i, random_state=r_state, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_test_tmp, y_test_tmp, shuffle=True, test_size=j, random_state=r_state, stratify=y_test_tmp)
    return(X_target, X_train, X_test, y_target, y_train, y_test)



def create_mia_data(clf:Any, 
                    xtrain:Iterable[float], 
                    xtest:Iterable[float], 
                    sort_probs:bool=False,
                    keep_top:int=-1):
    """
    This function predict the probability of train and test data
    (from building the model split) belonging to their
    corresponding train or test respectively
    and it also creates labels.

    clf: fitted classifier (target model)
    xtrain: training data (target model)
    xtest: test data (target model)
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


def run_membership_inference_attack(
    target_model:Any,
    shadow_model: Any,
    x_target_train:Iterable[float],
    x_shadow_train:Iterable[float], 
    x_test:Iterable[float], 
    mia_classifier:Any, 
    test_size:Optional[float]=0.5
):
    """
    This function performs MIA (membership inference attack).

    clf_name: name of the target model in which MIA is performed.
    model: trained target model in which MIA is performed.
    xtrain: train data of the target model.
    xtest: test data of the target model.
    mia_classifier: unfitted classifier for MIA.
    test_size: Size of the split for the test data. Default 0.5.
    """

    target_1, target_2 = train_test_split(x_target_train, train_size=(1-test_size))
    shadow_1, shadow_2 = train_test_split(x_shadow_train, train_size=(1-test_size))
    test_1, test_2 = train_test_split(x_test, train_size=(1-test_size))

    # Create data for training MIA
    mi_train_x, mi_train_y = create_mia_data(shadow_model, shadow_1, test_1, sort_probs=True)

    mi_clf = mia_classifier
    mi_clf.fit(mi_train_x, mi_train_y)

    # Create data for testing MIA
    mi_test_x, mi_test_y = create_mia_data(target_model, target_2, test_2, sort_probs=True)
    
            
    return(mi_test_x, mi_test_y, mi_clf)



def mia_salem_1(
    target_model: Any,
    shadow_clf: Any,
    X_target_train: Iterable[float],
    X_shadow_train: Iterable[float],
    y_shadow_train: Iterable,
    X_test: Iterable[float],
    #y_test:Iterable,
    #j:Optional[float]=0.5,
    #r_state:Optional[int]=5,
    mia_clf:Optional[Any]=RandomForestClassifier(),
    mia_test_split:Optional[float]=0.5):
    """
    Perform Salem adversary 1 type of attack. This attack assumes attacker has a dataset of the same distribution as the target's model tarining data. The shadow model mimic the target's model behaviour.

    shadow_clf: classifier for shadow model (not fitted)
    X_shadow_train: shadow model training data, which must be of the same distribution of the target data, e.g. could be a split of the one used for training/test the target model and not used to train the target.
    y_shadow_train: training data labels
    X_test: test data
    y_test: test data labels
    mia_clf: unfitted classifier for the MIA attack. Deafault: RandomForest.
    mia_test_split: proportion of data for the test split for MIA.
    """

    #1 shadow model training
    shadow_model = shadow_clf
    shadow_model.fit(X_shadow_train, y_shadow_train)

    #2 attack model training
    # Get prediction probabilities from the shadow model
    mi_test_x, mi_test_y, mi_clf = run_membership_inference_attack(
        target_model,
        shadow_model,
        X_target_train,
        X_shadow_train,
        X_test,mia_clf,
        mia_test_split
    )

    return(mi_test_x, mi_test_y, mi_clf, shadow_model)



def mia_salem_2(
    target_model: Any,
    shadow_clf: Any,
    X_target_train,
    X_shadow:Optional=None,
    y_shadow:Optional=None,
    #X_test:Optional=None,
    #y_test:Optional=None,
    j:Optional[float]=0.5,
    r_state:Optional[int]=5,
    mia_clf:Optional[Any]=RandomForestClassifier(),
    mia_test_split:Optional[float]=0.5):
    """
    Perform Salem adversary 2 type of attack. This attack assumes attacker does not have a dataset of the same distribution as the target's model tarining data. The adversary does not know the structure of the target mode.

    
    shadow_clf: classifier for shadow model (not fitted)
    X_shadow_train: shadow model training data, which must be of the same distribution of the target data, e.g. could be a split of the one used for training/test the target model and not used to train the target. Default: breast cancer data provided with sklearn datasets.
    y_shadow_train: training data labels. Default: labels breast cancer data provided with sklearn datasets.
    X_test: test data. Default: split of breast cancer data provided with sklearn datasets.
    y_test: test data labels. Default: labels from a split of breast cancer data provided with sklearn datasets.
    mia_clf: unfitted classifier for the MIA attack. Deafault: RandomForest.
    mia_test_split: proportion of data for the test split for MIA.
    """
    
    if not X_shadow:
        X_breast_cancer, y_breast_cancer = datasets.load_breast_cancer(return_X_y=True)
        X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(X_breast_cancer, y_breast_cancer, shuffle=True, test_size=j, random_state=r_state)
    else:
        X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(X_shadow, y_shadow, shuffle=True, test_size=j, random_state=r_state)
    
    
    #1 shadow model training
    shadow_model = shadow_clf
    shadow_model.fit(X_shadow_train, y_shadow_train)
    
    #2 attack model training
    # Get prediction probabilities from the shadow model   
    mi_test_x, mi_test_y, mi_clf = run_membership_inference_attack(
        target_model,
        shadow_model,
        X_target_train,
        X_shadow_train,
        X_shadow_test,
        mia_clf,
        mia_test_split
    )
    return(mi_test_x, mi_test_y, mi_clf, shadow_model, X_shadow_test, y_shadow_test)


def train_mia(mia_train_probs, mia_train_labels, mia_classifier):
    """
    Train the mia classifier
    """
    mia_classifier.fit(mia_train_probs, mia_train_labels)
    return mia_classifier

def worst_case_mia(target_model, X_target_train, X_test, prop_mia_train=0.5, mia_classifier=RandomForestClassifier()):
    """
    Worst case mia scenario
    """
    mi_probs, mi_labels = create_mia_data(target_model, X_target_train, X_test, sort_probs=True)
    mia_train_probs, mia_test_probs, mia_train_labels, mia_test_labels = train_test_split(
        mi_probs,
        mi_labels,
        stratify=mi_labels,
        train_size=prop_mia_train
    )

    mia_classifier = train_mia(mia_train_probs, mia_train_labels, mia_classifier)
    return mia_test_probs, mia_test_labels, mia_classifier

def salem(target_model, shadow_clf, X_target_train, X_shadow, y_shadow, X_test, prop_shadow_train=0.5, mia_classifier=RandomForestClassifier()):
    """
    Salem scenario 2
    """

    X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(X_shadow, y_shadow, train_size=prop_shadow_train)

    shadow_clf.fit(X_shadow_train, y_shadow_train)


    mia_train_probs, mia_train_labels = create_mia_data(shadow_clf, X_shadow_train, X_shadow_test, sort_probs=True)
    mia_test_probs, mia_test_labels = create_mia_data(target_model, X_target_train, X_test, sort_probs=True)

    _, n_class_shadow = mia_train_probs.shape
    _, n_class_target = mia_test_probs.shape

    min_classes = min(n_class_shadow, n_class_target)

    mia_train_probs = mia_train_probs[:, :min_classes]
    mia_test_probs = mia_test_probs[:, :min_classes]

    mia_classifier = train_mia(mia_train_probs, mia_train_labels, mia_classifier)
    return mia_test_probs, mia_test_labels, mia_classifier, shadow_clf, X_shadow_test, y_shadow_test


    
