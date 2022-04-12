'''
Functions to split data and perform different scenario
membership inference (MIA) attacks.
'''
from typing import Any, Iterable, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def split_target_data(X:Iterable[float],
                      y:Iterable,
                      i:Optional[float]=0.67,
                      j:Optional[float]=0.33,
                      r_state:Optional[int]=5):
    """
    This functions splits the data into 3 sets: target, train and test.
    The default split is 1/3 of the data for each of the 3 sets. The random state default is set to 5.

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
    sort_probs: sort rows by highest probability. Default: False.
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



def train_mia(mia_train_probs, mia_train_labels, mia_classifier):
    """
    Train the mia classifier.
    """
    mia_classifier.fit(mia_train_probs, mia_train_labels)
    return mia_classifier


def worst_case_mia(target_model,
                   X_target_train: Iterable[float],
                   X_test: Iterable[float],
                   prop_mia_train: float=0.5,
                   mia_classifier: Any=RandomForestClassifier()):
    """
    Worst case Membership Inference Attack (MIA) scenario.
    Creates MIA data, splits it and train the MIA model.

    X_target_train: train data of the target model.
    X_test: test data.
    prop_mia_train: Proportional data for training. Default:0.5.
    mia_clf: unfitted classifier for the MIA attack. Deafault: RandomForest.
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

def salem(target_model:Any,
          shadow_clf:str,
          X_target_train:Iterable[float],
          X_shadow: Iterable[float],
          y_shadow: Iterable,
          X_test: Iterable[float],
          prop_shadow_train: Optional[float]=0.5,
          mia_classifier: Optional[Any]=RandomForestClassifier()):
    """
    Salem scenarios 1 and 2. Both work in the same way. The difference
    is that while scenario 1 is performed with shadow data of the same
    distribution as the target data, Salem adversary 2 type of attack
    assumes that the attacker does not have a dataset of the same
    distribution as the target's model tarining data. The adversary does
    not know the structure of the target model eihter.

    target_model: fitted target model.
    shadow_clf: classifier for shadow model (not fitted).
    X_target_train
    X_shadow: shadow model dataset, which must be of the different
    distribution of the target data, e.g. could be a split of the one
    used for training/test the target model and not used to train the target.
    Default data: breast cancer data provided with sklearn datasets.
    y_shadow_train: shadow data labels. Default: labels breast cancer data
    provided with sklearn datasets.
    X_test: test data.
    prop_shadow_train: proportion of data for the test split for MIA.
    mia_classifier: unfitted classifier for the MIA attack. Deafault: RandomForest.
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
