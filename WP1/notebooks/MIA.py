import numpy as np
from sklearn.model_selection import train_test_split
from metricPlots import plotROC_classifier, plot_prob_test_train


def create_mia_data(clf, xtrain, xtest):
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

    miX, miY = create_mia_data(model, xtrain, xtest)
    mi_train_x, mi_test_x, mi_train_y, mi_test_y = train_test_split(miX, miY, test_size=0.2, stratify=miY)

    mi_rf = MIA_classifier
    mi_rf.fit(mi_train_x, mi_train_y)

    plotROC_classifier(mi_rf, mi_test_x, mi_test_y, clf_name+" - MIA "+MIA_name)

    mi_pred_y = mi_rf.predict_proba(mi_test_x)
    mi_pred_train_y = mi_rf.predict_proba(mi_train_x)

    plot_prob_test_train(mi_pred_y, mi_pred_train_y, clf_name+" - MIA "+MIA_name)
