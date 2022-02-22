"""This module contains unit tests for the SafeRandomForest wrapper."""

import pickle

import joblib
import numpy as np
from SafeModel import SafeRandomForest
from sklearn import datasets


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


def test_randomforest_unchanged():
    """SafeRandomForest using recommended values."""
    x, y = get_data()
    model = SafeRandomForest(random_state=1)
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_randomforest_recommended():
    """SafeRandomForest using recommended values."""
    x, y = get_data()
    model = SafeRandomForest(random_state=1)
    model.min_samples_leaf = 6
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_randomforest_unsafe_1():
    """SafeRandomForest with unsafe changes."""
    x, y = get_data()
    model = SafeRandomForest(random_state=1)
    model.bootstrap = False
    model.fit(x, y)
    assert model.score(x, y) == 0.9735099337748344
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True.\n"
        "Changed parameter bootstrap = True.\n"
    )
    assert msg == correct_msg
    assert disclosive is True


def test_randomforest_unsafe_2():
    """SafeRandomForest with unsafe changes."""
    model = SafeRandomForest(random_state=1)
    model.bootstrap = True
    model.min_samples_leaf = 2
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter min_samples_leaf = 2 identified as less than the recommended "
        "min value of 5.\n"
        "Changed parameter min_samples_leaf = 5.\n"
    )
    assert msg == correct_msg
    assert disclosive is True


def test_randomforest_unsafe_3():
    """SafeRandomForest with unsafe changes."""
    model = SafeRandomForest(random_state=1)
    model.bootstrap = False
    model.min_samples_leaf = 2
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True.\n"
        "Changed parameter bootstrap = True.\n"
        "- parameter min_samples_leaf = 2 identified as less than the recommended "
        "min value of 5.\n"
        "Changed parameter min_samples_leaf = 5.\n"
    )
    assert msg == correct_msg
    assert disclosive is True


def test_randomforest_save():
    """SafeRandomForest model saving."""
    x, y = get_data()
    model = SafeRandomForest(random_state=1, min_samples_leaf=50)
    model.fit(x, y)
    assert model.score(x, y) == 0.6622516556291391
    # test pickle
    model.save("rf_test.pkl")
    with open("rf_test.pkl", "rb") as file:
        pkl_model = pickle.load(file)
    assert pkl_model.score(x, y) == 0.6622516556291391
    # test joblib
    model.save("rf_test.sav")
    with open("rf_test.sav", "rb") as file:
        sav_model = joblib.load(file)
    assert sav_model.score(x, y) == 0.6622516556291391
