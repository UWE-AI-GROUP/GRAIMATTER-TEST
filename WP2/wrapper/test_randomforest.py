"""This module contains unit tests for the SafeRandomForest wrapper."""

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
    """SafeDecisionTree with unsafe changes."""
    x, y = get_data()
    model = SafeRandomForest(random_state=1)
    model.bootstrap = False
    model.fit(x, y)
    assert model.score(x, y) == 0.9735099337748344
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "Model parameters are within recommended ranges.\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True.\n"
        "Changed parameter bootstrap = True.\n"
    )
    assert msg == correct_msg
    assert disclosive is False


def test_randomforest_unsafe_2():
    """SafeDecisionTree with unsafe changes."""
    model = SafeRandomForest(random_state=1)
    model.bootstrap = True
    model.min_samples_leaf = 2
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "Model parameters are within recommended ranges.\n"
        "- parameter min_samples_leaf = 2 identified as less than the recommended "
        "min value of 5.\n"
        "Changed parameter min_samples_leaf = 5.\n"
    )
    assert msg == correct_msg
    assert disclosive is False


def test_randomforest_unsafe_3():
    """SafeDecisionTree with unsafe changes."""
    model = SafeRandomForest(random_state=1)
    model.bootstrap = False
    model.min_samples_leaf = 2
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "Model parameters are within recommended ranges.\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True.\n"
        "Changed parameter bootstrap = True.\n"
        "- parameter min_samples_leaf = 2 identified as less than the recommended "
        "min value of 5.\n"
        "Changed parameter min_samples_leaf = 5.\n"
    )
    assert msg == correct_msg
    assert disclosive is False
