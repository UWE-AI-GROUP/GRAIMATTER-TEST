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
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter bootstrap unchanged at recommended value True"
        "- parameter min_samples_leaf unchanged at recommended value 5\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is False


def test_randomforest_recommended():
    """SafeRandomForest using recommended values."""
    x, y = get_data()
    model = SafeRandomForest(random_state=1)
    model.min_samples_leaf = 6
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter bootstrap unchanged at recommended value True"
        "- parameter min_samples_leaf increased from recommended min value of 5 to 6. "
        "This is not problematic.\n\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is False


def test_randomforest_unsafe_1():
    """SafeDecisionTree with unsafe changes."""
    x, y = get_data()
    model = SafeRandomForest(random_state=1)
    model.bootstrap = False
    model.fit(x, y)
    assert model.score(x, y) == 0.9735099337748344
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter bootstrap changed from recommended fixed value of True to False. "
        "THIS IS POTENTIALLY PROBLEMATIC.\n"
        "- parameter min_samples_leaf unchanged at recommended value 5\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is True


def test_randomforest_unsafe_2():
    """SafeDecisionTree with unsafe changes."""
    model = SafeRandomForest(random_state=1)
    model.bootstrap = True
    model.min_samples_leaf = 2
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter bootstrap unchanged at recommended value True"
        "- parameter min_samples_leaf decreased from recommended min value of 5 to 2. "
        "THIS IS POTENTIALLY PROBLEMATIC.\n\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is True


def test_randomforest_unsafe_3():
    """SafeDecisionTree with unsafe changes."""
    model = SafeRandomForest(random_state=1)
    model.bootstrap = False
    model.min_samples_leaf = 2
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter bootstrap changed from recommended fixed value of True to False. "
        "THIS IS POTENTIALLY PROBLEMATIC.\n"
        "- parameter min_samples_leaf decreased from recommended min value of 5 to 2. "
        "THIS IS POTENTIALLY PROBLEMATIC.\n\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is True
