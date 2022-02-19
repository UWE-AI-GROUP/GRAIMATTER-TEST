"""This module contains unit tests for the SafeDecisionTree wrapper."""

import numpy as np
from SafeModel import SafeDecisionTree
from sklearn import datasets


def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


def test_decisiontree_unchanged():
    """SafeDecisionTree using unchanged values."""
    x, y = get_data()
    model = SafeDecisionTree(random_state=1)
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter min_samples_leaf unchanged at recommended value 5\n"
        "- parameter min_samples_leaf decreased from recommended "
        "max value of 500 to 5. This is not problematic.\n\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is False


def test_decisiontree_safe_recommended():
    """SafeDecisionTree using recommended values."""
    x, y = get_data()
    model = SafeDecisionTree(random_state=1)
    model.min_samples_leaf = 5
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter min_samples_leaf unchanged at recommended value 5\n"
        "- parameter min_samples_leaf decreased from recommended "
        "max value of 500 to 5. This is not problematic.\n\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is False


def test_decisiontree_safe():
    """SafeDecisionTree with safe changes."""
    x, y = get_data()
    model = SafeDecisionTree(random_state=1)
    model.min_samples_leaf = 10
    model.fit(x, y)
    assert model.score(x, y) == 0.9536423841059603
    model.preliminary_check()
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter min_samples_leaf increased from recommended "
        "min value of 5 to 10. This is not problematic.\n\n"
        "- parameter min_samples_leaf decreased from recommended "
        "max value of 500 to 10. This is not problematic.\n\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is False


def test_decisiontree_unsafe():
    """SafeDecisionTree with unsafe changes."""
    model = SafeDecisionTree(random_state=1)
    model.min_samples_leaf = 1
    msg, possibly_disclosive = model.preliminary_check()
    correct_msg = (
        "- parameter min_samples_leaf decreased from recommended "
        "min value of 5 to 1. THIS IS POTENTIALLY PROBLEMATIC.\n\n"
        "- parameter min_samples_leaf decreased from recommended "
        "max value of 500 to 1. This is not problematic.\n\n"
    )
    assert msg == correct_msg
    assert possibly_disclosive is True
