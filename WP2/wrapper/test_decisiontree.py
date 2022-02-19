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
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_safe_recommended():
    """SafeDecisionTree using recommended values."""
    x, y = get_data()
    model = SafeDecisionTree(random_state=1)
    model.min_samples_leaf = 5
    model.fit(x, y)
    assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_safe():
    """SafeDecisionTree with safe changes."""
    x, y = get_data()
    model = SafeDecisionTree(random_state=1)
    model.min_samples_leaf = 10
    model.fit(x, y)
    assert model.score(x, y) == 0.9536423841059603
    model.preliminary_check()
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_decisiontree_unsafe():
    """SafeDecisionTree with unsafe changes."""
    model = SafeDecisionTree(random_state=1)
    model.min_samples_leaf = 1
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "Model parameters are within recommended ranges.\n"
        "- parameter min_samples_leaf = 1 identified as less than the recommended "
        "min value of 5.\n"
        "Changed parameter min_samples_leaf = 5.\n"
    )
    assert msg == correct_msg
    assert disclosive is False
