"""This module contains unit tests for the SafeSVC."""

import pickle
import os
import sys
import joblib
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath("")))
sys.path.append(ROOT_DIR)

from safe_model.SafeModel import SafeSVC
from sklearn import datasets



def get_data():
    """Returns data for testing."""
    iris = datasets.load_iris()
    x = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.float64)
    x = np.vstack([x, (7, 2.0, 4.5, 1)])
    y = np.append(y, 4)
    return x, y


def test_svc_unchanged():
    """SafeSVC using recommended values."""
    x, y = get_data()
    model = SafeSVC(random_state=1)
    model.fit(x, y)
    rint (model.score)
    #assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_svc_recommended():
    """SafeSVC using recommended values."""
    x, y = get_data()
    model = SafeSVC(random_state=1)
    #model.min_samples_leaf = 6
    model.fit(x, y)
    print(f"model.dict={model.__dict__}")
    #assert model.score(x, y) == 0.9668874172185431
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False


def test_svc_unsafe_1():
    """SafeSVC with unsafe changes."""
    x, y = get_data()
    model = SafeSVC(random_state=1)
    #model.bootstrap = False
    model.fit(x, y)
    #assert model.score(x, y) == 0.9735099337748344
    msg, disclosive = model.preliminary_check()
    correct_msg = (
        "WARNING: model parameters may present a disclosure risk:\n"
        "- parameter bootstrap = False identified as different than the recommended "
        "fixed value of True."
    )
    assert msg == correct_msg
    assert disclosive is True


def test_svc_save():
    """SafeSVC model saving."""
    x, y = get_data()
    model = SafeSVC(random_state=1)
    model.fit(x, y)
    assert model.score(x, y) == 0.6622516556291391
    # test pickle
    model.save("svc_test.pkl")
    with open("svc_test.pkl", "rb") as file:
        pkl_model = pickle.load(file)
    #assert pkl_model.score(x, y) == 0.6622516556291391
    # test joblib
    model.save("svc_test.sav")
    with open("svc_test.sav", "rb") as file:
        sav_model = joblib.load(file)
    #assert sav_model.score(x, y) == 0.6622516556291391


def test_svc_hacked_postfit():
    """SafeSVC changes made to parameters after fit() called."""
    x, y = get_data()
    model = SafeSVC(random_state=1)
    model.bootstrap = False
    model.fit(x, y)
    #assert model.score(x, y) == 0.9735099337748344
    model.bootstrap = True
    msg, disclosive = model.preliminary_check()
    correct_msg = "Model parameters are within recommended ranges.\n"
    assert msg == correct_msg
    assert disclosive is False
    msg2, disclosive2 = model.posthoc_check()
    correct_msg2 = (
        "Warning: basic parameters differ in 1 places:\n"
        "parameter bootstrap changed from False to True after model was fitted\n"
    )
    print(msg2)
    assert msg2 == correct_msg2
    assert disclosive2 is True
