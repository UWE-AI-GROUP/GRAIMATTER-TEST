'''
Test the metrics
'''

import unittest
import numpy as np
from attacks.metrics import get_metrics
from attacks.mia_extremecase import min_max_disc

PREDICTED_CLASS = np.array([0, 1, 0, 0, 1, 1])
TRUE_CLASS = np.array([0, 0, 0, 1, 1, 1])
PREDICTED_PROBS = np.array([
    [0.9, 0.1],
    [0.4, 0.6],
    [0.8, 0.2],
    [0.55, 0.45],
    [0.1, 0.9],
    [0.01, 0.99]
])

class DummyClassifier:
    '''
    Mocks the predict and predict_proba methods
    '''
    def predict(self, X):
        '''
        return dummy predictions
        '''
        return PREDICTED_CLASS
    def predict_proba(self, X):
        '''
        return dummy predicted probabilities
        '''
        return PREDICTED_PROBS

class TestMetrics(unittest.TestCase):
    '''
    Test the metrics with some dummy predictions
    '''
    def test_metrics(self):
        '''
        Test each individual metric with dummy data
        '''
        clf = DummyClassifier()
        testX = []
        testy = TRUE_CLASS
        metrics = get_metrics(clf, testX, testy)
        self.assertAlmostEqual(metrics['TPR'], 2 / 3)
        self.assertAlmostEqual(metrics['FPR'], 1 / 3)
        self.assertAlmostEqual(metrics['FAR'], 1 / 3)
        self.assertAlmostEqual(metrics['TNR'], 2 / 3)
        self.assertAlmostEqual(metrics['PPV'], 2 / 3)
        self.assertAlmostEqual(metrics['NPV'], 2 / 3)
        self.assertAlmostEqual(metrics['FNR'], 1 / 3)
        self.assertAlmostEqual(metrics['ACC'], 4 / 6)
        self.assertAlmostEqual(metrics['F1score'], (8 / 9) / (2 / 3 + 2 / 3))
        self.assertAlmostEqual(metrics['Advantage'], 1 / 3)
        self.assertAlmostEqual(metrics['AUC'], 8 / 9)

class TestExtrete(unittest.TestCase):
    '''
    Test the extreme metrics
    '''
    def test_extreme_default(self):
        '''
        Tets with the dummy data
        '''
        pred_probs = DummyClassifier().predict_proba(None)[:, 1]
        maxd, mind, mmd, _ = min_max_disc(TRUE_CLASS, pred_probs)

        # 10% of 6 is 1 so:
        # maxd should be 1 (the highest one is predicted as1)
        # mind should be 0 (the lowest one is not predicted as1)
        self.assertAlmostEqual(maxd, 1.0)
        self.assertAlmostEqual(mind, 0.0)
        self.assertAlmostEqual(mmd, 1.0)

    def test_extreme_higer_prop(self):
        '''
        Tets with the dummy data but increase proportion to 0.5
        '''
        pred_probs = DummyClassifier().predict_proba(None)[:, 1]
        maxd, mind, mmd, _ = min_max_disc(TRUE_CLASS, pred_probs, x_prop=0.5)

        # 10% of 6 is 1 so:
        # maxd should be 1 (the highest one is predicted as1)
        # mind should be 0 (the lowest one is not predicted as1)
        self.assertAlmostEqual(maxd, 2 / 3)
        self.assertAlmostEqual(mind, 1 / 3)
        self.assertAlmostEqual(mmd, 1 / 3)
