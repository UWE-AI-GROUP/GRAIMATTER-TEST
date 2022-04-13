'''
Test the metrics
'''

import unittest
import numpy as np
from attacks.metrics import get_metrics

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
        
        
        

        
        
        
