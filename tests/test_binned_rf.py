'''
Test the binned probability random forest
'''
import unittest
import math
import numpy as np
from safemodel.classifiers import RFBinnedOutput
from safemodel.classifiers.rf_binned_output_probs import bin_probabilities
from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True)

def round_half_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n*multiplier - 0.5) / multiplier

class TestBinning(unittest.TestCase):
    '''Test the basic binning method'''
    def test_binning(self):
        '''Ten bins'''
        probs = np.array(
            [
                [0.22222, 1 - 0.22222],
                [0.65, 0.35],
                [0.94, 0.06],
                [0.95, 0.05]
            ]
        )
        binned_probs = bin_probabilities(probs, 10)
        self.assertEqual(binned_probs[0, 0], 0.2)
        self.assertEqual(binned_probs[0, 1], 0.8)
        self.assertAlmostEqual(binned_probs[1, 0], 0.6/0.9)
        self.assertAlmostEqual(binned_probs[1, 1], 0.3/0.9)
        self.assertEqual(binned_probs[2, 0], 0.9)
        self.assertEqual(binned_probs[2, 1], 0.1)
        self.assertEqual(binned_probs[3, 0], 1.0)
        self.assertEqual(binned_probs[3, 1], 0.0)


    def test_5_bins(self):
        '''5 bins'''
        probs = np.array(
            [
                [0.12, 0.88],
                [0.3, 0.7]
            ]
        )
        binned_probs = bin_probabilities(probs, 5)
        self.assertEqual(binned_probs[0, 0], 0.2)
        self.assertEqual(binned_probs[0, 1], 0.8)
        self.assertAlmostEqual(binned_probs[1, 0], 0.2 / 0.8)
        self.assertAlmostEqual(binned_probs[1, 1], 0.6 / 0.8)



class TestBinnedOutput(unittest.TestCase):
    '''Tests for the binned output'''
    def test_no_binning(self):
        '''Test that no binning gives the same as normal'''
        rf = RFBinnedOutput(n_probability_bins=0)
        rf.fit(X, y)
        probs_new = rf.predict_proba(X)
        probs_old = rf.original_predict_proba(X)
        self.assertTrue((probs_new == probs_old).all())

    def test_ten_bins(self):
        '''
        Test ten probability bins
        '''
        rf = RFBinnedOutput(n_probability_bins=10)
        rf.fit(X, y)
        probs_new = rf.predict_proba(X)
        probs_old = rf.original_predict_proba(X)
        self.assertFalse((probs_new == probs_old).all())

        for i, row in enumerate(probs_old):
            norm_row = []
            for j, val in enumerate(row):
                rounded_old_val = round_half_down(val, decimals=1)
                norm_row.append(rounded_old_val)
            norm_row = np.array(norm_row)
            norm_row /= norm_row.sum()
            # print(probs_old[i, :], norm_row, probs_new[i, :])
            for j, val in enumerate(norm_row):
                # print(probs_old[i, :], print(norm_row))#, print(probs_new[i, :]))
                self.assertAlmostEqual(val, probs_new[i, j])
