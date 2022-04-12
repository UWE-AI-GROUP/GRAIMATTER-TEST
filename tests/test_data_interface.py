'''
Test the data interface code
'''

import unittest
import os,sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "GRAIMatter"))
from data_preprocessing.data_interface import get_data_sklearn, UnknownDataset  # pylint: disable = import-error


class TestLoaders(unittest.TestCase):
    '''
    Test the data loaders
    '''
    def test_nursery(self):
        '''Nursery data'''
        feature_df, target_df = get_data_sklearn("nursery")
        self.assertIsInstance(feature_df, pd.DataFrame)
        self.assertIsInstance(target_df, pd.DataFrame)

    def test_unknown(self):
        '''
        Test that a nonsense string raises the correct exception
        '''
        with self.assertRaises(UnknownDataset):
            _, _ = get_data_sklearn("NONSENSE")

    def test_standard(self):
        '''
        Test that standardisation creates standard features
        '''
        feature_df, _ = get_data_sklearn("standard mimic2-iaccd")
        for column in feature_df.columns:
            temp = feature_df[column].mean()
            self.assertAlmostEqual(temp, 0.0)
            temp = feature_df[column].std()
            self.assertAlmostEqual(temp, 1.0)

    def test_minmax(self):
        '''
        Test the minmax scaling
        '''
        feature_df, _ = get_data_sklearn("minmax indian liver")
        for column in feature_df.columns:
            temp = feature_df[column].min()
            self.assertAlmostEqual(temp, 0.0)
            temp = feature_df[column].max()
            self.assertAlmostEqual(temp, 1.0)
