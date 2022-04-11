'''
Test thw scenario code
'''
import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from WP1.notebooks.scenarios import create_mia_data # pylint: disable = import-error


class TestSharedMethods(unittest.TestCase):
    '''
    Methods to test shared code
    '''
    def test_create_mia_data(self):
        '''
        Test method for creating mia data
        Checks that the mia data has the correct number of rows
        '''
        # Load iris and make it a binary classification problem
        iris_x, iris_y = load_iris(return_X_y=True)
        pos = np.where(iris_y == 2)
        iris_y[pos] = 1

        # Split into train and test sets
        train_x, test_x, train_y, _ = train_test_split(iris_x, iris_y, test_size=50)

        # Fit a logistic regression
        logreg = LogisticRegression()
        logreg.fit(train_x, train_y)

        # Test that the method with default params produces the correct output
        mi_x, mi_y = create_mia_data(logreg, train_x, test_x)

        self.assertTupleEqual(mi_x.shape, (len(iris_x), 2))
        self.assertTupleEqual(mi_y.shape, (len(iris_x),))

        mi_x, mi_y = create_mia_data(logreg, train_x, test_x, sort_probs=True)

        self.assertTupleEqual(mi_x.shape, (len(iris_x), 2))
        self.assertTupleEqual(mi_y.shape, (len(iris_x),))

        self.assertEqual(mi_x.shape[1], 2)

        mi_x, mi_y = create_mia_data(logreg, train_x, test_x, sort_probs=True, keep_top=1)

        self.assertTupleEqual(mi_x.shape, (len(iris_x), 1))
        self.assertTupleEqual(mi_y.shape, (len(iris_x),))
