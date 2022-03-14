'''
Template for how to put any estimator into the experimental pipeline
'''
from random import Random
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class GenericEstimator:
    '''
    Base (abstract) class. Just provided to raise errors if a child class does not have the methods.
    This just shows the methods any class needs
    '''
    def fit(self, train_features: Any, train_labels: Any) -> None:
        '''
        Fit the model to the provided data
        '''
        raise NotImplementedError

    def predict_proba(self, test_features: Any) -> np.ndarray:
        '''
        Produce predictive probabilities. Results should be a nparray with one row per row
        in test_features, and one column per class.
        '''
        raise NotImplementedError
    
    def predict(self, test_features: Any) -> np.ndarray:
        '''
        Produce hard predictions. Results should be a numpy ndarray with shape (n,) where n is the 
        number of rows in test_features
        '''
        raise NotImplementedError

    def set_params(self, **kwargs):
        '''
        Method to set the hyper-params of the model. All hyper-params should be passed as named
        arguments.
        '''
        raise NotImplementedError


class ExampleWrapper(GenericEstimator):
    '''
    Example for how to make a class that will fit in. This is just a wrapper for the sklearn
    Random Forest, but could be anything.
    '''
    def __init__(self):
        self.random_forest = RandomForestClassifier()

    def fit(self, train_features: Any, train_labels: Any):
        self.random_forest.fit(train_features, train_labels)

    def predict_proba(self, test_features: Any) -> np.ndarray:
        return self.random_forest.predict_proba(test_features)

    def predict(self, test_features: Any) -> np.ndarray:
        print(self.random_forest.predict(test_features).shape)
        return self.random_forest.predict(test_features)

    def set_params(self, **kwargs):
        self.random_forest.set_params(**kwargs)
