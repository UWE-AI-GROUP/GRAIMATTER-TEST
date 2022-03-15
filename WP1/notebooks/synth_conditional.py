## Wrapper to use synthetic data generation methods in a predictive-model like way
##
## James Liley
## 10/03/22

"""
Suppose a dataset Dn consists of $n$ random variables with common distribution D (possibly IID) with support Z coming from some space of distributions Q. We consider a synthetic-data generation S mechanism as a map from Dn to a distribution S(Dn) in Q. A classifier, by contrast, maps Dn to a function mapping Z to [0,1]. Our membership inference attacks are based on assessing classifiers, so we wish to represent a synthetic data generation mechanism as a classifier. 

We will suppose that data contain a dimension Y with values in {0,1}, call the remaining dimensions X, and presume that our classifier aims to predict Y|X.

Let g(D,m) represent m IID draws from distribution D in Q, and h(X,D) with X in the non-Y part of Z and D in Q represent the oracle classifier on distribution D; that is, h(X,D)=P_D(Y|X). Furthermore, let i(X,Dm) denote a classifier trained to data Dm~H^m with H in Q.

Our representation of the synthetic data generator is now

f_S(X) = h(X,S(Dn))
 = lim_{m to infinity} i(X, g(m,S(Dn)))

roughly, a perfect classifier on the synthesised distribution. The limit holds if the classifier i is sufficiently universal: that is, it can model h(X,S(Dn)) - and the parameters corresponding to h are identifiable (probably some other regulatory conditions).

This representation has the useful properties:
 1. A perfect synthetic data generating mechanism (S(Dn)~D) will correspond to an oracle classifier (a function f:Z->R with f(X)=P(Y=1|X))
 2. In some sense, the further S(Dn) is from D, the worse the classifier
 3. An 'overfitted' synthetic data generator, wherein S(Dn) has high weight only locally around instances of Dn and low weight elsewhere, will correspond to an 'overfitted' classifier, whe

Our representation of a synthetic data generator as a classifier has the downside that our synthetic data generating mechanism predicts (X,Y) and we only assess it on its prediction of Y|X. This is fairly common in the literature.

"""

## Libraries
from random import Random
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pate_gan import pategan

# Fit method for classifier for synthmethod. This is approximately f_S for S=synthmethod.
#def synthmethod_classifier.fit(Xtrain: np.array, # Train data X, part of Dn
#                               Ytrain: np.array, # Train data Y, part of Dn
#                               hyperparameters: list = defaults, # Parameters for synthmethod
#                               m: int, # m; this ideally converges to f_S as m-> infinity
#                               oracle_method: str, # Method to use for function i
#                               oracle_hyperparams: list)
  ## Call synthmethod with given hyperparameters to generate m synthetic data points
  ## Fit oracle_method to synthetic data points with parameters oracle_hyperparams
  ## Return fitted oracle method

class pategan_classifier:
    '''
    Class to treat the PATE-GAN synthetic data generation method as a classifier for MIA tests.
    '''
    
    
    def __init__(self):
        '''
        Set default parameters for synthetic data generation and for oracle model. Note 'lambda' is
        misspelled.
        '''
        self.params={'n_s': 1, 'batch_size': 64, 'k': 100,'epsilon': 100, 'delta': 0.0001,'lamda': 1}
        self.oracle_params={'bootstrap': True, 'min_samples_split': 10}
    
    
    def fit(self, 
            train_features: Any,      # Train data X, part of Dn
            train_labels: Any,        # Train data Y, part of Dn
            m=None)-> None: # ,
            #oracle_method: Any,      # Add these parameters to have a variable oracle predictor
            #oracle_hyperparameters: Any) -> None:
        '''
        1. Call pategan with given hyperparameters to generate m synthetic data points
        2. Fit oracle predictor to synthetic data points with parameters oracle_params
        3. Return fitted oracle predictor
        '''

        # By default, simulate 10x the original data volume
        if m is None:
            m=10*train_features.shape[0]
        
        print(m)
        print(self.params)
        
        # Potentially pre-process data to all-numeric features
        data=np.column_stack((train_features,train_labels))
        
        
        # Generate synthetic data using PATE-GAN with specified parameters
        synthdata=pategan(data,self.params,m)
        
        # Potentially post-process data back to original form
        self.synthX=synthdata[:,:(synthdata.shape[1]-1)]
        self.synthY=np.round(synthdata[:,synthdata.shape[1]-1])
        
        # Fit model to synthetic data
        self.predictor=RandomForestClassifier(**self.oracle_params)
        self.predictor.fit(self.synthX,self.synthY)
         

    def predict_proba(self, test_features: Any) -> np.ndarray:
        '''
        Produce predictive probabilities. Results should be a nparray with one row per row
        in test_features, and one column per class.
        '''
        return self.predictor.predict_proba(test_features)
    
    def predict(self, test_features: Any) -> np.ndarray:
        '''
        Produce hard predictions. Results should be a numpy ndarray with shape (n,) where n is the 
        number of rows in test_features
        '''
        return self.predictor.predict(test_features)

    def set_params(self, **kwargs):
        '''
        Sets hyperparameters of the synthetic data generator. All hyper-params should be passed as named
        arguments.
        '''
        self.params=kwargs

