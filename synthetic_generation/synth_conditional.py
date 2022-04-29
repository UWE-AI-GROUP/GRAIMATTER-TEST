## Wrapper to use synthetic data generation methods in a predictive-model like way
##
## James Liley
## 10/03/22

## Libraries - general
import numpy as np
import pandas as pd

## Libraries - SDG specific
import synthetic_generation.sdg_utilities.ron_gauss as ron_gauss
import sdv

## More specific functions
from random import Random
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from synthetic_generation.sdg_utilities.pate_gan import pategan


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
 3. An 'overfitted' synthetic data generator, wherein S(Dn) has high weight only locally around instances of Dn and low weight elsewhere, will correspond to an 'overfitted' classifier.

Our representation of a synthetic data generator as a classifier has the downside that our synthetic data generating mechanism predicts (X,Y) and we only assess it on its prediction of Y|X. 

In general, data-generator-as-classifier methods work for a data generating method synthmethod and a universally representative classifier oracle_method by
 - calling synthmethod with given hyperparameters to generate m synthetic data points
 - fiting oracle_method to synthetic data points with parameters oracle_hyperparams
 - returning a fitted oracle method

"""


##############################################################################
## PATE-GAN                                                                 ##
##############################################################################

## PATE-GAN (private aggregation of teacher ensembles)
## PATEGAN is a GAN-based method (https://openreview.net/pdf?id=S1zk9iRqF7). It uses a generator G to 
##  generate putative synthetic data and a discriminator D trained adversarially to recognise synthetic 
##  data from non-synthetic data. D consists of two parts: a set of 'teacher' discriminators {Ti}, each 
##  of which can see only one of a set of disjoint subsets of the data, and a 'student' discriminator S. 
##  S is an arbitrary non-DP classifier trained by taking public, unlabelled data and using {T_i} to 
##  generate labels for it: that is, it is trained to learn the behaviour of {Ti}. Output from the 
##  generator goes to {Ti}, so Ti are fitted to learn the behaviour of G. G is then optimised to 
##  minimise error in S. The reason for doing it this way is that the only bit that needs to be 
##  differentially private is the output of {Ti}: and since this is a set of discriminators, this can be
##  done by just assembles the aggregated results from each Ti and adding noise to this aggregate (so 
##  only adding noise to the output, rather than anything internal in the NN). The 'public, unlabelled 
##  data' used to train S come from the generator G. The algorithm outputs G. 
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
        
        # Convert arguments to np.array if not already; this saves some problems
        if type(train_features)==pd.core.frame.DataFrame:
            train_features=train_features.to_numpy()
        if type(train_labels)==pd.core.frame.DataFrame:
            train_labels=train_labels.to_numpy()
                    
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
        ak=list(kwargs.keys())
        av=list(kwargs.values())
        for i in range(0,len(ak)):
            self.params[ak[i]]=av[i]


## General synthetic data sampler
## Run to generate synthetic data samples
## For parameter explanations, see pate_gan.py
## X and y should be numpy arrays
## m is the number of synthetic data samples to generate, defaulting to the number of samples in X
def synth_pategan(X: Any,
                  y: Any,
                  params={'n_s': 1, 'batch_size': 64, 'k': 100,'epsilon': 100, 'delta': 0.0001,'lamda': 1},
                  m=None):

    if m is None:
        m=X.shape[0]
 
    # Convert arguments to np.array if not already; this saves some problems
    if type(X)==pd.core.frame.DataFrame:
            X=X.to_numpy()
    if type(y)==pd.core.frame.DataFrame:
            y=y.to_numpy()

    data=np.column_stack((X,y))
    synthdata=pategan(data,params,m)
    return synthdata



##############################################################################
## RON-GAUSS                                                                ##
##############################################################################

## RON-GAUSS is a method which makes use of an effect that a high-dimensional
##  linear projection of most dataset (under some regularity conditions) is 
##  approximately Gaussian. By resampling from the implied Gaussian and inverting
##  the transformation, synthetic data can be sampled, and noise can be readily
##  added for differential privacy. See https://arxiv.org/abs/1709.00054
class rongauss_classifier:
    '''
    Class to treat the RON-GAUSS synthetic data generation method as a classifier for MIA tests.
    '''
    def __init__(self):
        '''
        Set default parameters for synthetic data generation and for oracle model. For RON-GAUSS, 
        these are only DP parameters
        '''
        self.params={'epsilon': 100, 'delta': 0.0001}
        self.oracle_params={'bootstrap': True, 'min_samples_split': 10}
    
    
    def fit(self, 
            train_features: Any,      # Train data X, part of Dn
            train_labels: Any,        # Train data Y, part of Dn
            m=None)-> None: # ,
            #oracle_method: Any,      # Add these parameters to have a variable oracle predictor
            #oracle_hyperparameters: Any) -> None:
        '''
        1. Call ron-Gauss with given hyperparameters to generate m synthetic data points
        2. Fit oracle predictor to synthetic data points with parameters oracle_params
        3. Return fitted oracle predictor
        '''

        # By default, simulate 10x the original data volume
        if m is None:
            m=10*train_features.shape[0]
        
        # Initialise object
        model = ron_gauss.RONGauss(int(train_features.shape[1] / 4 + 1), 
                                   self.params['epsilon'], 
                                   self.params['delta'], 
                                   conditional=False)
        
        # Convert arguments to np.array if not already; this saves some problems
        if type(train_features)==pd.core.frame.DataFrame:
            train_features=train_features.to_numpy()
        if type(train_labels)==pd.core.frame.DataFrame:
            train_labels=train_labels.to_numpy()
                    
        # Potentially pre-process data to all-numeric features
        data=np.column_stack((train_features,train_labels))
                
        # Generate synthetic data using PATE-GAN with specified parameters
        X_syn, y_syn, mu_dp = model.generate(train_features, train_labels,n_samples=m,
                                             max_y=np.max(train_labels))
        
        # Turn y_syn back to binary
        y_syn=np.where(y_syn > 0.5, 1, 0)
        
        # Potentially post-process data back to original form
        self.synthX=X_syn
        self.synthY=y_syn
        
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
        ak=list(kwargs.keys())
        av=list(kwargs.values())
        for i in range(0,len(ak)):
            self.params[ak[i]]=av[i]

            
            
## General synthetic data sampler
## Run to generate synthetic data samples
## For parameter explanations, see pate_gan.py
## X and y should be numpy arrays
## m is the number of synthetic data samples to generate, defaulting to the number of samples in X
def synth_rongauss(X: Any,
                  y: Any,
                  params={'epsilon': 100, 'delta': 0.0001},
                  m=None):

    if m is None:
        m=X.shape[0]
    
    # Convert arguments to np.array if not already; this saves some problems
    if type(X)==pd.core.frame.DataFrame:
            X=X.to_numpy()
    if type(y)==pd.core.frame.DataFrame:
            y=y.to_numpy()
            
    model = ron_gauss.RONGauss(int(X.shape[1] / 4 + 1), 
                                   params['epsilon'], 
                                   params['delta'], 
                                   conditional=False)
        
    X_syn, y_syn, mu_dp = model.generate(X, y,n_samples=m,
                                         max_y=np.max(y))

    #print(y_syn[0:50])
    
    synthdata=np.column_stack((X_syn,y_syn))

    return synthdata
            
            
            
##############################################################################
## Vine Copula (non-GP)                                                     ##
##############################################################################

## Copula synthetic data generator. Empirical marginal CDFs of data F={F_i,i in 1:p} are generated. 
##  The function F_inv = {F_i^(-1) (X_i), i in 1:p} is then defined and applied to the data X={Xi},
##  meaning that F_inv(X) has a copula distribution (unit uniform marginals). This is approximated
##  as a Gaussian copula (that is, corresponding to a multivariate Gaussian distribution) whose 
##  covariances are then approximated. The copula is then sampled giving a value x' and a synthetic
##  data sample is generated as F(x'). See https://arxiv.org/abs/1812.01226
class sdv_copula_classifier:
    '''
    Class to treat the SDV copula method as a classifier for MIA tests.
    '''
    
    
    def __init__(self):
        '''
        Set default parameters for synthetic data generation and for oracle model. For RON-GAUSS, 
        these are only DP parameters
        '''
        self.params={'epsilon': 100, 'delta': 0.0001}
        self.oracle_params={'bootstrap': True, 'min_samples_split': 10}
    
    
    def fit(self, 
            train_features: Any,      # Train data X, part of Dn
            train_labels: Any,        # Train data Y, part of Dn
            m=None)-> None: # ,
            #oracle_method: Any,      # Add these parameters to have a variable oracle predictor
            #oracle_hyperparameters: Any) -> None:
        '''
        1. Call sdv.copula with given hyperparameters to generate m synthetic data points
        2. Fit oracle predictor to synthetic data points with parameters oracle_params
        3. Return fitted oracle predictor
        '''

        # By default, simulate 10x the original data volume
        if m is None:
            m=10*train_features.shape[0]
        
        # Initialise object
        model = sdv.tabular.GaussianCopula()
            
        # Convert arguments to a pandas data frame
        if type(train_features)==np.ndarray:
            train_features=pd.DataFrame(train_features)
        if type(train_labels)==np.ndarray:
            train_labels=pd.DataFrame(train_labels)
                    
        # Potentially pre-process data to all-numeric features
        data=pd.concat([train_features,train_labels],axis=1)
              
        # Columns need names
        if data.columns.values.dtype=='int64':
            CN=['X'+str(x) for x in range(train_features.shape[1])]
            CN.append('y')
            data.columns=CN
    
        # Fit synthetic data generating mechanism    
        model.fit(data)    
            
        # Generate synthetic data using PATE-GAN with specified parameters
        sdata=model.sample(num_rows=m)
        X_syn=sdata.to_numpy()[:,0:(sdata.shape[1]-1)]
        y_syn=sdata.to_numpy()[:,sdata.shape[1]-1]
       
        # Potentially post-process data back to original form
        self.synthX=X_syn
        self.synthY=y_syn
        
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
        ak=list(kwargs.keys())
        av=list(kwargs.values())
        for i in range(0,len(ak)):
            self.params[ak[i]]=av[i]

            
## General synthetic data sampler
## Run to generate synthetic data samples
## For parameter explanations, see 
## X and y should be numpy arrays
## m is the number of synthetic data samples to generate, defaulting to the number of samples in X
def synth_copula(X: Any,
                  y: Any,
                  params={'epsilon': 100, 'delta': 0.0001},
                  m=None):

    if m is None:
        m=X.shape[0]
    
    # Initialise object
    model = sdv.tabular.GaussianCopula()
            
    # Convert arguments to a pandas data frame
    if type(X)==np.ndarray:
        X=pd.DataFrame(X)
    if type(y)==np.ndarray:
        y=pd.DataFrame(y)
                    
    # Potentially pre-process data to all-numeric features
    data=pd.concat([X,y],axis=1)
              
    # Columns need names
    if data.columns.values.dtype=='int64':
        CN=['X'+str(x) for x in range(X.shape[1])]
        CN.append('y')
        data.columns=CN
    
    # Fit synthetic data generating mechanism    
    model.fit(data)    
            
    # Generate synthetic data using PATE-GAN with specified parameters
    sdata=model.sample(num_rows=m)
    X_syn=sdata.to_numpy()[:,0:(sdata.shape[1]-1)]
    y_syn=sdata.to_numpy()[:,sdata.shape[1]-1]

    synthdata=np.column_stack((X_syn,y_syn))

    return synthdata


## Utility function - ROC curve, for convenience
# Sensitivity and specificity at range of cutoffs
def roc_xy(ypred,y,res=0):
    yt=sum(y); yl=len(y)
    opred=np.argsort(ypred) 
    sy=y[opred]; sp=ypred[opred]

    sens=1- (np.cumsum(sy)/yt)
    spec=np.cumsum(1-sy)/(yl-yt)
    
    # coarsen; choose points regularly along arc length. Speeds up and privatises plot drawing
    if res>0:
       ds=np.cumsum(math.sqrt((spec[0:(yl-1)]-spec[1:yl])**2 + (sens[0:(yl-1)]-sens[1:yl])**2))
       ds=ds/ds[yl-1]
       lsp=list(range(1,yl-1))/yl
       sub=np.round(yl*np.interp(np.linspace(0,max(ds),num=res),ds,lsp))
       sens=sens[sub]
       spec=spec[sub]
    
    return np.column_stack((sens,spec))



# Example use
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from dp_svm import dp_svm

# Paths
import sys
sys.path.append('/home/ec2-user/GRAIMatter/WP1/models')
sys.path.append('/home/ec2-user/GRAIMatter/WP1/notebooks')

# Synthetic data generator wrappers
import synth_conditional as sc

n=400 # number of samples
p=3 # number of features

def logistic(x):
    return 1/(1+np.exp(-x))

# X,y are training data; X1,y1 are test data
coef=np.random.normal(0,1,p)
X = np.random.normal(0,1,n*p).reshape(n,p)
X1 = np.random.normal(0,1,n*p).reshape(n,p)
y = np.random.binomial(1,logistic(np.matmul(X,coef)),n).flatten()
y1 = np.random.binomial(1,logistic(np.matmul(X1,coef)),n).flatten()

# Set some parameters to save time
pategan_parameters={'n_s': 1, 'batch_size': 4, 'k': 10,'epsilon': 100, 'delta': 0.0001,'lamda': 1}

# Raw synthetic data
s1=sc.synth_pategan(X,y,params=pategan_parameters,m=n)
s2=sc.synth_rongauss(X,y,m=n)
s3=sc.synth_copula(X,y)

# Classifiers
c1=sc.pategan_classifier()
c1.set_params(n_s=1, batch_size=4, k=10,epsilon=100, delta=0.0001,lamda=1)
c1.fit(train_features=X,train_labels=y)
p1=c1.predict_proba(test_features=X1)

c2=sc.rongauss_classifier()
c2.set_params(epsilon=100,delta=0.0001)
c2.fit(train_features=X,train_labels=y)
p2=c2.predict_proba(test_features=X1)

c3=sc.sdv_copula_classifier()
c3.fit(train_features=X,train_labels=y)
p3=c3.predict_proba(test_features=X1)


# Plot: copula is not bad
import matplotlib.pyplot as plt
rx=sc.roc_xy(p3[:,1],y1)
plt.plot(1-rx[:,1],rx[:,0],label="Orig",color="black")
plt.plot([0,1],[0,1],linestyle="--",linewidth=1,color="black")

"""