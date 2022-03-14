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

roughly, a perfect classifier on the synthesised distribution. The limit holds if the classifier i is sufficiently universal: that is, it can model h(X,S(Dn)).

This representation has the useful properties:
 1. A perfect synthetic data generating mechanism (S(Dn)~D) will correspond to an oracle classifier (a function f:Z->R with f(X)=P(Y=1|X))
 2. In some sense, the further S(Dn) is from D, the worse the classifier
 3. An 'overfitted' synthetic data generator, wherein S(Dn) has high weight only locally around instances of Dn and low weight elsewhere, will correspond to an 'overfitted' classifier, whe

Our representation of a synthetic data generator as a classifier has the downside that our synthetic data generating mechanism predicts (X,Y) and we only assess it on its prediction of Y|X. This is fairly common in the literature.

"""

# Fit method for classifier for synthmethod. This is approximately f_S for S=synthmethod.
def synthmethod_classifier.fit(Xtrain: np.array, # Train data X, part of Dn
                               Ytrain: np.array, # Train data Y, part of Dn
                               hyperparameters: list = defaults, # Parameters for synthmethod
                               m: int, # m; this ideally converges to f_S as m-> infinity
                               oracle_method: str, # Method to use for function i
                               oracle_hyperparams: list)
  ## Call synthmethod with given hyperparameters to generate m synthetic data points
  ## Fit oracle_method to synthetic data points with parameters oracle_hyperparams
  ## Return fitted oracle method
                   
