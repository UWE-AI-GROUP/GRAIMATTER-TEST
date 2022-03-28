## Wrapper for differentially private SVM
##
## James Liley
## 10/03/22

"""
Wrapper for differentially private SVM, implemented according to the method in

https://arxiv.org/pdf/0911.5708.pdf

Essentially approximates an infinite-dimensional latent space (and corresponding kernel) with a finite dimensional latent space, and adds noise to the normal to the separating hyperplane in this latent space. 

Only currently implemented for a radial basis kernel, but could be extended.

More specifically
 - draws a set of dhat random vectors from a probability measure induced by the Fourier transform of the kernel function
 - approximates the kernel with a 2*dhat dimensional latent space
 - computes the separating hyperplane in this latent space with normal w
 - then adds Laplacian noise to w and returns it along with the map to the latent space.

The SKlearn SVM (see https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation) minimises the function 

(1/2) ||w||_2 + C sum(zeta_i)

where 1-zeta_i≤ y_i (w phi(x_i) + b), where phi maps x to the latent space and zeta_i≥0. 
This is equivalent to minimising

(1/2) ||w||_2 + C/n sum(l(y_i,f_w(x_i))) 

where l(x,y)=n*max(0,1- x.y), which is n-Lipschitz continuous in y (given x is in {-1,1})

"""


## Libraries
import math
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from typing import Any

class dp_svm:
    '''
    Initiator. Takes arguments
    dhat: dimension of latent space (divided by 2). The higher dhat is, the better the latent space will approximate the 
     latent space corresponding to the true basis funciton.
    eps: epsilon, for differential privacy
    C: penalty parameter
    gamma: kernel width for radial basis kernel
    '''
    
    def __init__(self):
        
        # Set default parameters. 
        self.params={'eps': 10, 'dhat': 1000, 'C': 1,'gamma': 1}

    def fit(self, train_features: Any, train_labels: Any):
        
        # We now set up the functions needed for DP-SVM
        # As per corollary 9 in the reference, for epsilon-DP we need lambda>2^(2.5) L C sqrt(dhat)/(eps n). 
        #  Our loss function is n-Lipschitz continuous so L=n, which cancels. Values of C, dhat, and n are given. 
        # Can run with no differential privacy guarantee by setting epsilon <= 0.
        if self.params['eps']>0:
            self.lambdaval = (2**2.5)*self.params['C']*np.sqrt(self.params['dhat'])/self.params['eps']
        if self.params['eps']<= 0:
            self.lambdaval = 0
        
        # Dimensions of train_features
        n=train_features.shape[0]
        p=train_features.shape[1]
        
        # Draw dhat random vectors rho from Fourier transform of RBF (which is Gaussian with SD 1/gamma)
        self.rho=np.random.normal(0,1/self.params['gamma'],p*self.params['dhat']).reshape((self.params['dhat'],p))
        
        # Define phi_hat: finite dimensional approximation to infinite feature space
        def dot1(x,y): # this is the diagonal elements of x.y^t
            return np.sum(x*y,axis=1)
        def phi_hat(s):
            dhat=self.params['dhat']
            st=np.outer(np.ones(dhat),s)
            vt1=dot1(self.rho,st) 
            vt=(dhat**(-0.5)) * np.column_stack((np.cos(vt1),np.sin(vt1)))
            return vt.reshape(2*dhat)
        self.phi_hat=phi_hat # need this later
        
        # Define finite=dimensional kernel corresponding to phi_hat
        def k_hat(x,y):
            return np.dot(phi_hat(x),phi_hat(y))
        
        # Define the version which is sent to sklearn.svm. AFAICT python/numpy
        #  doesn't have an 'outer' for arbitrary functions.
        def k_hat_svm(x,y):
            r = np.zeros((x.shape[0],y.shape[0]))
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    r[i,j] = k_hat(x[i,:], y[j,:]) 
            return r    
        
        # Fit support vector machine
        self.cls=svm.SVC(kernel=k_hat_svm,C=self.params['C'],gamma=self.params['gamma'])
        self.cls.fit(train_features, train_labels)
        
        # Get support vectors
        self.support=train_features[self.cls.support_,:]
        
        # Get separating hyperplane and intercept
        alpha=self.cls.dual_coef_ # alpha from solved dual, multiplied by labels (-1,1)
        xi=train_features[self.cls.support_,:]  # support vectors x_i
        w=np.zeros(2*self.params['dhat'])
        for i in range(alpha.shape[1]):
            w=w+alpha[0,i]*phi_hat(xi[i,:])
        self.w=w
        self.b=self.cls.intercept_
        
        # Add Laplacian noise
        self.w_noise=np.random.laplace(0,self.lambdaval,len(w))
        
        # Logistic transform for predict_proba (rough): generate predictions (DP) for training data
        ypredn=np.zeros(train_features.shape[0])
        for i in range(train_features.shape[0]):
            ypredn[i]=np.dot(self.phi_hat(train_features[i,:]),self.w+self.w_noise) + self.b
        lx=LogisticRegression()
        lx.fit(ypredn.reshape(-1,1),train_labels)
        self.ptransform=lx
 

        
    def predict_proba(self, test_features: Any) -> np.ndarray:
        # Separating hyperplane with added noise
        wn=self.w + self.w_noise

        # Return values
        outn=np.zeros(test_features.shape[0])
        for i in range(test_features.shape[0]):
            outn[i]=np.dot(self.phi_hat(test_features[i,:]),wn) + self.b
        
        # Push through logistic regression model
        pr=self.ptransform.predict_proba(outn.reshape(-1,1))
        
        return pr
        

    def predict(self, test_features: Any) -> np.ndarray:
        
        # Separating hyperplane with added noise
        wn=self.w + self.w_noise

        # Return values
        outn=np.zeros(test_features.shape[0])
        for i in range(test_features.shape[0]):
            outn[i]=np.dot(self.phi_hat(test_features[i,:]),wn) + self.b
        out=np.where(outn>0,1,0)
        return out # Predictions

    
    def set_params(self, **kwargs):
        # Given that we will force an RBF kernel, parameters other than eps, dhat, C and gamma will be ignored.
        
        ak=list(kwargs.keys())
        av=list(kwargs.values())
        for i in range(0,len(ak)):
            self.params[ak[i]]=av[i]






### Example
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from dp_svm import dp_svm

n=40 # number of samples
p=3 # number of features

def logistic(x):
    return 1/(1+np.exp(-x))

# X,y are training data; X1,y1 are test data
coef=np.random.normal(0,1,p)
X = np.random.normal(0,1,n*p).reshape(n,p)
X1 = np.random.normal(0,1,n*p).reshape(n,p)
y = np.random.binomial(1,logistic(np.matmul(X,coef)),n).flatten()
y1 = np.random.binomial(1,logistic(np.matmul(X1,coef)),n).flatten()

# Parameters
gamma=1    # Kernel width
C=1        # Penalty term
dhat=500   # Dimension of approximator
eps=50     # DP level (not very private)

# Kernel for approximator: equivalent to rbf.
def rbf(x,y,gamma=1):
    return np.exp(-np.sum((x-y)**2) / (2* gamma**2))
def rbf_svm(x,y,gamma=1):
    r = np.zeros((x.shape[0],y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            r[i,j] = rbf(x[i,:], y[j,:],gamma) 
    return r    


# Basic SVM fitted using RBF
clf0 = svm.SVC(probability=True,kernel='rbf',gamma=gamma,C=C)
clf0.fit(X, y)
c0=clf0.predict(X1)
p0=clf0.predict_proba(X1)

# SVM fitted using approximate finite-dimensional RBF kernel
clf1 = svm.SVC(probability=True,kernel=rbf_svm,gamma=gamma,C=C)
clf1.fit(X, y)
c1=clf1.predict(X1)
p1=clf1.predict_proba(X1)

# DP version with no DP level (predicted labels equivalent to clf1; predicted probabilities will not be)
clf2 = dp_svm()
clf2.set_params(eps=-1, dhat=dhat, C=C, gamma=gamma)
clf2.fit(X,y)
c2=clf2.predict(X1)
p2=clf2.predict_proba(X1)

# DP version with DP level (approximate)
clf3 = dp_svm()
clf3.set_params(eps=eps, dhat=dhat, C=C, gamma=gamma)
clf3.fit(X,y)
c3=clf3.predict(X1)
p3=clf3.predict_proba(X1)

# Values
print([c0,c1,c2,c3])

# Plot p0 vs p1: finite-dimensional approximator works OK
plt.subplot(1, 3, 1)
plt.style.use('seaborn-whitegrid')
plt.plot(p0, p1, 'o', color='black');

# Plot p1 vs p2: logistic-regression based predict_proba is roughly equivalent to Platt scaling, at least here
plt.subplot(1, 3, 2)
plt.style.use('seaborn-whitegrid')
plt.plot(p1, p2, 'o', color='black');

# Plot p2 vs p3: enforcing differential privacy means we don't match very well. Set higher DP level, 
#  higher N, or lower C to match better.
plt.subplot(1, 3, 3)
plt.style.use('seaborn-whitegrid')
plt.plot(p2, p3, 'o', color='black');

plt.show()


'''
