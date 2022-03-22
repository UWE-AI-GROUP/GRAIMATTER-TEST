'''
Differentially private SVC
'''
import logging
from typing import Any
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from estimator_template import GenericEstimator

local_logger = logging.getLogger(__file__)




# pylint: disable = invalid-name

class DPSVC(GenericEstimator):
    ## Wrapper for differentially private SVM
    ##
    ## James Liley
    ## 21/03/22

    """
    Wrapper for differentially private SVM, implemented according to the method in

    https://arxiv.org/pdf/0911.5708.pdf

    Essentially approximates an infinite-dimensional latent space (and corresponding kernel) with
    a finite dimensional latent space, and adds noise to the normal to the separating hyperplane
    in this latent space.

    Only currently implemented for a radial basis kernel, but could be extended.

    More specifically
    - draws a set of dhat random vectors from a probability measure induced by the Fourier
        transform of the kernel function
    - approximates the kernel with a 2*dhat dimensional latent space
    - computes the separating hyperplane in this latent space with normal w
    - then adds Laplacian noise to w and returns it along with the map to the latent space.

    The SKlearn SVM (see https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation)
    minimises the function

    (1/2) ||w||_2 + C sum(zeta_i)

    where 1-zeta_i≤ y_i (w phi(x_i) + b), where phi maps x to the latent space and zeta_i ≥ 0.
    This is equivalent to minimising

    (1/2) ||w||_2 + C/n sum(l(y_i,f_w(x_i)))

    where l(x,y)=n*max(0,1- x.y), which is n-Lipschitz continuous in y (given x is in {-1,1})

    """

    def __init__(self, C=1., gamma='scale', dhat=1000, eps=10, **kwargs):
        self.svc = None
        self.gamma = gamma
        self.dpsvc_gamma = None
        self.dhat = dhat
        self.eps = eps
        self.C = C
        self.lambdaval = None
        self.rho = None
        self.support = None
        self.platt_transform = LogisticRegression()
        self.b = None
        self.classes_ = [0, 1]
        self.intercept = None
        self.noisy_weights = None

    @staticmethod
    def dot1(x, y):
        '''
        this is the diagonal elements of x.y^t
        '''
        return np.sum(x * y, axis=1)



    def phi_hat(self, s):
        '''TBC'''
        st = np.outer(np.ones(self.dhat), s)
        vt1 = DPSVC.dot1(self.rho, st)
        vt = (self.dhat**(-0.5)) * np.column_stack((np.cos(vt1), np.sin(vt1)))
        return vt.reshape(2*self.dhat)

    def k_hat(self, x, y):
        '''
        Define finite=dimensional kernel corresponding to phi_hat
        '''
        return np.dot(self.phi_hat(x), self.phi_hat(y))

    def k_hat_svm(self, x, y):
        '''
        Define the version which is sent to sklearn.svm. AFAICT python/numpy
        doesn't have an 'outer' for arbitrary functions.
        '''
        gram_matrix = np.zeros((x.shape[0],y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                gram_matrix[i,j] = self.k_hat(x[i,:], y[j,:])
        return gram_matrix

    def fit(self, train_features: Any, train_labels: Any) -> None:
        '''
        Fit the model
        '''

        # Check that the data passed is np.ndarray
        if not isinstance(train_features, np.ndarray) or not isinstance(train_labels, np.ndarray):
            raise NotImplementedError("DPSCV needs np.ndarray inputs")

        n_data, n_features = train_features.shape

        # Check the data passed in train_labels
        unique_labels = np.unique(train_labels)
        local_logger.info(unique_labels)
        for label in unique_labels:
            if label not in [0, 1]:
                raise NotImplementedError(
                    (
                        "DP SVC can only handle binary classification with",
                        "labels = 0 and 1"
                    )
                )

        if self.eps > 0:
            self.lambdaval = (2**2.5) * self.C * np.sqrt(self.dhat) / self.eps
        else:
            self.lambdaval = 0

        # Mimic sklearn skale and auto params
        if self.gamma == 'scale':
            self.gamma = 1. / (n_features * train_features.var())
        elif self.gamma == 'auto':
            self.gamma = 1. / n_features


        self.dpsvc_gamma = 1. / np.sqrt(2. * self.gamma) # alternative parameterisation
        local_logger.info("Gamma = %f (dp parameterisation = %f)", self.gamma, self.dpsvc_gamma)

        # Draw dhat random vectors rho from Fourier transform of RBF
        # (which is Gaussian with SD 1/gamma)
        self.rho = np.random.normal(0, 1. / self.dpsvc_gamma, (self.dhat, n_features))
        local_logger.info("Sampled rho")

        # Fit support vector machine
        logging.info("Fitting base SVM")
        self.svc=SVC(kernel=self.k_hat_svm, C=self.C)
        self.svc.fit(train_features, train_labels)


        # Get separating hyperplane and intercept
        alpha = self.svc.dual_coef_ # alpha from solved dual, multiplied by labels (-1,1)
        xi = train_features[self.svc.support_, :]  # support vectors x_i
        weights = np.zeros(2*self.dhat)
        for i in range(alpha.shape[1]):
            weights = weights + alpha[0, i] * self.phi_hat(xi[i, :])

        self.intercept = self.svc.intercept_

        # Add Laplacian noise
        self.noisy_weights = weights + np.random.laplace(0, self.lambdaval, len(weights))

        # Logistic transform for predict_proba (rough): generate predictions (DP) for training data
        ypredn = np.zeros(n_data)
        for i in range(n_data):
            ypredn[i] = np.dot(self.phi_hat(train_features[i,:]), self.noisy_weights) +\
                self.intercept

        local_logger.info("Fitting Platt scaling")
        self.platt_transform.fit(ypredn.reshape(-1, 1), train_labels) # was called ptransform


    def set_params(self, **kwargs) -> None:
        '''
        Set params
        '''
        for key, value in kwargs.items():
            if key == 'gamma':
                self.gamma = value
            elif key == 'eps':
                self.eps = value
            elif key == 'dhat':
                self.dhat = value
            else:
                local_logger.warning("Unsupported parameter: %s", key)

    def predict(self, test_features: Any) -> np.ndarray:
        '''
        Make predictions
        '''
        n_data, _ = test_features.shape
        # Return values
        outn = np.zeros(n_data)
        for i in range(n_data):
            outn[i] = np.dot(self.phi_hat(test_features[i,:]), self.noisy_weights) +\
                self.intercept

        out = 1 * (outn > 0)
        return out # Predictions

    def predict_proba(self, test_features: Any) -> np.ndarray:
        '''
        Predictive probabilities
        '''

        n_data, _ = test_features.shape
        # Return values
        outn=np.zeros(n_data)
        for i in range(n_data):
            outn[i] = np.dot(self.phi_hat(test_features[i, :]), self.noisy_weights) +\
                self.intercept

        # Push through logistic regression model
        pred_probs = self.platt_transform.predict_proba(outn.reshape(-1, 1))

        return pred_probs
