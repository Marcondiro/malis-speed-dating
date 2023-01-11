import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator

class GMM(BaseEstimator):
    """
    A Gaussian Mixture Model based, sklearn-compatible estimator, inheriting the BaseEstimator class.
    """
    def __init__(self, n_components=1, covariance_type='full', pi='auto'):
        # save the values that have been passed
        # WARNING - this function should have NO logic
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.pi = pi


    def fit(self, X, y, **kwargs):
        # first, we implement the parameter logic
        # initialize the models
        self.gmm0_ = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)
        self.gmm1_ = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)
        if self.pi == 'auto':
            # then, we set pi as the proportion of True samples
            self.pi = np.sum(y)/len(y)

        # --- check that data dimensions are compatible --- #
        if X.shape[0] != y.shape[0]:
            raise ValueError('Input data has incompatible dimensions')

        # split X into X0 and X1
        X0 = X[y == 0]
        X1 = X[y == 1]
        # train gmm0 on X0 and gmm1 on X1
        self.gmm0_.fit(X0)
        self.gmm1_.fit(X1)
        return self


    def predict(self, X):
        # get log-likelihood for each gmm
        ll1 = self.gmm1_.score_samples(X)
        ll0 = self.gmm0_.score_samples(X)
        # compute the threshold
        t = -np.log(self.pi / (1 - self.pi))
        # get the score
        S = ll1-ll0
        predL = S > t
        return predL.astype(int)