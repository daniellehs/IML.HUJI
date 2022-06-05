from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # # define the classes
        # self.classes_ = np.unique(y)
        # # calculate mu and sigma over the classes
        # self.mu_ = np.ones((X.shape[1], X.shape[1]))
        #
        # self.cov_ = []
        # self._cov_inv = []
        # self.pi_ = np.zeros(len(self.classes_))
        # for i in range(len(self.classes_)):
        #     self.mu_[i] = (np.mean(X[y == self.classes_[i]]))
        #     self.cov_.append(np.cov(X[y == self.classes_[i]]))
        #     self.pi_[i] = (np.sum(X[y == self.classes_[i]]) / y.shape[0])
        # for i in range(len(self.cov_)):
        #     self._cov_inv.append(np.linalg.inv(self.cov_[i]))

        # define the classes
        self.classes_ = np.unique(y)
        # calculate mu and sigma over the classes
        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        self._cov_inv = []
        self.pi_ = np.zeros(len(self.classes_))
        for i in range(len(self.classes_)):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)
            self.cov_ += ((X[y == self.classes_[i]] - self.mu_[i]).T @ (X[y == self.classes_[i]] - self.mu_[i]))
            # self.cov_.append(np.cov(X[y == self.classes_[i]]))
            self.pi_[i] = (np.sum(y == self.classes_[i]) / y.shape[0])
        self.cov_ = self.cov_ / (X.shape[0] - len(self.classes_))
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self.likelihood(X), axis=1)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        a = self._cov_inv @ self.mu_.T
        b = np.log(self.pi_) - 0.5 * np.diagonal(self.mu_ @ self._cov_inv @ self.mu_.T)
        return X @ a + b


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        loss = loss_functions.misclassification_error(y_pred.flatten(), y.flatten())
        return loss
