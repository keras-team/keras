from __future__ import absolute_import
import numpy as np

class KerasClassifier(object):
    """
    Implementation of the scikit-learn classifier API for Keras.

    Parameters
    ----------
    model : object, optional
        A pre-compiled Keras model is required to use the scikit-learn wrapper.
    """
    def __init__(self, model=None):
        self.model = model
        self.classes_ = []
        self.config_ = []
        self.weights_ = []

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Dictionary of parameter names mapped to their values.
        """
        return {'model': self.model}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        params: dict
            Dictionary of parameter names mapped to their values.

        Returns
        -------
        self
        """
        for parameter, value in params.items():
            self.setattr(parameter, value)
        return self

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=0, shuffle=True):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        batch_size : int, optional
            Number of training samples evaluated at a time.
        nb_epochs : int, optional
            Number of training epochs.
        verbose : int, optional
            Verbosity level.
        shuffle : boolean, optional
            Indicator to shuffle the training data.

        Returns
        -------
        self : object
            Returns self.
        """
        if len(y.shape) == 1:
            self.classes_ = list(np.unique(y))
        else:
            self.classes_ = np.arange(0, y.shape[1])
        self.model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, shuffle=shuffle)
        self.config_ = self.model.get_config()
        self.weights_ = self.model.get_weights()
        return self

    def score(self, X, y, batch_size=128, verbose=0):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        batch_size : int, optional
            Number of test samples evaluated at a time.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        loss, accuracy = self.model.evaluate(X, y, batch_size=batch_size, show_accuracy=True, verbose=verbose)
        return accuracy

    def predict(self, X, batch_size=128, verbose=0):
        """
        Returns the class predictions for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        batch_size : int, optional
            Number of test samples evaluated at a time.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        preds : array-like, shape = (n_samples)
            Class predictions.
        """
        return self.model.predict_classes(X, batch_size=batch_size, verbose=verbose)

    def predict_proba(self, X, batch_size=128, verbose=0):
        """
        Returns class probability estimates for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        batch_size : int, optional
            Number of test samples evaluated at a time.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        proba : array-like, shape = (n_samples, n_outputs)
            Class probability estimates.
        """
        return self.model.predict_proba(X, batch_size=batch_size, verbose=verbose)
