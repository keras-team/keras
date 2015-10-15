from __future__ import absolute_import
import abc
import copy
import numpy as np

from ..utils.np_utils import to_categorical


class BaseWrapper(object):
    """
    Base class for the Keras scikit-learn wrapper.

    Warning: This class should not be used directly. Use derived classes instead.

    Parameters
    ----------
    train_batch_size : int, optional
        Number of training samples evaluated at a time.
    test_batch_size : int, optional
        Number of test samples evaluated at a time.
    nb_epochs : int, optional
        Number of training epochs.
    shuffle : boolean, optional
        Whether to shuffle the samples at each epoch.
    show_accuracy : boolean, optional
        Whether to display class accuracy in the logs at each epoch.
    validation_split : float [0, 1], optional
        Fraction of the data to use as held-out validation data.
    validation_data : tuple (X, y), optional
        Data to be used as held-out validation data. Will override validation_split.
    callbacks : list, optional
        List of callbacks to apply during training.
    verbose : int, optional
        Verbosity level.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, model, optimizer, loss,
                 train_batch_size=128, test_batch_size=128,
                 nb_epoch=100, shuffle=True, show_accuracy=False,
                 validation_split=0, validation_data=None, callbacks=None,
                 verbose=0,):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.compiled_model_ = None
        self.classes_ = []
        self.config_ = []
        self.weights_ = []

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.nb_epoch = nb_epoch
        self.shuffle = shuffle
        self.show_accuracy = show_accuracy
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.callbacks = [] if callbacks is None else callbacks

        self.verbose = verbose

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
        return {'model': self.model, 'optimizer': self.optimizer, 'loss': self.loss}

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
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Makes a copy of the un-compiled model definition to use for
        compilation and fitting, leaving the original definition
        intact.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        history : object
            Returns details about the training history at each epoch.
        """
        if len(y.shape) == 1:
            self.classes_ = list(np.unique(y))
            if self.loss == 'categorical_crossentropy':
                y = to_categorical(y)
        else:
            self.classes_ = np.arange(0, y.shape[1])

        self.compiled_model_ = copy.deepcopy(self.model)
        self.compiled_model_.compile(optimizer=self.optimizer, loss=self.loss)
        history = self.compiled_model_.fit(
            X, y, batch_size=self.train_batch_size, nb_epoch=self.nb_epoch, verbose=self.verbose,
            shuffle=self.shuffle, show_accuracy=self.show_accuracy,
            validation_split=self.validation_split, validation_data=self.validation_data,
            callbacks=self.callbacks)

        self.config_ = self.model.get_config()
        self.weights_ = self.model.get_weights()

        return history


class KerasClassifier(BaseWrapper):
    """
    Implementation of the scikit-learn classifier API for Keras.

    Parameters
    ----------
    model : object
        An un-compiled Keras model object is required to use the scikit-learn wrapper.
    optimizer : string
        Optimization method used by the model during compilation/training.
    loss : string
        Loss function used by the model during compilation/training.
    """
    def __init__(self, model, optimizer='adam', loss='categorical_crossentropy', **kwargs):
        super(KerasClassifier, self).__init__(model, optimizer, loss, **kwargs)

    def predict(self, X):
        """
        Returns the class predictions for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        preds : array-like, shape = (n_samples)
            Class predictions.
        """
        return self.compiled_model_.predict_classes(
            X, batch_size=self.test_batch_size, verbose=self.verbose)

    def predict_proba(self, X):
        """
        Returns class probability estimates for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        proba : array-like, shape = (n_samples, n_outputs)
            Class probability estimates.
        """
        return self.compiled_model_.predict_proba(
            X, batch_size=self.test_batch_size, verbose=self.verbose)

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of predictions on X wrt. y.
        """
        loss, accuracy = self.compiled_model_.evaluate(
            X, y, batch_size=self.test_batch_size, show_accuracy=True, verbose=self.verbose)
        return accuracy


class KerasRegressor(BaseWrapper):
    """
    Implementation of the scikit-learn regressor API for Keras.

    Parameters
    ----------
    model : object
        An un-compiled Keras model object is required to use the scikit-learn wrapper.
    optimizer : string
        Optimization method used by the model during compilation/training.
    loss : string
        Loss function used by the model during compilation/training.
    """
    def __init__(self, model, optimizer='adam', loss='mean_squared_error', **kwargs):
        super(KerasRegressor, self).__init__(model, optimizer, loss, **kwargs)

    def predict(self, X):
        """
        Returns predictions for the given test data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        preds : array-like, shape = (n_samples)
            Predictions.
        """
        return self.compiled_model_.predict(
            X, batch_size=self.test_batch_size, verbose=self.verbose).ravel()

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        y : array-like, shape = (n_samples)
            True labels for X.

        Returns
        -------
        score : float
            Loss from predictions on X wrt. y.
        """
        loss = self.compiled_model_.evaluate(
            X, y, batch_size=self.test_batch_size, show_accuracy=False, verbose=self.verbose)
        return loss
