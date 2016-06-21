from __future__ import absolute_import
import copy
import inspect
import types
import numpy as np

from ..utils.np_utils import to_categorical
from ..models import Sequential


class BaseWrapper(object):
    '''Base class for the Keras scikit-learn wrapper.

    Warning: This class should not be used directly.
    Use descendant classes instead.

    # Arguments
        build_fn: callable function or class instance
        sk_params: model parameters & fitting parameters

    The build_fn should construct, compile and return a Keras model, which
    will then be used to fit/predict. One of the following
    three values could be passed to build_fn:
    1. A function
    2. An instance of a class that implements the __call__ method
    3. None. This means you implement a class that inherits from either
    `KerasClassifier` or `KerasRegressor`. The __call__ method of the
    present class will then be treated as the default build_fn.

    `sk_params` takes both model parameters and fitting parameters. Legal model
    parameters are the arguments of `build_fn`. Note that like all other
    estimators in scikit-learn, 'build_fn' should provide default values for
    its arguments, so that you could create the estimator without passing any
    values to `sk_params`.

    `sk_params` could also accept parameters for calling `fit`, `predict`,
    `predict_proba`, and `score` methods (e.g., `nb_epoch`, `batch_size`).
    fitting (predicting) parameters are selected in the following order:

    1. Values passed to the dictionary arguments of
    `fit`, `predict`, `predict_proba`, and `score` methods
    2. Values passed to `sk_params`
    3. The default values of the `keras.models.Sequential`
    `fit`, `predict`, `predict_proba` and `score` methods

    When using scikit-learn's `grid_search` API, legal tunable parameters are
    those you could pass to `sk_params`, including fitting parameters.
    In other words, you could use `grid_search` to search for the best
    `batch_size` or `nb_epoch` as well as the model parameters.
    '''

    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.check_params(sk_params)

    def check_params(self, params):
        '''Check for user typos in "params" keys to avoid
        unwanted usage of default values

        # Arguments
            params: dictionary
                The parameters to be checked
        '''
        legal_params_fns = [Sequential.fit, Sequential.predict,
                            Sequential.predict_classes, Sequential.evaluate]
        if self.build_fn is None:
            legal_params_fns.append(self.__call__)
        elif not isinstance(self.build_fn, types.FunctionType):
            legal_params_fns.append(self.build_fn.__call__)
        else:
            legal_params_fns.append(self.build_fn)

        legal_params = []
        for fn in legal_params_fns:
            legal_params += inspect.getargspec(fn)[0]
        legal_params = set(legal_params)

        for params_name in params:
            if params_name not in legal_params:
                assert False, '{} is not a legal parameter'.format(params_name)

    def get_params(self, deep=True):
        '''Get parameters for this estimator.

        # Arguments
            deep: boolean, optional
                If True, will return the parameters for this estimator and
                contained sub-objects that are estimators.

        # Returns
            params : dict
                Dictionary of parameter names mapped to their values.
        '''
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        '''Set the parameters of this estimator.

        # Arguments
        params: dict
            Dictionary of parameter names mapped to their values.

        # Returns
            self
        '''
        self.check_params(params)
        self.sk_params.update(params)
        return self

    def fit(self, X, y, **kwargs):
        '''Construct a new model with build_fn and fit the model according
        to the given training data.

        # Arguments
            X : array-like, shape `(n_samples, n_features)`
                Training samples where n_samples in the number of samples
                and n_features is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for X.
            kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`

        # Returns
            history : object
                details about the training history at each epoch.
        '''

        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        history = self.model.fit(X, y, **fit_args)

        return history

    def filter_sk_params(self, fn, override={}):
        '''Filter sk_params and return those in fn's arguments

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override sk_params

        # Returns
            res : dictionary dictionary containing variables
                in both sk_params and fn's arguments.
        '''
        res = {}
        fn_args = inspect.getargspec(fn)[0]
        for name, value in self.sk_params.items():
            if name in fn_args:
                res.update({name: value})
        res.update(override)
        return res


class KerasClassifier(BaseWrapper):
    '''Implementation of the scikit-learn classifier API for Keras.
    '''

    def predict(self, X, **kwargs):
        '''Returns the class predictions for the given test data.

        # Arguments
            X: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.predict_classes`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        '''
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        return self.model.predict_classes(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        '''Returns class probability estimates for the given test data.

        # Arguments
            X: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.predict_classes`.

        # Returns
            proba: array-like, shape `(n_samples, n_outputs)`
                Class probability estimates.
                In the case of binary classification,
                tp match the scikit-learn API,
                will return an array of shape '(n_samples, 2)'
                (instead of `(n_sample, 1)` as in Keras).
        '''
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        probs = self.model.predict_proba(X, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, X, y, **kwargs):
        '''Returns the mean accuracy on the given test data and labels.

        # Arguments
            X: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for X.
            kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        # Returns
            score: float
                Mean accuracy of predictions on X wrt. y.
        '''
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
        outputs = self.model.evaluate(X, y, **kwargs)
        if type(outputs) is not list:
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise Exception('The model is not configured to compute accuracy. '
                        'You should pass `metrics=["accuracy"]` to '
                        'the `model.compile()` method.')


class KerasRegressor(BaseWrapper):
    '''Implementation of the scikit-learn regressor API for Keras.
    '''

    def predict(self, X, **kwargs):
        '''Returns predictions for the given test data.

        # Arguments
            X: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.predict`.
        # Returns
            preds: array-like, shape `(n_samples,)`
                Predictions.
        '''
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        return self.model.predict(X, **kwargs)

    def score(self, X, y, **kwargs):
        '''Returns the mean loss on the given test data and labels.

        # Arguments
            X: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            y: array-like, shape `(n_samples,)`
                True labels for X.
            kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        # Returns
            score: float
                Mean accuracy of predictions on X wrt. y.
        '''
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
        loss = self.model.evaluate(X, y, **kwargs)
        if type(loss) is list:
            return loss[0]
        return loss
