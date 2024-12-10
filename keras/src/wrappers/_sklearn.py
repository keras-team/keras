import copy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import check_is_fitted
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.metadata_routing import MetadataRequest

from keras.src.api_export import keras_export
from keras.src.models.cloning import clone_model
from keras.src.models.model import Model
from keras.src.wrappers.fixes import _routing_enabled
from keras.src.wrappers.fixes import _validate_data
from keras.src.wrappers.fixes import type_of_target
from keras.src.wrappers.random_state import tensorflow_random_state
from keras.src.wrappers.utils import TargetReshaper
from keras.src.wrappers.utils import _check_model


class KerasBase(BaseEstimator):
    """Base class for scikit-learn wrappers.

    Args:
        model: `Model`.
            An instance of `Model`, or a callable returning such an object.
            Note that if input is a `Model`, it will be cloned using
            `keras.models.clone_model` before being fitted, unless
            `warm_start=True`.
            The `Model` instance needs to be passed as already compiled.
            If callable, it must accept at least `X` and `y` as keyword
            arguments. Other arguments must be accepted if passed as
            `model_args` by the user.
        warm_start: bool, defaults to False.
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.
        model_args: dict, defaults to None.
            Keyword arguments passed to `model`, if `model` is callable.
        fit_args: dict, defaults to None.
            Keyword arguments passed to `model.fit`. These can also be passed
            directly to the `fit` method of the scikit-learn wrapper. The
            values passed directly to the `fit` method take precedence over
            these.
        random_state : int, np.random.RandomState, or None, defaults to None.
            Set the Tensorflow random number generators to a reproducible
            deterministic state using this seed. Pass an int for reproducible
            results across multiple function calls.

    Attributes:
        model_ : `Model`
            The fitted model.
    """

    def __init__(
        self,
        model,
        warm_start=False,
        model_args=None,
        fit_args=None,
        random_state=None,
    ):
        self.model = model
        self.warm_start = warm_start
        self.model_args = model_args
        self.fit_args = fit_args
        self.random_state = random_state

    def __sklearn_clone__(self):
        """Return a deep copy of the model.

        This is used by the `sklearn.base.clone` function.
        """
        model = (
            self.model if callable(self.model) else copy.deepcopy(self.model)
        )
        return type(self)(
            model=model,
            warm_start=self.warm_start,
            model_args=self.model_args,
            random_state=self.random_state,
        )

    @property
    def epoch_(self) -> int:
        """The current training epoch."""
        return getattr(self, "history_", {}).get("epoch", 0)

    def set_fit_request(self, **kwargs):
        """Set requested parameters by the fit method.

        Please see [scikit-learn's metadata routing](
        https://scikit-learn.org/stable/metadata_routing.html) for more
        details.


        Arguments:
            kwargs : dict
                Arguments should be of the form `param_name=alias`, and `alias`
                can be one of `{True, False, None, str}`.

        Returns:
            self
        """
        if not _routing_enabled():
            raise RuntimeError(
                "This method is only available when metadata routing is "
                "enabled. You can enable it using "
                "sklearn.set_config(enable_metadata_routing=True)."
            )

        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self

    def _get_loss(self, y):
        if self.loss:
            return self.loss
        target_type = type_of_target(y)
        if target_type == "binary":
            return "binary_crossentropy"
        elif target_type == "multiclass":
            return "categorical_crossentropy"
        else:
            raise ValueError(
                "Cannot automatically identify loss for target type: "
                f"{target_type}"
            )

    def _get_model(self, X, y):
        if isinstance(self.model, Model):
            return clone_model(self.model)
        else:
            args = self.model_args or {}
            rand = self._get_random_int()
            if rand is not None:
                with tensorflow_random_state(rand):
                    return self.model(X=X, y=y, **args)
            else:
                return self.model(X=X, y=y, **args)

    def _get_random_int(self):
        if isinstance(self.random_state, np.random.RandomState):
            # Keras needs an integer
            # we sample an integer and use that as a seed
            # Given the same RandomState, the seed will always be
            # the same, thus giving reproducible results
            state = self.random_state.get_state()
            r = np.random.RandomState()
            r.set_state(state)
            return r.randint(low=1)

        # int or None
        return self.random_state

    def fit(self, X, y, **kwargs):
        """Fit the model.

        Args:
            X: array-like, shape=(n_samples, n_features)
                The input samples.
            y: array-like, shape=(n_samples,) or (n_samples, n_outputs)
                The targets.
            **kwargs: keyword arguments passed to `model.fit`
        """
        X, y = _validate_data(self, X, y)
        y = self._process_target(y, reset=True)
        model = self._get_model(X, y)
        _check_model(model)

        rand = self._get_random_int()
        fit_args = self.fit_args or {}
        fit_args.update(kwargs)
        if rand is not None:
            with tensorflow_random_state(rand):
                self.history_ = model.fit(X, y, **fit_args)
        else:
            self.history_ = model.fit(X, y, **fit_args)

        self.model_ = model
        return self

    def predict(self, X):
        """Predict using the model."""
        check_is_fitted(self)
        X = _validate_data(self, X, reset=False)
        raw_output = self.model_.predict(X)
        return self._reverse_process_target(raw_output)

    def _process_target(self, y, reset=False):
        """Regressors are NOOP here, classifiers do OHE."""
        # This is here to raise the right error in case of invalid target
        type_of_target(y, raise_unknown=True)
        if reset:
            self._target_encoder = TargetReshaper().fit(y)
        return self._target_encoder.transform(y)

    def _reverse_process_target(self, y):
        """Regressors are NOOP here, classifiers reverse OHE."""
        return self._target_encoder.inverse_transform(y)


@keras_export("keras.wrappers.KerasClassifier")
class KerasClassifier(ClassifierMixin, KerasBase):
    """scikit-learn compatible classifier wrapper for Keras models.

    Args:
        model: `Model`.
            An instance of `Model`, or a callable returning such an object.
            Note that if input is a `Model`, it will be cloned using
            `keras.models.clone_model` before being fitted, unless
            `warm_start=True`.
            The `Model` instance needs to be passed as already compiled.
            If callable, it must accept at least `X` and `y` as keyword
            arguments. Other arguments must be accepted if passed as
            `model_args` by the user.
        warm_start: bool, defaults to False.
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.
        model_args: dict, defaults to None.
            Keyword arguments passed to `model`, if `model` is callable.
        fit_args: dict, defaults to None.
            Keyword arguments passed to `model.fit`. These can also be passed
            directly to the `fit` method of the scikit-learn wrapper. The
            values passed directly to the `fit` method take precedence over
            these.
        random_state : int, np.random.RandomState, or None, defaults to None.
            Set the Tensorflow random number generators to a reproducible
            deterministic state using this seed. Pass an int for reproducible
            results across multiple function calls.

    Attributes:
        model_ : `Model`
            The fitted model.
        classes_ : array-like, shape=(n_classes,)
            The classes labels.

    Example:
    Here we use a function which creates a basic MLP model dynamically choosing
    the input and output shapes. We will use this to create our scikit-learn
    model.

    ``` python
    from keras.src.layers import Dense, Input, Model

    def dynamic_model(X, y, loss, layers=[10]):
        # Creates a basic MLP model dynamically choosing the input and output
        # shapes.
        n_features_in = X.shape[1]
        inp = Input(shape=(n_features_in,))

        hidden = inp
        for layer_size in layers:
            hidden = Dense(layer_size, activation="relu")(hidden)

        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        out = [Dense(n_outputs, activation="softmax")(hidden)]
        model = Model(inp, out)
        model.compile(loss=loss, optimizer="rmsprop")

        return model
    ```

    You can then use this function to create a scikit-learn compatible model
    and fit it on some data.

    ``` python
    from sklearn.datasets import make_classification
    from keras.wrappers import KerasClassifier

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3)
    est = KerasClassifier(
        model=dynamic_model,
        random_state=42,
        model_args={
            "loss": "categorical_crossentropy",
            "layers": [20, 20, 20],
        },
    )

    est.fit(X, y, epochs=5)
    ```
    """

    def _process_target(self, y, reset=False):
        """Classifiers do OHE."""
        target_type = type_of_target(y, raise_unknown=True)
        if target_type not in ["binary", "multiclass"]:
            raise ValueError(
                "Only binary and multiclass target types are supported."
                f" Target type: {target_type}"
            )
        if reset:
            self._target_encoder = make_pipeline(
                TargetReshaper(), OneHotEncoder(sparse_output=False)
            ).fit(y)
            self.classes_ = np.unique(y)
            if len(self.classes_) == 1:
                raise ValueError(
                    "Classifier can't train when only one class is present."
                )
        return self._target_encoder.transform(y)

    def _more_tags(self):
        # required to be compatible with scikit-learn<1.6
        return {
            "poor_score": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


@keras_export("keras.wrappers.KerasRegressor")
class KerasRegressor(RegressorMixin, KerasBase):
    """scikit-learn compatible regressor wrapper for Keras models.

    Args:
        model: `Model`.
            An instance of `Model`, or a callable returning such an object.
            Note that if input is a `Model`, it will be cloned using
            `keras.models.clone_model` before being fitted, unless
            `warm_start=True`.
            The `Model` instance needs to be passed as already compiled.
            If callable, it must accept at least `X` and `y` as keyword
            arguments. Other arguments must be accepted if passed as
            `model_args` by the user.
        warm_start: bool, defaults to False.
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.
        model_args: dict, defaults to None.
            Keyword arguments passed to `model`, if `model` is callable.
        fit_args: dict, defaults to None.
            Keyword arguments passed to `model.fit`. These can also be passed
            directly to the `fit` method of the scikit-learn wrapper. The
            values passed directly to the `fit` method take precedence over
            these.
        random_state : int, np.random.RandomState, or None, defaults to None.
            Set the Tensorflow random number generators to a reproducible
            deterministic state using this seed. Pass an int for reproducible
            results across multiple function calls.

    Attributes:
        model_ : `Model`
            The fitted model.

    Example:
    Here we use a function which creates a basic MLP model dynamically choosing
    the input and output shapes. We will use this to create our scikit-learn
    model.

    ``` python
    from keras.src.layers import Dense, Input, Model

    def dynamic_model(X, y, loss, layers=[10]):
        # Creates a basic MLP model dynamically choosing the input and output
        # shapes.
        n_features_in = X.shape[1]
        inp = Input(shape=(n_features_in,))

        hidden = inp
        for layer_size in layers:
            hidden = Dense(layer_size, activation="relu")(hidden)

        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        out = [Dense(n_outputs, activation="softmax")(hidden)]
        model = Model(inp, out)
        model.compile(loss=loss, optimizer="rmsprop")

        return model
    ```

    You can then use this function to create a scikit-learn compatible model
    and fit it on some data.

    ``` python
    from sklearn.datasets import make_classification
    from keras.wrappers import KerasClassifier

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3)
    est = KerasClassifier(
        model=dynamic_model,
        random_state=42,
        model_args={
            "loss": "categorical_crossentropy",
            "layers": [20, 20, 20],
        },
    )

    est.fit(X, y, epochs=5)
    ```
    """

    def _more_tags(self):
        # required to be compatible with scikit-learn<1.6
        return {
            "poor_score": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags


@keras_export("keras.wrappers.KerasTransformer")
class KerasTransformer(TransformerMixin, KerasBase):
    """scikit-learn compatible transformer wrapper for Keras models.

    Args:
        model: `Model`
            An instance of `Model`. Needs to be compiled, have a loss, and
            optimizer. Note that the model will be cloned using `clone_model`
            before being fitted, unless `warm_start=True`.

        warm_start: bool, default=False
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.

    Attributes:
        model_ : `Model`
            The fitted model.
    """

    def transform(self, X):
        """Transform the data.

        Args:
            X: array-like, shape=(n_samples, n_features)
                The input samples.

        Returns:
            X_transformed: array-like, shape=(n_samples, n_features)
                The transformed data.
        """
        check_is_fitted(self)
        X = _validate_data(self, X, reset=False)
        return self.model_.predict(X)

    def _more_tags(self):
        # required to be compatible with scikit-learn<1.6
        return {
            "preserves_dtype": [],
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = []
        return tags
