import copy

import numpy as np

from keras.src.api_export import keras_export
from keras.src.models.cloning import clone_model
from keras.src.models.model import Model
from keras.src.wrappers.fixes import _routing_enabled
from keras.src.wrappers.fixes import _validate_data
from keras.src.wrappers.fixes import type_of_target
from keras.src.wrappers.utils import TargetReshaper
from keras.src.wrappers.utils import _check_model
from keras.src.wrappers.utils import assert_sklearn_installed

try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn.base import ClassifierMixin
    from sklearn.base import RegressorMixin
    from sklearn.base import TransformerMixin
except ImportError:
    sklearn = None

    class BaseEstimator:
        pass

    class ClassifierMixin:
        pass

    class RegressorMixin:
        pass

    class TransformerMixin:
        pass


class SKLBase(BaseEstimator):
    """Base class for scikit-learn wrappers.

    Note that there are sources of randomness in model initialization and
    training. Refer to [Reproducibility in Keras Models](
    https://keras.io/examples/keras_recipes/reproducibility_recipes/) on how to
    control randomness.

    Args:
        model: `Model`.
            An instance of `Model`, or a callable returning such an object.
            Note that if input is a `Model`, it will be cloned using
            `keras.models.clone_model` before being fitted, unless
            `warm_start=True`.
            The `Model` instance needs to be passed as already compiled.
            If callable, it must accept at least `X` and `y` as keyword
            arguments. Other arguments must be accepted if passed as
            `model_kwargs` by the user.
        warm_start: bool, defaults to `False`.
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.
        model_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model`, if `model` is callable.
        fit_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model.fit`. These can also be passed
            directly to the `fit` method of the scikit-learn wrapper. The
            values passed directly to the `fit` method take precedence over
            these.

    Attributes:
        model_ : `Model`
            The fitted model.
        history_ : dict
            The history of the fit, returned by `model.fit`.
    """

    def __init__(
        self,
        model,
        warm_start=False,
        model_kwargs=None,
        fit_kwargs=None,
    ):
        assert_sklearn_installed(self.__class__.__name__)
        self.model = model
        self.warm_start = warm_start
        self.model_kwargs = model_kwargs
        self.fit_kwargs = fit_kwargs

    def _more_tags(self):
        return {"non_deterministic": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

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
            model_kwargs=self.model_kwargs,
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

        self._metadata_request = sklearn.utils.metadata_routing.MetadataRequest(
            owner=self.__class__.__name__
        )
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self

    def _get_model(self, X, y):
        if isinstance(self.model, Model):
            return clone_model(self.model)
        else:
            args = self.model_kwargs or {}
            return self.model(X=X, y=y, **args)

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

        fit_kwargs = self.fit_kwargs or {}
        fit_kwargs.update(kwargs)
        self.history_ = model.fit(X, y, **fit_kwargs)

        self.model_ = model
        return self

    def predict(self, X):
        """Predict using the model."""
        sklearn.base.check_is_fitted(self)
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


@keras_export("keras.wrappers.SKLearnClassifier")
class SKLearnClassifier(ClassifierMixin, SKLBase):
    """scikit-learn compatible classifier wrapper for Keras models.

    Note that there are sources of randomness in model initialization and
    training. Refer to [Reproducibility in Keras Models](
    https://keras.io/examples/keras_recipes/reproducibility_recipes/) on how to
    control randomness.

    Args:
        model: `Model`.
            An instance of `Model`, or a callable returning such an object.
            Note that if input is a `Model`, it will be cloned using
            `keras.models.clone_model` before being fitted, unless
            `warm_start=True`.
            The `Model` instance needs to be passed as already compiled.
            If callable, it must accept at least `X` and `y` as keyword
            arguments. Other arguments must be accepted if passed as
            `model_kwargs` by the user.
        warm_start: bool, defaults to `False`.
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.
        model_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model`, if `model` is callable.
        fit_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model.fit`. These can also be passed
            directly to the `fit` method of the scikit-learn wrapper. The
            values passed directly to the `fit` method take precedence over
            these.

    Attributes:
        model_ : `Model`
            The fitted model.
        history_ : dict
            The history of the fit, returned by `model.fit`.
        classes_ : array-like, shape=(n_classes,)
            The classes labels.

    Example:
    Here we use a function which creates a basic MLP model dynamically
    choosing the input and output shapes. We will use this to create our
    scikit-learn model.

    ``` python
    from keras.src.layers import Dense, Input, Model

    def dynamic_model(X, y, loss, layers=[10]):
        # Creates a basic MLP model dynamically choosing the input and
        # output shapes.
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
    from keras.wrappers import SKLearnClassifier

    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3)
    est = SKLearnClassifier(
        model=dynamic_model,
        model_kwargs={
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
            self._target_encoder = sklearn.pipeline.make_pipeline(
                TargetReshaper(),
                sklearn.preprocessing.OneHotEncoder(sparse_output=False),
            ).fit(y)
            self.classes_ = np.unique(y)
            if len(self.classes_) == 1:
                raise ValueError(
                    "Classifier can't train when only one class is present."
                )
        return self._target_encoder.transform(y)

    def _more_tags(self):
        # required to be compatible with scikit-learn<1.6
        return {"poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


@keras_export("keras.wrappers.SKLearnRegressor")
class SKLearnRegressor(RegressorMixin, SKLBase):
    """scikit-learn compatible regressor wrapper for Keras models.

    Note that there are sources of randomness in model initialization and
    training. Refer to [Reproducibility in Keras Models](
    https://keras.io/examples/keras_recipes/reproducibility_recipes/) on how to
    control randomness.

    Args:
        model: `Model`.
            An instance of `Model`, or a callable returning such an object.
            Note that if input is a `Model`, it will be cloned using
            `keras.models.clone_model` before being fitted, unless
            `warm_start=True`.
            The `Model` instance needs to be passed as already compiled.
            If callable, it must accept at least `X` and `y` as keyword
            arguments. Other arguments must be accepted if passed as
            `model_kwargs` by the user.
        warm_start: bool, defaults to `False`.
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.
        model_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model`, if `model` is callable.
        fit_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model.fit`. These can also be passed
            directly to the `fit` method of the scikit-learn wrapper. The
            values passed directly to the `fit` method take precedence over
            these.

    Attributes:
        model_ : `Model`
            The fitted model.

    Example:
    Here we use a function which creates a basic MLP model dynamically
    choosing the input and output shapes. We will use this to create our
    scikit-learn model.

    ``` python
    from keras.src.layers import Dense, Input, Model

    def dynamic_model(X, y, loss, layers=[10]):
        # Creates a basic MLP model dynamically choosing the input and
        # output shapes.
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
    from sklearn.datasets import make_regression
    from keras.wrappers import SKLearnRegressor

    X, y = make_regression(n_samples=1000, n_features=10)
    est = SKLearnRegressor(
        model=dynamic_model,
        model_kwargs={
            "loss": "mse",
            "layers": [20, 20, 20],
        },
    )

    est.fit(X, y, epochs=5)
    ```
    """

    def _more_tags(self):
        # required to be compatible with scikit-learn<1.6
        return {"poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags


@keras_export("keras.wrappers.SKLearnTransformer")
class SKLearnTransformer(TransformerMixin, SKLBase):
    """scikit-learn compatible transformer wrapper for Keras models.

    Note that this is a scikit-learn compatible transformer, and not a
    transformer in the deep learning sense.

    Also note that there are sources of randomness in model initialization and
    training. Refer to [Reproducibility in Keras Models](
    https://keras.io/examples/keras_recipes/reproducibility_recipes/) on how to
    control randomness.

    Args:
        model: `Model`.
            An instance of `Model`, or a callable returning such an object.
            Note that if input is a `Model`, it will be cloned using
            `keras.models.clone_model` before being fitted, unless
            `warm_start=True`.
            The `Model` instance needs to be passed as already compiled.
            If callable, it must accept at least `X` and `y` as keyword
            arguments. Other arguments must be accepted if passed as
            `model_kwargs` by the user.
        warm_start: bool, defaults to `False`.
            Whether to reuse the model weights from the previous fit. If `True`,
            the given model won't be cloned and the weights from the previous
            fit will be reused.
        model_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model`, if `model` is callable.
        fit_kwargs: dict, defaults to `None`.
            Keyword arguments passed to `model.fit`. These can also be passed
            directly to the `fit` method of the scikit-learn wrapper. The
            values passed directly to the `fit` method take precedence over
            these.

    Attributes:
        model_ : `Model`
            The fitted model.
        history_ : dict
            The history of the fit, returned by `model.fit`.

    Example:
    A common use case for a scikit-learn transformer, is to have a step
    which gives you the embedding of your data. Here we assume
    `my_package.my_model` is a Keras model which takes the input and gives
    embeddings of the data, and `my_package.my_data` is your dataset loader.

    ``` python
    from my_package import my_model, my_data
    from keras.wrappers import SKLearnTransformer
    from sklearn.frozen import FrozenEstimator # requires scikit-learn>=1.6
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = my_data()

    trs = FrozenEstimator(SKLearnTransformer(model=my_model))
    pipe = make_pipeline(trs, HistGradientBoostingClassifier())
    pipe.fit(X, y)
    ```

    Note that in the above example, `FrozenEstimator` prevents any further
    training of the transformer step in the pipeline, which can be the case
    if you don't want to change the embedding model at hand.
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
        sklearn.base.check_is_fitted(self)
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
