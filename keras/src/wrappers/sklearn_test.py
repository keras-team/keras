"""Tests using Scikit-Learn's bundled estimator_checks."""

import unittest
from contextlib import contextmanager

import pytest
import sklearn
from packaging.version import parse as parse_version
from sklearn.utils.estimator_checks import parametrize_with_checks

import keras
from keras.src.backend import floatx
from keras.src.backend import set_floatx
from keras.src.layers import Dense
from keras.src.layers import Input
from keras.src.models import Model
from keras.src.wrappers import SKLearnClassifier
from keras.src.wrappers import SKLearnRegressor
from keras.src.wrappers import SKLearnTransformer


def wrapped_parametrize_with_checks(
    estimators,
    *,
    legacy=True,
    expected_failed_checks=None,
):
    """Wrapped `parametrize_with_checks` handling backwards compat."""
    sklearn_version = parse_version(
        parse_version(sklearn.__version__).base_version
    )

    if sklearn_version >= parse_version("1.6"):
        return parametrize_with_checks(
            estimators,
            legacy=legacy,
            expected_failed_checks=expected_failed_checks,
        )

    def patched_more_tags(estimator, expected_failed_checks):
        import copy

        original_tags = copy.deepcopy(sklearn.utils._tags._safe_tags(estimator))

        def patched_more_tags(self):
            original_tags.update({"_xfail_checks": expected_failed_checks})
            return original_tags

        estimator.__class__._more_tags = patched_more_tags
        return estimator

    estimators = [
        patched_more_tags(estimator, expected_failed_checks(estimator))
        for estimator in estimators
    ]

    # legacy is not supported and ignored
    return parametrize_with_checks(estimators)


def dynamic_model(X, y, loss, layers=[10]):
    """Creates a basic MLP classifier dynamically choosing binary/multiclass
    classification loss and output activations.
    """
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


@contextmanager
def use_floatx(x):
    """Context manager to temporarily
    set the keras backend precision.
    """
    _floatx = floatx()
    set_floatx(x)
    try:
        yield
    finally:
        set_floatx(_floatx)


EXPECTED_FAILED_CHECKS = {
    "SKLearnClassifier": {
        "check_classifiers_regression_target": "not an issue in sklearn>=1.6",
        "check_parameters_default_constructible": (
            "not an issue in sklearn>=1.6"
        ),
        "check_classifiers_one_label_sample_weights": (
            "0 sample weight is not ignored"
        ),
        "check_classifiers_classes": (
            "with small test cases the estimator returns not all classes "
            "sometimes"
        ),
        "check_classifier_data_not_an_array": (
            "This test assumes reproducibility in fit."
        ),
        "check_supervised_y_2d": "This test assumes reproducibility in fit.",
        "check_fit_idempotent": "This test assumes reproducibility in fit.",
    },
    "SKLearnRegressor": {
        "check_parameters_default_constructible": (
            "not an issue in sklearn>=1.6"
        ),
    },
    "SKLearnTransformer": {
        "check_parameters_default_constructible": (
            "not an issue in sklearn>=1.6"
        ),
    },
}


@wrapped_parametrize_with_checks(
    estimators=[
        SKLearnClassifier(
            model=dynamic_model,
            model_kwargs={
                "loss": "categorical_crossentropy",
                "layers": [20, 20, 20],
            },
            fit_kwargs={"epochs": 5},
        ),
        SKLearnRegressor(
            model=dynamic_model,
            model_kwargs={"loss": "mse"},
        ),
        SKLearnTransformer(
            model=dynamic_model,
            model_kwargs={"loss": "mse"},
        ),
    ],
    expected_failed_checks=lambda estimator: EXPECTED_FAILED_CHECKS[
        type(estimator).__name__
    ],
)
def test_sklearn_estimator_checks(estimator, check):
    """Checks that can be passed with sklearn's default tolerances
    and in a single epoch.
    """
    try:
        check(estimator)
    except Exception as exc:
        if keras.config.backend() in ["numpy", "openvino"] and (
            isinstance(exc, NotImplementedError)
            or "NotImplementedError" in str(exc)
        ):
            pytest.xfail("Backend not implemented")
        elif isinstance(exc, unittest.SkipTest):
            # Workaround for https://github.com/pytest-dev/pytest/issues/13895
            pytest.skip(str(exc))
        else:
            raise
