"""Tests using Scikit-Learn's bundled estimator_checks."""

from contextlib import contextmanager

import pytest

import keras
from keras.src.backend import floatx
from keras.src.backend import set_floatx
from keras.src.layers import Dense
from keras.src.layers import Input
from keras.src.models import Model
from keras.src.wrappers import KerasClassifier
from keras.src.wrappers import KerasRegressor
from keras.src.wrappers import KerasTransformer
from keras.src.wrappers.fixes import parametrize_with_checks


def dynamic_model(X, y, loss, layers=[10]):
    """Creates a basic MLP classifier dynamically choosing binary/multiclass
    classification loss and ouput activations.
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
def use_floatx(x: str):
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
    "KerasClassifier": {
        "check_classifiers_regression_target": ("not an issue in sklearn>=1.6"),
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
    },
    "KerasRegressor": {
        "check_parameters_default_constructible": (
            "not an issue in sklearn>=1.6"
        ),
    },
    "KerasTransformer": {
        "check_parameters_default_constructible": (
            "not an issue in sklearn>=1.6"
        ),
    },
}


@parametrize_with_checks(
    estimators=[
        KerasClassifier(
            model=dynamic_model,
            random_state=42,
            model_args={
                "loss": "categorical_crossentropy",
                "layers": [20, 20, 20],
            },
            fit_args={"epochs": 5},
        ),
        KerasRegressor(
            model=dynamic_model,
            random_state=42,
            model_args={"loss": "mse"},
        ),
        KerasTransformer(
            model=dynamic_model,
            random_state=42,
            model_args={"loss": "mse"},
        ),
    ],
    expected_failed_checks=lambda estimator: EXPECTED_FAILED_CHECKS[
        type(estimator).__name__
    ],
)
def test_fully_compliant_estimators_low_precision(estimator, check):
    """Checks that can be passed with sklearn's default tolerances
    and in a single epoch.
    """
    try:
        check(estimator)
    except NotImplementedError:
        if keras.config.backend() == "numpy":
            pytest.skip("Backend not implemented")
        else:
            raise
