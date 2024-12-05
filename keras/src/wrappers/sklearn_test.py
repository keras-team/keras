"""Tests using Scikit-Learn's bundled estimator_checks."""

from contextlib import contextmanager

from keras.src.models import Model
from keras.src.backend import floatx, set_floatx
from keras.src.wrappers import KerasClassifier, KerasRegressor, KerasTransformer
from keras.src.layers import Dense, Input
from keras.src.models import Model

from sklearn.utils.estimator_checks import parametrize_with_checks


def dynamic_model(X, y, loss: str):
    """Creates a basic MLP classifier dynamically choosing binary/multiclass
    classification loss and ouput activations.
    """
    n_features_in = X.shape[1]
    inp = Input(shape=(n_features_in,))

    hidden = inp
    for layer_size in [10]:
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


@parametrize_with_checks(
    estimators=[
        KerasClassifier(
            model=dynamic_model,
            random_state=42,
            model_args={"loss": "categorical_crossentropy"},
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
)
def test_fully_compliant_estimators_low_precision(estimator, check):
    """Checks that can be passed with sklearn's default tolerances
    and in a single epoch.
    """
    check(estimator)
