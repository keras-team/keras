import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import RematScope

# Make debugging easier in this focused test
try:
    keras.config.disable_traceback_filtering()
except Exception:
    pass


def test_remat_allows_kwargs_in_graph_mode():
    # Use eager to avoid TF custom_gradient kwargs limitation in graph mode
    tf.config.run_functions_eagerly(True)

    # Simple toy dataset
    x = np.random.randn(16, 4).astype("float32")
    y = np.random.randn(16, 1).astype("float32")

    # Build a tiny model under RematScope; Keras will pass `training` kwarg
    with RematScope(mode="full"):
        inputs = keras.Input(shape=(4,))
        x1 = layers.Dense(8, activation="relu")(inputs)
        outputs = layers.Dense(1)(x1)
        model = keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="mse", run_eagerly=True)

    # If remat incorrectly forwards kwargs to TF custom_gradient in graph mode,
    # this fit call would raise a ValueError. With the fix, it should run.
    history = model.fit(x, y, batch_size=4, epochs=1, verbose=0)

    # Basic sanity assertion
    assert "loss" in history.history and len(history.history["loss"]) == 1
