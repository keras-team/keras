# File: keras/src/layers/pooling/test_training_adaptive_pooling.py
import numpy as np
import pytest

from keras.src import backend as K
from keras.src import layers
from keras.src import models

np.random.seed(42)
x_train = np.random.randn(1000, 32, 32, 3).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.int32)
x_val = np.random.randn(200, 32, 32, 3).astype(np.float32)
y_val = np.random.randint(0, 10, 200).astype(np.int32)


def make_model(pool_type="avg"):
    pool_layer = (
        layers.AdaptiveAveragePooling2D((4, 4))
        if pool_type == "avg"
        else layers.AdaptiveMaxPooling2D((4, 4))
    )
    return models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            pool_layer,
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )


@pytest.mark.parametrize("pool", ["avg", "max"])
def test_training_adaptive_pooling(pool):
    # Skip backends where training is unsupported
    if K.backend() in ["numpy", "openvino", "tensorflow"]:
        pytest.skip(
            f"fit or adaptive pooling not supported for backend: {K.backend()}"
        )

    model = make_model(pool)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=1,
        batch_size=32,
        verbose=0,
    )

    # Basic assertions
    assert "accuracy" in history.history
    preds = model.predict(
        np.random.randn(1, 32, 32, 3).astype(np.float32), verbose=0
    )
    assert preds.shape == (1, 10)
