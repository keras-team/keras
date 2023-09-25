import numpy as np
import pytest

import keras
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras import testing


class MyModel(keras.Model):
    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(hidden_dim, activation="relu")
        self.dense3 = layers.Dense(output_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


@pytest.mark.requires_trainable_backend
class BasicFlowTest(testing.TestCase):
    def test_basic_fit(self):
        model = MyModel(hidden_dim=256, output_dim=16)

        x = np.random.random((50000, 128))
        y = np.random.random((50000, 16))
        batch_size = 32
        epochs = 3

        model.compile(
            optimizer=optimizers.SGD(learning_rate=0.001),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        output_before_fit = model(x)
        history = model.fit(
            x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2
        )
        output_after_fit = model(x)

        print("History:")
        print(history.history)

        self.assertNotAllClose(output_before_fit, output_after_fit)
