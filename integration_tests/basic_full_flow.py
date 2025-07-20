import os

import numpy as np
import pytest

import keras
from keras.src import layers
from keras.src import losses
from keras.src import metrics
from keras.src import optimizers
from keras.src import testing


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


class BasicFlowTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basic_fit(self):
        model = MyModel(hidden_dim=2, output_dim=1)

        x = np.random.random((128, 4))
        y = np.random.random((128, 4))
        batch_size = 32
        epochs = 3

        model.compile(
            optimizer=optimizers.SGD(learning_rate=0.001),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        output_before_fit = model(x)
        model.fit(
            x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2
        )
        output_after_fit = model(x)

        self.assertNotAllClose(output_before_fit, output_after_fit)

    def test_basic_fit_no_training(self):
        model = MyModel(hidden_dim=2, output_dim=1)
        x = np.random.random((128, 4))
        model.predict(x)
        model(x)

    @pytest.mark.skipif(
        os.environ.get("KERAS_NNX_ENABLED") != "true",
        reason="Test only runs with NNX enabled",
    )
    def test_bare_ops_functional(self):
        """Test that functional models work correctly with bare ops."""
        # Create input
        inputs = keras.Input(shape=(10,))

        # Add a layer
        x = layers.Dense(5, activation="relu")(inputs)

        # Add a bare op (not a layer)
        x = keras.ops.add(x, 2.0)

        # Add another layer
        outputs = layers.Dense(1)(x)

        # Create functional model
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        test_input = np.random.random((2, 10))
        output = model(test_input)

        # Verify output shape and values
        self.assertEqual(output.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(output)))

        # Test with multiple bare ops
        inputs2 = keras.Input(shape=(5,))
        x2 = layers.Dense(3, activation="relu")(inputs2)
        x2 = keras.ops.add(x2, 1.0)
        x2 = keras.ops.multiply(x2, 2.0)
        x2 = keras.ops.subtract(x2, 0.5)
        outputs2 = layers.Dense(1)(x2)

        model2 = keras.Model(inputs=inputs2, outputs=outputs2)
        test_input2 = np.random.random((3, 5))
        output2 = model2(test_input2)

        # Verify output shape and values
        self.assertEqual(output2.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(output2)))
