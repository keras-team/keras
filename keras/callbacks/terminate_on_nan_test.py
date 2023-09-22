import numpy as np
import pytest

from keras import callbacks
from keras import initializers
from keras import layers
from keras import testing
from keras.models import Sequential
from keras.utils import numerical_utils


class TerminateOnNaNTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_TerminateOnNaN(self):
        TRAIN_SAMPLES = 10
        TEST_SAMPLES = 10
        INPUT_DIM = 3
        NUM_CLASSES = 2
        BATCH_SIZE = 4

        np.random.seed(1337)
        x_train = np.random.random((TRAIN_SAMPLES, INPUT_DIM))
        y_train = np.random.choice(np.arange(NUM_CLASSES), size=TRAIN_SAMPLES)
        x_test = np.random.random((TEST_SAMPLES, INPUT_DIM))
        y_test = np.random.choice(np.arange(NUM_CLASSES), size=TEST_SAMPLES)

        y_test = numerical_utils.to_categorical(y_test)
        y_train = numerical_utils.to_categorical(y_train)
        model = Sequential()
        initializer = initializers.Constant(value=1e5)
        for _ in range(5):
            model.add(
                layers.Dense(
                    2,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        model.add(layers.Dense(NUM_CLASSES))
        model.compile(loss="mean_squared_error", optimizer="sgd")

        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[callbacks.TerminateOnNaN()],
            epochs=20,
        )
        loss = history.history["loss"]
        self.assertEqual(len(loss), 1)
        self.assertTrue(np.isnan(loss[0]) or np.isinf(loss[0]))
