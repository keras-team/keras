import numpy as np
import pytest

import keras
from keras.src import testing


@pytest.mark.requires_trainable_backend
class BasicFlowTest(testing.TestCase):
    def test_basic_fit(self):
        def get_expected_result(x1, x2):
            z = x1 + x2
            return 1.0 / (1.0 + np.exp(-z))

        x1 = keras.Input(shape=(2,), dtype="float32")
        x2 = keras.Input(shape=(2,), dtype="float32")
        add = keras.layers.Add()([x1, x2])
        sigmoid = keras.activations.sigmoid(add)
        model = keras.Model(inputs=[x1, x2], outputs=sigmoid)

        x = np.random.random((128, 4))
        y = np.random.random((128, 4))

        result = model.predict([x, y])
        expected = get_expected_result(x, y)
        self.assertAllClose(result, expected, rtol=1e-05)
