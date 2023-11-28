import numpy as np
import pytest

from keras import testing
from keras.layers.activations import hard_swish


class HardSwishTest(testing.TestCase):
    def test_config(self):
        hard_swish_layer = hard_swish.HardSwish()
        self.run_class_serialization_test(hard_swish_layer)

    @pytest.mark.requires_trainable_backend
    def test_hard_swish(self):
        self.run_layer_test(
            hard_swish.HardSwish,
            init_kwargs={},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_correctness(self):
        def np_hard_swish(x):
            return x * np.minimum(np.maximum(0, x + 3.0), 6.0) * (1.0 / 6.0)

        x = np.random.uniform(-3, 3, (2, 2, 5)).astype("float32")
        hard_swish_layer = hard_swish.HardSwish()
        self.assertAllClose(hard_swish_layer(x), np_hard_swish(x))

        x = np.random.uniform(3, 6, (2, 2, 5)).astype("float32")
        self.assertAllClose(hard_swish_layer(x), x)

        x = np.random.uniform(-6, -3, (2, 2, 5)).astype("float32")
        self.assertAllClose(hard_swish_layer(x), np.zeros_like(x))
