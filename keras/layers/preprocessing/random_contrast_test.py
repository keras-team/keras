import numpy as np
import pytest
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import testing


class RandomContrastTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomContrast,
            init_kwargs={
                "factor": 0.75,
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_contrast(self):
        seed = 9809
        np.random.seed(seed)
        inputs = np.random.random((12, 8, 16, 3))
        layer = layers.RandomContrast(factor=0.5, seed=seed)
        outputs = layer(inputs)

        # Actual contrast arithmetic
        np.random.seed(seed)
        factor = np.random.uniform(0.5, 1.5)
        inp_mean = np.mean(inputs, axis=-3, keepdims=True)
        inp_mean = np.mean(inp_mean, axis=-2, keepdims=True)
        actual_outputs = (inputs - inp_mean) * factor + inp_mean
        outputs = backend.convert_to_numpy(outputs)
        actual_outputs = np.clip(outputs, 0, 255)

        self.assertAllClose(outputs, actual_outputs)

    def test_tf_data_compatibility(self):
        layer = layers.RandomContrast(factor=0.5, seed=1337)
        input_data = np.random.random((2, 8, 8, 3))
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
