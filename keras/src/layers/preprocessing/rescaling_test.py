import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RescalingTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_rescaling_basics(self):
        self.run_layer_test(
            layers.Rescaling,
            init_kwargs={"scale": 1.0 / 255, "offset": 0.5},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @pytest.mark.requires_trainable_backend
    def test_rescaling_dtypes(self):
        # int scale
        self.run_layer_test(
            layers.Rescaling,
            init_kwargs={"scale": 2, "offset": 0.5},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # int offset
        self.run_layer_test(
            layers.Rescaling,
            init_kwargs={"scale": 1.0, "offset": 2},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # int inputs
        self.run_layer_test(
            layers.Rescaling,
            init_kwargs={"scale": 1.0 / 255, "offset": 0.5},
            input_shape=(2, 3),
            input_dtype="int16",
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    def test_rescaling_correctness(self):
        layer = layers.Rescaling(scale=1.0 / 255, offset=0.5)
        x = np.random.random((3, 10, 10, 3)) * 255
        out = layer(x)
        self.assertAllClose(out, x / 255 + 0.5)

    def test_tf_data_compatibility(self):
        layer = layers.Rescaling(scale=1.0 / 255, offset=0.5)
        x = np.random.random((3, 10, 10, 3)) * 255
        ds = tf_data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_rescaling_with_channels_first_and_vector_scale(self):
        config = backend.image_data_format()
        backend.set_image_data_format("channels_first")
        layer = layers.Rescaling(
            scale=[1.0 / 255, 1.5 / 255, 2.0 / 255], offset=0.5
        )
        x = np.random.random((2, 3, 10, 10)) * 255
        layer(x)
        backend.set_image_data_format(config)
