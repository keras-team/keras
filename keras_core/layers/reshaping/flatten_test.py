import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import ops
from keras_core import testing


class FlattenTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_flatten(self):
        inputs = np.random.random((10, 3, 5, 5)).astype("float32")

        # Test default data_format and channels_last
        expected_output = ops.convert_to_tensor(
            np.reshape(inputs, (-1, 5 * 5 * 3))
        )
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={},
            input_data=inputs,
            expected_output=expected_output,
        )
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={"data_format": "channels_last"},
            input_data=inputs,
            expected_output=expected_output,
        )

        # Test channels_first
        expected_output = ops.convert_to_tensor(
            np.reshape(np.transpose(inputs, (0, 2, 3, 1)), (-1, 5 * 5 * 3))
        )
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={"data_format": "channels_first"},
            input_data=inputs,
            expected_output=expected_output,
        )

    @pytest.mark.requires_trainable_backend
    def test_flatten_with_scalar_channels(self):
        inputs = np.random.random((10,)).astype("float32")
        expected_output = ops.convert_to_tensor(np.expand_dims(inputs, -1))

        # Test default data_format and channels_last
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={},
            input_data=inputs,
            expected_output=expected_output,
        )
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={"data_format": "channels_last"},
            input_data=inputs,
            expected_output=expected_output,
        )

        # Test channels_first
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={"data_format": "channels_first"},
            input_data=inputs,
            expected_output=expected_output,
        )

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes",
    )
    def test_flatten_with_dynamic_batch_size(self):
        input_layer = layers.Input(batch_shape=(None, 2, 3))
        flattened = layers.Flatten()(input_layer)
        self.assertEqual(flattened.shape, (None, 2 * 3))

    @pytest.mark.skipif(
        not backend.DYNAMIC_SHAPES_OK,
        reason="Backend does not support dynamic shapes",
    )
    def test_flatten_with_dynamic_dimension(self):
        input_layer = layers.Input(batch_shape=(5, 2, None))
        flattened = layers.Flatten()(input_layer)
        self.assertEqual(flattened.shape, (5, None))
