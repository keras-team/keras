import numpy as np
import pytest
from absl.testing import parameterized

from keras import backend
from keras import layers
from keras import ops
from keras import testing


class FlattenTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        [
            {"testcase_name": "dense", "sparse": False},
            {"testcase_name": "sparse", "sparse": True},
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_flatten(self, sparse):
        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            pytest.skip("Backend does not support sparse tensors.")

        inputs = np.random.random((10, 3, 5, 5)).astype("float32")
        # Make the ndarray relatively sparse
        inputs = np.multiply(inputs, inputs >= 0.8)
        expected_output_channels_last = ops.convert_to_tensor(
            np.reshape(inputs, (-1, 5 * 5 * 3))
        )
        expected_output_channels_first = ops.convert_to_tensor(
            np.reshape(np.transpose(inputs, (0, 2, 3, 1)), (-1, 5 * 5 * 3))
        )
        if sparse:
            import tensorflow as tf

            inputs = tf.sparse.from_dense(inputs)
            expected_output_channels_last = tf.sparse.from_dense(
                expected_output_channels_last
            )
            expected_output_channels_first = tf.sparse.from_dense(
                expected_output_channels_first
            )

        # Test default data_format and channels_last
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={},
            input_data=inputs,
            input_sparse=True,
            expected_output=expected_output_channels_last
            if backend.config.image_data_format() == "channels_last"
            else expected_output_channels_first,
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={"data_format": "channels_last"},
            input_data=inputs,
            input_sparse=True,
            expected_output=expected_output_channels_last,
            expected_output_sparse=sparse,
            run_training_check=not sparse,
        )

        # Test channels_first
        self.run_layer_test(
            layers.Flatten,
            init_kwargs={"data_format": "channels_first"},
            input_data=inputs,
            input_sparse=True,
            expected_output=expected_output_channels_first,
            expected_output_sparse=sparse,
            run_training_check=not sparse,
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

    def test_flatten_with_dynamic_batch_size(self):
        input_layer = layers.Input(batch_shape=(None, 2, 3))
        flattened = layers.Flatten()(input_layer)
        self.assertEqual(flattened.shape, (None, 2 * 3))

    def test_flatten_with_dynamic_dimension(self):
        input_layer = layers.Input(batch_shape=(5, 2, None))
        flattened = layers.Flatten()(input_layer)
        self.assertEqual(flattened.shape, (5, None))
