import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_core import layers
from keras_core import testing


@pytest.mark.requires_trainable_backend
class GlobalAveragePoolingBasicTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        ("channels_last", False, (3, 5, 4), (3, 4)),
        ("channels_last", True, (3, 5, 4), (3, 1, 4)),
        ("channels_first", False, (3, 5, 4), (3, 5)),
    )
    def test_global_average_pooling1d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalAveragePooling1D,
            init_kwargs={
                "data_format": data_format,
                "keepdims": keepdims,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @parameterized.parameters(
        ("channels_last", False, (3, 5, 6, 4), (3, 4)),
        ("channels_last", True, (3, 5, 6, 4), (3, 1, 1, 4)),
        ("channels_first", False, (3, 5, 6, 4), (3, 5)),
    )
    def test_global_average_pooling2d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalAveragePooling2D,
            init_kwargs={
                "data_format": data_format,
                "keepdims": keepdims,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        ("channels_last", False, (3, 5, 6, 5, 4), (3, 4)),
        ("channels_last", True, (3, 5, 6, 5, 4), (3, 1, 1, 1, 4)),
        ("channels_first", False, (3, 5, 6, 5, 4), (3, 5)),
    )
    def test_global_average_pooling3d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalAveragePooling3D,
            init_kwargs={
                "data_format": data_format,
                "keepdims": keepdims,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
        )


class GlobalAveragePoolingCorrectnessTest(
    testing.TestCase, parameterized.TestCase
):
    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
    )
    def test_global_average_pooling1d(self, data_format, keepdims):
        inputs = np.arange(24, dtype="float32").reshape((2, 3, 4))

        layer = layers.GlobalAveragePooling1D(
            data_format=data_format,
            keepdims=keepdims,
        )
        tf_keras_layer = tf.keras.layers.GlobalAveragePooling1D(
            data_format=data_format,
            keepdims=keepdims,
        )

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)

        if data_format == "channels_last":
            mask = np.array([[1, 1, 0], [0, 1, 0]], dtype="int32")
        else:
            mask = np.array([[1, 1, 0, 0], [0, 1, 0, 1]], dtype="int32")
        outputs = layer(inputs, mask)
        expected = tf_keras_layer(inputs, mask)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
    )
    def test_global_average_pooling2d(self, data_format, keepdims):
        inputs = np.arange(96, dtype="float32").reshape((2, 3, 4, 4))

        layer = layers.GlobalAveragePooling2D(
            data_format=data_format,
            keepdims=keepdims,
        )
        tf_keras_layer = tf.keras.layers.GlobalAveragePooling2D(
            data_format=data_format,
            keepdims=keepdims,
        )

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
    )
    def test_global_average_pooling3d(self, data_format, keepdims):
        inputs = np.arange(360, dtype="float32").reshape((2, 3, 3, 5, 4))

        layer = layers.GlobalAveragePooling3D(
            data_format=data_format,
            keepdims=keepdims,
        )
        tf_keras_layer = tf.keras.layers.GlobalAveragePooling3D(
            data_format=data_format,
            keepdims=keepdims,
        )

        outputs = layer(inputs)
        expected = tf_keras_layer(inputs)
        self.assertAllClose(outputs, expected)
