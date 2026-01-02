import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import testing

SKIP_BACKENDS = ["openvino"]

pytestmark = pytest.mark.skipif(
    backend.backend() in SKIP_BACKENDS,
    reason=(
        "Adaptive pooling tests not supported for backend: {}".format(
            backend.backend()
        )
    ),
)


class AdaptivePooling3DLayerTest(testing.TestCase):
    """Tests for AdaptiveAveragePooling3D and AdaptiveMaxPooling3D."""

    def _run_layer_test(self, layer_class, x_np, output_size, data_format):
        """Helper: test layer output shape matches compute_output_shape()."""
        layer = layer_class(output_size=output_size, data_format=data_format)
        y = layer(x_np)
        expected_shape = layer.compute_output_shape(x_np.shape)
        self.assertEqual(y.shape, expected_shape)

    def test_average_pooling_basic_shapes(self):
        """Test AdaptiveAveragePooling3D basic shape transformation."""
        shape = (2, 3, 8, 8, 8)  # N,C,D,H,W
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveAveragePooling3D,
            x,
            output_size=4,
            data_format="channels_first",
        )

    def test_max_pooling_basic_shapes(self):
        """Test AdaptiveMaxPooling3D basic shape transformation."""
        shape = (2, 3, 8, 8, 8)
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveMaxPooling3D,
            x,
            output_size=4,
            data_format="channels_first",
        )

    def test_average_pooling_channels_last(self):
        """Test AdaptiveAveragePooling3D with channels_last format."""
        shape = (2, 8, 8, 8, 3)  # N,D,H,W,C
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveAveragePooling3D,
            x,
            output_size=4,
            data_format="channels_last",
        )

    def test_max_pooling_channels_last(self):
        """Test AdaptiveMaxPooling3D with channels_last format."""
        shape = (2, 8, 8, 8, 3)
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveMaxPooling3D,
            x,
            output_size=4,
            data_format="channels_last",
        )

    def test_average_pooling_tuple_output_size(self):
        """Test AdaptiveAveragePooling3D with tuple output_size."""
        shape = (2, 8, 8, 8, 3)
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveAveragePooling3D,
            x,
            output_size=(4, 4, 4),
            data_format="channels_last",
        )

    def test_max_pooling_tuple_output_size(self):
        """Test AdaptiveMaxPooling3D with tuple output_size."""
        shape = (2, 8, 8, 8, 3)
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveMaxPooling3D,
            x,
            output_size=(2, 4, 4),
            data_format="channels_last",
        )

    def test_average_pooling_compute_output_shape(self):
        """Test compute_output_shape() for AdaptiveAveragePooling3D."""
        layer = layers.AdaptiveAveragePooling3D(
            output_size=8, data_format="channels_last"
        )
        input_shape = (None, 32, 32, 32, 3)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (None, 8, 8, 8, 3))

    def test_max_pooling_compute_output_shape(self):
        """Test compute_output_shape() for AdaptiveMaxPooling3D."""
        layer = layers.AdaptiveMaxPooling3D(
            output_size=(4, 8, 8), data_format="channels_first"
        )
        input_shape = (2, 3, 32, 32, 32)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (2, 3, 4, 8, 8))

    def test_average_pooling_get_config(self):
        """Test get_config() serialization for AdaptiveAveragePooling3D."""
        layer = layers.AdaptiveAveragePooling3D(
            output_size=16, data_format="channels_first"
        )
        config = layer.get_config()
        self.assertEqual(config["output_size"], (16, 16, 16))
        self.assertEqual(config["data_format"], "channels_first")

    def test_max_pooling_get_config(self):
        """Test get_config() serialization for AdaptiveMaxPooling3D."""
        layer = layers.AdaptiveMaxPooling3D(
            output_size=(8, 16, 16), data_format="channels_last"
        )
        config = layer.get_config()
        self.assertEqual(config["output_size"], (8, 16, 16))
        self.assertEqual(config["data_format"], "channels_last")

    def test_average_pooling3d_numerical(self):
        """Test AdaptiveAveragePooling3D numerical correctness."""
        inputs = np.array(
            [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
            dtype="float32",
        )
        expected = np.array(
            [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
            dtype="float32",
        )

        layer = layers.AdaptiveAveragePooling3D(
            output_size=2, data_format="channels_first"
        )
        outputs = layer(inputs)
        self.assertAllClose(outputs, expected, atol=1e-4)

    def test_max_pooling3d_numerical(self):
        """Test AdaptiveMaxPooling3D numerical correctness."""
        inputs = np.array(
            [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
            dtype="float32",
        )
        expected = np.array(
            [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
            dtype="float32",
        )

        layer = layers.AdaptiveMaxPooling3D(
            output_size=2, data_format="channels_first"
        )
        outputs = layer(inputs)
        self.assertAllClose(outputs, expected, atol=1e-4)
