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


class AdaptivePooling1DLayerTest(testing.TestCase):
    """Tests for AdaptiveAveragePooling1D and AdaptiveMaxPooling1D."""

    def _run_layer_test(self, layer_class, x_np, output_size, data_format):
        """Helper: test layer output shape matches compute_output_shape()."""
        layer = layer_class(output_size=output_size, data_format=data_format)
        y = layer(x_np)
        expected_shape = layer.compute_output_shape(x_np.shape)
        self.assertEqual(y.shape, expected_shape)

    def test_average_pooling_basic_shapes(self):
        """Test AdaptiveAveragePooling1D basic shape transformation."""
        shape = (2, 3, 8)  # N,C,L
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveAveragePooling1D,
            x,
            output_size=4,
            data_format="channels_first",
        )

    def test_max_pooling_basic_shapes(self):
        """Test AdaptiveMaxPooling1D basic shape transformation."""
        shape = (2, 3, 8)
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveMaxPooling1D,
            x,
            output_size=4,
            data_format="channels_first",
        )

    def test_average_pooling_channels_last(self):
        """Test AdaptiveAveragePooling1D with channels_last format."""
        shape = (2, 8, 3)  # N,L,C
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveAveragePooling1D,
            x,
            output_size=4,
            data_format="channels_last",
        )

    def test_max_pooling_channels_last(self):
        """Test AdaptiveMaxPooling1D with channels_last format."""
        shape = (2, 8, 3)
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveMaxPooling1D,
            x,
            output_size=4,
            data_format="channels_last",
        )

    def test_average_pooling_compute_output_shape(self):
        """Test compute_output_shape() for AdaptiveAveragePooling1D."""
        layer = layers.AdaptiveAveragePooling1D(
            output_size=16, data_format="channels_last"
        )
        input_shape = (None, 64, 3)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (None, 16, 3))

    def test_max_pooling_compute_output_shape(self):
        """Test compute_output_shape() for AdaptiveMaxPooling1D."""
        layer = layers.AdaptiveMaxPooling1D(
            output_size=16, data_format="channels_first"
        )
        input_shape = (2, 3, 64)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (2, 3, 16))

    def test_average_pooling_get_config(self):
        """Test get_config() serialization for AdaptiveAveragePooling1D."""
        layer = layers.AdaptiveAveragePooling1D(
            output_size=32, data_format="channels_first"
        )
        config = layer.get_config()
        self.assertEqual(config["output_size"], (32,))
        self.assertEqual(config["data_format"], "channels_first")

    def test_max_pooling_get_config(self):
        """Test get_config() serialization for AdaptiveMaxPooling1D."""
        layer = layers.AdaptiveMaxPooling1D(
            output_size=32, data_format="channels_last"
        )
        config = layer.get_config()
        self.assertEqual(config["output_size"], (32,))
        self.assertEqual(config["data_format"], "channels_last")

    def test_average_pooling_numerical(self):
        """Test AdaptiveAveragePooling1D numerical correctness."""
        inputs = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype="float32")
        expected = np.array([[[2.0, 5.0]]], dtype="float32")

        layer = layers.AdaptiveAveragePooling1D(
            output_size=2, data_format="channels_first"
        )

        outputs = layer(inputs)
        self.assertAllClose(outputs, expected, atol=1e-4)

    def test_max_pooling_numerical(self):
        """Test AdaptiveMaxPooling1D numerical correctness."""
        inputs = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype="float32")
        expected = np.array([[[3.0, 6.0]]], dtype="float32")

        layer = layers.AdaptiveMaxPooling1D(
            output_size=2, data_format="channels_first"
        )

        outputs = layer(inputs)
        self.assertAllClose(outputs, expected, atol=1e-4)
