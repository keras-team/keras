"""Tests for Adaptive Average Pooling 2D layer."""

import numpy as np
import pytest

from keras.src import layers
from keras.src import ops
from keras.src import testing

# Only import torch if available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AdaptiveAveragePooling2DTest(testing.TestCase):
    """Test suite for AdaptiveAveragePooling2D layer."""

    def test_adaptive_avg_pooling_2d_basic(self):
        """Test basic functionality with square output."""
        layer = layers.AdaptiveAveragePooling2D(output_size=4)
        x = np.random.randn(2, 8, 8, 3).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 4, 4, 3))

    def test_adaptive_avg_pooling_2d_rectangular(self):
        """Test with rectangular output size."""
        layer = layers.AdaptiveAveragePooling2D(output_size=(2, 4))
        x = np.random.randn(2, 8, 8, 3).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 2, 4, 3))

    def test_adaptive_avg_pooling_2d_channels_first(self):
        """Test channels_first data format."""
        layer = layers.AdaptiveAveragePooling2D(
            output_size=4, data_format="channels_first"
        )
        x = np.random.randn(2, 3, 8, 8).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 4, 4))

    def test_adaptive_avg_pooling_2d_output_shape(self):
        """Test compute_output_shape method."""
        layer = layers.AdaptiveAveragePooling2D(output_size=(2, 4))
        x_shape = (2, 8, 8, 3)
        output_shape = layer.compute_output_shape(x_shape)
        self.assertEqual(output_shape, (2, 2, 4, 3))

    def test_adaptive_avg_pooling_2d_invalid_output_size(self):
        """Test error handling for invalid output_size."""
        with self.assertRaisesRegex(ValueError, "`output_size` must be"):
            layers.AdaptiveAveragePooling2D(output_size=(2, 3, 4))

    def test_adaptive_avg_pooling_2d_invalid_data_format(self):
        """Test error handling for invalid data_format."""
        with self.assertRaisesRegex(ValueError, "Invalid data_format"):
            layer = layers.AdaptiveAveragePooling2D(
                output_size=4, data_format="invalid"
            )
            x = np.random.randn(2, 8, 8, 3).astype("float32")
            layer(x)

    def test_adaptive_avg_pooling_2d_get_config(self):
        """Test layer serialization."""
        layer = layers.AdaptiveAveragePooling2D(
            output_size=(3, 5), data_format="channels_first"
        )
        config = layer.get_config()
        self.assertEqual(config["output_size"], (3, 5))
        self.assertEqual(config["data_format"], "channels_first")

        # Test reconstruction from config
        new_layer = layers.AdaptiveAveragePooling2D.from_config(config)
        self.assertEqual(new_layer.output_size, (3, 5))
        self.assertEqual(new_layer.data_format, "channels_first")


class AdaptiveMaxPooling2DTest(testing.TestCase):
    """Test suite for AdaptiveMaxPooling2D layer."""

    def test_adaptive_max_pooling_2d_basic(self):
        """Test basic functionality with square output."""
        layer = layers.AdaptiveMaxPooling2D(output_size=4)
        x = np.random.randn(2, 8, 8, 3).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 4, 4, 3))

    def test_adaptive_max_pooling_2d_rectangular(self):
        """Test with rectangular output size."""
        layer = layers.AdaptiveMaxPooling2D(output_size=(3, 5))
        x = np.random.randn(2, 9, 15, 3).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 5, 3))

    def test_adaptive_max_pooling_2d_channels_first(self):
        """Test channels_first data format."""
        layer = layers.AdaptiveMaxPooling2D(
            output_size=4, data_format="channels_first"
        )
        x = np.random.randn(2, 3, 8, 8).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 4, 4))

    def test_adaptive_max_pooling_2d_output_shape(self):
        """Test compute_output_shape method."""
        layer = layers.AdaptiveMaxPooling2D(output_size=(3, 5))
        x_shape = (2, 9, 15, 3)
        output_shape = layer.compute_output_shape(x_shape)
        self.assertEqual(output_shape, (2, 3, 5, 3))

    def test_adaptive_max_pooling_2d_get_config(self):
        """Test layer serialization."""
        layer = layers.AdaptiveMaxPooling2D(
            output_size=(3, 5), data_format="channels_first"
        )
        config = layer.get_config()
        self.assertEqual(config["output_size"], (3, 5))
        self.assertEqual(config["data_format"], "channels_first")

        # Test reconstruction from config
        new_layer = layers.AdaptiveMaxPooling2D.from_config(config)
        self.assertEqual(new_layer.output_size, (3, 5))
        self.assertEqual(new_layer.data_format, "channels_first")


# Parameterized tests as standalone functions (OUTSIDE classes)
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "output_size", [(4, 4), (2, 2), (3, 5), (1, 1), (7, 9)]
)
def test_adaptive_avg_pooling2d_matches_torch(output_size):
    """Test numerical accuracy against PyTorch implementation."""
    x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)

    # PyTorch
    x_torch = torch.tensor(x_np)
    y_torch = torch.nn.functional.adaptive_avg_pool2d(x_torch, output_size)

    # Keras/JAX
    x_keras = ops.convert_to_tensor(x_np)
    y_keras = ops.adaptive_avg_pool(
        x_keras, output_size=output_size, data_format="channels_first"
    )

    y_keras_np = np.asarray(y_keras)

    np.testing.assert_allclose(
        y_keras_np, y_torch.numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize(
    "output_size", [(4, 4), (2, 2), (3, 5), (1, 1), (7, 9)]
)
def test_adaptive_max_pooling2d_matches_torch(output_size):
    """Test numerical accuracy against PyTorch implementation."""
    x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)

    # PyTorch
    x_torch = torch.tensor(x_np)
    y_torch = torch.nn.functional.adaptive_max_pool2d(x_torch, output_size)

    # Keras/JAX
    x_keras = ops.convert_to_tensor(x_np)
    y_keras = ops.adaptive_max_pool(
        x_keras, output_size=output_size, data_format="channels_first"
    )

    y_keras_np = np.asarray(y_keras)

    np.testing.assert_allclose(
        y_keras_np, y_torch.numpy(), rtol=1e-5, atol=1e-5
    )
