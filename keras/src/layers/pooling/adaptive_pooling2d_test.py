"""Tests for Adaptive Average and Max Pooling 2D layer."""

import numpy as np
import pytest

from keras.src import backend as K
from keras.src import layers
from keras.src import ops
from keras.src import testing

SKIP_BACKENDS = ["openvino", "numpy"]

pytestmark = pytest.mark.skipif(
    K.backend() in SKIP_BACKENDS,
    reason=(
        "Adaptive pooling tests not supported for backend: {}".format(
            K.backend()
        )
    ),
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AdaptivePooling2DLayerTest(testing.TestCase):
    """Basic tests for AdaptiveAveragePooling2D and AdaptiveMaxPooling2D."""

    def _run_layer_test(self, layer_class, x_np, output_size, data_format):
        layer = layer_class(output_size=output_size, data_format=data_format)
        y = layer(x_np)
        expected_shape = layer.compute_output_shape(x_np.shape)
        self.assertEqual(y.shape, expected_shape)

    def test_average_pooling_basic_shapes(self):
        shape = (2, 3, 8, 8)  # N,C,H,W
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveAveragePooling2D,
            x,
            output_size=4,
            data_format="channels_first",
        )

    def test_max_pooling_basic_shapes(self):
        shape = (2, 3, 8, 8)
        x = np.random.randn(*shape).astype("float32")
        self._run_layer_test(
            layers.AdaptiveMaxPooling2D,
            x,
            output_size=4,
            data_format="channels_first",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize("output_size", [1, 2, 3, 4])
def test_adaptive_avg_pool2d_matches_torch(output_size):
    x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    x_torch = torch.tensor(x_np)
    y_torch = torch.nn.functional.adaptive_avg_pool2d(x_torch, output_size)

    x_keras = ops.convert_to_tensor(x_np)
    y_keras = ops.adaptive_avg_pool(
        x_keras, output_size=output_size, data_format="channels_first"
    )
    y_keras_np = np.asarray(y_keras)

    np.testing.assert_allclose(
        y_keras_np, y_torch.numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize("output_size", [1, 2, 3, 4])
def test_adaptive_max_pool2d_matches_torch(output_size):
    x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
    x_torch = torch.tensor(x_np)
    y_torch = torch.nn.functional.adaptive_max_pool2d(x_torch, output_size)

    x_keras = ops.convert_to_tensor(x_np)
    y_keras = ops.adaptive_max_pool(
        x_keras, output_size=output_size, data_format="channels_first"
    )
    y_keras_np = np.asarray(y_keras)

    np.testing.assert_allclose(
        y_keras_np, y_torch.numpy(), rtol=1e-5, atol=1e-5
    )
