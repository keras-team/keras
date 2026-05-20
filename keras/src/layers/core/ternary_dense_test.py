import numpy as np

from keras.src import layers
from keras.src import ops
from keras.src import testing


class TernaryDenseTest(testing.TestCase):
    def test_output_shape(self):
        layer = layers.TernaryDense(8)
        x = np.random.rand(4, 16).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (4, 8))

    def test_effective_kernel_is_ternary(self):
        layer = layers.TernaryDense(16)
        x = np.ones((1, 8), dtype="float32")
        layer(x)
        k = ops.convert_to_numpy(layer._ternary_kernel())
        # STE: round to nearest int for fp tolerance
        rounded = set(np.round(k).astype(np.int32).flat)
        self.assertTrue(
            rounded <= {-1, 0, 1},
            f"Expected values near {{-1, 0, 1}}, got {np.unique(k)}",
        )

    def test_threshold_zero_no_zeros(self):
        # threshold=0.0: |w| > 0 for all non-zero glorot weights → all ±1
        layer = layers.TernaryDense(16, threshold=0.0)
        x = np.ones((1, 8), dtype="float32")
        layer(x)
        k = ops.convert_to_numpy(layer._ternary_kernel())
        self.assertNotIn(0.0, np.unique(k).tolist())

    def test_threshold_large_all_zeros(self):
        layer = layers.TernaryDense(4, threshold=1e9)
        x = np.ones((1, 4), dtype="float32")
        y = layer(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y),
            np.zeros((1, 4), dtype="float32"),
            atol=1e-6,
        )

    def test_forward_matches_manual(self):
        # 2-input, 2-output layer, no bias.
        # kernel = [[1, -1], [0, 1]]  (will be forced via direct assignment)
        # input  = [[2, 3]]
        # expected output = [[2*1 + 3*0, 2*(-1) + 3*1]] = [[2, 1]]
        layer = layers.TernaryDense(2, use_bias=False)
        layer.build((None, 2))
        layer.kernel.assign(
            np.array([[1.0, -1.0], [0.0, 1.0]], dtype="float32")
        )
        layer.threshold = 0.5  # only |w|>0.5 survive: 0.0 → 0, ±1.0 → ±1
        x = np.array([[2.0, 3.0]], dtype="float32")
        y = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(y, [[2.0, 1.0]], atol=1e-6)

    def test_no_bias(self):
        layer = layers.TernaryDense(4, use_bias=False)
        layer.build((None, 4))
        self.assertIsNone(layer.bias)

    def test_activation(self):
        layer = layers.TernaryDense(8, activation="relu")
        x = np.random.randn(2, 4).astype("float32")
        y = ops.convert_to_numpy(layer(x))
        self.assertTrue(np.all(y >= 0), "relu output should be non-negative")

    def test_nd_input(self):
        # Works on 3D input (batch, seq, dim)
        layer = layers.TernaryDense(16)
        x = np.random.rand(2, 5, 8).astype("float32")
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 16))

    def test_get_config_roundtrip(self):
        layer = layers.TernaryDense(
            32,
            threshold=0.01,
            activation="relu",
            use_bias=False,
        )
        config = layer.get_config()
        self.assertEqual(config["units"], 32)
        self.assertAlmostEqual(config["threshold"], 0.01)
        self.assertEqual(config["use_bias"], False)

        restored = layers.TernaryDense.from_config(config)
        self.assertEqual(restored.units, 32)
        self.assertAlmostEqual(restored.threshold, 0.01)

    def test_serialization(self):
        layer = layers.TernaryDense(8)
        x = np.random.rand(2, 4).astype("float32")
        layer(x)
        config = layer.get_config()
        restored = layers.TernaryDense.from_config(config)
        self.assertEqual(restored.units, layer.units)

    def test_invalid_units(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            layers.TernaryDense(0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            layers.TernaryDense(-4)

    def test_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            layers.TernaryDense(4, threshold=-0.1)

    def test_beta_scaling_applied_when_threshold_none(self):
        # threshold=None → output *= beta=mean(|kernel|) per BitNet b1.58 §3.1
        layer = layers.TernaryDense(2, use_bias=False)
        layer.build((None, 2))
        kernel = np.array([[0.5, -0.5], [0.5, 0.5]], dtype="float32")
        layer.kernel.assign(kernel)
        # t=0.25; all |w|=0.5 > t → all ±1; beta=0.5
        # k_ternary = [[1, -1], [1, 1]]; beta = mean(|kernel|) = 0.5
        # x=[[1,1]] @ k_ternary = [[2, 0]]; scaled = [[1.0, 0.0]]
        x = np.array([[1.0, 1.0]], dtype="float32")
        y = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(y, [[1.0, 0.0]], atol=1e-5)

    def test_compute_output_shape(self):
        layer = layers.TernaryDense(10)
        self.assertEqual(layer.compute_output_shape((None, 5)), (None, 10))
        self.assertEqual(layer.compute_output_shape((2, 3, 5)), (2, 3, 10))
