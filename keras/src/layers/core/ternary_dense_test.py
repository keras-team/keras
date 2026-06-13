import os

import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
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

    # Quantization: real ternary export + inference.

    def test_quantize_matches_float_forward(self):
        # Freezing the ternary kernel must not change the layer's outputs:
        # the packed kernel holds exactly the values the float path produces.
        layer = layers.TernaryDense(16)
        layer.build((None, 11))
        x = np.random.rand(4, 11).astype("float32")
        y_float = layer(x)

        layer.quantize("ternary")
        self.assertEqual(layer.quantization_mode, "ternary")
        # The float kernel is gone; only the packed kernel + scale remain.
        self.assertFalse(hasattr(layer, "kernel"))
        self.assertEqual(
            backend.standardize_dtype(layer._packed_kernel.dtype), "uint8"
        )

        y_quantized = layer(x)
        self.assertAllClose(y_float, y_quantized)

    def test_quantize_fixed_threshold_matches_float_forward(self):
        # Fixed-threshold mode applies no beta rescaling (scale == 1.0).
        layer = layers.TernaryDense(8, threshold=0.02, use_bias=False)
        layer.build((None, 9))
        x = np.random.rand(3, 9).astype("float32")
        y_float = layer(x)

        layer.quantize("ternary")
        self.assertAllClose(ops.convert_to_numpy(layer.kernel_scale), 1.0)
        self.assertAllClose(y_float, layer(x))

    def test_quantized_kernel_is_packed_at_floor(self):
        # input_dim=40 -> ceil(40/5)=8 packed rows; 8 bytes encode 40 trits.
        layer = layers.TernaryDense(32)
        layer.build((None, 40))
        layer.quantize("ternary")

        self.assertEqual(tuple(layer._packed_kernel.shape), (8, 32))

        n_weights = 40 * 32
        n_bytes = 8 * 32
        bits_per_weight = 8 * n_bytes / n_weights
        self.assertEqual(bits_per_weight, 1.6)
        # Strictly denser than int4 (would need n_weights / 2 bytes).
        self.assertLess(n_bytes, n_weights // 2)

    def test_quantized_model_save_load(self):
        layer = layers.TernaryDense(16)
        layer.build((None, 8))
        x = np.random.random((2, 8))
        y_float = layer(x)
        layer.quantize("ternary")
        y_quantized = layer(x)
        self.assertAllClose(y_float, y_quantized)

        # Full model save / load round-trip.
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_ternary_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertEqual(new_model.layers[0].quantization_mode, "ternary")
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Weights-only save / load round-trip.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_ternary_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential([layers.TernaryDense(16)])
        new_model.build((None, 8))
        new_model.quantize("ternary")
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_model_quantize_ternary(self):
        model = models.Sequential([layers.TernaryDense(8)])
        model.build((None, 6))
        x = np.random.rand(3, 6).astype("float32")
        y_float = model.predict(x)
        model.quantize("ternary")
        self.assertEqual(model.layers[0].quantization_mode, "ternary")
        self.assertAllClose(y_float, model.predict(x))

    def test_quantize_unbuilt_raises(self):
        layer = layers.TernaryDense(4)
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            layer.quantize("ternary")

    def test_quantize_twice_raises(self):
        layer = layers.TernaryDense(4)
        layer.build((None, 6))
        layer.quantize("ternary")
        with self.assertRaisesRegex(ValueError, "already quantized"):
            layer.quantize("ternary")

    def test_quantize_invalid_mode_raises(self):
        layer = layers.TernaryDense(4)
        layer.build((None, 6))
        with self.assertRaises(NotImplementedError):
            layer.quantize("int8")
