import numpy as np
import pytest

import keras
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.gptq import GPTQ
from keras.src.quantizers.gptq import _stable_permutation
from keras.src.quantizers.gptq import gptq_quantize_matrix
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.gptq_quantizer import dequantize
from keras.src.quantizers.gptq_quantizer import quantize


def _get_mock_layer(layer_type, kernel_shape, rng):
    if layer_type == "Dense":
        layer = layers.Dense(units=kernel_shape[1])
        layer.build(input_shape=(None, kernel_shape[0]))
    elif layer_type == "EinsumDense":
        output_shape = (kernel_shape[1], kernel_shape[2])
        layer = layers.EinsumDense(
            equation="...h,hio->...io", output_shape=output_shape
        )
        dummy_input = rng.standard_normal(size=(1, 1, kernel_shape[0]))
        layer(dummy_input)
        layer.kernel.assign(
            rng.standard_normal(size=kernel_shape).astype("float32")
        )
    else:
        layer = layers.Layer()
    return layer


@pytest.mark.requires_trainable_backend
class GPTQTest(testing.TestCase):
    def test_initialization_with_dense_layer(self):
        rng = np.random.default_rng(seed=42)

        mock_layer = _get_mock_layer("Dense", kernel_shape=(64, 128), rng=rng)

        gptq_instance = GPTQ(mock_layer)
        self.assertEqual(gptq_instance.rows, 64)
        self.assertEqual(gptq_instance.columns, 128)
        self.assertEqual(gptq_instance.hessian.shape, (64, 64))

    def test_initialization_with_einsumdense_3d(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer(
            "EinsumDense", kernel_shape=(64, 4, 32), rng=rng
        )
        gptq_instance = GPTQ(mock_layer)
        self.assertEqual(gptq_instance.rows, 64)
        self.assertEqual(gptq_instance.columns, 4 * 32)
        self.assertEqual(gptq_instance.hessian.shape, (64, 64))

    def test_update_hessian(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer("Dense", kernel_shape=(16, 32), rng=rng)
        gptq_instance = GPTQ(mock_layer)
        batch1 = rng.standard_normal(size=(8, 16)).astype("float32")
        gptq_instance.update_hessian_with_batch(batch1)
        self.assertEqual(gptq_instance.num_samples, 8)
        H1 = np.copy(ops.convert_to_numpy(gptq_instance.hessian))
        batch2 = rng.standard_normal(size=(4, 16)).astype("float32")
        gptq_instance.update_hessian_with_batch(batch2)
        self.assertEqual(gptq_instance.num_samples, 12)
        H2 = np.copy(ops.convert_to_numpy(gptq_instance.hessian))
        self.assertFalse(np.allclose(H1, H2))

    def test_full_quantization_process(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer("Dense", kernel_shape=(16, 32), rng=rng)
        original_weights = np.copy(ops.convert_to_numpy(mock_layer.kernel))

        gptq_instance = GPTQ(
            mock_layer,
            GPTQConfig(
                dataset=None,
                tokenizer=None,
                weight_bits=4,
                symmetric=False,
                group_size=-1,
            ),
        )
        calibration_data = rng.standard_normal(size=(128, 16)).astype("float32")
        gptq_instance.update_hessian_with_batch(calibration_data)
        gptq_instance.quantize_and_correct_block()

        quantized_weights = ops.convert_to_numpy(mock_layer.kernel)
        self.assertFalse(np.allclose(original_weights, quantized_weights))

        gptq_instance.free()
        self.assertIsNone(gptq_instance.hessian)

    def test_unsupported_layer_error(self):
        rng = np.random.default_rng(seed=42)
        unsupported_layer = _get_mock_layer(
            "Unsupported", kernel_shape=None, rng=rng
        )
        with self.assertRaisesRegex(TypeError, "Unsupported layer type"):
            GPTQ(unsupported_layer)

    def test_update_hessian_invalid_input(self):
        rng = np.random.default_rng(seed=42)
        mock_layer = _get_mock_layer("Dense", kernel_shape=(16, 32), rng=rng)
        gptq_instance = GPTQ(mock_layer)
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            gptq_instance.update_hessian_with_batch(None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            gptq_instance.update_hessian_with_batch(np.empty((0, 16)))
        with self.assertRaisesRegex(ValueError, "match input features"):
            bad_input = rng.standard_normal(size=(8, 99))
            gptq_instance.update_hessian_with_batch(bad_input)

    def test_streaming_equals_big_batch(self):
        """Tests that streaming updates match big batch updates."""
        # dummy inputs
        x = ops.array(np.random.randn(100, 7), "float32")

        # One-shot hessian update
        layer_1 = layers.Dense(5, use_bias=False)
        layer_1.build(input_shape=(None, 7))

        g1 = GPTQ(layer_1)
        g1.update_hessian_with_batch(x)

        # Streamed hessian update
        layer_2 = layers.Dense(5, use_bias=False)
        layer_2.build(input_shape=(None, 7))
        g2 = GPTQ(layer_2)
        g2.update_hessian_with_batch(x[:50])
        g2.update_hessian_with_batch(x[50:])

        # Both the one-shot and streamed hessian updates should match
        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_hessian_matches_closed_form(self):
        """Tests that the Hessian matches the closed-form solution."""
        x = ops.array(np.random.randn(128, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))
        g = GPTQ(layer)
        g.update_hessian_with_batch(x)

        expected = ops.multiply(
            ops.divide(2.0, x.shape[0]), ops.matmul(ops.transpose(x), x)
        )
        self.assertAllClose(g.hessian, expected, rtol=1e-6, atol=1e-6)

    def test_higher_rank_inputs_are_reshaped(self):
        """Tests that higher-rank inputs are reshaped correctly."""
        # x: [batch, time, feat]
        x = ops.array(np.random.randn(10, 4, 7), "float32")
        x_flat = ops.reshape(x, (-1, ops.shape(x)[-1]))

        layer1 = layers.Dense(5, use_bias=False)
        layer1.build((None, 7))
        g1 = GPTQ(layer1)
        g1.update_hessian_with_batch(x)

        layer2 = layers.Dense(5, use_bias=False)
        layer2.build((None, 7))
        g2 = GPTQ(layer2)
        g2.update_hessian_with_batch(x_flat)

        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_raises_on_feature_mismatch(self):
        x = ops.array(np.random.randn(8, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 6))  # wrong in_features
        g = GPTQ(layer)

        with self.assertRaisesRegex(ValueError, "do not match input features"):
            g.update_hessian_with_batch(x)

        with self.assertRaisesRegex(ValueError, "cannot be None"):
            g.update_hessian_with_batch(None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            g.update_hessian_with_batch(
                ops.array(np.empty((0, 7), dtype="float32"))
            )

    def test_num_samples_accumulates_correctly(self):
        """Tests that the number of samples is accumulated correctly when
        streaming updates are used."""
        x = ops.array(np.random.randn(64, 7), "float32")
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))
        g = GPTQ(layer)

        g.update_hessian_with_batch(x[:5])
        g.update_hessian_with_batch(x[5:30])
        g.update_hessian_with_batch(x[30:])

        self.assertEqual(g.num_samples, 64)

    def test_numeric_stability_large_values(self):
        """Tests numeric stability of hessian update with large input values."""
        x = ops.multiply(ops.array(np.random.randn(32, 7), "float32"), 1e6)
        layer = layers.Dense(5, use_bias=False)
        layer.build((None, 7))

        g = GPTQ(layer)
        g.update_hessian_with_batch(x)

        # Should be finite and symmetric
        self.assertTrue(ops.all(ops.isfinite(g.hessian)))
        self.assertTrue(ops.all(ops.equal(g.hessian, ops.transpose(g.hessian))))

    def test_einsumdense_2d_kernel_hessian_shape(self):
        x = layers.Input((7,))
        y = layers.EinsumDense("ab,bc->ac", output_shape=(5,))(x)
        model = keras.Model(x, y)
        einsum_dense_layer = next(
            l for l in model.layers if isinstance(l, layers.EinsumDense)
        )

        g = GPTQ(einsum_dense_layer)

        # should infer rows==7
        self.assertEqual(ops.shape(g.hessian), (7, 7))

    def test_einsumdense_3d_kernel_streaming_equals_big_batch(self):
        """Tests that streaming updates to the Hessian are equivalent to a big
        batch update."""
        # Construct a tiny attention-like einsum with 3D kernel
        x = layers.Input((7,))
        qkv = layers.EinsumDense("bf,fhk->bhk", output_shape=(2, 3))(
            x
        )  # heads=2, head_dim=3
        model = keras.Model(x, qkv)
        einsum_dense_layer = next(
            l for l in model.layers if isinstance(l, layers.EinsumDense)
        )

        x = ops.array(np.random.randn(50, 7), "float32")

        g1 = GPTQ(einsum_dense_layer)
        g1.update_hessian_with_batch(x)

        g2 = GPTQ(einsum_dense_layer)
        g2.update_hessian_with_batch(x[:20])
        g2.update_hessian_with_batch(x[20:])

        self.assertAllClose(g1.hessian, g2.hessian, rtol=1e-6, atol=1e-6)

    def test_identity_inv_hessian_matches_direct_quantization(self):
        """Tests that the matrix quantization without error correction
        matches the direct implementation."""
        in_features, out_features = 16, 8
        weights = ops.reshape(
            ops.linspace(
                -0.9, 1.1, in_features * out_features, dtype="float32"
            ),
            (in_features, out_features),
        )
        weights_transpose = ops.transpose(weights)

        # inverse_hessian = identity; no cross-feature correction
        # (since all off-diagonal elements are zero), which means
        # there is no interaction between different features
        inverse_hessian = ops.eye(in_features, dtype="float32")

        dequantized_weights = gptq_quantize_matrix(
            weights_transpose,
            inverse_hessian,
            blocksize=128,
            group_size=-1,
            activation_order=False,
            compute_scale_zero=_compute_scale_zero,
        )

        # Compare function output with columnwise direct application
        # of quantization.
        out = ops.zeros_like(weights_transpose)
        for j in range(ops.shape(weights_transpose)[1]):
            column = weights_transpose[:, j : j + 1]
            scale, zero, maxq = _compute_scale_zero(column)
            quantized_col = quantize(column, scale, zero, maxq)
            dequantized = dequantize(quantized_col, scale, zero)
            out = ops.slice_update(
                out, (0, j), ops.expand_dims(dequantized[:, 0], 1)
            )

        self.assertAllClose(dequantized_weights, out, atol=1e-6)

    def test_activation_order_permutation_is_undone(self):
        in_features, out_features = 8, 6
        layer = layers.Dense(out_features, use_bias=False)
        layer.build((None, in_features))
        weights = ops.array(
            np.random.randn(in_features, out_features), "float32"
        )
        layer.set_weights([weights])

        # generate a non-trivial order metric.
        diag = ops.linspace(10.0, 1.0, in_features, dtype="float32")
        diag = ops.random.shuffle(diag)
        H = ops.diag(diag)

        # Ensure it generates a non-trivial permutation
        perm = _stable_permutation(diag)
        self.assertFalse(ops.all(ops.equal(perm, ops.arange(in_features))))

        # Quantize with activation order
        g1 = GPTQ(
            layer,
            GPTQConfig(
                dataset=None,
                tokenizer=None,
                group_size=-1,
                activation_order=True,
            ),
        )
        g1.hessian = H
        g1.quantize_and_correct_block()

        # Quantize without activation order
        layer2 = layers.Dense(out_features, use_bias=False)
        layer2.build((None, in_features))
        layer2.set_weights([ops.copy(weights)])

        g2 = GPTQ(
            layer2,
            GPTQConfig(
                dataset=None,
                tokenizer=None,
                group_size=-1,
                activation_order=False,
            ),
        )
        g2.hessian = H
        g2.quantize_and_correct_block()

        # The weights should be identical since permutation is undone
        self.assertAllClose(layer.get_weights()[0], layer2.get_weights()[0])


def _compute_scale_zero(x, **_):
    # Per-column asymmetric int4 example
    # scale = (max-min)/maxq, zero = round(-min/scale)
    maxq = 15.0
    xmin = ops.min(x, axis=0, keepdims=True)
    xmax = ops.max(x, axis=0, keepdims=True)
    scale = ops.divide(ops.subtract(xmax, xmin), ops.add(maxq, 1e-8))
    zero = ops.round(ops.divide(ops.negative(xmin), ops.add(scale, 1e-8)))
    return scale, zero, maxq
