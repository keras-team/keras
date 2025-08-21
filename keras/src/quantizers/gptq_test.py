import numpy as np
import pytest

from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.gptq import GPTQ
from keras.src.quantizers.gptq_quant import GPTQQuantization


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

        gptq_instance = GPTQ(mock_layer)
        gptq_instance.quantizer = GPTQQuantization(
            weight_bits=4, symmetric=False
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
