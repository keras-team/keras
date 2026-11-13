"""The calibration modes' shared chassis, tested once per mode."""

import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.quantization_test_utils import assert_calibrated
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.testing.test_utils import named_product

MODES = ["gptq", "awq"]


def _config(mode, **kwargs):
    if mode == "awq":
        kwargs.setdefault("num_grid_points", 5)
        kwargs.setdefault("apply_clip", False)
    return calibration_config(mode, **kwargs)


def _calibrate(layer, mode, x, **kwargs):
    """Calibrates the float `layer` with `mode` on `x`.

    The calibrator's solve swaps the layer for the quantized one. Returns
    the calibrator.
    """
    return calibrate_layer(layer, _config(mode, **kwargs), x)


def _float_dense(rng):
    """A built float `Dense(16)` on 12 features, and 32 rows of inputs."""
    layer = layers.Dense(16)
    layer.build((None, 12))
    layer.kernel.assign(rng.standard_normal(layer.kernel.shape) * 0.1)
    return layer, rng.standard_normal((32, 12)).astype("float32")


class CalibrationSwapTest(testing.TestCase):
    """A calibration mode quantizes a float layer in one step."""

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_calibrated_layer_holds_no_config(self, mode):
        # The policy names the bit width and the group size; the run's
        # config, with its dataset, stays with the caller.
        rng = np.random.default_rng(0)
        layer, x = _float_dense(rng)
        weights = [w.path for w in layer.weights]
        _calibrate(layer, mode, x, weight_bits=4, group_size=4)
        self.assertEqual(layer.dtype_policy.name, f"{mode}/4/4_from_float32")
        assert_calibrated(self, layer, mode)
        self.assertIsNone(layer.get_config()["quantization_config"])
        # The bias keeps its place; the codes and their parameters follow.
        self.assertEqual(layer.weights[0].path, weights[1])
        self.assertLen(layer.weights, 5 if mode == "gptq" else 6)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_config_from_a_file_is_dropped(self, mode):
        # A config deserialized with a calibrated layer agrees with the
        # policy name, or the build refuses it. The layer drops it: the
        # variables follow the policy, and the layer serializes no config
        # again.
        rng = np.random.default_rng(0)
        layer, x = _float_dense(rng)
        _calibrate(layer, mode, x, weight_bits=4, group_size=4)
        layer_config = layer.get_config()
        layer_config["quantization_config"] = saving.serialize_keras_object(
            _config(mode, weight_bits=4, group_size=4)
        )
        loaded = layers.Dense.from_config(layer_config)
        loaded.build((None, 12))
        self.assertIsNone(loaded.quantization_config)
        self.assertIsNone(loaded.get_config()["quantization_config"])
        self.assertEqual(loaded.kernel_scale.shape, layer.kernel_scale.shape)
        loaded.set_weights(layer.get_weights())
        self.assertAllEqual(loaded(x), layer(x))

    @parameterized.named_parameters(
        named_product(mode=MODES, kind=["dense", "einsum"])
    )
    def test_config_is_dropped_at_build(self, mode, kind):
        # Under `<mode>/4/8`, a layer lays out 4-bit codes in groups of 8
        # rows and drops the config it was constructed with. The config
        # agrees with the name, or the build refuses it.
        config = _config(mode, weight_bits=4, group_size=8)
        dtype = f"{mode}/4/8_from_float32"
        if kind == "dense":
            layer = layers.Dense(8, dtype=dtype, quantization_config=config)
        else:
            layer = layers.EinsumDense(
                "ab,bc->ac",
                output_shape=(8,),
                dtype=dtype,
                quantization_config=config,
            )
        layer.build((None, 16))
        self.assertIsNone(layer.quantization_config)
        self.assertEqual(tuple(layer.quantized_kernel.shape), (16, 4))
        self.assertEqual(tuple(layer.kernel_scale.shape), (2, 8))
        self.assertEqual(layer._quantized_weight().scheme.code_range, (0, 15))
