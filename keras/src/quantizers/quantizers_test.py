import sys

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import quantizers
from keras.src import random
from keras.src import testing
from keras.src.quantizers.quantizers import compute_quantization_parameters
from keras.src.quantizers.quantizers import dequantize_with_sz_map
from keras.src.quantizers.quantizers import dequantize_with_zero_point
from keras.src.quantizers.quantizers import quantize_with_sz_map
from keras.src.quantizers.quantizers import quantize_with_zero_point
from keras.src.testing.test_utils import named_product


class QuantizersTest(testing.TestCase):
    def test_get_method(self):
        quantizer = quantizers.get("abs_max_quantizer", axis=-1)
        self.assertTrue(quantizer, quantizers.AbsMaxQuantizer)

        quantizer = quantizers.get(None)
        self.assertEqual(quantizer, None)

        with self.assertRaises(ValueError):
            quantizers.get("typo")

    def test_abs_max_quantizer(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1, dtype="float32")
        quantizer = quantizers.AbsMaxQuantizer(axis=-1)

        # Test quantizing
        quantized_values, scale = quantizer(values)
        self.assertDType(quantized_values, "int8")
        self.assertDType(scale, "float32")
        self.assertEqual(tuple(quantized_values.shape), (3, 4, 5))
        self.assertEqual(tuple(scale.shape), (3, 4, 1))
        self.assertLessEqual(ops.max(quantized_values), 127)
        self.assertGreaterEqual(ops.min(quantized_values), -127)

        # Test dequantizing
        dequantized_values = ops.divide(quantized_values, scale)
        rmse = ops.sqrt(
            ops.mean(ops.square(ops.subtract(values, dequantized_values)))
        )
        self.assertLess(rmse, 1e-1)  # loose assertion

        # Test serialization
        self.run_class_serialization_test(quantizer)

        # Test bfloat16 & float16 dtype
        values = random.uniform(
            [3, 4, 5], minval=-1, maxval=1, dtype="bfloat16"
        )
        quantized_values, scale = quantizer(values)
        self.assertDType(quantized_values, "int8")
        self.assertDType(scale, "bfloat16")
        values = random.uniform([3, 4, 5], minval=-1, maxval=1, dtype="float16")
        quantized_values, scale = quantizer(values)
        self.assertDType(quantized_values, "int8")
        self.assertDType(scale, "float16")

    def test_abs_max_quantizer_to_numpy(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1, dtype="float32")
        quantized_values, scale = quantizers.abs_max_quantize(
            values, axis=-1, to_numpy=True
        )
        ref_quantized_values, ref_scale = quantizers.abs_max_quantize(
            values, axis=-1
        )
        self.assertAllClose(quantized_values, ref_quantized_values)
        self.assertAllClose(scale, ref_scale)

    def test_compute_float8_scale(self):
        amax = 3.0
        scale = 4.0
        dtype_max = 448.0  # float8_e4m3fn
        # The algorithm for computing the new scale is sourced from
        # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.update_fp8_metas
        expected_scale = 1.0 / (dtype_max / amax) / (2**0)

        computed_scale = quantizers.compute_float8_scale(amax, scale, dtype_max)
        self.assertAllClose(computed_scale, expected_scale)

    def test_compute_float8_amax_history(self):
        values = random.uniform([3, 4, 5], minval=-1, maxval=1)
        amax_history = random.uniform([123])
        amax_from_values = ops.max(ops.abs(values))

        computed_amax_history = quantizers.compute_float8_amax_history(
            values, amax_history
        )
        self.assertAllClose(computed_amax_history[0], amax_from_values)
        # Shift to left with 1 step
        self.assertAllClose(
            computed_amax_history[1:], ops.roll(amax_history, -1)[1:]
        )

    def test_quantize_and_dequantize(self):
        scale = 1.0 / 100.0
        values = random.uniform([3, 4, 5], minval=-1, maxval=1)
        qdq_values = quantizers.quantize_and_dequantize(
            values, scale, "float8_e4m3fn", "float32"
        )
        # A loose assertion due to an expected quantization error
        self.assertAllClose(qdq_values, values, atol=1e-1)

        qdq_values = quantizers.quantize_and_dequantize(
            values, scale, "float8_e5m2", "float32"
        )
        # A loose assertion due to an expected quantization error
        self.assertAllClose(qdq_values, values, atol=5e-1)

    SHAPE_AXIS_SCENARIOS = [
        # 1. 2D Tensors
        # Covers the unpack fast path (rank=2, axis=0) for both parities
        {"testcase_name": "2d_axis0_odd", "shape": (5, 8), "axis": 0},
        {"testcase_name": "2d_axis0_even", "shape": (4, 8), "axis": 0},
        # Covers the general path and a negative axis for 2D tensors
        {"testcase_name": "2d_axis1_odd", "shape": (8, 7), "axis": 1},
        {"testcase_name": "2d_axis_neg1_even", "shape": (8, 6), "axis": -1},
        # 2. Higher-Rank Tensors
        # Covers a middle axis for a complex shape with both parities
        {"testcase_name": "4d_axis1_odd", "shape": (2, 5, 4, 6), "axis": 1},
        {"testcase_name": "4d_axis2_even", "shape": (2, 4, 8, 6), "axis": 2},
        # Covers the last axis of a complex shape with a negative index
        {
            "testcase_name": "4d_axis_neg1_odd",
            "shape": (2, 4, 6, 7),
            "axis": -1,
        },
    ]

    DTYPE_PARAMS = [
        {"testcase_name": "int8", "dtype": "int8", "minval": -8, "maxval": 8},
        {"testcase_name": "uint8", "dtype": "uint8", "minval": 0, "maxval": 16},
    ]

    @parameterized.named_parameters(
        named_product(SHAPE_AXIS_SCENARIOS, DTYPE_PARAMS)
    )
    def test_pack_unpack_int4(self, shape, axis, dtype, minval, maxval):
        # Create a random tensor with int4 values in the specified range and
        # dtype
        arr = ops.cast(
            ops.floor(random.uniform(shape, minval=minval, maxval=maxval)),
            dtype,
        )

        # Pack the tensor using the specified dtype
        packed, packed_shape, orig_len = quantizers.pack_int4(
            arr, axis=axis, dtype=dtype
        )

        # Unpack the tensor using the specified dtype
        unpacked = quantizers.unpack_int4(
            packed, orig_len, axis=axis, dtype=dtype
        )

        # Verify that the packed tensor has the correct dtype
        self.assertDType(packed, dtype)

        # Verify that the unpacked tensor has the correct dtype
        self.assertDType(unpacked, dtype)

        # The unpacked tensor should be the same as the original tensor
        self.assertAllClose(unpacked, arr)

        # Test the packed shape
        expected_packed_shape = list(shape)
        expected_packed_shape[axis] = (expected_packed_shape[axis] + 1) // 2
        self.assertEqual(
            list(ops.convert_to_numpy(packed_shape)), expected_packed_shape
        )

    @parameterized.named_parameters(
        ("per_tensor", None),
        ("per_channel", -1),
    )
    def test_fake_quant_with_min_max_vars_symbolic(self, axis):
        x = backend.KerasTensor((2, 3, 4))
        y = quantizers.fake_quant_with_min_max_vars(x, -3.0, 3.0, axis=axis)

        self.assertIsInstance(y, backend.KerasTensor)
        self.assertEqual(y.shape, (2, 3, 4))

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "wide_8bits_input_mins_0.0_input_maxs_255.0",
                "narrow_range": False,
                "input_mins": [0.0],
                "input_maxs": [255.0],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [255.0],
                "expected_steps": [1.0],
                "axis": None,
            },
            {
                "testcase_name": "wide_8bits_input_mins_0.5_input_maxs_128.0",
                "narrow_range": False,
                "input_mins": [0.5],
                "input_maxs": [128.0],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [127.5],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "wide_8bits_input_mins_-128.0_input_maxs_-0.5",
                "narrow_range": False,
                "input_mins": [-128.0],
                "input_maxs": [-0.5],
                "num_bits": 8,
                "expected_nudged_input_mins": [-127.5],
                "expected_nudged_input_maxs": [0.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "wide_8bits_input_mins_-0.1_input_maxs_127.4",
                "narrow_range": False,
                "input_mins": [-0.1],
                "input_maxs": [127.4],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [127.5],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "narrow_8bits_input_mins_0.0_input_maxs_254.0",
                "narrow_range": True,
                "input_mins": [0.0],
                "input_maxs": [254.0],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [254.0],
                "expected_steps": [1.0],
                "axis": None,
            },
            {
                "testcase_name": "narrow_8bits_input_mins_0.1_input_maxs_127.1",
                "narrow_range": True,
                "input_mins": [0.1],
                "input_maxs": [127.1],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [127.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": (
                    "narrow_8bits_input_mins_-127.1_input_maxs_-0.1"
                ),
                "narrow_range": True,
                "input_mins": [-127.1],
                "input_maxs": [-0.1],
                "num_bits": 8,
                "expected_nudged_input_mins": [-127.0],
                "expected_nudged_input_maxs": [0.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": (
                    "narrow_8bits_input_mins_-0.1_input_maxs_126.9"
                ),
                "narrow_range": True,
                "input_mins": [-0.1],
                "input_maxs": [126.9],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [127.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "wide_7bits_input_mins_0.0_input_maxs_127.0",
                "narrow_range": False,
                "input_mins": [0.0],
                "input_maxs": [127.0],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [127.0],
                "expected_steps": [1.0],
                "axis": None,
            },
            {
                "testcase_name": "wide_7bits_input_mins_0.5_input_maxs_64.0",
                "narrow_range": False,
                "input_mins": [0.5],
                "input_maxs": [64.0],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [63.5],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "wide_7bits_input_mins_-64.0_input_maxs_-0.5",
                "narrow_range": False,
                "input_mins": [-64.0],
                "input_maxs": [-0.5],
                "num_bits": 7,
                "expected_nudged_input_mins": [-63.5],
                "expected_nudged_input_maxs": [0.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "wide_7bits_input_mins_-0.1_input_maxs_63.4",
                "narrow_range": False,
                "input_mins": [-0.1],
                "input_maxs": [63.4],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [63.5],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "narrow_7bits_input_mins_0.0_input_maxs_126.0",
                "narrow_range": True,
                "input_mins": [0.0],
                "input_maxs": [126.0],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [126.0],
                "expected_steps": [1.0],
                "axis": None,
            },
            {
                "testcase_name": "narrow_7bits_input_mins_0.1_input_maxs_63.1",
                "narrow_range": True,
                "input_mins": [0.1],
                "input_maxs": [63.1],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [63.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": (
                    "narrow_7bits_input_mins_-63.1_input_maxs_-0.1"
                ),
                "narrow_range": True,
                "input_mins": [-63.1],
                "input_maxs": [-0.1],
                "num_bits": 7,
                "expected_nudged_input_mins": [-63.0],
                "expected_nudged_input_maxs": [0.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "narrow_7bits_input_mins_-0.1_input_maxs_62.9",
                "narrow_range": True,
                "input_mins": [-0.1],
                "input_maxs": [62.9],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0],
                "expected_nudged_input_maxs": [63.0],
                "expected_steps": [0.5],
                "axis": None,
            },
            {
                "testcase_name": "wide_8bits_multi_channel",
                "narrow_range": False,
                "input_mins": [0.0, 0.5, -128.0, -0.1],
                "input_maxs": [255.0, 128.0, -0.5, 127.4],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0, 0.0, -127.5, 0.0],
                "expected_nudged_input_maxs": [255.0, 127.5, 0.0, 127.5],
                "expected_steps": [1.0, 0.5, 0.5, 0.5],
                "axis": 1,
            },
            {
                "testcase_name": "narrow_8bits_multi_channel",
                "narrow_range": True,
                "input_mins": [0.0, 0.1, -127.1, -0.1],
                "input_maxs": [254.0, 127.1, -0.1, 126.9],
                "num_bits": 8,
                "expected_nudged_input_mins": [0.0, 0.0, -127.0, 0.0],
                "expected_nudged_input_maxs": [254.0, 127.0, 0.0, 127.0],
                "expected_steps": [1.0, 0.5, 0.5, 0.5],
                "axis": 1,
            },
            {
                "testcase_name": "wide_7bits_multi_channel",
                "narrow_range": False,
                "input_mins": [0.0, 0.5, -64.0, -0.1],
                "input_maxs": [127.0, 64.0, -0.5, 63.4],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0, 0.0, -63.5, 0.0],
                "expected_nudged_input_maxs": [127.0, 63.5, 0.0, 63.5],
                "expected_steps": [1.0, 0.5, 0.5, 0.5],
                "axis": 1,
            },
            {
                "testcase_name": "narrow_7bits_multi_channel",
                "narrow_range": True,
                "input_mins": [0.0, 0.1, -63.1, -0.1],
                "input_maxs": [126.0, 63.1, -0.1, 62.9],
                "num_bits": 7,
                "expected_nudged_input_mins": [0.0, 0.0, -63.0, 0.0],
                "expected_nudged_input_maxs": [126.0, 63.0, 0.0, 63.0],
                "expected_steps": [1.0, 0.5, 0.5, 0.5],
                "axis": 1,
            },
        ]
    )
    @pytest.mark.skipif(
        backend.backend() not in ("tensorflow", "jax", "torch"),
        reason=f"{backend.backend()} doesn't support `custom_gradient`.",
    )
    def test_fake_quant_with_min_max_vars(
        self,
        input_mins,
        input_maxs,
        num_bits,
        narrow_range,
        axis,
        expected_nudged_input_mins,
        expected_nudged_input_maxs,
        expected_steps,
    ):
        num_channels = len(input_mins)
        inputs_list = []
        expected_list = []
        initial_gradients_list = []
        expected_backprops_wrt_input_list = []
        for i in range(num_channels):
            expected_nudged_input_min = expected_nudged_input_mins[i]
            expected_nudged_input_max = expected_nudged_input_maxs[i]
            expected_step = expected_steps[i]

            inputs_list.append(
                [
                    expected_nudged_input_min - expected_step,
                    expected_nudged_input_min - 0.01,
                    expected_nudged_input_min,
                    expected_nudged_input_min + 0.01,
                    expected_nudged_input_min + expected_step - 0.01,
                    expected_nudged_input_min + expected_step,
                    expected_nudged_input_min + expected_step + 0.01,
                    expected_nudged_input_max - 0.01,
                    expected_nudged_input_max,
                    expected_nudged_input_max + 0.01,
                    expected_nudged_input_max + expected_step,
                ]
            )
            expected_list.append(
                [
                    expected_nudged_input_min,
                    expected_nudged_input_min,
                    expected_nudged_input_min,
                    expected_nudged_input_min,
                    expected_nudged_input_min + expected_step,
                    expected_nudged_input_min + expected_step,
                    expected_nudged_input_min + expected_step,
                    expected_nudged_input_max,
                    expected_nudged_input_max,
                    expected_nudged_input_max,
                    expected_nudged_input_max,
                ]
            )
            initial_gradients_list.append(
                list(range(1, len(inputs_list[-1]) + 1))
            )
            expected_backprops_wrt_input_list.append(
                [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0]
            )
        inputs = ops.transpose(ops.array(inputs_list, dtype="float32"))
        expected = ops.transpose(ops.array(expected_list, dtype="float32"))
        expected_backprops_wrt_input = ops.transpose(
            ops.array(expected_backprops_wrt_input_list, dtype="float32")
        )
        input_min = ops.array(input_mins, dtype="float32")
        input_max = ops.array(input_maxs, dtype="float32")
        initial_gradients = ops.transpose(
            ops.array(initial_gradients_list, dtype="float32")
        )

        # Test gradients.
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            @tf.function(jit_compile=True)
            def test_op(
                inputs, input_mins, input_maxs, num_bits, narrow_range, axis
            ):
                with tf.GradientTape() as tape:
                    tape.watch(inputs)
                    result = quantizers.fake_quant_with_min_max_vars(
                        inputs,
                        input_mins,
                        input_maxs,
                        num_bits,
                        narrow_range,
                        axis,
                    )
                return initial_gradients * tape.gradient(result, inputs)

        if backend.backend() == "torch":
            import torch

            def test_op(
                inputs, input_mins, input_maxs, num_bits, narrow_range, axis
            ):
                # Create tensor and enable gradient tracking
                inputs = torch.tensor(
                    inputs, dtype=torch.float32, requires_grad=True
                )

                # Apply the quantization operation
                result = quantizers.fake_quant_with_min_max_vars(
                    inputs, input_mins, input_maxs, num_bits, narrow_range, axis
                )

                # Compute gradients
                result.backward(torch.ones_like(result))

                return initial_gradients * inputs.grad

        if backend.backend() == "jax":
            import jax

            def test_op(
                inputs, input_mins, input_maxs, num_bits, narrow_range, axis
            ):
                # Define the function to compute gradients for
                def quantize_fn(x):
                    return quantizers.fake_quant_with_min_max_vars(
                        x, input_mins, input_maxs, num_bits, narrow_range, axis
                    )

                _, f_vjp = jax.vjp(quantize_fn, inputs)

                if getattr(jax.config, "jax_vjp3", False):
                    input_gradients = f_vjp.opaque_residuals[0]
                elif sys.version_info >= (3, 10):
                    input_gradients = f_vjp.args[0].args[0][0]
                else:
                    input_gradients = f_vjp.args[0].args[0][1]

                return ops.multiply(initial_gradients, input_gradients)

        gradients = test_op(
            inputs, input_min, input_max, num_bits, narrow_range, axis
        )
        if backend.backend() != "jax" or not testing.jax_uses_gpu():
            # JAX GPU produces less precise numbers, causing the CI to fail.
            # For example, 127.5 / 255.0 results in 0.49999997 instead of 0.5.
            self.assertAllClose(gradients, expected_backprops_wrt_input)

        # Test outputs.
        outputs = quantizers.fake_quant_with_min_max_vars(
            inputs,
            input_min,
            input_max,
            num_bits=num_bits,
            narrow_range=narrow_range,
            axis=axis,
        )
        self.assertAllClose(outputs, expected)

        # Test bfloat16 & float16 dtype
        outputs = quantizers.fake_quant_with_min_max_vars(
            ops.cast(inputs, "bfloat16"),
            input_min,
            input_max,
            num_bits=num_bits,
            narrow_range=narrow_range,
            axis=axis,
        )
        self.assertDType(outputs, "bfloat16")
        self.assertAllClose(outputs, expected)

        outputs = quantizers.fake_quant_with_min_max_vars(
            ops.cast(inputs, "float16"),
            input_min,
            input_max,
            num_bits=num_bits,
            narrow_range=narrow_range,
            axis=axis,
        )
        self.assertDType(outputs, "float16")
        self.assertAllClose(outputs, expected)


class GPTQQuantizerTest(testing.TestCase):
    @parameterized.named_parameters(
        ("bits_2_sym_False", 2, False),
        ("bits_4_sym_False", 4, False),
        ("bits_8_sym_False", 8, False),
        ("bits_2_sym_True", 2, True),
        ("bits_4_sym_True", 4, True),
        ("bits_8_sym_True", 8, True),
    )
    def test_quantize_dequantize_roundtrip_error_bound_per_tensor(
        self, bits, symmetric
    ):
        """
        For finite inputs and positive scales, the reconstruction error
        |x_hat - clip(x)| is bounded by 0.5 * scale elementwise.
        """
        rng = np.random.default_rng(0)
        x = ops.array(rng.standard_normal((64, 32)), "float32")
        scale = ops.array(0.05)  # per-tensor scale
        maxq = ops.array(ops.subtract(ops.power(2, bits), 1), "float32")
        zero = ops.array(maxq / 2.0 if symmetric else 3.0, "float32")

        quantized = quantize_with_zero_point(x, scale, zero, maxq)
        dequantized = dequantize_with_zero_point(quantized, scale, zero)

        # Representable dequantization range:
        # [scale*(0 - zero), scale*(maxq - zero)]
        lo = ops.multiply(scale, ops.subtract(ops.array(0.0), zero))
        hi = ops.multiply(scale, ops.subtract(maxq, zero))
        x_clipped = ops.clip(x, lo, hi)

        err = ops.abs(dequantized - x_clipped)
        self.assertTrue(
            ops.all(err <= (ops.add(ops.multiply(0.5, scale), 1e-7)))
        )

    def test_quantize_clipping_behavior_extremes(self):
        """
        Very negative q == 0 ; very positive q == maxq.
        """
        maxq = ops.array(15.0)
        scale = ops.array(0.1)
        zero = ops.array(7.0)

        x = ops.array([[-1e6, 1e6]], "float32")
        quantized = quantize_with_zero_point(x, scale, zero, maxq)

        self.assertEqual(quantized.shape, (1, 2))
        self.assertEqual(quantized[0, 0], 0.0)
        self.assertEqual(quantized[0, 1], maxq)

    def test_zero_scale_guard_no_nans_for_finite_inputs(self):
        """
        If scale == 0, quantize should not produce NaNs (uses epsilon
        replacement).
        """
        x = ops.array([[0.0, 1.0, -2.0]])
        scale = ops.array(0.0)  # triggers epsilon path
        zero = ops.array(5.0)
        maxq = ops.array(15.0)

        q = quantize_with_zero_point(x, scale, zero, maxq)
        self.assertFalse(ops.any(ops.isnan(q)))

        # Dequantize should also be finite
        x_hat = dequantize_with_zero_point(q, scale, zero)
        self.assertTrue(ops.all(ops.isfinite(x_hat)))

    @parameterized.parameters(4, 8)
    def test_idempotent_quantize_when_input_is_already_levels(self, bits):
        """
        If input is already exactly on representable dequantized grid,
        quantize→dequantize should return the same values (within float eps).
        """
        scale = ops.array(0.125)
        maxq = ops.array(ops.subtract(ops.power(2, bits), 1), "float32")
        zero = ops.array(ops.divide(maxq, 2.0))

        # Build dequantized grid points: x = scale * (k - zero), k in [0..maxq]
        ks = ops.arange(0, ops.add(maxq, 1))
        x_vals = ops.multiply(scale, ops.subtract(ks, zero))
        x = ops.reshape(x_vals, (1, -1))

        q = quantize_with_zero_point(x, scale, zero, maxq)
        x_hat = dequantize_with_zero_point(q, scale, zero)
        self.assertAllClose(x_hat, x, rtol=0, atol=1e-6)


class ComputeScaleZeroTest(testing.TestCase):
    def test_error_when_x_is_none(self):
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            compute_quantization_parameters(None, bits=4)

    def test_error_when_x_is_empty(self):
        x = ops.array([], "float32")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            compute_quantization_parameters(x, bits=4)

    def test_error_when_weight_rank_too_low(self):
        x = ops.array([1.0, 2.0], "float32")  # rank-1
        with self.assertRaisesRegex(ValueError, "rank of at least 2"):
            compute_quantization_parameters(x, bits=4, weight=True)

    @parameterized.named_parameters(
        ("bits2_asym", 2, False),
        ("bits4_asym", 4, False),
        ("bits8_asym", 8, False),
        ("bits2_sym", 2, True),
        ("bits4_sym", 4, True),
        ("bits8_sym", 8, True),
    )
    def test_per_tensor_shapes_and_basic_invariants(self, bits, symmetric):
        """Test per-tensor shapes and basic invariants."""
        x = ops.array(
            np.random.default_rng(0).standard_normal((7, 5), dtype="float32")
        )
        scale, zero, maxq = compute_quantization_parameters(
            x, bits=bits, symmetric=symmetric, per_channel=False, weight=False
        )

        # Shapes (per-tensor): (1,) for scale/zero
        self.assertEqual(scale.shape, (1,))
        self.assertEqual(zero.shape, (1,))

        # Scale must be strictly positive
        self.assertTrue(ops.all(scale > 0.0))

        if symmetric:
            # zero should be (maxq + 1)/2 for symmetric
            expected_zero = ops.divide(ops.add(maxq, 1.0), 2.0)
            self.assertAllClose(zero, expected_zero)
        else:
            # Asymmetric: zero ~ round(-min/scale) on the flattened input
            flat = ops.reshape(x, (1, -1))
            min_val = ops.min(flat, axis=1)
            expected_zero = ops.round(ops.divide(ops.negative(min_val), scale))
            self.assertAllClose(zero, expected_zero)

    def test_per_tensor_symmetric_on_constant_input_uses_safe_range(self):
        """Ensures safe range adjustment if entries are equal"""
        x = ops.array(np.full((3, 4), 0.0, dtype=np.float32))
        scale, zero, maxq = compute_quantization_parameters(
            x, bits=4, symmetric=True, per_channel=False, weight=False
        )
        # With symmetric=True and constant input, zero = (maxq+1)/2
        self.assertAllClose(zero, ops.array((float(maxq) + 1.0) / 2.0))
        self.assertTrue(ops.all(ops.greater(scale, 0.0)))

    def test_weight_per_tensor_tiles_rows(self):
        """Tests that scales/zeros tensors are properly tiled when
        per-channel quantization is not used."""
        x = ops.array(
            np.random.default_rng(1).standard_normal((8, 16)), "float32"
        )
        scale, zero, _ = compute_quantization_parameters(
            x, bits=4, symmetric=False, per_channel=False, weight=True
        )
        # When weight=True and per_channel=False, shapes are (rows, 1)
        self.assertEqual(scale.shape, (8, 1))
        self.assertEqual(zero.shape, (8, 1))

        # All elements in the scale and zero tensors must be equal due to
        # tiling.
        self.assertTrue(ops.all(scale == scale[0, 0]))
        self.assertTrue(ops.all(zero == zero[0, 0]))

    def test_weight_per_channel_ungrouped_shapes(self):
        """Tests that scales/zeros tensors have the correct shape when
        per-channel quantization is used without grouping."""
        x = ops.array(
            np.random.default_rng(2).standard_normal((6, 10)), "float32"
        )
        scale, zero, _ = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=-1,
            weight=True,
        )
        # Per-channel (ungrouped): one scale per output row -> (rows, 1)
        self.assertEqual(scale.shape, (6, 1))
        self.assertEqual(zero.shape, (6, 1))
        self.assertTrue(ops.all(ops.greater(scale, 0.0)))

        # Each channel should have roughly unique scales and zeros
        self.assertFalse(ops.all(scale == scale[0, 0]))
        self.assertFalse(ops.all(zero == zero[0, 0]))

    def test_weight_per_channel_grouped_shapes_and_count(self):
        """Tests that scales/zeros have the correct shape and count when
        per-channel quantization is used with grouping."""
        rows, cols, groups = 8, 16, 4
        x = ops.array(
            np.random.default_rng(3).standard_normal((rows, cols)), "float32"
        )
        scale, zero, _ = compute_quantization_parameters(
            x,
            bits=4,
            symmetric=False,
            per_channel=True,
            group_size=groups,
            weight=True,
        )
        # Grouped path reshapes to [-1, group_size]
        # number of groups = rows*cols / groups
        num_groups = (rows * cols) // groups
        self.assertEqual(scale.shape, (num_groups, 1))
        self.assertEqual(zero.shape, (num_groups, 1))
        self.assertTrue(ops.all(ops.greater(scale, 0.0)))

    @parameterized.named_parameters(
        ("sym_true", True),
        ("sym_false", False),
    )
    def test_dtype_and_finiteness(self, symmetric):
        x = ops.array(
            np.random.default_rng(4).standard_normal((5, 7)).astype("float32")
        )
        scale, zero, maxq = compute_quantization_parameters(
            x,
            bits=8,
            symmetric=symmetric,
            per_channel=True,
            group_size=-1,
            weight=True,
        )
        # All outputs should be all finite
        self.assertTrue(ops.all(ops.isfinite(scale)))
        self.assertTrue(ops.all(ops.isfinite(zero)))
        self.assertTrue(ops.all(ops.isfinite(maxq)))

    def test_dequantize_with_sz_map_logic(self):
        """Validates the vectorized dequantization logic against a
        manual implementation."""
        out_features, in_features, group_size = 4, 16, 4
        n_groups = in_features // group_size

        # Create dummy quantized weights
        q_weights = ops.cast(
            ops.array(
                np.random.randint(0, 15, size=(out_features, in_features))
            ),
            "uint8",
        )

        # Create dummy scales and zeros
        scale = ops.abs(
            ops.array(
                np.random.random((out_features, n_groups)).astype("float32")
            )
        )
        zero = ops.cast(
            ops.array(np.random.randint(0, 15, size=(out_features, n_groups))),
            "uint8",
        )

        # Create group index mapping
        g_idx = ops.array(np.arange(in_features) // group_size, dtype="int32")

        # Get the result from the function under test
        dequantized_result = dequantize_with_sz_map(
            q_weights, scale, zero, g_idx
        )

        # Manually compute the expected result
        expected_dequantized = np.zeros(
            (out_features, in_features), dtype="float32"
        )

        for i in range(out_features):
            for j in range(in_features):
                group = g_idx[j]
                s = scale[i, group]
                z = zero[i, group]
                # Dequantization formula: (q_val - z) * s
                expected_dequantized[i, j] = ops.multiply(
                    ops.subtract(q_weights[i, j], ops.cast(z, "float32")), s
                )

        self.assertAllClose(dequantized_result, expected_dequantized)

    def test_quantize_with_sz_map_logic(self):
        """Validates the vectorized quantization logic against a
        manual implementation."""
        out_features, in_features, group_size = 4, 16, 4
        n_groups = in_features // group_size

        # Create dummy float weights
        weights = ops.array(
            np.random.default_rng(5).standard_normal(
                (out_features, in_features)
            ),
            "float32",
        )

        # Create dummy scales and zeros
        scale = ops.abs(
            ops.array(
                np.random.random((out_features, n_groups)).astype("float32")
            )
        )
        zero = ops.cast(
            ops.array(np.random.randint(0, 15, size=(out_features, n_groups))),
            "uint8",
        )

        maxq = ops.array(15.0)

        # Create group index mapping
        g_idx = ops.array(np.arange(in_features) // group_size, dtype="int32")

        # Get the result from the function under test
        quantized_result = quantize_with_sz_map(
            weights, scale, zero, g_idx, maxq
        )

        # Manually compute the expected result
        expected_quantized = np.zeros(
            (out_features, in_features), dtype="uint8"
        )

        for i in range(out_features):
            for j in range(in_features):
                group = g_idx[j]
                s = scale[i, group]
                z = zero[i, group]
                # Quantization formula: clip(round(x/s + z), 0, maxq)
                q_val = ops.round(ops.add(ops.divide(weights[i, j], s), z))
                q_val_clipped = ops.clip(q_val, 0.0, maxq)
                expected_quantized[i, j] = ops.cast(q_val_clipped, "uint8")

        self.assertAllClose(quantized_result, expected_quantized)
