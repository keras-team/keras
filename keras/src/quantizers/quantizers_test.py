from keras.src import backend
from keras.src import ops
from keras.src import quantizers
from keras.src import random
from keras.src import testing


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

    def _TestOp(
        self,
        input_mins,
        input_maxs,
        num_bits,
        narrow_range,
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
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            @tf.function(jit_compile=True)
            def test_op(inputs, input_mins, input_maxs, num_bits, narrow_range):
                with tf.GradientTape() as tape:
                    tape.watch(inputs)
                    result = (
                        quantizers.fake_quant_with_min_max_vars_per_channel(
                            inputs,
                            input_mins,
                            input_maxs,
                            num_bits,
                            narrow_range,
                        )
                    )
                return initial_gradients * tape.gradient(result, inputs)

            gradients = test_op(
                inputs, input_mins, input_maxs, num_bits, narrow_range
            )
            # test gradients
            self.assertAllClose(gradients, expected_backprops_wrt_input)

        if backend.backend() == "torch":
            import torch

            def test_op(inputs, input_min, input_max, num_bits, narrow_range):
                # Create tensor and enable gradient tracking
                inputs = torch.tensor(
                    inputs, dtype=torch.float32, requires_grad=True
                )

                # Apply the quantization operation
                result = quantizers.fake_quant_with_min_max_vars_per_channel(
                    inputs, input_mins, input_maxs, num_bits, narrow_range
                )

                # Compute gradients
                result.backward(torch.ones_like(result))

                return initial_gradients * inputs.grad

            gradients = test_op(
                inputs, input_min, input_max, num_bits, narrow_range
            )
            # test gradients
            self.assertAllClose(gradients, expected_backprops_wrt_input)

        if backend.backend() == "jax":
            import jax

            def test_op(inputs, input_mins, input_maxs, num_bits, narrow_range):
                # Define the function to compute gradients for
                def quantize_fn(x):
                    return quantizers.fake_quant_with_min_max_vars_per_channel(
                        x, input_mins, input_maxs, num_bits, narrow_range
                    )

                # Get the gradient function
                grad_fn = jax.jit(jax.grad(lambda x: ops.sum(quantize_fn(x))))

                # Compute gradients
                input_gradients = grad_fn(inputs)

                return initial_gradients * input_gradients

            gradients = test_op(
                inputs, input_min, input_max, num_bits, narrow_range
            )
            # test gradients
            self.assertAllClose(gradients, expected_backprops_wrt_input)
        outputs = quantizers.fake_quant_with_min_max_vars_per_channel(
            inputs,
            input_min,
            input_max,
            num_bits=num_bits,
            narrow_range=narrow_range,
        )
        self.assertAllClose(outputs, expected)

    def test_fakeQuantWithMinMax_8BitsNoSclngNoNdgng(self):
        self._TestOp(
            [0.0],
            [255.0],
            8,
            False,
            [0.0],
            [255.0],
            [1.0],
        )

    def test_fakeQuantWithMinMax_8BitsSclngAndNdgngDown(self):
        self._TestOp(
            [0.5],
            [128.0],
            8,
            False,
            [0.0],
            [127.5],
            [0.5],
        )

    def test_fakeQuantWithMinMax_8BitsSclngAndNdgngUp(self):
        self._TestOp(
            [-128.0],
            [-0.5],
            8,
            False,
            [-127.5],
            [0.0],
            [0.5],
        )

    def test_fakeQuantWithMinMax_8BitsSclngAndNdgngBtwn(self):
        self._TestOp(
            [-0.1],
            [127.4],
            8,
            False,
            [0.0],
            [127.5],
            [0.5],
        )

    # 8 bits, narrow range.
    def test_fakeQuantWithMinMax_8BitsNrrwRangeNoSclngNoNdgng(self):
        self._TestOp(
            [0.0],
            [254.0],
            8,
            True,
            [0.0],
            [254.0],
            [1.0],
        )

    def test_fakeQuantWithMinMax_8BitsNrrwRangeSclngAndNdgngDown(self):
        self._TestOp(
            [0.1],
            [127.1],
            8,
            True,
            [0.0],
            [127.0],
            [0.5],
        )

    def test_fakeQuantWithMinMax_8BitsNrrwRangeSclngAndNdgngUp(self):
        self._TestOp(
            [-127.1],
            [-0.1],
            8,
            True,
            [-127.0],
            [0.0],
            [0.5],
        )

    def test_fakeQuantWithMinMax_8BitsNrrwRangeSclngAndNdgngBtwn(self):
        self._TestOp(
            [-0.1],
            [126.9],
            8,
            True,
            [0.0],
            [127.0],
            [0.5],
        )

    # 7 bits, wide range.
    def test_fakeQuantWithMinMax_7BitsNoSclngNoNdgng(self):
        self._TestOp(
            [0.0],
            [127.0],
            7,
            False,
            [0.0],
            [127.0],
            [1.0],
        )

    def test_fakeQuantWithMinMax_7BitsSclngAndNdgngDown(self):
        self._TestOp(
            [0.5],
            [64.0],
            7,
            False,
            [0.0],
            [63.5],
            [0.5],
        )

    def test_fakeQuantWithMinMax_7BitsSclngAndNdgngUp(self):
        self._TestOp(
            [-64.0],
            [-0.5],
            7,
            False,
            [-63.5],
            [0.0],
            [0.5],
        )

    def test_fakeQuantWithMinMax_7BitsSclngAndNdgngBtwn(self):
        self._TestOp(
            [-0.1],
            [63.4],
            7,
            False,
            [0.0],
            [63.5],
            [0.5],
        )

    # 7 bits, narrow range.
    def test_fakeQuantWithMinMax_7BitsNrrwRangeNoSclngNoNdgng(self):
        self._TestOp(
            [0.0],
            [126.0],
            7,
            True,
            [0.0],
            [126.0],
            [1.0],
        )

    def test_fakeQuantWithMinMax_7BitsNrrwRangeSclngAndNdgngDown(self):
        self._TestOp(
            [0.1],
            [63.1],
            7,
            True,
            [0.0],
            [63.0],
            [0.5],
        )

    def test_fakeQuantWithMinMax_7BitsNrrwRangeSclngAndNdgngUp(self):
        self._TestOp(
            [-63.1],
            [-0.1],
            7,
            True,
            [-63.0],
            [0.0],
            [0.5],
        )

    def test_fakeQuantWithMinMax_7BitsNrrwRangeSclngAndNdgngBtwn(self):
        self._TestOp(
            [-0.1],
            [62.9],
            7,
            True,
            [0.0],
            [63.0],
            [0.5],
        )

    # 8 bits, wide range.
    def test_fakeQuantWithMinMax_8Bits(self):
        self._TestOp(
            [0.0, 0.5, -128.0, -0.1],
            [255.0, 128.0, -0.5, 127.4],
            8,
            False,
            [0.0, 0.0, -127.5, 0.0],
            [255.0, 127.5, 0.0, 127.5],
            [1.0, 0.5, 0.5, 0.5],
        )

    # 8 bits, narrow range.
    def test_fakeQuantWithMinMax_8BitsNarrowRange(self):
        self._TestOp(
            [0.0, 0.1, -127.1, -0.1],
            [254.0, 127.1, -0.1, 126.9],
            8,
            True,
            [0.0, 0.0, -127.0, 0.0],
            [254.0, 127.0, 0.0, 127.0],
            [1.0, 0.5, 0.5, 0.5],
        )

    # 7 bits, wide range.
    def test_fakeQuantWithMinMax_7Bits(self):
        self._TestOp(
            [0.0, 0.5, -64.0, -0.1],
            [127.0, 64.0, -0.5, 63.4],
            7,
            False,
            [0.0, 0.0, -63.5, 0.0],
            [127.0, 63.5, 0.0, 63.5],
            [1.0, 0.5, 0.5, 0.5],
        )

    # 7 bits, narrow range.
    def test_fakeQuantWithMinMax_7BitsNarrowRange(self):
        self._TestOp(
            [0.0, 0.1, -63.1, -0.1],
            [126.0, 63.1, -0.1, 62.9],
            7,
            True,
            [0.0, 0.0, -63.0, 0.0],
            [126.0, 63.0, 0.0, 63.0],
            [1.0, 0.5, 0.5, 0.5],
        )
