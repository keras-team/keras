import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import quantizers
from keras.src import testing
from keras.src.quantizers.fake_quant import FakeQuantWithMinMaxVars


class FakeQuantTest(testing.TestCase):
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
        ("num_bits", {"num_bits": 2}, -1.0, 1.0),
        ("narrow_range", {"narrow_range": True}, -1.0, 1.0),
        (
            "per_channel",
            {"num_bits": 3, "narrow_range": True, "axis": -1},
            [-1.0, -0.5, -0.25, -0.75],
            [1.0, 0.5, 0.75, 0.25],
        ),
    )
    def test_fake_quant_with_min_max_vars_functional_model(
        self, kwargs, min_vals, max_vals
    ):
        # The symbolic call keeps `num_bits`, `narrow_range` and `axis`.
        min_vals = np.array(min_vals, dtype="float32")
        max_vals = np.array(max_vals, dtype="float32")
        x = np.linspace(-1.0, 1.0, 24, dtype="float32").reshape(6, 4)
        expected = quantizers.fake_quant_with_min_max_vars(
            x, min_vals, max_vals, **kwargs
        )

        inputs = layers.Input((4,))
        outputs = quantizers.fake_quant_with_min_max_vars(
            inputs, min_vals, max_vals, **kwargs
        )
        model = models.Model(inputs, outputs)
        restored = models.Model.from_config(model.get_config())

        for m in (model, restored):
            (op,) = [
                o
                for o in m.operations
                if isinstance(o, FakeQuantWithMinMaxVars)
            ]
            self.assertEqual(op.num_bits, kwargs.get("num_bits", 8))
            self.assertEqual(op.narrow_range, kwargs.get("narrow_range", False))
            # The symbolic call stores the axis canonicalized.
            axis = kwargs.get("axis")
            self.assertEqual(op.axis, None if axis is None else axis % x.ndim)
            self.assertAllClose(m(x), expected)

    def test_fake_quant_with_min_max_vars_symbolic_invalid_axis(self):
        # An invalid axis is refused while the graph is built.
        x = backend.KerasTensor((2, 3, 4))
        with self.assertRaisesRegex(ValueError, "axis"):
            quantizers.fake_quant_with_min_max_vars(x, -3.0, 3.0, axis=3)

    def test_fake_quant_with_min_max_vars_symbolic_range(self):
        # A symbolic range makes the call symbolic too.
        x = np.zeros((2, 3, 4), dtype="float32")
        min_vals = backend.KerasTensor((4,))
        max_vals = backend.KerasTensor((4,))
        y = quantizers.fake_quant_with_min_max_vars(
            x, min_vals, max_vals, axis=-1
        )

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
                "testcase_name": (
                    "wide_8bits_scalar_input_mins_0.0_input_maxs_255.0"
                ),
                "narrow_range": False,
                "input_mins": 0.0,
                "input_maxs": 255.0,
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
        not backend.SUPPORTS_GRADIENT,
        reason="Backend does not support gradients.",
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
        num_channels = len(expected_nudged_input_mins)
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
        def quantize_fn(x):
            return quantizers.fake_quant_with_min_max_vars(
                x, input_min, input_max, num_bits, narrow_range, axis
            )

        gradients = initial_gradients * ops.grad(quantize_fn)(inputs)
        if not testing.jax_uses_gpu():
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
