"""The calibration modes' shared chassis, tested once per mode."""

import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.awq import MAX_CLIP_SAMPLE_ROWS
from keras.src.quantizers.geometry import ContractionView
from keras.src.quantizers.geometry import KernelAxes
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.quantization_test_utils import assert_calibrated
from keras.src.quantizers.quantization_test_utils import assert_float
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import calibration_statistic
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


def _einsum(equation, output_shape, input_shape, kernel=None):
    layer = layers.EinsumDense(equation, output_shape=output_shape)
    layer.build(input_shape)
    if kernel is not None:
        layer.kernel.assign(kernel)
    return layer


def _dense(kernel):
    layer = layers.Dense(kernel.shape[1])
    layer.build((None, kernel.shape[0]))
    layer.kernel.assign(kernel)
    return layer


def _dequantized(layer):
    return ops.convert_to_numpy(layer._quantized_weight().dequantize("float32"))


def _assert_relatively_close(test, actual, desired, tolerance=0.1):
    """`actual` is within relative L2 distance `tolerance` of `desired`."""
    actual = ops.convert_to_numpy(actual)
    desired = ops.convert_to_numpy(desired)
    error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)
    test.assertLess(error, tolerance)


class AxesGeometry(ProjectionGeometry):
    """`AxesProjection`'s geometry: declared kernel axes, its own einsum."""

    @property
    def kernel_axes(self):
        return self.layer.declared_axes

    def contraction_view(self):
        if self.layer.input_batch_axes is None:
            return super().contraction_view()
        return ContractionView(
            self.weight_shape,
            self.kernel_axes,
            input_batch_axes=self.layer.input_batch_axes,
            input_contracted_axes=(-1,),
        )

    def contract(self, inputs, kernel):
        return ops.einsum(self.layer.equation, inputs, kernel)


class AxesProjection(layers.Layer):
    """A third-party N-D projection written to the quantization protocol.

    It is not an `EinsumDense`: its geometry declares the roles of the
    kernel's axes by hand and contracts with `equation`, and the layer
    lists the calibration modes in its `variable_serialization_spec`.
    With `input_batch_axes`, the geometry also overrides
    `contraction_view` to name the input axes of the kernel's batch axes.
    """

    def __init__(
        self,
        equation,
        kernel_shape,
        kernel_axes,
        input_batch_axes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        self.declared_shape = tuple(kernel_shape)
        self.declared_axes = kernel_axes
        self.input_batch_axes = input_batch_axes
        self.bias = None
        self.activation = None

    def build(self, input_shape):
        self.kernel_shape = self.declared_shape
        if self.quantization_mode:
            self.quantized_build(
                self.kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        if not self._strategy_owns_weight_storage():
            self._kernel = self.add_weight(
                name="kernel", shape=self.kernel_shape
            )

    @property
    def kernel(self):
        quantized_weight = self._quantized_weight()
        if quantized_weight is None:
            return self._kernel
        return quantized_weight.unpack()

    def call(self, inputs):
        return ops.einsum(self.equation, inputs, self._kernel)

    def _quantization_geometry(self):
        return AxesGeometry(self)

    @property
    def variable_serialization_spec(self):
        stored = ["quantized_kernel", "kernel_scale", "kernel_zero"]
        return {
            None: ["kernel"],
            "gptq": stored + ["g_idx"],
            "awq": stored + ["awq_scales", "g_idx"],
        }

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "kernel")


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


class CalibrationEinsumLayoutTest(testing.TestCase):
    """GPTQ and AWQ split an einsum kernel by its equation."""

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_kernel_only_axes_share_one_statistic(self, mode):
        # Gemma's query projection `[heads, d_model, head_dim]` against the
        # same weights laid out `[d_model, heads, head_dim]`.
        input_shape, output_shape = (None, 5, 8), (None, 2, 4)
        rng = np.random.default_rng(seed=3)
        layer = _einsum("btd,ndh->btnh", output_shape, input_shape)
        kernel = ops.convert_to_numpy(layer.kernel)
        reference = _einsum(
            "btd,dnh->btnh",
            output_shape,
            input_shape,
            kernel=np.transpose(kernel, (1, 0, 2)),
        )
        x = rng.standard_normal((2, 5, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        reference_calibrator = _calibrate(reference, mode, x, group_size=4)

        # One statistic over the model width, shared by every head, and the
        # same weights up to the transpose.
        statistic = calibration_statistic(calibrator)
        self.assertEqual(ops.shape(statistic)[0], 8)
        self.assertAllClose(
            statistic, calibration_statistic(reference_calibrator)
        )
        self.assertAllClose(
            _dequantized(layer),
            np.transpose(_dequantized(reference), (1, 0, 2)),
        )
        self.assertAllClose(layer(x), reference(x))
        # Stored as `(d_model, packed columns)`, like the reference.
        self.assertEqual(tuple(layer.quantized_kernel.shape), (8, 4))
        self.assertEqual(tuple(layer.g_idx.shape), (8,))

    @parameterized.named_parameters(
        named_product(
            [
                dict(testcase_name="one_group", inner=6, group_size=-1),
                # The last group of every expert is shorter.
                dict(testcase_name="ragged", inner=7, group_size=3),
            ],
            mode=MODES,
        )
    )
    def test_batch_axis_calibrates_each_expert_alone(
        self, mode, inner, group_size
    ):
        # A mixture-of-experts down projection `[experts, inner, d_model]`:
        # the expert axis is shared by the inputs, so every expert is its
        # own problem, calibrated from its own slice of the inputs.
        experts, width = 3, 8
        kwargs = dict(group_size=group_size)
        if group_size != -1:
            # GPTQ's permuted groups and AWQ's clipping sample, per expert.
            if mode == "gptq":
                kwargs["activation_order"] = True
            else:
                kwargs["apply_clip"] = True
        rng = np.random.default_rng(seed=11)
        kernel = rng.standard_normal((experts, inner, width)).astype("float32")
        layer = _einsum(
            "btei,eid->bted",
            (None, experts, width),
            (None, 5, experts, inner),
            kernel=kernel,
        )
        x = rng.standard_normal((2, 5, experts, inner)).astype("float32")
        calibrator = _calibrate(layer, mode, x, **kwargs)
        statistic = ops.convert_to_numpy(calibration_statistic(calibrator))
        self.assertEqual(statistic.shape[:2], (experts, inner))

        dequantized = _dequantized(layer)
        outputs = ops.convert_to_numpy(layer(x))
        n_groups = 1 if group_size == -1 else -(-inner // group_size)
        g_idx = ops.convert_to_numpy(layer.g_idx).reshape(experts, inner)
        for expert in range(experts):
            dense = _dense(kernel[expert])
            x_expert = x[:, :, expert, :]
            dense_calibrator = _calibrate(
                dense, mode, x_expert.reshape(-1, inner), **kwargs
            )
            self.assertAllClose(
                statistic[expert], calibration_statistic(dense_calibrator)
            )
            if mode == "gptq":
                self.assertAllClose(dequantized[expert], _dequantized(dense))
                self.assertAllClose(outputs[:, :, expert, :], dense(x_expert))
            else:
                # The statistics above match exactly. AWQ's grid searches
                # then pick discrete candidates, and the batched and the
                # per-expert contractions sum in different orders, so a near
                # tie can pick a neighbouring scale for an expert.
                _assert_relatively_close(
                    self, dequantized[expert], _dequantized(dense)
                )
                _assert_relatively_close(
                    self, outputs[:, :, expert, :], dense(x_expert)
                )
            # The experts stack along the rows, each with its own groups.
            self.assertAllClose(
                g_idx[expert],
                ops.convert_to_numpy(dense.g_idx) + expert * n_groups,
            )
        self.assertEqual(
            tuple(layer.quantized_kernel.shape),
            (experts * inner, width // 2),
        )
        self.assertEqual(
            tuple(layer.kernel_scale.shape), (experts * n_groups, width)
        )

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_kernel_without_a_free_axis_matches_dense(self, mode):
        # A per-head gate `btnh,nh->btn`: every head is a problem of one
        # column.
        heads = 3
        rng = np.random.default_rng(seed=23)
        layer = _einsum("btnh,nh->btn", (None, heads), (None, 5, heads, 8))
        kernel = ops.convert_to_numpy(layer.kernel)
        x = rng.standard_normal((2, 5, heads, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        statistic = ops.convert_to_numpy(calibration_statistic(calibrator))
        self.assertEqual(tuple(layer.quantized_kernel.shape), (heads * 8, 1))

        dequantized = _dequantized(layer)
        outputs = ops.convert_to_numpy(layer(x))
        for head in range(heads):
            dense = _dense(kernel[head].reshape(8, 1))
            x_head = x[:, :, head, :]
            dense_calibrator = _calibrate(dense, mode, x_head, group_size=4)
            self.assertAllClose(
                statistic[head].reshape(8, -1),
                ops.reshape(calibration_statistic(dense_calibrator), (8, -1)),
            )
            self.assertAllClose(dequantized[head], _dequantized(dense)[:, 0])
            self.assertAllClose(outputs[:, :, head], dense(x_head)[..., 0])

    def test_awq_clip_sample_is_capped_per_expert(self):
        # Over two batches of 300 tokens, each expert keeps its own
        # `MAX_CLIP_SAMPLE_ROWS` rows for the clipping search.
        experts, inner = 2, 8
        rng = np.random.default_rng(seed=31)
        layer = _einsum(
            "btei,eid->bted", (None, experts, 4), (None, 300, experts, inner)
        )
        batches = [
            rng.standard_normal((1, 300, experts, inner)).astype("float32")
            for _ in range(2)
        ]
        config = _config("awq", group_size=4, apply_clip=True)
        calibrator = calibrate_layer(layer, config, *batches)
        self.assertEqual(calibrator._clip_sample_rows, MAX_CLIP_SAMPLE_ROWS)
        self.assertEqual(
            tuple(ops.concatenate(calibrator._clip_samples, axis=-2).shape),
            (experts, MAX_CLIP_SAMPLE_ROWS, inner),
        )


class ThirdPartyGeometryTest(testing.TestCase):
    """A third-party N-D projection calibrates through its geometry."""

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_declared_axes_calibrate_like_einsum_dense(self, mode):
        # Gemma's query projection `[heads, d_model, head_dim]`, its axes
        # declared by hand, calibrates like the `EinsumDense` that derives
        # them from the equation.
        equation = "btd,ndh->btnh"
        kernel_axes = KernelAxes(contracted=(1,), free=(0, 2))
        rng = np.random.default_rng(seed=19)
        layer = AxesProjection(equation, (2, 8, 4), kernel_axes)
        layer.build((None, 5, 8))
        reference = _einsum(
            equation,
            (None, 2, 4),
            (None, 5, 8),
            kernel=ops.convert_to_numpy(layer.kernel),
        )
        x = rng.standard_normal((2, 5, 8)).astype("float32")
        calibrator = _calibrate(layer, mode, x, group_size=4)
        reference_calibrator = _calibrate(reference, mode, x, group_size=4)
        assert_calibrated(self, layer, mode)
        self.assertAllClose(
            calibration_statistic(calibrator),
            calibration_statistic(reference_calibrator),
        )
        self.assertAllClose(_dequantized(layer), _dequantized(reference))
        self.assertAllClose(layer(x), reference(x))
        self.assertEqual(tuple(layer.quantized_kernel.shape), (8, 4))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_batch_axis_needs_a_view_override(self, mode):
        # A mixture-of-experts down projection `[experts, inner, d_model]`.
        # The default view names no input batch axis, so the layer is
        # refused until its geometry overrides `contraction_view`; with the
        # inputs' expert axis named, it calibrates like `EinsumDense`.
        equation = "btei,eid->bted"
        kernel_axes = KernelAxes(contracted=(1,), free=(2,), batch=(0,))
        refused = AxesProjection(equation, (3, 6, 4), kernel_axes)
        refused.build((None, 5, 3, 6))
        with self.assertRaisesRegex(
            NotImplementedError, "must override `contraction_view`"
        ):
            refused.quantize(mode, config=_config(mode, group_size=-1))
        assert_float(self, refused)

        rng = np.random.default_rng(seed=37)
        layer = AxesProjection(
            equation, (3, 6, 4), kernel_axes, input_batch_axes=(2,)
        )
        layer.build((None, 5, 3, 6))
        reference = _einsum(
            equation,
            (None, 3, 4),
            (None, 5, 3, 6),
            kernel=ops.convert_to_numpy(layer.kernel),
        )
        x = rng.standard_normal((2, 5, 3, 6)).astype("float32")
        _calibrate(layer, mode, x, group_size=-1)
        _calibrate(reference, mode, x, group_size=-1)
        self.assertAllClose(_dequantized(layer), _dequantized(reference))
        self.assertAllClose(layer(x), reference(x))

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_view_batch_axes_must_match_the_inputs(self, mode):
        # The override names the token axis (5) for the 3 experts.
        layer = AxesProjection(
            "btei,eid->bted",
            (3, 6, 4),
            KernelAxes(contracted=(1,), free=(2,), batch=(0,)),
            input_batch_axes=(1,),
        )
        layer.build((None, 5, 3, 6))
        x = np.ones((2, 5, 3, 6), dtype="float32")
        with self.assertRaisesRegex(ValueError, "has 3 problems"):
            _calibrate(layer, mode, x)
        assert_float(self, layer)

    @parameterized.named_parameters(named_product(mode=MODES))
    def test_mis_declared_axes_raise(self, mode):
        # Axis 0 of the kernel has no role. The author's mistake reaches
        # the caller instead of reading as a layer without support.
        layer = AxesProjection(
            "btd,ndh->btnh", (2, 8, 4), KernelAxes(contracted=(1,), free=(2,))
        )
        layer.build((None, 5, 8))
        with self.assertRaisesRegex(ValueError, "partition the kernel axes"):
            layer.quantize(mode, config=_config(mode))
        assert_float(self, layer)
