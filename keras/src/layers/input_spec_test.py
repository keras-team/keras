from unittest import mock

from keras.src import testing
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.input_spec import assert_input_compatibility


class InputSpecTest(testing.TestCase):
    def test_basic_init(self):
        spec = InputSpec(dtype="float32", shape=(None, 10))
        self.assertEqual(spec.dtype, "float32")
        self.assertEqual(spec.shape, (None, 10))
        self.assertEqual(spec.ndim, 2)

    def test_init_ndim_only(self):
        spec = InputSpec(ndim=3)
        self.assertEqual(spec.ndim, 3)
        self.assertIsNone(spec.shape)

    def test_init_min_max_ndim(self):
        spec = InputSpec(min_ndim=2, max_ndim=4)
        self.assertEqual(spec.min_ndim, 2)
        self.assertEqual(spec.max_ndim, 4)

    def test_init_axes(self):
        spec = InputSpec(ndim=3, axes={-1: 10})
        self.assertEqual(spec.axes, {-1: 10})

    def test_init_axes_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            InputSpec(axes="invalid")

    def test_init_axes_exceeds_max_dim_raises(self):
        with self.assertRaises(ValueError):
            InputSpec(ndim=2, axes={5: 10})

    def test_init_axes_exceeds_max_ndim_raises(self):
        with self.assertRaises(ValueError):
            InputSpec(max_ndim=3, axes={5: 10})

    def test_optional_default_false(self):
        spec = InputSpec()
        self.assertFalse(spec.optional)

    def test_optional_true(self):
        spec = InputSpec(optional=True)
        self.assertTrue(spec.optional)

    def test_repr_with_dtype(self):
        spec = InputSpec(dtype="float32")
        self.assertIn("dtype=float32", repr(spec))

    def test_repr_with_shape(self):
        spec = InputSpec(shape=(None, 10))
        self.assertIn("shape=(None, 10)", repr(spec))

    def test_repr_with_ndim(self):
        spec = InputSpec(ndim=3)
        self.assertIn("ndim=3", repr(spec))

    def test_repr_empty(self):
        spec = InputSpec()
        self.assertEqual(repr(spec), "InputSpec()")

    def test_get_config(self):
        spec = InputSpec(
            dtype="float32",
            shape=(None, 10),
            max_ndim=None,
            min_ndim=None,
            axes=None,
            optional=False,
        )
        config = spec.get_config()
        self.assertEqual(config["dtype"], "float32")
        self.assertEqual(config["shape"], (None, 10))
        self.assertEqual(config["ndim"], 2)
        self.assertFalse(config["optional"])

    def test_from_config(self):
        config = {
            "dtype": "float32",
            "shape": (None, 10),
            "ndim": None,
            "max_ndim": None,
            "min_ndim": None,
            "axes": None,
            "optional": False,
        }
        spec = InputSpec.from_config(config)
        self.assertEqual(spec.dtype, "float32")
        self.assertEqual(spec.shape, (None, 10))

    def test_allow_last_axis_squeeze(self):
        spec = InputSpec(shape=(None, 28, 28, 1), allow_last_axis_squeeze=True)
        self.assertTrue(spec.allow_last_axis_squeeze)


class AssertInputCompatibilityTest(testing.TestCase):
    def _make_tensor_like(self, shape, dtype="float32"):
        """Create a simple object with shape and dtype attributes."""
        t = mock.MagicMock()
        t.shape = shape
        t.dtype = dtype
        return t

    def test_none_spec_passes(self):
        # Should not raise
        assert_input_compatibility(None, [self._make_tensor_like((2, 3))], "l")

    def test_empty_spec_passes(self):
        assert_input_compatibility([], [self._make_tensor_like((2, 3))], "l")

    def test_matching_ndim(self):
        spec = InputSpec(ndim=2)
        assert_input_compatibility(
            spec, self._make_tensor_like((3, 5)), "test_layer"
        )

    def test_wrong_ndim_raises(self):
        spec = InputSpec(ndim=3)
        with self.assertRaises(ValueError):
            assert_input_compatibility(
                spec, self._make_tensor_like((3, 5)), "test_layer"
            )

    def test_matching_dtype(self):
        spec = InputSpec(dtype="float32")
        assert_input_compatibility(
            spec, self._make_tensor_like((3, 5), "float32"), "test_layer"
        )

    def test_wrong_dtype_raises(self):
        spec = InputSpec(dtype="float32")
        with self.assertRaises(ValueError):
            assert_input_compatibility(
                spec, self._make_tensor_like((3, 5), "int32"), "test_layer"
            )

    def test_min_ndim_satisfied(self):
        spec = InputSpec(min_ndim=2)
        assert_input_compatibility(
            spec, self._make_tensor_like((3, 5, 7)), "test_layer"
        )

    def test_min_ndim_violated(self):
        spec = InputSpec(min_ndim=3)
        with self.assertRaises(ValueError):
            assert_input_compatibility(
                spec, self._make_tensor_like((3, 5)), "test_layer"
            )

    def test_max_ndim_satisfied(self):
        spec = InputSpec(max_ndim=3)
        assert_input_compatibility(
            spec, self._make_tensor_like((3, 5)), "test_layer"
        )

    def test_max_ndim_violated(self):
        spec = InputSpec(max_ndim=2)
        with self.assertRaises(ValueError):
            assert_input_compatibility(
                spec, self._make_tensor_like((3, 5, 7)), "test_layer"
            )

    def test_axes_matching(self):
        spec = InputSpec(ndim=3, axes={-1: 10})
        assert_input_compatibility(
            spec, self._make_tensor_like((2, 5, 10)), "test_layer"
        )

    def test_axes_mismatch_raises(self):
        spec = InputSpec(ndim=3, axes={-1: 10})
        with self.assertRaises(ValueError):
            assert_input_compatibility(
                spec, self._make_tensor_like((2, 5, 7)), "test_layer"
            )

    def test_shape_matching(self):
        spec = InputSpec(shape=(None, 10))
        assert_input_compatibility(
            spec, self._make_tensor_like((5, 10)), "test_layer"
        )

    def test_shape_mismatch_raises(self):
        spec = InputSpec(shape=(None, 10))
        with self.assertRaises(ValueError):
            assert_input_compatibility(
                spec, self._make_tensor_like((5, 7)), "test_layer"
            )

    def test_multiple_inputs_multiple_specs(self):
        specs = [InputSpec(ndim=2), InputSpec(ndim=3)]
        inputs = [
            self._make_tensor_like((3, 5)),
            self._make_tensor_like((3, 5, 7)),
        ]
        assert_input_compatibility(specs, inputs, "test_layer")

    def test_input_count_mismatch_raises(self):
        specs = [InputSpec(ndim=2), InputSpec(ndim=3)]
        inputs = [self._make_tensor_like((3, 5))]
        with self.assertRaises(ValueError):
            assert_input_compatibility(specs, inputs, "test_layer")

    def test_non_tensor_input_raises(self):
        spec = InputSpec(ndim=2)
        with self.assertRaises(ValueError):
            assert_input_compatibility(spec, "not_a_tensor", "test_layer")

    def test_optional_input_accepts_none(self):
        spec = InputSpec(ndim=2, optional=True)
        assert_input_compatibility([spec], [None], "test_layer")

    def test_dict_inputs_with_named_specs(self):
        specs = [
            InputSpec(ndim=2, name="input_a"),
            InputSpec(ndim=3, name="input_b"),
        ]
        inputs = {
            "input_a": self._make_tensor_like((3, 5)),
            "input_b": self._make_tensor_like((3, 5, 7)),
        }
        assert_input_compatibility(specs, inputs, "test_layer")

    def test_dict_inputs_missing_key_raises(self):
        specs = [
            InputSpec(ndim=2, name="input_a"),
            InputSpec(ndim=3, name="input_b"),
        ]
        inputs = {"input_a": self._make_tensor_like((3, 5))}
        with self.assertRaises(ValueError):
            assert_input_compatibility(specs, inputs, "test_layer")

    def test_allow_last_axis_squeeze(self):
        spec = InputSpec(shape=(None, 28, 28), allow_last_axis_squeeze=True)
        # Input with extra trailing 1 should pass
        assert_input_compatibility(
            spec, self._make_tensor_like((5, 28, 28, 1)), "test_layer"
        )
