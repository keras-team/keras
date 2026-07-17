import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.layers.core.input_layer import Input
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.input_spec import assert_input_compatibility
from keras.src.models import Model


class AssertInputCompatibilityTest(testing.TestCase):
    def test_single_spec_2d_tensor_passes(self):
        spec = InputSpec(min_ndim=2)
        x = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
        assert_input_compatibility(spec, x, "dense")

    def test_single_spec_list_of_one_2d_tensor_passes(self):
        # The general path (list of one spec) must also accept a 2D tensor.
        spec = [InputSpec(min_ndim=2)]
        x = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
        assert_input_compatibility(spec, x, "dense")

    def test_single_spec_axes_check_passes(self):
        spec = InputSpec(min_ndim=2, axes={-1: 3})
        x = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
        assert_input_compatibility(spec, x, "dense")

    def test_single_spec_shape_check_passes(self):
        spec = InputSpec(shape=(None, 3))
        x = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
        assert_input_compatibility(spec, x, "dense")

    def test_none_input_spec_returns_immediately(self):
        assert_input_compatibility(None, np.array([1.0]), "any")

    def test_empty_list_spec_returns_immediately(self):
        assert_input_compatibility([], np.array([1.0]), "any")

    def test_optional_spec_none_input_returns(self):
        spec = InputSpec(optional=True)
        assert_input_compatibility(spec, None, "optional")

    def test_single_spec_1d_tensor_raises_min_ndim(self):
        spec = InputSpec(min_ndim=2)
        x = backend.convert_to_tensor(np.array([1.0, 2.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "min_ndim=2"):
            assert_input_compatibility(spec, x, "dense")

    def test_single_spec_ndim2_1d_tensor_raises_ndim(self):
        spec = InputSpec(ndim=2)
        x = backend.convert_to_tensor(np.array([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "ndim=2"):
            assert_input_compatibility(spec, x, "dense")

    def test_allow_last_axis_squeeze_1d_passes(self):
        # Functional models set `allow_last_axis_squeeze=True`: a model
        # built with `Input(shape=[1])` accepts inputs of shape `(1,)`.
        spec = InputSpec(shape=(None, 1), allow_last_axis_squeeze=True)
        x = backend.convert_to_tensor(np.array([5.0]))
        assert_input_compatibility(spec, x, "functional")

    def test_allow_last_axis_squeeze_list_spec_1d_passes(self):
        spec = [InputSpec(shape=(None, 1), allow_last_axis_squeeze=True)]
        x = backend.convert_to_tensor(np.array([5.0]))
        assert_input_compatibility(spec, x, "functional")

    def test_allow_last_axis_squeeze_rank_n_plus_1_passes(self):
        # A rank N+1 input with a last axis of size 1 is compatible with a
        # rank N spec.
        spec = InputSpec(shape=(None, 3), allow_last_axis_squeeze=True)
        x = backend.convert_to_tensor(np.array([[[1.0], [2.0], [3.0]]]))
        assert_input_compatibility(spec, x, "squeeze")
        # Also when the spec itself ends with an axis of size 1.
        spec = InputSpec(shape=(None, 3, 1), allow_last_axis_squeeze=True)
        x = backend.convert_to_tensor(np.zeros((1, 3, 1, 1)))
        assert_input_compatibility(spec, x, "squeeze")

    def test_allow_last_axis_squeeze_same_rank_mismatch_raises(self):
        # The squeeze only applies to rank mismatches of exactly one: inputs
        # of the same rank as the spec are compared as-is, including their
        # last axis.
        spec = InputSpec(shape=(None, 3), allow_last_axis_squeeze=True)
        x = backend.convert_to_tensor(np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            assert_input_compatibility(spec, x, "squeeze")
        spec = InputSpec(shape=(None, 1), allow_last_axis_squeeze=True)
        x = backend.convert_to_tensor(np.zeros((2, 5)))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            assert_input_compatibility(spec, x, "squeeze")

    def test_allow_last_axis_squeeze_wrong_squeezed_dim_raises(self):
        # A rank N+1 input squeezed to rank N must match all N spec
        # dimensions, including a spec last axis of size 1.
        spec = InputSpec(shape=(None, 3, 1), allow_last_axis_squeeze=True)
        x = backend.convert_to_tensor(np.zeros((2, 3, 99, 1)))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            assert_input_compatibility(spec, x, "squeeze")
        # A rank N+1 input whose last axis is not 1 is not squeezed and is
        # checked against the full spec.
        x = backend.convert_to_tensor(np.zeros((2, 3, 7, 9)))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            assert_input_compatibility(spec, x, "squeeze")

    def test_wrong_axes_raises(self):
        spec = InputSpec(min_ndim=2, axes={-1: 3})
        x = backend.convert_to_tensor(np.array([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "axis"):
            assert_input_compatibility(spec, x, "dense")

    def test_axes_out_of_bounds_raises(self):
        # An axis in `spec.axes` that does not exist on the input must raise
        # a `ValueError`, not an `IndexError`.
        x = backend.convert_to_tensor(np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            assert_input_compatibility(InputSpec(axes={3: 5}), x, "axes")
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            assert_input_compatibility(InputSpec(axes={-3: 5}), x, "axes")

    def test_wrong_shape_raises(self):
        spec = InputSpec(shape=(None, 3))
        x = backend.convert_to_tensor(np.array([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "shape"):
            assert_input_compatibility(spec, x, "dense")

    def test_wrong_dtype_raises(self):
        spec = InputSpec(dtype="float32")
        x = backend.convert_to_tensor(np.array([[1, 2, 3]], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "dtype"):
            assert_input_compatibility(spec, x, "dense")

    def test_max_ndim_exceeded_raises(self):
        spec = InputSpec(max_ndim=2)
        x = backend.convert_to_tensor(np.array([[[1.0, 2.0]]]))
        with self.assertRaisesRegex(ValueError, "max_ndim"):
            assert_input_compatibility(spec, x, "dense")

    def test_non_tensor_input_raises(self):
        # Inputs without a `shape` attribute (e.g. a `Layer` passed in the
        # Functional API) must raise a clear error.
        spec = InputSpec(min_ndim=2)
        with self.assertRaisesRegex(ValueError, "tensors"):
            assert_input_compatibility(spec, object(), "dense")

    def test_two_specs_one_input_raises(self):
        specs = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]
        x = backend.convert_to_tensor(np.array([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "2 input"):
            assert_input_compatibility(specs, x, "multi")

    def test_bare_and_list_wrapped_spec_equivalent(self):
        # The fast path (bare spec) and the general path (list-wrapped spec)
        # must agree on pass/raise and on the exact error message.
        cases = [
            (InputSpec(min_ndim=2), np.array([[1.0, 2.0, 3.0]]), False),
            (InputSpec(min_ndim=2), np.array([1.0, 2.0, 3.0]), True),
            (InputSpec(ndim=2), np.array([1.0, 2.0]), True),
            (InputSpec(shape=(None, 3)), np.array([[1.0, 2.0]]), True),
            (
                InputSpec(dtype="float32"),
                np.array([[1, 2]], dtype=np.int32),
                True,
            ),
        ]
        for spec, arr, expect_raise in cases:
            x = backend.convert_to_tensor(arr)
            bare_error = None
            list_error = None
            try:
                assert_input_compatibility(spec, x, "layer")
            except ValueError as e:
                bare_error = str(e)
            try:
                assert_input_compatibility([spec], x, "layer")
            except ValueError as e:
                list_error = str(e)
            self.assertEqual(bare_error, list_error)
            if expect_raise:
                self.assertIsNotNone(bare_error)
            else:
                self.assertIsNone(bare_error)


class InputSpecIntegrationTest(testing.TestCase):
    def test_dense_2d_input_passes(self):
        layer = layers.Dense(4)
        result = layer(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(result.shape[-1], 4)

    def test_dense_1d_input_raises(self):
        layer = layers.Dense(4)
        with self.assertRaisesRegex(Exception, "min_ndim|ndim"):
            layer(np.array([1.0, 2.0, 3.0]))

    def test_functional_model_1d_input_passes(self):
        # Functional models use `allow_last_axis_squeeze=True`, so a model
        # built with `Input(shape=[1])` accepts a 1D input.
        inp = Input(shape=[1])
        out = layers.Dense(1)(inp)
        model = Model(inp, out)
        result = model(np.array([5.0]))
        self.assertIsNotNone(result)

    def test_functional_model_2d_input_passes(self):
        inp = Input(shape=[1])
        out = layers.Dense(1)(inp)
        model = Model(inp, out)
        result = model(np.array([[5.0]]))
        self.assertIsNotNone(result)

    def test_functional_model_rank_mismatch_raises(self):
        # Rank mismatches that cannot be resolved by squeezing a last axis
        # of size 1 are left to `Functional._adjust_input_rank`, which
        # raises an error that includes the input path.
        inp = Input(shape=(3,), name="foo")
        out = layers.Dense(3)(inp)
        model = Model(inp, out)
        with self.assertRaisesRegex(ValueError, "Invalid input shape"):
            model(np.zeros((2, 3, 4)))
