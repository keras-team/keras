"""Tests for assert_input_compatibility.

Behavior pinned by these tests (KERAS_BACKEND=torch):
  - Bare Dense layer (InputSpec(min_ndim=2)) RAISES for 1-D inputs.
  - Functional model InputSpec has allow_last_axis_squeeze=True; 1-D inputs
    PASS via the squeeze path (no ndim check when allow_last_axis_squeeze=True).
  - The fast path (single-InputSpec + single-tensor) must match the general
    path exactly.
"""

import numpy as np

import keras
from keras.src import backend
from keras.src import testing
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.input_spec import assert_input_compatibility

# ---------------------------------------------------------------------------
# Direct assert_input_compatibility tests
# ---------------------------------------------------------------------------


class AssertInputCompatibilityFastPathTest(testing.TestCase):
    """Pin behavior for single-InputSpec + single-tensor."""

    # -- correct input accepts --

    def test_single_spec_2d_tensor_passes(self):
        spec = InputSpec(min_ndim=2)
        x = backend.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
        # Must not raise
        assert_input_compatibility(spec, x, "dense")

    def test_single_spec_list_of_one_2d_tensor_passes(self):
        """General path (list of one spec) must also accept 2-D tensor."""
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
        """InputSpec(optional=True) with None input returns cleanly."""
        spec = InputSpec(optional=True)
        assert_input_compatibility(spec, None, "optional")

    # -- 1-D input rejects on bare Dense spec (RAISES) --

    def test_single_spec_1d_tensor_raises_min_ndim(self):
        """Dense sets InputSpec(min_ndim=2); 1-D must raise."""
        spec = InputSpec(min_ndim=2)
        x = backend.convert_to_tensor(np.array([1.0, 2.0, 3.0]))  # shape (3,)
        with self.assertRaisesRegex(ValueError, "min_ndim=2"):
            assert_input_compatibility(spec, x, "dense")

    def test_single_spec_ndim2_1d_tensor_raises_ndim(self):
        """InputSpec(ndim=2): 1-D input must raise."""
        spec = InputSpec(ndim=2)
        x = backend.convert_to_tensor(np.array([1.0, 2.0]))  # shape (2,)
        with self.assertRaisesRegex(ValueError, "ndim=2"):
            assert_input_compatibility(spec, x, "dense")

    # -- allow_last_axis_squeeze=True: 1-D input PASSES --
    # (Functional model behavior)

    def test_allow_last_axis_squeeze_1d_passes(self):
        """Functional model InputSpec has allow_last_axis_squeeze=True.
        Accepts np.array([5]) (shape (1,)) for Input(shape=[1]) models.
        """
        spec = InputSpec(shape=(None, 1), allow_last_axis_squeeze=True)
        x = backend.convert_to_tensor(np.array([5.0]))  # shape (1,)
        # Must NOT raise
        assert_input_compatibility(spec, x, "functional")

    def test_allow_last_axis_squeeze_list_spec_1d_passes(self):
        """Same via list-wrapped spec (Functional model call path)."""
        spec = [InputSpec(shape=(None, 1), allow_last_axis_squeeze=True)]
        x = backend.convert_to_tensor(np.array([5.0]))  # shape (1,)
        assert_input_compatibility(spec, x, "functional")

    def test_allow_last_axis_squeeze_rank_n_plus_1_passes(self):
        """Rank N+1 input with trailing axis 1 squeezes to match rank N spec."""
        spec = InputSpec(shape=(None, 3), allow_last_axis_squeeze=True)
        # shape (1, 3, 1); last axis 1 is squeezed to (1, 3), matching spec.
        x = backend.convert_to_tensor(np.array([[[1.0], [2.0], [3.0]]]))
        assert_input_compatibility(spec, x, "squeeze")

    # -- wrong inputs must raise --

    def test_wrong_axes_raises(self):
        spec = InputSpec(min_ndim=2, axes={-1: 3})
        x = backend.convert_to_tensor(
            np.array([[1.0, 2.0]])
        )  # last dim=2 not 3
        with self.assertRaisesRegex(ValueError, "axis"):
            assert_input_compatibility(spec, x, "dense")

    def test_wrong_shape_raises(self):
        spec = InputSpec(shape=(None, 3))
        x = backend.convert_to_tensor(
            np.array([[1.0, 2.0]])
        )  # dim 1 is 2 not 3
        with self.assertRaisesRegex(ValueError, "shape"):
            assert_input_compatibility(spec, x, "dense")

    def test_wrong_dtype_raises(self):
        spec = InputSpec(dtype="float32")
        x = backend.convert_to_tensor(np.array([[1, 2, 3]], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "dtype"):
            assert_input_compatibility(spec, x, "dense")

    def test_max_ndim_exceeded_raises(self):
        spec = InputSpec(max_ndim=2)
        x = backend.convert_to_tensor(np.array([[[1.0, 2.0]]]))  # ndim=3
        with self.assertRaisesRegex(ValueError, "max_ndim"):
            assert_input_compatibility(spec, x, "dense")

    def test_non_tensor_input_raises(self):
        """Passing a Layer (no .shape) must raise clearly."""
        spec = InputSpec(min_ndim=2)
        with self.assertRaisesRegex(ValueError, "tensors"):
            assert_input_compatibility(spec, object(), "dense")

    # -- count mismatch --

    def test_two_specs_one_input_raises(self):
        specs = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]
        x = backend.convert_to_tensor(np.array([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "2 input"):
            assert_input_compatibility(specs, x, "multi")

    # -- bare spec vs list-wrapped spec equivalence --

    def test_bare_and_list_wrapped_spec_equivalent(self):
        """A bare spec and the same spec in a list must behave identically.

        The fast path (bare spec) and the general path (list-wrapped) must
        agree on pass/raise and on the exact error message.
        """
        cases = [
            # (spec, input array, expect_raise)
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


# ---------------------------------------------------------------------------
# Integration: Dense layer + Sequential model
# ---------------------------------------------------------------------------


class DenseLayerInputCompatibilityTest(testing.TestCase):
    """Behavior through the actual Dense layer call."""

    def test_dense_2d_input_passes(self):
        layer = keras.layers.Dense(4)
        result = layer(np.array([[1.0, 2.0, 3.0]]))
        self.assertEqual(result.shape[-1], 4)

    def test_dense_1d_input_raises(self):
        layer = keras.layers.Dense(4)
        with self.assertRaisesRegex(Exception, "min_ndim|ndim"):
            layer(np.array([1.0, 2.0, 3.0]))

    def test_functional_model_1d_input_passes(self):
        """Functional model with Input(shape=[1]) accepts np.array([5.0]).
        This is the saving-test fixture scenario
        (allow_last_axis_squeeze=True).
        """
        inp = keras.Input(shape=[1])
        out = keras.layers.Dense(1)(inp)
        model = keras.Model(inp, out)
        # Must NOT raise
        result = model(np.array([5.0]))
        self.assertIsNotNone(result)

    def test_functional_model_2d_input_passes(self):
        inp = keras.Input(shape=[1])
        out = keras.layers.Dense(1)(inp)
        model = keras.Model(inp, out)
        result = model(np.array([[5.0]]))
        self.assertIsNotNone(result)
