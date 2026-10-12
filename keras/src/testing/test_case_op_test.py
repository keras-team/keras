import functools
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops.operation import Operation
from keras.src.testing import test_case

X = np.array([[1, 2, 3], [3, 2, 1]])


class WrongRank(Operation):
    """Squares `x`; the symbolic output has an extra dimension."""

    def call(self, x):
        return ops.square(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape + (1,), dtype=x.dtype)


class WrongDtype(Operation):
    """Squares `x`; the symbolic output dtype is always float32."""

    def call(self, x):
        return ops.square(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="float32")


class NeedsStaticShape(Operation):
    """Squares `x`; the symbolic path rejects unknown dimensions."""

    def call(self, x):
        return ops.square(x)

    def compute_output_spec(self, x):
        if None in x.shape:
            raise ValueError("Needs a static shape")
        return KerasTensor(x.shape, dtype=x.dtype)


class LazyShape(Operation):
    """Squares `x`; one unknown input dim makes every output dim unknown."""

    def call(self, x):
        return ops.square(x)

    def compute_output_spec(self, x):
        if None in x.shape:
            return KerasTensor([None] * len(x.shape), dtype=x.dtype)
        return KerasTensor(x.shape, dtype=x.dtype)


class LazyShapeNoSizeOne(LazyShape):
    """`LazyShape` whose symbolic path also rejects dimensions of size 1."""

    def compute_output_spec(self, x):
        if 1 in x.shape:
            raise ValueError("Size-1 dimensions are not supported")
        return super().compute_output_spec(x)


class WrongRankAndValues(WrongRank):
    """`WrongRank` whose eager values are also off by one."""

    def call(self, x):
        return ops.square(x) + 1


def function_form(op):
    """Wraps an `Operation` instance as a function, like `knp.square`."""
    return lambda *args, **kwargs: op(*args, **kwargs)


def run_as_test(fn):
    """Runs `fn(test_case)` as a test and returns its `unittest.TestResult`.

    `subTest` failures and skips are recorded on the result, not raised.
    """

    class Case(testing.TestCase):
        def runTest(self):
            fn(self)

    result = unittest.TestResult()
    Case().run(result)
    return result


def failed_checks(result):
    """Returns the distinct `check` names of the failed or errored subtests."""
    return sorted(
        {t.params["check"] for t, _ in result.failures + result.errors}
    )


class RunOpTestTest(testing.TestCase):
    def test_failed_check_skips_the_checks_refining_it(self):
        result = run_as_test(
            lambda t: t.run_op_test(function_form(WrongRank()), np.square, (X,))
        )
        self.assertEqual(failed_checks(result), ["symbolic_static"])
        self.assertIn(
            "Output rank differs: symbolic=(2, 3, 1), eager=(2, 3) "
            "for inputs int64[2, 3]",
            result.failures[0][1],
        )
        self.assertEqual(
            [r for _, r in result.skipped], ["symbolic_static failed"] * 2
        )

    def test_failing_check_does_not_hide_independent_ones(self):
        result = run_as_test(
            lambda t: t.run_op_test(
                function_form(WrongRankAndValues()), np.square, (X,)
            )
        )
        self.assertEqual(failed_checks(result), ["eager", "symbolic_static"])

    def test_unknown_dims_checks(self):
        result = run_as_test(
            lambda t: t.run_op_test(
                function_form(NeedsStaticShape()), np.square, (X,)
            )
        )
        self.assertEqual(
            failed_checks(result), ["symbolic_dynamic", "symbolic_partial"]
        )
        # All dims at once, then one mask per dim of `X`.
        self.assertEqual(
            [t.params.get("shapes") for t, _ in result.errors],
            [None, "(None, 3)", "(2, None)"],
        )

    def test_partial_masks(self):
        shapes = []

        def op(x, y):
            shapes.append((tuple(x.shape), tuple(y.shape)))
            return ops.add(x, y)

        self.run_op_test(op, np.add, (X, X))
        self.assertEqual(
            [s for s in shapes if None in s[0] + s[1]],
            [
                ((None, None), (None, None)),
                ((None, 3), (2, 3)),
                ((2, None), (2, 3)),
                ((2, 3), (None, 3)),
                ((2, 3), (2, None)),
                ((None, 3), (None, 3)),
            ],
        )

    def test_lazy_shape_fails_precision(self):
        op = LazyShape()
        self.run_op_symbolic_check(op, (X,), {}, none_dims={(0, 0), (0, 1)})
        with self.assertRaisesRegex(AssertionError, "does not depend on"):
            self.run_op_symbolic_check(op, (X,), {}, none_dims={(0, 0)})

    def test_invalid_probe_skips_precision(self):
        self.run_op_symbolic_check(
            LazyShapeNoSizeOne(), (X,), {}, none_dims={(0, 0)}
        )

    def test_disabled_dynamic_shape_checks_are_skipped(self):
        result = run_as_test(
            lambda t: t.run_op_test(
                function_form(NeedsStaticShape()),
                np.square,
                (X,),
                run_dynamic_shape_checks=False,
            )
        )
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(
            [r for _, r in result.skipped], ["Disabled by the test"] * 2
        )

    def test_operation_instance_runs_eager_only(self):
        # The symbolic checks would fail on `WrongRank`, so they didn't run.
        result = run_as_test(
            lambda t: t.run_op_test(WrongRank(), np.square, (X,))
        )
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.skipped, [])

    def test_no_array_inputs_runs_eager_only(self):
        # `ones` has no symbolic path, so a symbolic check would fail.
        result = run_as_test(
            lambda t: t.run_op_test(ops.ones, np.ones, ((2, 3),))
        )
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.skipped, [])

    def test_output_count_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError, "output count differs"):
            self.run_op_eager_check(
                lambda x: ops.stack([x, x]), lambda x: (x, x), (X,), {}
            )

    def test_symbolic_dtype_mismatch_fails(self):
        with self.assertRaisesRegex(
            AssertionError, "Output dtype differs: symbolic=float32, eager=int"
        ):
            self.run_op_symbolic_check(WrongDtype(), (X,), {})

    def test_complex_outputs_compare_imaginary(self):
        close = functools.partial(
            test_case._assert_outputs_close, self, atol=1e-6, rtol=1e-6, msg=""
        )
        close(np.array([1 + 2j]), np.array([1 + 2j]))
        with self.assertRaises(AssertionError):
            close(np.array([1j]), np.array([-1j]))

    def test_python_scalar_inputs(self):
        self.run_op_test(ops.add, np.add, (X, 1))
        self.run_op_input_dtypes_check(
            ops.add, np.add, jnp.add, (X, 1), {}, ("int8",)
        )


class RunOpInputDtypesCheckTest(testing.TestCase):
    def test_dtype_mismatch_vs_jax_fails(self):
        def op(x):
            return ops.cast(ops.square(x), "int32")

        result = run_as_test(
            lambda t: t.run_op_input_dtypes_check(
                op, np.square, jnp.square, (X,), {}, ("uint8",)
            )
        )
        self.assertEqual(failed_checks(result), ["jax_dtype"])
        self.assertIn(
            "Output dtype differs from JAX: keras=int32, jax=uint8 "
            "for inputs uint8[2, 3]",
            result.failures[0][1],
        )

    def test_skipped_dtype_has_reason(self):
        skipped = {backend.backend(): {"int8": "no int8 here"}}
        with mock.patch.dict(test_case._SKIPPED_DTYPES, skipped):
            with self.assertRaisesRegex(unittest.SkipTest, "int8: no int8"):
                self.run_op_input_dtypes_check(
                    ops.square, np.square, jnp.square, (X,), {}, ("int8",)
                )

    def test_dtype_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.run_op_input_dtypes_check(
                ops.add, np.add, jnp.add, (X, X), {}, ("int8",)
            )
