import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops.operation import Operation
from keras.src.testing import test_case

X = np.array([[1, 2, 3], [3, 2, 1]])


class Square(Operation):
    """Squares `x`. The subclasses below break its symbolic path."""

    def call(self, x):
        return ops.square(x)


class WrongRank(Square):
    """The symbolic output has an extra dimension."""

    def compute_output_spec(self, x):
        return KerasTensor(x.shape + (1,), dtype=x.dtype)


class WrongDtype(Square):
    """The symbolic output dtype is always float32."""

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="float32")


class NeedsStaticShape(Square):
    """The symbolic path rejects unknown dimensions."""

    def compute_output_spec(self, x):
        if None in x.shape:
            raise ValueError("Needs a static shape")
        return KerasTensor(x.shape, dtype=x.dtype)


class LazyShape(Square):
    """One unknown input dim makes every output dim unknown."""

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
        self.assertRegex(
            result.failures[0][1],
            r"Output differs: symbolic=int\d\d\[2, 3, 1\], "
            r"eager=int\d\d\[2, 3\] for inputs int64\[2, 3\]",
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
        # All dims at once, then one unknown dim at a time.
        self.assertEqual(
            [t.params["shapes"] for t, _ in result.errors],
            ["(None, None)", "(None, 3)", "(2, None)"],
        )

    def test_unknown_dim_variants(self):
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
        self.run_op_symbolic_check(op, (X,), {}, [(None, None)])
        with self.assertRaisesRegex(AssertionError, "does not depend on"):
            self.run_op_symbolic_check(op, (X,), {}, [(None, 3)])

    def test_invalid_probe_skips_precision(self):
        self.run_op_symbolic_check(LazyShapeNoSizeOne(), (X,), {}, [(None, 3)])

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
            AssertionError, "Output differs: symbolic=float32"
        ):
            self.run_op_symbolic_check(WrongDtype(), (X,), {})

    def test_python_scalar_inputs(self):
        self.run_op_test(ops.add, np.add, (X, 1))
        self.run_op_input_dtypes_check(
            ops.add, np.add, jnp.add, (X, 1), {}, ("int8",)
        )

    def test_excluded_check_is_skipped(self):
        def run(t):
            excluded = {test_case._nodeid(t): [{"check": "eager"}]}
            with mock.patch.dict(test_case.EXCLUDED_CHECKS, excluded):
                t.run_op_test(
                    function_form(WrongRankAndValues()), np.square, (X,)
                )

        result = run_as_test(run)
        self.assertEqual(failed_checks(result), ["symbolic_static"])
        self.assertEqual(
            [r for _, r in result.skipped],
            [f"Not supported operation by {backend.backend()} backend"]
            + ["symbolic_static failed"] * 2,
        )

    def test_excluded_check_params_are_a_subset_match(self):
        def run(t):
            excluded = {test_case._nodeid(t): [{"check": "symbolic_partial"}]}
            with mock.patch.dict(test_case.EXCLUDED_CHECKS, excluded):
                t.run_op_test(ops.square, np.square, (X,))

        result = run_as_test(run)
        self.assertTrue(result.wasSuccessful())
        # Both partial variants are skipped, the dynamic one still ran.
        self.assertEqual(
            [t.params["check"] for t, _ in result.skipped],
            ["symbolic_partial"] * 2,
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


class ParseExcludedTestTest(testing.TestCase):
    NODEID = "keras/src/ops/x_test.py::FooTest::test_op"

    @parameterized.named_parameters(
        ("whole_test", "", None),
        ("parameterized_name", "_('int8', 'bool')", None),
        ("nested_parens", "_(1, (2, 3))", None),
        ("one_check", " (check='eager')", {"check": "eager"}),
        (
            "two_params",
            " (check='symbolic_partial', shapes='(None, 3) x (2, 3)')",
            {"check": "symbolic_partial", "shapes": "(None, 3) x (2, 3)"},
        ),
        (
            "nested_parens_and_check",
            "_(1, (2, 3)) (check='eager')",
            {"check": "eager"},
        ),
    )
    def test_parse(self, suffix, params):
        line = self.NODEID + suffix
        nodeid = line.split(" (check=")[0] if params else line
        self.assertEqual(test_case.parse_excluded_test(line), (nodeid, params))
