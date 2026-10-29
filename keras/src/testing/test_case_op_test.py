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
from keras.src.testing.test_case import X
from keras.src.testing.test_utils import named_dtype_combinations


class Square(Operation):
    """Squares `x`. The subclasses below break its symbolic path."""

    def call(self, x):
        return ops.square(x)


class WrongRank(Square):
    def compute_output_spec(self, x):
        return KerasTensor(x.shape + (1,), dtype=x.dtype)


class WrongDtype(Square):
    def compute_output_spec(self, x):
        return KerasTensor(x.shape, dtype="float32")


class NeedsStaticShape(Square):
    def compute_output_spec(self, x):
        if None in x.shape:
            raise ValueError("Needs a static shape")
        return KerasTensor(x.shape, dtype=x.dtype)


class WrongRankAndValues(WrongRank):
    def call(self, x):
        return ops.square(x) + 1


def function_form(op):
    """Wraps an `Operation` instance as a function, like `knp.square`."""
    return lambda *args, **kwargs: op(*args, **kwargs)


def methods_of(cls):
    return sorted(n for n in dir(cls) if n.startswith("test"))


def outcomes(cls):
    """Runs the tests of `cls`: `{name: passed / skipped: … / failed: …}`."""
    tests = list(unittest.TestLoader().loadTestsFromTestCase(cls))
    result = unittest.TestResult()
    unittest.TestSuite(tests).run(result)
    out = {test._testMethodName: "passed" for test in tests}
    for test, traceback in result.failures + result.errors:
        error = traceback.strip().rsplit("\n", 1)[-1]
        out[test._testMethodName] = f"failed: {error}"
    for test, reason in result.skipped:
        out[test._testMethodName] = f"skipped: {reason}"
    return out


class OpTestTest(testing.TestCase):
    def test_generated_names(self):
        class Unary(testing.OpTest):
            op, op_class = ops.square, Square
            reference_fn, jax_fn = np.square, jnp.square
            inputs = (X,)

        names = methods_of(Unary)
        self.assertEqual(
            [n for n in names if "input_dtypes" not in n],
            [
                "test_eager",
                "test_eager_class",
                "test_symbolic_dynamic",
                "test_symbolic_partial",
                "test_symbolic_static",
            ],
        )
        self.assertLen([n for n in names if "input_dtypes" in n], 10)

        class Binary(testing.OpTest):
            op, reference_fn, jax_fn = ops.add, np.add, jnp.add
            cases = {"a": ((X, X), {}), "b": ((X, X[:1]), {})}
            python_scalars = True

        names = methods_of(Binary)
        self.assertIn("test_eager_a", names)
        self.assertIn("test_symbolic_partial_b", names)
        self.assertNotIn("test_eager_class_a", names)  # No `op_class`.
        self.assertLen(
            [n for n in names if "input_dtypes" in n],
            len(named_dtype_combinations(2)),
        )
        self.assertLen([n for n in names if "python_types" in n], 20)
        self.assertIn("test_python_types_int8_float", names)

    def test_not_generated_when_inapplicable(self):
        class NoArrays(testing.OpTest):  # `ones` has no symbolic path.
            op, reference_fn, jax_fn = ops.ones, np.ones, jnp.ones
            inputs = ((2, 3),)

        self.assertEqual(outcomes(NoArrays), {"test_eager": "passed"})

        class NoJax(testing.OpTest):
            op, reference_fn = ops.square, np.square
            inputs = (X,)
            python_scalars = True  # Needs two inputs: nothing generated.

        self.assertNotIn("test_input_dtypes_int8", methods_of(NoJax))
        self.assertNotIn("test_python_types_int8_int", methods_of(NoJax))

    def test_override_is_kept(self):
        class Case(testing.OpTest):
            op, reference_fn = ops.square, np.square
            inputs = (X,)

            def test_eager(self):
                self.skipTest("overridden")

        class Subclass(Case):
            pass

        self.assertEqual(outcomes(Case)["test_eager"], "skipped: overridden")
        self.assertIs(Subclass.test_eager, Case.test_eager)

    def test_functions_are_unbound(self):
        class Case(testing.OpTest):
            op, reference_fn, jax_fn = ops.square, np.square, jnp.square
            inputs = (X,)

        case = Case("test_eager")
        self.assertAllClose(case.op(X), case.reference_fn(X))
        self.assertAllClose(case.jax_fn(X), np.square(X))

    def test_disabled_dynamic_checks_are_skipped(self):
        class Case(testing.OpTest):
            op, reference_fn = function_form(NeedsStaticShape()), np.square
            inputs = (X,)
            run_dynamic_shape_checks = False

        self.assertEqual(
            outcomes(Case),
            {
                "test_eager": "passed",
                "test_symbolic_static": "passed",
                "test_symbolic_dynamic": "skipped: Disabled by the test",
                "test_symbolic_partial": "skipped: Disabled by the test",
            },
        )

    def test_failed_prerequisite_skips_dependents(self):
        class Case(testing.OpTest):
            op, op_class = function_form(WrongRank()), WrongRank
            reference_fn = np.square
            inputs = (X,)

        out = outcomes(Case)
        self.assertEqual(out["test_eager"], "passed")
        self.assertEqual(out["test_eager_class"], "passed")
        self.assertRegex(
            out["test_symbolic_static"],
            r"^failed: AssertionError: Output differs: "
            r"symbolic=int\d\d\[2, 3, 1\], eager=int\d\d\[2, 3\] "
            r"for inputs int32\[2, 3\]$",
        )
        self.assertEqual(
            out["test_symbolic_dynamic"], "skipped: symbolic_static failed"
        )
        self.assertEqual(
            out["test_symbolic_partial"], "skipped: symbolic_static failed"
        )

        class EagerCase(Case):
            op = function_form(WrongRankAndValues())
            op_class = WrongRankAndValues

        out = outcomes(EagerCase)
        self.assertRegex(out["test_eager"], "^failed: ")
        self.assertEqual(out["test_eager_class"], "skipped: eager failed")
        self.assertRegex(out["test_symbolic_static"], "^failed: ")
        self.assertEqual(
            out["test_symbolic_dynamic"], "skipped: symbolic_static failed"
        )

    def test_unknown_dims_checks(self):
        class Case(testing.OpTest):
            op, reference_fn = function_form(NeedsStaticShape()), np.square
            inputs = (X,)

        out = outcomes(Case)
        self.assertEqual(out["test_symbolic_static"], "passed")
        # The failing symbolic shapes are added to the error.
        self.assertEqual(
            out["test_symbolic_dynamic"],
            "failed: Symbolic input shapes: [(None, None)]",
        )
        self.assertEqual(
            out["test_symbolic_partial"],
            "failed: Symbolic input shapes: [(None, 3)]",
        )

    def test_unknown_dim_shapes(self):
        shapes = [(2, 3), (2,)]
        self.assertEqual(
            test_case._all_dims_unknown(shapes), [(None, None), (None,)]
        )
        self.assertEqual(
            test_case._one_dim_unknown(shapes),
            [
                [(None, 3), (2,)],
                [(2, None), (2,)],
                [(2, 3), (None,)],
            ],
        )

    def test_output_count_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError, "output count differs"):
            self.run_op_eager_check(
                lambda x: ops.stack([x, x]), lambda x: (x, x), (X,), {}
            )
        with self.assertRaisesRegex(AssertionError, "Output differs"):
            self.run_op_symbolic_check(  # Two eager outputs, one symbolic.
                lambda x: x if isinstance(x, KerasTensor) else (x, x), (X,), {}
            )

    def test_symbolic_dtype_mismatch_fails(self):
        with self.assertRaisesRegex(
            AssertionError, "Output differs: symbolic=float32"
        ):
            self.run_op_symbolic_check(WrongDtype(), (X,), {})

    def test_unknown_output_dims_are_accepted(self):
        self.run_op_symbolic_check(ops.square, (X,), {}, [(None, 3)])
        self.run_op_symbolic_check(ops.add, (X, X[:1]), {}, [(None, 3)] * 2)

    def test_presets(self):
        for preset in (
            testing.UnaryOpTest,
            testing.BinaryOpTest,
            testing.ReductionOpTest,
        ):
            self.assertEqual(methods_of(preset), [])

        class Unary(testing.UnaryOpTest):
            op, reference_fn, jax_fn = ops.square, np.square, jnp.square

        class Binary(testing.BinaryOpTest):
            op, reference_fn, jax_fn = ops.add, np.add, jnp.add

        class Reduction(testing.ReductionOpTest):
            op, reference_fn, jax_fn = ops.sum, np.sum, jnp.sum

        self.assertIn("test_symbolic_partial", methods_of(Unary))
        self.assertIn("test_symbolic_partial_broadcast", methods_of(Binary))
        self.assertIn("test_python_types_int8_float", methods_of(Binary))
        reduction = methods_of(Reduction)
        self.assertIn("test_symbolic_partial_axis1_keepdims", reduction)
        self.assertLen([n for n in reduction if n.startswith("test_eager")], 10)
        failed = {k: v for k, v in outcomes(Unary).items() if "failed" in v}
        self.assertEqual(failed, {})


class RunOpInputDtypesCheckTest(testing.TestCase):
    def test_dtype_mismatch_vs_jax_fails(self):
        def op(x):
            return ops.cast(ops.square(x), "int32")

        with self.assertRaisesRegex(
            AssertionError,
            r"Output dtype differs from JAX for inputs uint8\[2, 3\]",
        ):
            self.run_op_input_dtypes_check(op, jnp.square, (X,), {}, ("uint8",))

    def test_skipped_dtype_has_reason(self):
        skipped = {backend.backend(): {"int8": "no int8 here"}}
        with mock.patch.dict(test_case._SKIPPED_DTYPES, skipped):
            with self.assertRaisesRegex(unittest.SkipTest, "int8: no int8"):
                self.run_op_input_dtypes_check(
                    ops.square, jnp.square, (X,), {}, ("int8",)
                )

    def test_dtype_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.run_op_input_dtypes_check(
                ops.add, jnp.add, (X, X), {}, ("int8",)
            )

    def test_python_scalar_input(self):
        self.run_op_input_dtypes_check(ops.add, jnp.add, (X, 1), {}, ("int8",))
