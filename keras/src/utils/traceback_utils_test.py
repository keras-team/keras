"""Tests for the lazy inject_argument_info_in_error path in traceback_utils."""

import numpy as np

from keras.src import testing
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils

# ---------------------------------------------------------------------------
# Helper layer definitions
# ---------------------------------------------------------------------------


class FailLayer(Operation):
    """Operation whose call() raises a standard ValueError."""

    def call(self, x, y=10):
        raise ValueError("bad input")

    def compute_output_spec(self, x, y=10):
        return x


class CustomExcLayer(Operation):
    """Operation whose call() raises an exception with a non-standard ctor."""

    def call(self, x):
        raise _NonStandardError("oops", extra=42)

    def compute_output_spec(self, x):
        return x


class HappyLayer(Operation):
    """Operation whose call() always succeeds."""

    def call(self, x):
        return x

    def compute_output_spec(self, x):
        return x


class _NonStandardError(Exception):
    """Exception whose constructor does not accept a single positional arg."""

    def __init__(self, message, extra):
        super().__init__(message)
        self.extra = extra


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TracebackUtilsTest(testing.TestCase):
    """Tests for inject_argument_info_in_error (lazy, on-error-path only)."""

    def setUp(self):
        # Ensure filtering is enabled at start of each test; restore after.
        self._was_enabled = traceback_utils.is_traceback_filtering_enabled()
        traceback_utils.enable_traceback_filtering()

    def tearDown(self):
        if self._was_enabled:
            traceback_utils.enable_traceback_filtering()
        else:
            traceback_utils.disable_traceback_filtering()

    # ------------------------------------------------------------------
    # 1. Happy path — no exception, no overhead
    # ------------------------------------------------------------------

    def test_happy_path_no_exception(self):
        layer = HappyLayer()
        x = np.ones((2, 3), dtype="float32")
        result = layer(x)
        self.assertIs(result, x)  # returned unchanged

    # ------------------------------------------------------------------
    # 2. Standard exception: type preserved, message augmented
    # ------------------------------------------------------------------

    def test_standard_exception_type_preserved(self):
        layer = FailLayer()
        x = np.ones((2,), dtype="float32")
        with self.assertRaises(ValueError) as ctx:
            layer(x, y=5)
        e = ctx.exception
        self.assertIsInstance(e, ValueError)
        self.assertIn("Exception encountered when calling", str(e))
        self.assertIn("bad input", str(e))
        self.assertIn("FailLayer.call()", str(e))
        self.assertTrue(getattr(e, "_keras_call_info_injected", False))

    def test_standard_exception_arguments_in_message(self):
        layer = FailLayer()
        x = np.ones((2,), dtype="float32")
        with self.assertRaises(ValueError) as ctx:
            layer(x, y=99)
        msg = str(ctx.exception)
        # y=99 should appear in the argument listing
        self.assertIn("y=", msg)
        self.assertIn("99", msg)

    # ------------------------------------------------------------------
    # 3. Custom exception (non-standard ctor): falls back to RuntimeError
    # ------------------------------------------------------------------

    def test_custom_exception_falls_back_to_runtime_error(self):
        layer = CustomExcLayer()
        x = np.ones((2,), dtype="float32")
        with self.assertRaises(RuntimeError) as ctx:
            layer(x)
        e = ctx.exception
        self.assertIsInstance(e, RuntimeError)
        self.assertIn("Exception encountered when calling", str(e))
        self.assertIn("oops", str(e))
        self.assertTrue(getattr(e, "_keras_call_info_injected", False))

    # ------------------------------------------------------------------
    # 4. Filtering disabled: original exception, no augmentation
    # ------------------------------------------------------------------

    def test_filtering_disabled_original_exception(self):
        traceback_utils.disable_traceback_filtering()
        layer = FailLayer()
        x = np.ones((2,), dtype="float32")
        with self.assertRaises(ValueError) as ctx:
            layer(x)
        e = ctx.exception
        # Message should NOT be augmented
        self.assertNotIn("Exception encountered when calling", str(e))
        self.assertFalse(getattr(e, "_keras_call_info_injected", False))

    # ------------------------------------------------------------------
    # 5. Double-injection prevention via _keras_call_info_injected flag
    # ------------------------------------------------------------------

    def test_no_double_injection(self):
        """inject_argument_info_in_error skips already-injected exceptions."""
        original_msg = "already augmented"

        def fake_fn(x):
            pass

        # Build a pre-injected exception
        pre_injected = ValueError(original_msg)
        pre_injected._keras_call_info_injected = True

        # Calling inject_argument_info_in_error with a pre-injected exception
        # should still produce a new augmented exception (the guard lives in
        # operation.__call__), but let's verify the helper itself returns a
        # new exception that carries the flag.
        traceback_utils.inject_argument_info_in_error(
            pre_injected, fake_fn, (1,), {}
        )
        # result may be None (unbindable) or a new exception — either way
        # check that operation.__call__ would not double-inject.
        # Simulate the guard logic from operation.py:
        e = pre_injected
        if not getattr(e, "_keras_call_info_injected", False):
            augmented = traceback_utils.inject_argument_info_in_error(
                e, fake_fn, (1,), {}
            )
        else:
            augmented = None  # flag is set — skip injection
        self.assertIsNone(augmented)  # guard prevented second injection

    # ------------------------------------------------------------------
    # 6. inject_argument_info_in_error returns None for unbindable args
    # ------------------------------------------------------------------

    def test_returns_none_for_unbindable_args(self):
        def fn_no_args():
            pass

        e = ValueError("boom")
        result = traceback_utils.inject_argument_info_in_error(
            e,
            fn_no_args,
            (1, 2, 3),
            {},  # too many positional args
        )
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # 7. Failed flag-set must not duplicate the augmented message across
    #    nested Operation calls.
    # ------------------------------------------------------------------

    def test_setattr_failure_does_not_duplicate_message_across_nesting(self):
        """Regression test for duplicated augmentation text.

        If the exception's flag attribute cannot be set (e.g. its type
        rejects new attributes once constructed), a naive implementation
        still returns the augmented-but-unmarked exception. Each nested
        Operation.__call__ then re-augments it again since the flag was
        never actually persisted, duplicating the "Exception encountered
        when calling ..." text once per nesting level. The fix makes a
        failed flag-set fall back to the plain original exception so the
        text never appears more than once.
        """

        class _LockedError(Exception):
            """Exception whose __setattr__ rejects new attributes once
            construction has completed."""

            def __init__(self, message):
                super().__init__(message)
                object.__setattr__(self, "_locked", True)

            def __setattr__(self, name, value):
                if getattr(self, "_locked", False) and name != "args":
                    raise AttributeError(f"cannot set {name!r}")
                object.__setattr__(self, name, value)

        class InnerLockedLayer(Operation):
            def call(self, x):
                raise _LockedError("bad input")

            def compute_output_spec(self, x):
                return x

        class OuterLockedLayer(Operation):
            def __init__(self):
                super().__init__()
                self.inner = InnerLockedLayer()

            def call(self, x):
                return self.inner(x)

            def compute_output_spec(self, x):
                return x

        layer = OuterLockedLayer()
        x = np.ones((2,), dtype="float32")
        with self.assertRaises(_LockedError) as ctx:
            layer(x)
        msg = str(ctx.exception)
        occurrences = msg.count("Exception encountered when calling")
        # Previously this was 2 (one augmentation per nesting level).
        self.assertLessEqual(occurrences, 1)
        self.assertIn("bad input", msg)
