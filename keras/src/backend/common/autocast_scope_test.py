from absl.testing import parameterized

from keras.src import ops
from keras.src import testing
from keras.src.backend.common.autocast_scope import AutocastScope
from keras.src.backend.common.autocast_scope import get_autocast_scope


class AutocastScopeTest(testing.TestCase):
    def test_init_with_float_dtype(self):
        scope = AutocastScope("bfloat16")
        self.assertEqual(scope.dtype, "bfloat16")
        self.assertIsNone(scope.original_scope)

    def test_init_with_none_dtype(self):
        scope = AutocastScope(None)
        self.assertIsNone(scope.dtype)
        self.assertIsNone(scope.original_scope)

    def test_autocast_scope_with_non_float_dtype(self):
        with self.assertRaisesRegex(ValueError, "floating-point"):
            AutocastScope("int32")

    def test_init_with_invalid_dtype(self):
        with self.assertRaisesRegex(ValueError, "Invalid dtype"):
            AutocastScope("invalid_dtype")

    def test_get_autocast_scope_default(self):
        self.assertIsNone(get_autocast_scope())

    def test_context_manager_sets_and_restores_scope(self):
        self.assertIsNone(get_autocast_scope())
        scope = AutocastScope("float16")
        with scope:
            self.assertIs(get_autocast_scope(), scope)
            self.assertEqual(get_autocast_scope().dtype, "float16")
            self.assertIsNone(scope.original_scope)
        self.assertIsNone(get_autocast_scope())

    def test_nested_scopes(self):
        inner = AutocastScope("float16")
        outer = AutocastScope("bfloat16")
        with outer:
            with inner:
                self.assertIs(get_autocast_scope(), inner)
                self.assertIs(inner.original_scope, outer)
            self.assertIs(get_autocast_scope(), outer)
        self.assertIsNone(get_autocast_scope())

    def test_restores_scope_on_exception(self):
        outer = AutocastScope("float32")
        with outer:
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with AutocastScope("float16"):
                    self.assertEqual(get_autocast_scope().dtype, "float16")
                    raise RuntimeError("nested")
            self.assertIs(get_autocast_scope(), outer)
        self.assertIsNone(get_autocast_scope())

    @parameterized.named_parameters(
        ("float32_to_float16", "float32", "float16"),
        ("float32_to_bfloat16", "float32", "bfloat16"),
        ("float16_to_float32", "float16", "float32"),
    )
    def test_maybe_cast_float_tensor(self, source_dtype, target_dtype):
        scope = AutocastScope(target_dtype)
        value = ops.convert_to_tensor([1.5, -2.5], dtype=source_dtype)
        casted = scope.maybe_cast(value)
        self.assertDType(casted, target_dtype)
        self.assertAllClose(casted, [1.5, -2.5])

    def test_maybe_cast_non_float_tensor_unchanged(self):
        scope = AutocastScope("float16")
        value = ops.convert_to_tensor([1, 2], dtype="int32")
        result = scope.maybe_cast(value)
        self.assertIs(result, value)
        self.assertDType(result, "int32")

    def test_maybe_cast_none_scope_unchanged(self):
        scope = AutocastScope(None)
        value = ops.convert_to_tensor([1.5, -2.5], dtype="float32")
        result = scope.maybe_cast(value)
        self.assertIs(result, value)
        self.assertDType(result, "float32")
