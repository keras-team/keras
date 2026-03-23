from keras.src import testing
from keras.src.utils.argument_validation import standardize_padding
from keras.src.utils.argument_validation import standardize_tuple
from keras.src.utils.argument_validation import validate_string_arg


class StandardizeTupleTest(testing.TestCase):
    def test_int_input(self):
        result = standardize_tuple(3, 2, "kernel_size")
        self.assertEqual(result, (3, 3))

    def test_int_input_single(self):
        result = standardize_tuple(5, 1, "strides")
        self.assertEqual(result, (5,))

    def test_int_input_triple(self):
        result = standardize_tuple(2, 3, "kernel_size")
        self.assertEqual(result, (2, 2, 2))

    def test_tuple_input(self):
        result = standardize_tuple((3, 5), 2, "kernel_size")
        self.assertEqual(result, (3, 5))

    def test_list_input(self):
        result = standardize_tuple([1, 2, 3], 3, "strides")
        self.assertEqual(result, (1, 2, 3))

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            standardize_tuple((1, 2), 3, "kernel_size")

    def test_wrong_length_too_many(self):
        with self.assertRaises(ValueError):
            standardize_tuple((1, 2, 3), 2, "kernel_size")

    def test_non_iterable_raises(self):
        with self.assertRaises(ValueError):
            standardize_tuple(None, 2, "kernel_size")

    def test_negative_value_raises(self):
        with self.assertRaises(ValueError):
            standardize_tuple(-1, 2, "kernel_size")

    def test_zero_value_raises_by_default(self):
        with self.assertRaises(ValueError):
            standardize_tuple(0, 2, "kernel_size")

    def test_zero_value_allowed(self):
        result = standardize_tuple(0, 2, "padding", allow_zero=True)
        self.assertEqual(result, (0, 0))

    def test_negative_value_with_allow_zero_raises(self):
        with self.assertRaises(ValueError):
            standardize_tuple(-1, 2, "padding", allow_zero=True)

    def test_float_in_tuple_accepted(self):
        # float values are accepted since int() can convert them
        result = standardize_tuple((1.5, 2), 2, "kernel_size")
        self.assertEqual(result, (1.5, 2))

    def test_none_in_tuple_raises(self):
        with self.assertRaises(ValueError):
            standardize_tuple((None, 2), 2, "kernel_size")

    def test_mixed_valid_invalid_raises(self):
        with self.assertRaises(ValueError):
            standardize_tuple((1, -2), 2, "kernel_size")

    def test_large_tuple(self):
        result = standardize_tuple(1, 10, "padding")
        self.assertEqual(result, (1,) * 10)

    def test_error_message_contains_name(self):
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            standardize_tuple(None, 2, "kernel_size")


class StandardizePaddingTest(testing.TestCase):
    def test_valid_padding(self):
        self.assertEqual(standardize_padding("valid"), "valid")

    def test_same_padding(self):
        self.assertEqual(standardize_padding("same"), "same")

    def test_case_insensitive(self):
        self.assertEqual(standardize_padding("VALID"), "valid")
        self.assertEqual(standardize_padding("Same"), "same")

    def test_causal_not_allowed_by_default(self):
        with self.assertRaises(ValueError):
            standardize_padding("causal")

    def test_causal_allowed(self):
        self.assertEqual(
            standardize_padding("causal", allow_causal=True), "causal"
        )

    def test_invalid_padding_raises(self):
        with self.assertRaises(ValueError):
            standardize_padding("full")

    def test_list_passthrough(self):
        result = standardize_padding([(1, 1), (2, 2)])
        self.assertEqual(result, [(1, 1), (2, 2)])

    def test_tuple_passthrough(self):
        result = standardize_padding(((0, 1), (0, 1)))
        self.assertEqual(result, ((0, 1), (0, 1)))


class ValidateStringArgTest(testing.TestCase):
    def test_valid_string(self):
        # Should not raise
        validate_string_arg(
            "relu", {"relu", "sigmoid", "tanh"}, "Dense", "activation"
        )

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            validate_string_arg(
                "invalid", {"relu", "sigmoid"}, "Dense", "activation"
            )

    def test_none_not_allowed_by_default(self):
        with self.assertRaises(ValueError):
            validate_string_arg(
                None, {"relu", "sigmoid"}, "Dense", "activation"
            )

    def test_none_allowed(self):
        # Should not raise
        validate_string_arg(
            None, {"relu"}, "Dense", "activation", allow_none=True
        )

    def test_callable_not_allowed_by_default(self):
        with self.assertRaises(ValueError):
            validate_string_arg(lambda x: x, {"relu"}, "Dense", "activation")

    def test_callable_allowed(self):
        # Should not raise
        validate_string_arg(
            lambda x: x,
            {"relu"},
            "Dense",
            "activation",
            allow_callables=True,
        )

    def test_error_message_contains_arg_name(self):
        with self.assertRaisesRegex(ValueError, "activation"):
            validate_string_arg("bad", {"relu"}, "Dense", "activation")

    def test_error_message_contains_caller_name(self):
        with self.assertRaisesRegex(ValueError, "Dense"):
            validate_string_arg("bad", {"relu"}, "Dense", "activation")
