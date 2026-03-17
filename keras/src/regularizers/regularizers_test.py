import numpy as np

from keras.src import backend, regularizers, testing
from keras.src.regularizers.regularizers import validate_float_arg


class RegularizersTest(testing.TestCase):
    def test_config(self):
        reg = regularizers.L1(0.1)
        self.run_class_serialization_test(reg)

        reg = regularizers.L2(0.1)
        self.run_class_serialization_test(reg)

        reg = regularizers.L1L2(l1=0.1, l2=0.2)
        self.run_class_serialization_test(reg)

        reg = regularizers.OrthogonalRegularizer(factor=0.1, mode="rows")
        self.run_class_serialization_test(reg)

    def test_l1(self):
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y = regularizers.L1(0.1)(x)
        self.assertAllClose(y, 0.1 * np.sum(np.abs(value)))

    def test_l2(self):
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y = regularizers.L2(0.1)(x)
        self.assertAllClose(y, 0.1 * np.sum(np.square(value)))

    def test_l1_l2(self):
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y = regularizers.L1L2(l1=0.1, l2=0.2)(x)
        self.assertAllClose(
            y, 0.1 * np.sum(np.abs(value)) + 0.2 * np.sum(np.square(value))
        )

    def test_orthogonal_regularizer(self):
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y = regularizers.OrthogonalRegularizer(factor=0.1, mode="rows")(x)

        l2_norm = np.linalg.norm(value, axis=1, keepdims=True)
        inputs = value / l2_norm
        self.assertAllClose(
            y,
            0.1
            * 0.5
            * np.sum(
                np.abs(np.dot(inputs, np.transpose(inputs)) * (1.0 - np.eye(4)))
            )
            / (4.0 * (4.0 - 1.0) / 2.0),
            tpu_atol=1e-4,
            tpu_rtol=1e-4,
        )

    def test_get_method(self):
        obj = regularizers.get("l1l2")
        self.assertIsInstance(obj, regularizers.L1L2)

        obj = regularizers.get("l1")
        self.assertIsInstance(obj, regularizers.L1)

        obj = regularizers.get("l2")
        self.assertIsInstance(obj, regularizers.L2)

        obj = regularizers.get("orthogonal_regularizer")
        self.assertIsInstance(obj, regularizers.OrthogonalRegularizer)

        obj = regularizers.get(None)
        self.assertEqual(obj, None)

        with self.assertRaises(ValueError):
            regularizers.get("typo")

    def test_l1l2_get_config(self):
        l1 = 0.01
        l2 = 0.02
        reg = regularizers.L1L2(l1=l1, l2=l2)
        config = reg.get_config()

        self.assertEqual(config, {"l1": l1, "l2": l2})

        reg_from_config = regularizers.L1L2.from_config(config)
        config_from_config = reg_from_config.get_config()

        self.assertDictEqual(config, config_from_config)
        self.assertEqual(reg_from_config.l1, l1)
        self.assertEqual(reg_from_config.l2, l2)

    def test_orthogonal_regularizer_mode_validation(self):
        with self.assertRaises(ValueError) as context:
            regularizers.OrthogonalRegularizer(factor=0.01, mode="invalid_mode")

        expected_message = (
            'Invalid value for argument `mode`. Expected one of {"rows", '
            '"columns"}. Received: mode=invalid_mode'
        )
        self.assertEqual(str(context.exception), expected_message)

    def test_orthogonal_regularizer_input_rank_validation(self):
        with self.assertRaises(ValueError) as context:
            value = np.random.random((4, 4, 4)).astype(np.float32)
            x = backend.Variable(value)
            regularizers.OrthogonalRegularizer(factor=0.1)(x)

        expected_message = (
            "Inputs to OrthogonalRegularizer must have rank 2. "
            f"Received: inputs.shape={(4, 4, 4)}"
        )
        self.assertEqual(str(context.exception), expected_message)

    def test_orthogonal_regularizer_get_config(self):
        factor = 0.01
        mode = "columns"
        regularizer = regularizers.OrthogonalRegularizer(
            factor=factor, mode=mode
        )
        config = regularizer.get_config()

        self.assertAlmostEqual(config["factor"], factor, 7)
        self.assertEqual(config["mode"], mode)

        reg_from_config = regularizers.OrthogonalRegularizer.from_config(config)
        config_from_config = reg_from_config.get_config()

        self.assertAlmostEqual(config_from_config["factor"], factor, 7)
        self.assertEqual(config_from_config["mode"], mode)

    # ------------------------------------------------------------------ #
    # NEW: L1 edge cases                                                  #
    # ------------------------------------------------------------------ #

    def test_l1_with_zero_weights(self):
        # L1 regularization of all-zero weights must be exactly 0
        value = np.zeros((4, 4), dtype=np.float32)
        x = backend.Variable(value)
        y = regularizers.L1(0.1)(x)
        self.assertAllClose(y, 0.0)

    def test_l1_with_negative_weights(self):
        # L1 uses absolute values — sign must not affect the result
        pos_value = np.ones((4, 4), dtype=np.float32)
        neg_value = -np.ones((4, 4), dtype=np.float32)
        x_pos = backend.Variable(pos_value)
        x_neg = backend.Variable(neg_value)
        y_pos = regularizers.L1(0.1)(x_pos)
        y_neg = regularizers.L1(0.1)(x_neg)
        self.assertAllClose(y_pos, y_neg)

    def test_l1_zero_coefficient(self):
        # L1 with coefficient 0.0 should always return 0 regardless of input
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y = regularizers.L1(0.0)(x)
        self.assertAllClose(y, 0.0)

    # ------------------------------------------------------------------ #
    # NEW: L2 edge cases                                                  #
    # ------------------------------------------------------------------ #

    def test_l2_with_zero_weights(self):
        # L2 regularization of all-zero weights must be exactly 0
        value = np.zeros((4, 4), dtype=np.float32)
        x = backend.Variable(value)
        y = regularizers.L2(0.1)(x)
        self.assertAllClose(y, 0.0)

    def test_l2_zero_coefficient(self):
        # L2 with coefficient 0.0 should always return 0 regardless of input
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y = regularizers.L2(0.0)(x)
        self.assertAllClose(y, 0.0)

    def test_l2_is_larger_than_l1_for_large_weights(self):
        # For weights > 1, L2 penalty grows faster than L1
        value = np.ones((4, 4), dtype=np.float32) * 10.0
        x = backend.Variable(value)
        l1_val = float(backend.convert_to_numpy(regularizers.L1(1.0)(x)))
        l2_val = float(backend.convert_to_numpy(regularizers.L2(1.0)(x)))
        self.assertGreater(l2_val, l1_val)

    def test_l2_is_smaller_than_l1_for_small_weights(self):
        # For weights < 1, L2 penalty grows slower than L1
        value = np.ones((4, 4), dtype=np.float32) * 0.1
        x = backend.Variable(value)
        l1_val = float(backend.convert_to_numpy(regularizers.L1(1.0)(x)))
        l2_val = float(backend.convert_to_numpy(regularizers.L2(1.0)(x)))
        self.assertLess(l2_val, l1_val)

    # ------------------------------------------------------------------ #
    # NEW: L1L2 edge cases                                                #
    # ------------------------------------------------------------------ #

    def test_l1l2_zero_coefficients(self):
        # L1L2 with both l1=0 and l2=0 should always return exactly 0
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y = regularizers.L1L2(l1=0.0, l2=0.0)(x)
        self.assertAllClose(y, 0.0)

    def test_l1l2_only_l1(self):
        # L1L2 with l2=0 should equal pure L1
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y_l1l2 = regularizers.L1L2(l1=0.1, l2=0.0)(x)
        y_l1 = regularizers.L1(0.1)(x)
        self.assertAllClose(y_l1l2, y_l1)

    def test_l1l2_only_l2(self):
        # L1L2 with l1=0 should equal pure L2
        value = np.random.random((4, 4)).astype(np.float32)
        x = backend.Variable(value)
        y_l1l2 = regularizers.L1L2(l1=0.0, l2=0.1)(x)
        y_l2 = regularizers.L2(0.1)(x)
        self.assertAllClose(y_l1l2, y_l2)

    def test_l1l2_with_zero_weights(self):
        # L1L2 of all-zero weights must always be 0
        value = np.zeros((4, 4), dtype=np.float32)
        x = backend.Variable(value)
        y = regularizers.L1L2(l1=0.1, l2=0.2)(x)
        self.assertAllClose(y, 0.0)


class ValidateFloatArgTest(testing.TestCase):
    def test_validate_float_with_valid_args(self):
        self.assertEqual(validate_float_arg(1, "test"), 1.0)
        self.assertEqual(validate_float_arg(1.0, "test"), 1.0)

    def test_validate_float_with_invalid_types(self):
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg("not_a_number", "test")

    def test_validate_float_with_nan(self):
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg(float("nan"), "test")

    def test_validate_float_with_inf(self):
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg(float("inf"), "test")
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg(-float("inf"), "test")

    def test_validate_float_with_negative_number(self):
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg(-1, "test")

    # ------------------------------------------------------------------ #
    # NEW: validate_float_arg boundary & type tests                       #
    # ------------------------------------------------------------------ #

    def test_validate_float_with_zero(self):
        # Zero is a valid non-negative float — both int 0 and float 0.0
        self.assertEqual(validate_float_arg(0, "test"), 0.0)
        self.assertEqual(validate_float_arg(0.0, "test"), 0.0)

    def test_validate_float_with_very_small_positive(self):
        # Very small positive floats near machine epsilon should be valid
        result = validate_float_arg(1e-10, "test")
        self.assertAlmostEqual(result, 1e-10)

    def test_validate_float_returns_float_type(self):
        # Result must always be a Python float regardless of input type
        result = validate_float_arg(1, "test")
        self.assertIsInstance(result, float)

        result = validate_float_arg(1.0, "test")
        self.assertIsInstance(result, float)

    def test_validate_float_with_large_positive(self):
        # Large positive numbers should be valid (not rejected as inf)
        result = validate_float_arg(1e6, "test")
        self.assertAlmostEqual(result, 1e6)

    def test_validate_float_with_boolean_true(self):
        # In Python, True == 1 and is a subclass of int, should be valid
        result = validate_float_arg(True, "test")
        self.assertEqual(result, 1.0)

    def test_validate_float_with_negative_float(self):
        # Negative float (not just int) should also raise
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg(-0.001, "test")

    def test_validate_float_with_none(self):
        # None is not a valid float argument
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg(None, "test")

    def test_validate_float_with_list(self):
        # Lists should not be valid float arguments
        with self.assertRaisesRegex(
            ValueError, "expected a non-negative float"
        ):
            validate_float_arg([0.1], "test")