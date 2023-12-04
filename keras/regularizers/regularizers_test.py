import numpy as np

from keras import backend
from keras import regularizers
from keras import testing
from keras.regularizers.regularizers import validate_float_arg


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
        value = np.random.random((4, 4))
        x = backend.Variable(value)
        y = regularizers.L1(0.1)(x)
        self.assertAllClose(y, 0.1 * np.sum(np.abs(value)))

    def test_l2(self):
        value = np.random.random((4, 4))
        x = backend.Variable(value)
        y = regularizers.L2(0.1)(x)
        self.assertAllClose(y, 0.1 * np.sum(np.square(value)))

    def test_l1_l2(self):
        value = np.random.random((4, 4))
        x = backend.Variable(value)
        y = regularizers.L1L2(l1=0.1, l2=0.2)(x)
        self.assertAllClose(
            y, 0.1 * np.sum(np.abs(value)) + 0.2 * np.sum(np.square(value))
        )

    def test_orthogonal_regularizer(self):
        value = np.random.random((4, 4))
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
            value = np.random.random((4, 4, 4))
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
