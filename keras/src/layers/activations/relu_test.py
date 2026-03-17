import numpy as np
import pytest

from keras.src import testing
from keras.src.layers.activations import relu


class ReLUTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_relu(self):
        self.run_layer_test(
            relu.ReLU,
            init_kwargs={
                "max_value": 10,
                "negative_slope": 1,
                "threshold": 0.5,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
            assert_built_after_instantiation=True,
        )

    def test_normal_relu_correctness(self):
        relu_layer = relu.ReLU(max_value=10, negative_slope=0.0, threshold=0)
        input = np.array([-10, -5, 0.0, 5, 10])
        expected_output = np.array([0.0, 0.0, 0.0, 5.0, 10.0])
        result = relu_layer(input)
        self.assertAllClose(result, expected_output)

    def test_leaky_relu_correctness(self):
        relu_layer = relu.ReLU(max_value=10, negative_slope=0.5, threshold=0)
        input = np.array([-10, -5, 0.0, 5, 10])
        expected_output = np.array([-5.0, -2.5, 0.0, 5.0, 10.0])
        result = relu_layer(input)
        self.assertAllClose(result, expected_output)

    def test_threshold_relu_correctness(self):
        relu_layer = relu.ReLU(max_value=8, negative_slope=0.0, threshold=5)
        input = np.array([6.0, 7.0, 0.0, 5, 10])
        expected_output = np.array([6.0, 7.0, 0.0, 0.0, 8.0])
        result = relu_layer(input)
        self.assertAllClose(result, expected_output)

    def test_invalid_usage(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_value of a ReLU layer cannot be a negative value",
        ):
            self.run_layer_test(
                relu.ReLU,
                init_kwargs={
                    "max_value": -10,
                    "negative_slope": 1,
                    "threshold": 0.5,
                },
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "negative_slope of a ReLU layer cannot be None or a negative value",
        ):
            self.run_layer_test(
                relu.ReLU,
                init_kwargs={
                    "max_value": 10,
                    "negative_slope": -10,
                    "threshold": 0.5,
                },
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "threshold of a ReLU layer cannot be None or a negative value",
        ):
            self.run_layer_test(
                relu.ReLU,
                init_kwargs={
                    "max_value": 10,
                    "negative_slope": 1,
                    "threshold": -10,
                },
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

    def test_get_config(self):
        # Test that get_config returns correct values
        relu_layer = relu.ReLU(
            max_value=5.0, negative_slope=0.2, threshold=1.0
        )
        config = relu_layer.get_config()
        self.assertEqual(config["max_value"], 5.0)
        self.assertEqual(config["negative_slope"], 0.2)
        self.assertEqual(config["threshold"], 1.0)

    def test_get_config_default_params(self):
        # Test get_config with default parameters
        relu_layer = relu.ReLU()
        config = relu_layer.get_config()
        self.assertIsNone(config["max_value"])
        self.assertEqual(config["negative_slope"], 0.0)
        self.assertEqual(config["threshold"], 0.0)

    def test_from_config(self):
        # Test that a layer can be recreated from its config
        relu_layer = relu.ReLU(
            max_value=5.0, negative_slope=0.2, threshold=1.0
        )
        config = relu_layer.get_config()
        restored_layer = relu.ReLU.from_config(config)
        self.assertEqual(restored_layer.max_value, relu_layer.max_value)
        self.assertEqual(
            restored_layer.negative_slope, relu_layer.negative_slope
        )
        self.assertEqual(restored_layer.threshold, relu_layer.threshold)

    def test_compute_output_shape(self):
        # Test that output shape equals input shape for various shapes
        relu_layer = relu.ReLU()
        self.assertEqual(relu_layer.compute_output_shape((2, 3)), (2, 3))
        self.assertEqual(
            relu_layer.compute_output_shape((4, 5, 6)), (4, 5, 6)
        )
        self.assertEqual(relu_layer.compute_output_shape((1,)), (1,))

    def test_default_relu_no_args(self):
        # Test standard ReLU with all default parameters
        relu_layer = relu.ReLU()
        input = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected_output = np.array([0.0, 0.0, 0.0, 1.0, 3.0])
        result = relu_layer(input)
        self.assertAllClose(result, expected_output)

    def test_relu_none_max_value_is_unlimited(self):
        # Test that max_value=None means no upper cap on activation
        relu_layer = relu.ReLU(max_value=None)
        input = np.array([0.0, 100.0, 1000.0, 99999.0])
        result = relu_layer(input)
        self.assertAllClose(result, input)

    def test_relu_all_zeros_input(self):
        # Edge case: all-zeros input should produce all-zeros output
        relu_layer = relu.ReLU()
        input = np.zeros((3, 4))
        result = relu_layer(input)
        self.assertAllClose(result, np.zeros((3, 4)))

    def test_relu_multidimensional_input(self):
        # Test ReLU on 2D and 3D inputs
        relu_layer = relu.ReLU()
        input_2d = np.array([[-1.0, 2.0], [3.0, -4.0]])
        expected_2d = np.array([[0.0, 2.0], [3.0, 0.0]])
        self.assertAllClose(relu_layer(input_2d), expected_2d)

        input_3d = np.ones((2, 3, 4)) * -1.0
        expected_3d = np.zeros((2, 3, 4))
        self.assertAllClose(relu_layer(input_3d), expected_3d)

    def test_negative_slope_none_raises(self):
        # negative_slope=None should raise a ValueError with clear message
        with self.assertRaisesRegex(
            ValueError,
            "negative_slope of a ReLU layer cannot be None or a negative value",
        ):
            relu.ReLU(negative_slope=None)

    def test_threshold_none_raises(self):
        # threshold=None should raise a ValueError with clear message
        with self.assertRaisesRegex(
            ValueError,
            "threshold of a ReLU layer cannot be None or a negative value",
        ):
            relu.ReLU(threshold=None)