import numpy as np
import pytest
from absl.testing import parameterized

from keras import backend
from keras import layers
from keras import ops
from keras import testing


class Cropping2DTest(testing.TestCase, parameterized.TestCase):
    @parameterized.product(
        (
            # different cropping values
            {"cropping": ((1, 2), (3, 4)), "expected_ranges": ((1, 5), (3, 5))},
            # same cropping values with 2 tuples
            {"cropping": ((2, 2), (2, 2)), "expected_ranges": ((2, 5), (2, 7))},
            # same cropping values with 1 tuple
            {"cropping": (2, 2), "expected_ranges": ((2, 5), (2, 7))},
            # same cropping values with an integer
            {"cropping": 2, "expected_ranges": ((2, 5), (2, 7))},
            # cropping right only in both dimensions
            {"cropping": ((0, 2), (0, 4)), "expected_ranges": ((0, 5), (0, 5))},
            # cropping left only in both dimensions
            {"cropping": ((1, 0), (3, 0)), "expected_ranges": ((1, 7), (3, 9))},
            # cropping left only in rows dimension
            {"cropping": ((1, 0), (3, 4)), "expected_ranges": ((1, 7), (3, 5))},
            # cropping left only in cols dimension
            {"cropping": ((1, 2), (3, 0)), "expected_ranges": ((1, 5), (3, 9))},
        ),
        (
            {"data_format": "channels_first"},
            {"data_format": "channels_last"},
        ),
    )
    @pytest.mark.requires_trainable_backend
    def test_cropping_2d(self, cropping, data_format, expected_ranges):
        if data_format == "channels_first":
            inputs = np.random.rand(3, 5, 7, 9)
            expected_output = ops.convert_to_tensor(
                inputs[
                    :,
                    :,
                    expected_ranges[0][0] : expected_ranges[0][1],
                    expected_ranges[1][0] : expected_ranges[1][1],
                ]
            )
        else:
            inputs = np.random.rand(3, 7, 9, 5)
            expected_output = ops.convert_to_tensor(
                inputs[
                    :,
                    expected_ranges[0][0] : expected_ranges[0][1],
                    expected_ranges[1][0] : expected_ranges[1][1],
                    :,
                ]
            )

        self.run_layer_test(
            layers.Cropping2D,
            init_kwargs={"cropping": cropping, "data_format": data_format},
            input_data=inputs,
            expected_output=expected_output,
        )

    def test_cropping_2d_with_dynamic_spatial_dim(self):
        if backend.config.image_data_format() == "channels_last":
            input_layer = layers.Input(batch_shape=(1, 7, None, 5))
        else:
            input_layer = layers.Input(batch_shape=(1, 5, 7, None))
        cropped = layers.Cropping2D(((1, 2), (3, 4)))(input_layer)
        if backend.config.image_data_format() == "channels_last":
            self.assertEqual(cropped.shape, (1, 4, None, 5))
        else:
            self.assertEqual(cropped.shape, (1, 5, 4, None))

    @parameterized.product(
        (
            {"cropping": ((3, 6), (0, 0))},
            {"cropping": ((0, 0), (5, 4))},
        ),
        (
            {"data_format": "channels_first"},
            {"data_format": "channels_last"},
        ),
    )
    def test_cropping_2d_errors_if_cropping_more_than_available(
        self, cropping, data_format
    ):
        input_layer = layers.Input(batch_shape=(3, 7, 9, 5))
        with self.assertRaises(ValueError):
            layers.Cropping2D(cropping=cropping, data_format=data_format)(
                input_layer
            )

    def test_cropping_2d_errors_if_cropping_argument_invalid(self):
        with self.assertRaises(ValueError):
            layers.Cropping2D(cropping=(1,))
        with self.assertRaises(ValueError):
            layers.Cropping2D(cropping=(1, 2, 3))
        with self.assertRaises(ValueError):
            layers.Cropping2D(cropping="1")
        with self.assertRaises(ValueError):
            layers.Cropping2D(cropping=((1, 2), (3, 4, 5)))
        with self.assertRaises(ValueError):
            layers.Cropping2D(cropping=((1, 2), (3, -4)))
        with self.assertRaises(ValueError):
            layers.Cropping2D(cropping=((1, 2), "3"))

    @parameterized.product(
        (
            {"cropping": ((4, 5), (0, 0)), "input_shape": (3, 8, 9, 5)},
            {"cropping": ((0, 0), (5, 5)), "input_shape": (3, 8, 9, 5)},
            {"cropping": ((6, 3), (0, 0)), "input_shape": (3, 8, 9, 5)},
            {"cropping": ((0, 0), (7, 3)), "input_shape": (3, 8, 9, 5)},
        ),
        (
            {"data_format": "channels_first"},
            {"data_format": "channels_last"},
        ),
    )
    def test_cropping_2d_error_on_excessive_cropping(
        self, cropping, input_shape, data_format
    ):
        inputs = np.random.rand(*input_shape)

        with self.assertRaisesRegex(
            ValueError,
            "Values in `cropping` argument should be smaller than the "
            "corresponding spatial dimension of the input.",
        ):
            layer = layers.Cropping2D(
                cropping=cropping, data_format=data_format
            )
            _ = layer(inputs)
