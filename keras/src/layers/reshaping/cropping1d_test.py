import numpy as np
import pytest

from keras.src import layers
from keras.src import ops
from keras.src import testing


class Cropping1DTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_cropping_1d(self):
        inputs = np.random.rand(3, 5, 7)

        # Cropping with different values on the left and the right.
        self.run_layer_test(
            layers.Cropping1D,
            init_kwargs={"cropping": (1, 2)},
            input_data=inputs,
            expected_output=ops.convert_to_tensor(inputs[:, 1:3, :]),
        )
        # Same cropping on the left and the right.
        self.run_layer_test(
            layers.Cropping1D,
            init_kwargs={"cropping": (1, 1)},
            input_data=inputs,
            expected_output=ops.convert_to_tensor(inputs[:, 1:4, :]),
        )
        # Same cropping on the left and the right provided as an int.
        self.run_layer_test(
            layers.Cropping1D,
            init_kwargs={"cropping": 1},
            input_data=inputs,
            expected_output=ops.convert_to_tensor(inputs[:, 1:4, :]),
        )
        # Cropping on the right only.
        self.run_layer_test(
            layers.Cropping1D,
            init_kwargs={"cropping": (0, 1)},
            input_data=inputs,
            expected_output=ops.convert_to_tensor(inputs[:, 0:4, :]),
        )
        # Cropping on the left only.
        self.run_layer_test(
            layers.Cropping1D,
            init_kwargs={"cropping": (1, 0)},
            input_data=inputs,
            expected_output=ops.convert_to_tensor(inputs[:, 1:5, :]),
        )

    @pytest.mark.requires_trainable_backend
    def test_cropping_1d_with_dynamic_spatial_dim(self):
        input_layer = layers.Input(batch_shape=(1, None, 7))
        cropped = layers.Cropping1D((1, 2))(input_layer)
        self.assertEqual(cropped.shape, (1, None, 7))

    def test_cropping_1d_errors_if_cropping_argument_invalid(self):
        with self.assertRaises(ValueError):
            layers.Cropping1D(cropping=(1,))
        with self.assertRaises(ValueError):
            layers.Cropping1D(cropping=(1, 2, 3))
        with self.assertRaises(ValueError):
            layers.Cropping1D(cropping="1")

    def test_cropping_1d_errors_if_cropping_more_than_available(self):
        with self.assertRaisesRegex(
            ValueError,
            "`cropping` parameter of `Cropping1D` layer must be smaller than",
        ):
            input_layer = layers.Input(batch_shape=(3, 5, 7))
            layers.Cropping1D(cropping=(2, 3))(input_layer)

    def test_cropping_1d_error_on_excessive_cropping(self):
        inputs = np.random.rand(3, 5, 7)

        with self.assertRaisesRegex(
            ValueError,
            "`cropping` parameter of `Cropping1D` layer must be smaller than",
        ):
            layer = layers.Cropping1D(cropping=(3, 3))
            _ = layer(inputs)
