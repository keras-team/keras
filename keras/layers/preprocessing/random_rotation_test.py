import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import testing


class RandomRotationTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("random_rotate_neg4", -0.4),
        ("random_rotate_neg2", -0.2),
        ("random_rotate_4", 0.4),
        ("random_rotate_2", 0.2),
        ("random_rotate_tuple", (-0.2, 0.4)),
    )
    def test_random_rotation_shapes(self, factor):
        self.run_layer_test(
            layers.RandomRotation,
            init_kwargs={
                "factor": factor,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 4),
            supports_masking=False,
            run_training_check=False,
        )

    def test_random_rotation_correctness(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (1, 5, 5, 1)
        else:
            input_shape = (1, 1, 5, 5)
        input_image = np.reshape(np.arange(0, 25), input_shape)
        layer = layers.RandomRotation(factor=(0.5, 0.5))
        actual_output = layer(input_image)
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        ).reshape(input_shape)

        self.assertAllClose(
            backend.convert_to_tensor(expected_output), actual_output, atol=1e-5
        )

    def test_training_false(self):
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        layer = layers.RandomRotation(factor=(0.5, 0.5))
        actual_output = layer(input_image, training=False)
        self.assertAllClose(actual_output, input_image)

    def test_tf_data_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (1, 5, 5, 1)
        else:
            input_shape = (1, 1, 5, 5)
        input_image = np.reshape(np.arange(0, 25), input_shape)
        layer = layers.RandomRotation(factor=(0.5, 0.5))

        ds = tf_data.Dataset.from_tensor_slices(input_image).map(layer)
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        ).reshape(input_shape[1:])
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(expected_output, output)
