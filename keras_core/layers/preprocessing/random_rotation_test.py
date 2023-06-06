import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing


class RandomRotationTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("random_rotate_4", 0.4),
        ("random_rotate_3", 0.3),
        ("random_rotate_tuple_factor", (-0.5, 0.4)),
    )
    def test_random_rotation(self, factor):
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
        input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1))
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        )
        expected_output = backend.convert_to_tensor(
            np.reshape(expected_output, (1, 5, 5, 1))
        )
        self.run_layer_test(
            layers.RandomRotation,
            init_kwargs={
                "factor": (0.5, 0.5),
            },
            input_shape=None,
            input_data=input_image,
            expected_output=expected_output,
            supports_masking=False,
            run_training_check=False,
        )

    def test_tf_data_compatibility(self):
        layer = layers.RandomRotation(0.2)
        input_data = np.random.random((1, 4, 4, 3))
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(1).map(layer)
        for output in ds.take(1):
            output.numpy()
