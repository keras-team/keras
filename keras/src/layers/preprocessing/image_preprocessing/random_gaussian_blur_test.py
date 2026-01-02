import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend import convert_to_tensor


class RandomGaussianBlurTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomGaussianBlur,
            init_kwargs={
                "factor": 1.0,
                "kernel_size": 3,
                "sigma": 0,
                "value_range": (0, 255),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_erasing_inference(self):
        seed = 3481
        layer = layers.RandomGaussianBlur()

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_erasing_no_op(self):
        seed = 3481
        layer = layers.RandomGaussianBlur(factor=0)

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

    def test_random_erasing_basic(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.ones((1, 2, 2, 3))
            expected_output = np.asarray(
                [
                    [
                        [[0.7273, 0.7273, 0.7273], [0.7273, 0.7273, 0.7273]],
                        [[0.7273, 0.7273, 0.7273], [0.7273, 0.7273, 0.7273]],
                    ]
                ]
            )

        else:
            inputs = np.ones((1, 3, 2, 2))
            expected_output = np.asarray(
                [
                    [
                        [[0.7273, 0.7273], [0.7273, 0.7273]],
                        [[0.7273, 0.7273], [0.7273, 0.7273]],
                        [[0.7273, 0.7273], [0.7273, 0.7273]],
                    ]
                ]
            )

        layer = layers.RandomGaussianBlur(data_format=data_format)

        transformation = {
            "blur_factor": convert_to_tensor([0.3732, 0.8654]),
            "should_apply_blur": convert_to_tensor([True]),
        }
        output = layer.transform_images(inputs, transformation)

        self.assertAllClose(
            expected_output,
            output,
            atol=1e-4,
            rtol=1e-4,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomGaussianBlur(data_format=data_format)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
