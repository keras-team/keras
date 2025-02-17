import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomErasingTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomErasing,
            init_kwargs={
                "factor": 1.0,
                "scale": 0.5,
                "fill_value": 0,
                "value_range": (0, 255),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_erasing_inference(self):
        seed = 3481
        layer = layers.RandomErasing()

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_erasing_no_op(self):
        seed = 3481
        layer = layers.RandomErasing(factor=0)

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

        layer = layers.RandomErasing(scale=0)

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

    def test_random_erasing_basic(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.ones((2, 2, 1))
            expected_output = np.array([[[[0.0], [1.0]], [[1.0], [1.0]]]])

        else:
            inputs = np.ones((1, 2, 2))

            expected_output = np.array(
                [[[[0.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]
            )

        layer = layers.RandomErasing(data_format=data_format)

        transformation = {
            "apply_erasing": np.asarray([True]),
            "batch_masks": np.asarray(
                [[[[True], [False]], [[False], [False]]]]
            ),
            "fill_value": 0,
        }

        output = layer.transform_images(inputs, transformation)

        print(output)

        self.assertAllClose(expected_output, output)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomErasing(data_format=data_format)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
