import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class RandomElasticTransformTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.RandomElasticTransform,
            init_kwargs={
                "factor": 1.0,
                "scale": 0.5,
                "interpolation": "bilinear",
                "fill_mode": "reflect",
                "fill_value": 0,
                "value_range": (0, 255),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
            run_training_check=False,
        )

    def test_random_elastic_transform_inference(self):
        seed = 3481
        layer = layers.RandomElasticTransform()

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_elastic_transform_no_op(self):
        seed = 3481
        layer = layers.RandomElasticTransform(factor=0)

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

        layer = layers.RandomElasticTransform(scale=0)

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

    def test_random_elastic_transform_basic(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            inputs = np.zeros((8, 8, 1))
            inputs[3:5, 3:5, :] = 1.0
        else:
            inputs = np.zeros((1, 8, 8))
            inputs[:, 3:5, 3:5] = 1.0

        layer = layers.RandomElasticTransform(data_format=data_format)

        transformation = {
            "apply_transform": np.array([True]),
            "distortion_factor": np.float32(0.9109325),
            "seed": 42,
        }

        output = layer.transform_images(inputs, transformation)

        self.assertNotAllClose(inputs, output)
        self.assertEqual(inputs.shape, output.shape)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomElasticTransform(data_format=data_format)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            print("Output shape:", output.shape)  # Debugging line
            output_numpy = output.numpy()
            print("Output numpy shape:", output_numpy.shape)
