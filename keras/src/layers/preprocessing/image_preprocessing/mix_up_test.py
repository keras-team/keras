import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend import convert_to_tensor


class MixUpTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.MixUp,
            init_kwargs={
                "alpha": 0.2,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
            # StatelessRandomGammaV3 is not supported on XLA_GPU_JIT
            run_training_check=not testing.tensorflow_uses_gpu(),
        )

    def test_mix_up_inference(self):
        seed = 3481
        layer = layers.MixUp(alpha=0.2)
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_mix_up_basic_functionality(self):
        image = np.random.random((64, 64, 3))
        mix_up_layer = layers.MixUp(alpha=1)
        transformation = {"mix_weight": 1, "permutation_order": [0]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )[0]
        self.assertAllClose(output, image)

        image = np.random.random((4, 64, 64, 3))
        mix_up_layer = layers.MixUp(alpha=0.2)
        transformation = {"mix_weight": 0.2, "permutation_order": [1, 0, 2, 3]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )
        self.assertNotAllClose(output, image)
        self.assertAllClose(output.shape, image.shape)

    def test_mix_up_basic_functionality_channel_first(self):
        image = np.random.random((3, 64, 64))
        mix_up_layer = layers.MixUp(alpha=1)
        transformation = {"mix_weight": 1, "permutation_order": [0]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )[0]
        self.assertAllClose(output, image)

        image = np.random.random((4, 3, 64, 64))
        mix_up_layer = layers.MixUp(alpha=0.2)
        transformation = {"mix_weight": 0.2, "permutation_order": [1, 0, 2, 3]}
        output = mix_up_layer.transform_images(
            image, transformation=transformation
        )
        self.assertNotAllClose(output, image)
        self.assertAllClose(output.shape, image.shape)

    def test_tf_data_compatibility(self):
        layer = layers.MixUp()
        input_data = np.random.random((2, 8, 8, 3))
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_mix_up_bounding_boxes(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image_shape = (10, 8, 3)
        else:
            image_shape = (3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [2, 1, 4, 3],
                    [6, 4, 8, 6],
                ]
            ),
            "labels": np.array([1, 2]),
        }
        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}

        expected_boxes = [[2, 1, 4, 3, 6, 4, 8, 6], [6, 4, 8, 6, 2, 1, 4, 3]]

        random_flip_layer = layers.MixUp(
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "mix_weight": convert_to_tensor([0.5, 0.5]),
            "permutation_order": convert_to_tensor([1, 0]),
        }
        output = random_flip_layer.transform_bounding_boxes(
            input_data["bounding_boxes"],
            transformation=transformation,
            training=True,
        )
        self.assertAllClose(output["boxes"], expected_boxes)

    def test_mix_up_tf_data_bounding_boxes(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image_shape = (1, 10, 8, 3)
        else:
            image_shape = (1, 3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [
                        [2, 1, 4, 3],
                        [6, 4, 8, 6],
                    ]
                ]
            ),
            "labels": np.array([[1, 2]]),
        }

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}
        expected_boxes = [[2, 1, 4, 3, 6, 4, 8, 6], [6, 4, 8, 6, 2, 1, 4, 3]]

        ds = tf_data.Dataset.from_tensor_slices(input_data)
        layer = layers.MixUp(
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "mix_weight": convert_to_tensor([0.5, 0.5]),
            "permutation_order": convert_to_tensor([1, 0]),
        }
        ds = ds.map(
            lambda x: layer.transform_bounding_boxes(
                x["bounding_boxes"],
                transformation=transformation,
                training=True,
            )
        )

        output = next(iter(ds))
        expected_boxes = np.array(expected_boxes)
        self.assertAllClose(output["boxes"], expected_boxes)
