import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class CutMixTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_layer(self):
        self.run_layer_test(
            layers.CutMix,
            init_kwargs={
                "factor": 1.0,
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
            # StatelessRandomGammaV3 is not supported on XLA_GPU_JIT
            run_training_check=not testing.tensorflow_uses_gpu(),
        )

    def test_cut_mix_inference(self):
        seed = 3481
        layer = layers.CutMix()

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_cut_mix_basic(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image1 = np.ones((2, 2, 1))
            image2 = np.zeros((2, 2, 1))
            inputs = np.asarray([image1, image2])
            expected_output = np.array(
                [
                    [[[1.0], [1.0]], [[1.0], [1.0]]],
                    [[[0.0], [0.0]], [[0.0], [0.0]]],
                ]
            )
        else:
            image1 = np.ones((1, 2, 2))
            image2 = np.zeros((1, 2, 2))
            inputs = np.asarray([image1, image2])
            expected_output = np.asarray(
                [
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                ]
            )

        layer = layers.CutMix(data_format=data_format)

        transformation = {
            "batch_masks": np.asarray(
                [
                    [[[False], [True]], [[False], [False]]],
                    [[[False], [False]], [[True], [False]]],
                ]
            ),
            "mix_weight": np.asarray([[[[0.7826548]]], [[[0.8133545]]]]),
            "permutation_order": np.asarray([0, 1]),
        }

        output = layer.transform_images(inputs, transformation)

        self.assertAllClose(expected_output, output)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.CutMix(data_format=data_format)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()
