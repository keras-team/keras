import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_core import Sequential
from keras_core import backend
from keras_core import layers
from keras_core import testing


class ResizingTest(testing.TestCase, parameterized.TestCase):
    def test_resizing_basics(self):
        self.run_layer_test(
            layers.Resizing,
            init_kwargs={
                "height": 6,
                "width": 6,
                "data_format": "channels_last",
                "interpolation": "bicubic",
                "crop_to_aspect_ratio": True,
            },
            input_shape=(2, 12, 12, 3),
            expected_output_shape=(2, 6, 6, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )
        self.run_layer_test(
            layers.Resizing,
            init_kwargs={
                "height": 6,
                "width": 6,
                "data_format": "channels_first",
                "interpolation": "bilinear",
                "crop_to_aspect_ratio": True,
            },
            input_shape=(2, 3, 12, 12),
            expected_output_shape=(2, 3, 6, 6),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )
        self.run_layer_test(
            layers.Resizing,
            init_kwargs={
                "height": 6,
                "width": 6,
                "data_format": "channels_last",
                "interpolation": "nearest",
                "crop_to_aspect_ratio": False,
            },
            input_shape=(2, 12, 12, 3),
            expected_output_shape=(2, 6, 6, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    @pytest.mark.skipif(
        backend.backend() == "torch", reason="Torch does not support lanczos."
    )
    def test_resizing_basics_lanczos5(self):
        self.run_layer_test(
            layers.Resizing,
            init_kwargs={
                "height": 6,
                "width": 6,
                "data_format": "channels_first",
                "interpolation": "lanczos5",
                "crop_to_aspect_ratio": False,
            },
            input_shape=(2, 3, 12, 12),
            expected_output_shape=(2, 3, 6, 6),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters(
        [
            ((5, 7), "channels_first", True),
            ((5, 7), "channels_last", True),
            ((6, 8), "channels_first", False),
            ((6, 8), "channels_last", False),
        ]
    )
    def test_resizing_correctness(
        self, size, data_format, crop_to_aspect_ratio
    ):
        # batched case
        if data_format == "channels_first":
            img = np.random.random((2, 3, 9, 11))
        else:
            img = np.random.random((2, 9, 11, 3))
        out = layers.Resizing(
            size[0],
            size[1],
            data_format=data_format,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )(img)
        if data_format == "channels_first":
            img_transpose = np.transpose(img, (0, 2, 3, 1))

            ref_out = tf.transpose(
                tf.keras.layers.Resizing(
                    size[0], size[1], crop_to_aspect_ratio=crop_to_aspect_ratio
                )(img_transpose),
                (0, 3, 1, 2),
            )
        else:
            ref_out = tf.keras.layers.Resizing(
                size[0], size[1], crop_to_aspect_ratio=crop_to_aspect_ratio
            )(img)
        self.assertAllClose(ref_out, out)

        # unbatched case
        if data_format == "channels_first":
            img = np.random.random((3, 9, 11))
        else:
            img = np.random.random((9, 11, 3))
        out = layers.Resizing(
            size[0],
            size[1],
            data_format=data_format,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )(img)
        if data_format == "channels_first":
            img_transpose = np.transpose(img, (1, 2, 0))
            ref_out = tf.transpose(
                tf.keras.layers.Resizing(
                    size[0], size[1], crop_to_aspect_ratio=crop_to_aspect_ratio
                )(img_transpose),
                (2, 0, 1),
            )
        else:
            ref_out = tf.keras.layers.Resizing(
                size[0], size[1], crop_to_aspect_ratio=crop_to_aspect_ratio
            )(img)
        self.assertAllClose(ref_out, out)

    def test_tf_data_compatibility(self):
        layer = layers.Resizing(8, 9)
        input_data = np.random.random((2, 10, 12, 3))
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(list(output.shape), [2, 8, 9, 3])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Sequential + tf.data only works with TF backend",
    )
    def test_tf_data_compatibility_sequential(self):
        # Test compatibility when wrapping in a Sequential
        # https://github.com/keras-team/keras-core/issues/347
        layer = layers.Resizing(8, 9)
        input_data = np.random.random((2, 10, 12, 3))
        ds = (
            tf.data.Dataset.from_tensor_slices(input_data)
            .batch(2)
            .map(Sequential([layer]))
        )
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(list(output.shape), [2, 8, 9, 3])
