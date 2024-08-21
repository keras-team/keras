import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import Sequential
from keras.src import backend
from keras.src import layers
from keras.src import testing


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

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_down_sampling_numeric(self, data_format):
        img = np.reshape(np.arange(0, 16), (1, 4, 4, 1)).astype(np.float32)
        if data_format == "channels_first":
            img = img.transpose(0, 3, 1, 2)
        out = layers.Resizing(
            height=2, width=2, interpolation="nearest", data_format=data_format
        )(img)
        ref_out = (
            np.asarray([[5, 7], [13, 15]])
            .astype(np.float32)
            .reshape((1, 2, 2, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(0, 3, 1, 2)
        self.assertAllClose(ref_out, out)

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_up_sampling_numeric(self, data_format):
        img = np.reshape(np.arange(0, 4), (1, 2, 2, 1)).astype(np.float32)
        if data_format == "channels_first":
            img = img.transpose(0, 3, 1, 2)
        out = layers.Resizing(
            height=4,
            width=4,
            interpolation="nearest",
            data_format=data_format,
        )(img)
        ref_out = (
            np.asarray([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
            .astype(np.float32)
            .reshape((1, 4, 4, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(0, 3, 1, 2)
        self.assertAllClose(ref_out, out)

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_crop_to_aspect_ratio(self, data_format):
        img = np.reshape(np.arange(0, 16), (1, 4, 4, 1)).astype("float32")
        if data_format == "channels_first":
            img = img.transpose(0, 3, 1, 2)
        out = layers.Resizing(
            height=4,
            width=2,
            interpolation="nearest",
            data_format=data_format,
            crop_to_aspect_ratio=True,
        )(img)
        ref_out = (
            np.asarray(
                [
                    [1, 2],
                    [5, 6],
                    [9, 10],
                    [13, 14],
                ]
            )
            .astype("float32")
            .reshape((1, 4, 2, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(0, 3, 1, 2)
        self.assertAllClose(ref_out, out)

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_unbatched_image(self, data_format):
        img = np.reshape(np.arange(0, 16), (4, 4, 1)).astype("float32")
        if data_format == "channels_first":
            img = img.transpose(2, 0, 1)
        out = layers.Resizing(
            2, 2, interpolation="nearest", data_format=data_format
        )(img)
        ref_out = (
            np.asarray(
                [
                    [5, 7],
                    [13, 15],
                ]
            )
            .astype("float32")
            .reshape((2, 2, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(2, 0, 1)
        self.assertAllClose(ref_out, out)

    def test_tf_data_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        layer = layers.Resizing(8, 9)
        input_data = np.random.random(input_shape)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Sequential + tf.data only works with TF backend",
    )
    def test_tf_data_compatibility_sequential(self):
        # Test compatibility when wrapping in a Sequential
        # https://github.com/keras-team/keras/issues/347
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        layer = layers.Resizing(8, 9)
        input_data = np.random.random(input_shape)
        ds = (
            tf_data.Dataset.from_tensor_slices(input_data)
            .batch(2)
            .map(Sequential([layer]))
        )
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    @parameterized.parameters(
        [((15, 10), "channels_last"), ((15, 100), "channels_last")]
    )
    def test_data_stretch(self, size, data_format):
        img = np.random.rand(1, 1, 4, 4)
        output = layers.Resizing(
            size[0], size[1], data_format=data_format, crop_to_aspect_ratio=True
        )(img)
        self.assertEqual(output.shape, (1, *size, 4))
