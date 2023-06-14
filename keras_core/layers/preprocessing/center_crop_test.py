import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import layers
from keras_core import testing


class CenterCropTest(testing.TestCase, parameterized.TestCase):
    def test_center_crop_basics(self):
        self.run_layer_test(
            layers.CenterCrop,
            init_kwargs={
                "height": 6,
                "width": 6,
                "data_format": "channels_last",
            },
            input_shape=(2, 12, 12, 3),
            expected_output_shape=(2, 6, 6, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.CenterCrop,
            init_kwargs={
                "height": 7,
                "width": 7,
                "data_format": "channels_first",
            },
            input_shape=(2, 3, 13, 13),
            expected_output_shape=(2, 3, 7, 7),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @parameterized.parameters(
        [
            ((5, 7), "channels_first"),
            ((5, 7), "channels_last"),
            ((15, 10), "channels_first"),
            ((10, 17), "channels_last"),
        ]
    )
    def test_center_crop_correctness(self, size, data_format):
        # batched case
        if data_format == "channels_first":
            img = np.random.random((2, 3, 9, 11))
        else:
            img = np.random.random((2, 9, 11, 3))
        out = layers.CenterCrop(
            size[0],
            size[1],
            data_format=data_format,
        )(img)
        if data_format == "channels_first":
            img_transpose = np.transpose(img, (0, 2, 3, 1))

            ref_out = tf.transpose(
                tf.keras.layers.CenterCrop(size[0], size[1])(img_transpose),
                (0, 3, 1, 2),
            )
        else:
            ref_out = tf.keras.layers.CenterCrop(size[0], size[1])(img)
        self.assertAllClose(ref_out, out)

        # unbatched case
        if data_format == "channels_first":
            img = np.random.random((3, 9, 11))
        else:
            img = np.random.random((9, 11, 3))
        out = layers.CenterCrop(
            size[0],
            size[1],
            data_format=data_format,
        )(img)
        if data_format == "channels_first":
            img_transpose = np.transpose(img, (1, 2, 0))
            ref_out = tf.transpose(
                tf.keras.layers.CenterCrop(
                    size[0],
                    size[1],
                )(img_transpose),
                (2, 0, 1),
            )
        else:
            ref_out = tf.keras.layers.CenterCrop(
                size[0],
                size[1],
            )(img)
        self.assertAllClose(ref_out, out)

    def test_tf_data_compatibility(self):
        layer = layers.CenterCrop(8, 9)
        input_data = np.random.random((2, 10, 12, 3))
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(list(output.shape), [2, 8, 9, 3])

    def test_list_compatibility(self):
        images = [
            np.random.rand(10, 10, 3),
            np.random.rand(10, 10, 3),
        ]
        output = layers.CenterCrop(height=6, width=5)(images)
        ref_output = tf.keras.layers.CenterCrop(6, 5)(images)
        self.assertListEqual(list(output.shape), [2, 6, 5, 3])
        self.assertAllClose(ref_output, output)
