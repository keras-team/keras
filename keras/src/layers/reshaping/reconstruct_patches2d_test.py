import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


def _gradient_image(H, W, C=1, batch=1):
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype("float32")
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


class ReconstructPatches2DTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # The layer only supports channels_last; pin it so the suite is
        # independent of the backend's default image_data_format (the torch
        # CI config defaults to channels_first).
        self._original_data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self._original_data_format)

    @parameterized.parameters(
        # (H, W, C, size, padding)
        (64, 64, 1, (8, 8), "valid"),
        (64, 64, 3, (8, 8), "valid"),
        (32, 48, 1, (4, 8), "valid"),
        (59, 55, 3, (8, 8), "same"),
        (33, 41, 2, (5, 7), "same"),
        (7, 11, 1, (3, 5), "same"),
    )
    def test_extract_then_reconstruct_roundtrip(self, H, W, C, size, padding):
        x = _gradient_image(H, W, C, batch=2)
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(x_t, size=size, padding=padding)
        layer = layers.ReconstructPatches2D(
            size=size,
            output_size=(H, W),
            padding=padding,
        )
        recon = layer(patches)
        self.assertEqual(tuple(recon.shape), x.shape)
        self.assertAllClose(recon, x, atol=1e-6)

    def test_dynamic_spatial_dim(self):
        size = (4, 4)
        flat = size[0] * size[1] * 3
        input_layer = layers.Input(batch_shape=(1, None, None, flat))
        recon = layers.ReconstructPatches2D(
            size=size,
            output_size=(16, 16),
            padding="valid",
        )(input_layer)
        self.assertEqual(recon.shape, (1, 16, 16, 3))

    def test_get_config(self):
        layer = layers.ReconstructPatches2D(
            size=(3, 4),
            output_size=(12, 16),
            padding="valid",
        )
        config = layer.get_config()
        restored = layers.ReconstructPatches2D.from_config(config)
        self.assertEqual(restored.size, (3, 4))
        self.assertEqual(restored.output_size, (12, 16))
        self.assertEqual(restored.padding, "valid")

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "length 2"):
            layers.ReconstructPatches2D(size=(2, 3, 4), output_size=(10, 15))

    def test_int_size(self):
        # int size is normalized to (size, size).
        layer = layers.ReconstructPatches2D(size=4, output_size=(16, 16))
        self.assertEqual(layer.size, (4, 4))

    def test_invalid_output_size(self):
        with self.assertRaisesRegex(ValueError, "length 2"):
            layers.ReconstructPatches2D(size=(2, 2), output_size=(8, 8, 8))

    def test_invalid_padding(self):
        with self.assertRaisesRegex(ValueError, "'same' or 'valid'"):
            layers.ReconstructPatches2D(
                size=(2, 2),
                output_size=(8, 8),
                padding="reflect",
            )

    def test_channels_first_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "channels_first"):
            layers.ReconstructPatches2D(
                size=(2, 2),
                output_size=(8, 8),
                data_format="channels_first",
            )
