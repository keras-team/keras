import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import testing


class BaseImagePreprocessingLayerTest(testing.TestCase):
    @parameterized.named_parameters(
        ("aug_mix", layers.AugMix, {}),
        ("auto_contrast", layers.AutoContrast, {}),
        ("center_crop", layers.CenterCrop, {"height": 4, "width": 4}),
        (
            "clahe",
            layers.ContrastLimitedAdaptiveHistogramEqualization,
            {},
        ),
        ("cut_mix", layers.CutMix, {}),
        ("equalization", layers.Equalization, {}),
        (
            "max_num_bounding_boxes",
            layers.MaxNumBoundingBoxes,
            {"max_number": 5},
        ),
        ("mix_up", layers.MixUp, {}),
        ("rand_augment", layers.RandAugment, {}),
        ("random_brightness", layers.RandomBrightness, {"factor": 0.5}),
        (
            "random_color_degeneration",
            layers.RandomColorDegeneration,
            {"factor": 0.5},
        ),
        ("random_color_jitter", layers.RandomColorJitter, {}),
        ("random_contrast", layers.RandomContrast, {"factor": 0.5}),
        ("random_crop", layers.RandomCrop, {"height": 4, "width": 4}),
        ("random_elastic_transform", layers.RandomElasticTransform, {}),
        ("random_erasing", layers.RandomErasing, {}),
        ("random_flip", layers.RandomFlip, {}),
        ("random_gaussian_blur", layers.RandomGaussianBlur, {}),
        ("random_grayscale", layers.RandomGrayscale, {}),
        ("random_hue", layers.RandomHue, {"factor": 0.5}),
        ("random_invert", layers.RandomInvert, {}),
        ("random_perspective", layers.RandomPerspective, {}),
        ("random_posterization", layers.RandomPosterization, {"factor": 4}),
        ("random_rotation", layers.RandomRotation, {"factor": 0.5}),
        ("random_saturation", layers.RandomSaturation, {"factor": 0.5}),
        ("random_sharpness", layers.RandomSharpness, {"factor": 0.5}),
        ("random_shear", layers.RandomShear, {}),
        (
            "random_translation",
            layers.RandomTranslation,
            {"height_factor": 0.2, "width_factor": 0.2},
        ),
        ("random_zoom", layers.RandomZoom, {"height_factor": 0.2}),
        ("resizing", layers.Resizing, {"height": 4, "width": 4}),
        ("solarization", layers.Solarization, {}),
    )
    def test_data_format_serialization(self, layer_cls, init_kwargs):
        # `data_format` was previously dropped from `get_config` in layers
        # that rely on the base class to save it, so a save/load round-trip
        # silently reset it to the global image data format.
        data_format = (
            "channels_first"
            if backend.image_data_format() == "channels_last"
            else "channels_last"
        )
        layer = layer_cls(data_format=data_format, **init_kwargs)
        config = layer.get_config()
        self.assertIn("data_format", config)
        revived = layer_cls.from_config(config)
        self.assertEqual(revived.data_format, data_format)

    def test_segmentation_mask_dtype_is_preserved(self):
        # Regression for https://github.com/keras-team/keras/issues/20857:
        # uint8 segmentation masks passed through a preprocessing layer
        # were silently cast to float32 because subclassed
        # `transform_segmentation_masks` implementations delegated to
        # `transform_images`, which casts to the layer's compute_dtype.
        images = np.zeros((1, 4, 4, 3), dtype="uint8")
        masks = np.array(
            [
                [
                    [[0], [1], [2], [0]],
                    [[1], [1], [2], [2]],
                    [[0], [0], [1], [2]],
                    [[2], [0], [1], [0]],
                ]
            ],
            dtype="uint8",
        )
        layer = layers.RandomFlip("horizontal", seed=0)
        out = layer({"images": images, "segmentation_masks": masks})
        self.assertEqual(
            backend.standardize_dtype(out["segmentation_masks"].dtype), "uint8"
        )
