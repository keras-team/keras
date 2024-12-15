import keras.src.layers.preprocessing.image_preprocessing.random_brightness as random_brightness
import keras.src.layers.preprocessing.image_preprocessing.random_contrast as random_contrast
import keras.src.layers.preprocessing.image_preprocessing.random_hue as random_hue
import keras.src.layers.preprocessing.image_preprocessing.random_saturation as random_saturation
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomColorJitter")
class RandomColorJitter(BaseImagePreprocessingLayer):
    """RandomColorJitter class randomly apply brightness, contrast, saturation
    and hue image processing operation sequentially and randomly on the
    input. It expects input as RGB image. The expected image should be
    `(0-255)` pixel ranges.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is set up.
        brightness_factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The
            factor is used to determine the lower bound and upper bound of the
            brightness adjustment. A float value will be chosen randomly between
            the limits. When -1.0 is chosen, the output image will be black, and
            when 1.0 is chosen, the image will be fully white.
            When only one float is provided, eg, 0.2,
            then -0.2 will be used for lower bound and 0.2
            will be used for upper bound.
        contrast_factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound.
            When represented as a single float, lower = upper.
            The contrast factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
            the output will be `(x - mean) * factor + mean`
            where `mean` is the mean value of the channel.
        saturation_factor: A tuple of two floats or a single float.
            `factor` controls the extent to which the image saturation
            is impacted. `factor=0.5` makes this layer perform a no-op
            operation. `factor=0.0` makes the image fully grayscale.
            `factor=1.0` makes the image fully saturated. Values should
            be between `0.0` and `1.0`. If a tuple is used, a `factor`
            is sampled between the two values for every image augmented.
            If a single float is used, a value between `0.0` and the passed
            float is sampled. To ensure the value is always the same,
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        hue_factor: A single float or a tuple of two floats.
            `factor` controls the extent to which the
            image hue is impacted. `factor=0.0` makes this layer perform a
            no-op operation, while a value of `1.0` performs the most aggressive
            contrast adjustment available. If a tuple is used, a `factor` is
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    color_jitter = keras.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(-0.2, 0.5),
            contrast_factor=(0.5, 0.9),
            saturation_factor=(0.5, 0.9),
            hue_factor=(0.5, 0.9),
    )
    augmented_images = color_jitter(images)
    ```
    """

    def __init__(
        self,
        value_range,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        hue_factor,
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.value_range = value_range
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.seed = seed
        self.generator = SeedGenerator(seed)

        self.random_brightness = random_brightness.RandomBrightness(
            factor=self.brightness_factor, value_range=(0, 255), seed=self.seed
        )
        self.random_contrast = random_contrast.RandomContrast(
            factor=self.contrast_factor, value_range=(0, 255), seed=self.seed
        )
        self.random_saturation = random_saturation.RandomSaturation(
            factor=self.saturation_factor, seed=self.seed
        )
        self.random_hue = random_hue.RandomHue(
            factor=self.hue_factor, value_range=(0, 255), seed=self.seed
        )

    def transform_images(self, images, transformation, training=True):
        images = self._transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        images = self.random_brightness(images)
        images = self.random_contrast(images)
        images = self.random_saturation(images)
        images = self.random_hue(images)
        images = self._transform_value_range(
            images,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "value_range": self.value_range,
            "brightness_factor": self.brightness_factor,
            "contrast_factor": self.contrast_factor,
            "saturation_factor": self.saturation_factor,
            "hue_factor": self.hue_factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
