import random

import keras.src.layers as layers
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator
from keras.src.utils import backend_utils


@keras_export("keras.layers.RandAugment")
class RandAugment(BaseImagePreprocessingLayer):
    """RandAugment performs the Rand Augment operation on input images.

    This layer can be thought of as an all-in-one image augmentation layer. The
    policy implemented by this layer has been benchmarked extensively and is
    effective on a wide variety of datasets.

    References:
        - [RandAugment](https://arxiv.org/abs/1909.13719)

    Args:
        value_range: The range of values the input image can take.
            Default is `(0, 255)`. Typically, this would be `(0, 1)`
            for normalized images or `(0, 255)` for raw images.
        num_ops: The number of augmentation operations to apply sequentially
            to each image. Default is 2.
        factor: The strength of the augmentation as a normalized value
            between 0 and 1. Default is 0.5.
        interpolation: The interpolation method to use for resizing operations.
            Options include `nearest`, `bilinear`. Default is `bilinear`.
        seed: Integer. Used to create a random seed.

    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

    _AUGMENT_LAYERS = [
        "random_shear",
        "random_translation",
        "random_rotation",
        "random_brightness",
        "random_color_degeneration",
        "random_contrast",
        "random_sharpness",
        "random_posterization",
        "solarization",
        "auto_contrast",
        "equalization",
    ]

    def __init__(
        self,
        value_range=(0, 255),
        num_ops=2,
        factor=0.5,
        interpolation="bilinear",
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)

        self.value_range = value_range
        self.num_ops = num_ops
        self._set_factor(factor)
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)

        self.random_shear = layers.RandomShear(
            x_factor=self.factor,
            y_factor=self.factor,
            interpolation=interpolation,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.random_translation = layers.RandomTranslation(
            height_factor=self.factor,
            width_factor=self.factor,
            interpolation=interpolation,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.random_rotation = layers.RandomRotation(
            factor=self.factor,
            interpolation=interpolation,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.random_brightness = layers.RandomBrightness(
            factor=self.factor,
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.random_color_degeneration = layers.RandomColorDegeneration(
            factor=self.factor,
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.random_contrast = layers.RandomContrast(
            factor=self.factor,
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.random_sharpness = layers.RandomSharpness(
            factor=self.factor,
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.solarization = layers.Solarization(
            addition_factor=self.factor,
            threshold_factor=self.factor,
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.random_posterization = layers.RandomPosterization(
            factor=max(1, int(8 * self.factor[1])),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
            **kwargs,
        )

        self.auto_contrast = layers.AutoContrast(
            value_range=self.value_range, data_format=data_format, **kwargs
        )

        self.equalization = layers.Equalization(
            value_range=self.value_range, data_format=data_format, **kwargs
        )

    def build(self, input_shape):
        for layer_name in self._AUGMENT_LAYERS:
            augmentation_layer = getattr(self, layer_name)
            augmentation_layer.build(input_shape)

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if backend_utils.in_tf_graph():
            self.backend.set_backend("tensorflow")

            for layer_name in self._AUGMENT_LAYERS:
                augmentation_layer = getattr(self, layer_name)
                augmentation_layer.backend.set_backend("tensorflow")

        transformation = {}
        random.shuffle(self._AUGMENT_LAYERS)
        for layer_name in self._AUGMENT_LAYERS[: self.num_ops]:
            augmentation_layer = getattr(self, layer_name)
            transformation[layer_name] = (
                augmentation_layer.get_random_transformation(
                    data,
                    training=training,
                    seed=self._get_seed_generator(self.backend._backend),
                )
            )

        return transformation

    def transform_images(self, images, transformation, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)

            for layer_name, transformation_value in transformation.items():
                augmentation_layer = getattr(self, layer_name)
                images = augmentation_layer.transform_images(
                    images, transformation_value
                )

        images = self.backend.cast(images, self.compute_dtype)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        if training:
            for layer_name, transformation_value in transformation.items():
                augmentation_layer = getattr(self, layer_name)
                bounding_boxes = augmentation_layer.transform_bounding_boxes(
                    bounding_boxes, transformation_value, training=training
                )
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "value_range": self.value_range,
            "num_ops": self.num_ops,
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
