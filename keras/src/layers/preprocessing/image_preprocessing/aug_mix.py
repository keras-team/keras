import random as py_random

import keras.src.layers as layers
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator
from keras.src.utils import backend_utils

AUGMENT_LAYERS_ALL = [
    "random_shear",
    "random_translation",
    "random_rotation",
    "random_posterization",
    "solarization",
    "auto_contrast",
    "equalization",
    "random_brightness",
    "random_color_degeneration",
    "random_contrast",
    "random_sharpness",
]

AUGMENT_LAYERS = [
    "random_shear",
    "random_translation",
    "random_rotation",
    "random_posterization",
    "solarization",
    "auto_contrast",
    "equalization",
]


@keras_export("keras.layers.AugMix")
class AugMix(BaseImagePreprocessingLayer):
    """Performs the AugMix data augmentation technique.

    AugMix aims to produce images with variety while preserving the image
    semantics and local statistics. During the augmentation process,
    the same augmentation is applied across all images in the batch
    in num_chains different ways, with each chain consisting of
    chain_depth augmentations.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written (low, high).
            This is typically either `(0, 1)` or `(0, 255)` depending
            on how your preprocessing pipeline is set up.
        num_chains: an integer representing the number of different chains to
            be mixed, defaults to 3.
        chain_depth: an integer representing the maximum number of
            transformations to be applied in each chain. The actual number
            of transformations in each chain will be sampled randomly
            from the range `[0, `chain_depth`]`. Defaults to 3.
        factor: The strength of the augmentation as a normalized value
            between 0 and 1. Default is 0.3.
        alpha: a float value used as the probability coefficients for the
            Beta and Dirichlet distributions, defaults to 1.0.
        all_ops: Use all operations (including random_brightness,
            random_color_degeneration, random_contrast and random_sharpness).
            Default is True.
        interpolation: The interpolation method to use for resizing operations.
            Options include `"nearest"`, `"bilinear"`. Default is `"bilinear"`.
        seed: Integer. Used to create a random seed.

    References:
        - [AugMix paper](https://arxiv.org/pdf/1912.02781)
        - [Official Code](https://github.com/google-research/augmix)
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

    def __init__(
        self,
        value_range=(0, 255),
        num_chains=3,
        chain_depth=3,
        factor=0.3,
        alpha=1.0,
        all_ops=True,
        interpolation="bilinear",
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)

        self.value_range = value_range
        self.num_chains = num_chains
        self.chain_depth = chain_depth
        self._set_factor(factor)
        self.alpha = alpha
        self.all_ops = all_ops
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)

        if self.all_ops:
            self._augment_layers = AUGMENT_LAYERS_ALL
        else:
            self._augment_layers = AUGMENT_LAYERS

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

        if self.all_ops:
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

    def build(self, input_shape):
        for layer_name in self._augment_layers:
            augmentation_layer = getattr(self, layer_name)
            augmentation_layer.build(input_shape)

    def _sample_from_dirichlet(self, shape, alpha, seed):
        gamma_sample = self.backend.random.gamma(
            shape=shape,
            alpha=alpha,
            seed=seed,
        )
        return gamma_sample / self.backend.numpy.sum(
            gamma_sample, axis=-1, keepdims=True
        )

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if backend_utils.in_tf_graph():
            self.backend.set_backend("tensorflow")

            for layer_name in self._augment_layers:
                augmentation_layer = getattr(self, layer_name)
                augmentation_layer.backend.set_backend("tensorflow")

        seed = seed or self._get_seed_generator(self.backend._backend)

        chain_mixing_weights = self._sample_from_dirichlet(
            [self.num_chains], self.alpha, seed
        )
        weight_sample = self.backend.random.beta(
            shape=(),
            alpha=self.alpha,
            beta=self.alpha,
            seed=seed,
        )

        chain_transforms = []
        for _ in range(self.num_chains):
            depth_transforms = []
            for _ in range(self.chain_depth):
                layer_name = py_random.choice(self._augment_layers + [None])
                if layer_name is None:
                    continue
                augmentation_layer = getattr(self, layer_name)
                depth_transforms.append(
                    {
                        "layer_name": layer_name,
                        "transformation": (
                            augmentation_layer.get_random_transformation(
                                data,
                                seed=self._get_seed_generator(
                                    self.backend._backend
                                ),
                            )
                        ),
                    }
                )
            chain_transforms.append(depth_transforms)

        transformation = {
            "chain_mixing_weights": chain_mixing_weights,
            "weight_sample": weight_sample,
            "chain_transforms": chain_transforms,
        }

        return transformation

    def transform_images(self, images, transformation, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)

            chain_mixing_weights = self.backend.cast(
                transformation["chain_mixing_weights"], dtype=self.compute_dtype
            )
            weight_sample = self.backend.cast(
                transformation["weight_sample"], dtype=self.compute_dtype
            )
            chain_transforms = transformation["chain_transforms"]

            aug_images = self.backend.numpy.zeros_like(images)
            for idx, chain_transform in enumerate(chain_transforms):
                copied_images = self.backend.numpy.copy(images)
                for depth_transform in chain_transform:
                    layer_name = depth_transform["layer_name"]
                    layer_transform = depth_transform["transformation"]

                    augmentation_layer = getattr(self, layer_name)
                    copied_images = augmentation_layer.transform_images(
                        copied_images, layer_transform
                    )
                aug_images += copied_images * chain_mixing_weights[idx]
            images = weight_sample * images + (1 - weight_sample) * aug_images

            images = self.backend.numpy.clip(
                images, self.value_range[0], self.value_range[1]
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
            "num_chains": self.chain_depth,
            "chain_depth": self.num_chains,
            "factor": self.factor,
            "alpha": self.alpha,
            "all_ops": self.all_ops,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
