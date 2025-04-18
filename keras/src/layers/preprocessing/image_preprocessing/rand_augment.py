import keras.src.layers as layers
import keras.src.ops as ops
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator, shuffle
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

        self.augmentations = [
            layers.RandomShear(
                x_factor=self.factor,
                y_factor=self.factor,
                interpolation=interpolation,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.RandomTranslation(
                height_factor=self.factor,
                width_factor=self.factor,
                interpolation=interpolation,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.RandomRotation(
                factor=self.factor,
                interpolation=interpolation,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.RandomBrightness(
                factor=self.factor,
                value_range=self.value_range,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.RandomColorDegeneration(
                factor=self.factor,
                value_range=self.value_range,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.RandomContrast(
                factor=self.factor,
                value_range=self.value_range,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.RandomSharpness(
                factor=self.factor,
                value_range=self.value_range,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.Solarization(
                addition_factor=self.factor,
                threshold_factor=self.factor,
                value_range=self.value_range,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.RandomPosterization(
                factor=max(1, int(8 * self.factor[1])),
                value_range=self.value_range,
                seed=self.seed,
                data_format=data_format,
                **kwargs,
            ),
            layers.AutoContrast(
                value_range=self.value_range, data_format=data_format, **kwargs
            ),
            layers.Equalization(
                value_range=self.value_range, data_format=data_format, **kwargs
            )
        ]
        self.num_layers = len(self.augmentations)

    def build(self, input_shape):
        for augmentation_layer in self.augmentations:
            augmentation_layer.build(input_shape)

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if backend_utils.in_tf_graph():
            self.backend.set_backend("tensorflow")

            for augmentation_layer in self.augmentations:
                augmentation_layer.backend.set_backend("tensorflow")

        transformation = []
        idx = self.backend.random.shuffle(
            self.backend.numpy.arange(self.num_layers, dtype="int32"),
            seed=self._get_seed_generator(self.backend._backend),
        )
        
        for i in range(self.num_layers):
            augmentation_layer = self.augmentations[i]
            transformation.append(
                augmentation_layer.get_random_transformation(
                    data,
                    training=training,
                    seed=self._get_seed_generator(self.backend._backend),
                )
            )

        return idx, transformation
    
    def _apply_augs(self, transformation, func_name, inputs):
        aug_index, transforms = transformation
        

        def get_fn(aug, xform):
            def func(x):
                if isinstance(x, dict):
                    z = tree.map_structure(self.backend.numpy.copy, x)
                    return getattr(aug, func_name)(z, xform)
                return getattr(aug, func_name)(x, xform)
            return func
        
        def body(i, loop_var):
            idx = aug_index[i]
            return self.backend.core.switch(
                idx,
                [get_fn(aug, xform) for aug, xform in zip(self.augmentations, transforms)],
                loop_var,
            )
        
        return self.backend.core.fori_loop(0, self.num_ops, body, inputs)
    
    def transform_images(self, images, transformation, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)

            images = self._apply_augs(transformation, "transform_images", images)
            
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
            bounding_boxes = self._apply_augs(transformation, "transform_bounding_boxes", bounding_boxes)
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
