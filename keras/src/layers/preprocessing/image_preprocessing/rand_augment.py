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

    The policy operates as follows:

    For each augmentation in the range `[0, augmentations_per_image]`,
    the policy selects a random operation from a list of operations.
    It then samples a random number and if that number is less than
    `rate` applies it to the given image.

    References:
        - [RandAugment](https://arxiv.org/abs/1909.13719)

    """

    _AUGMENT_LAYERS = ["Identity", "random_shear", "random_translation", "random_rotation",
                       "random_brightness", "random_color_degeneration", "random_contrast",
                       "random_sharpness", "random_posterization", "solarization", "auto_contrast", "equalization"]

    def __init__(
            self,
            value_range=(0, 255),
            num_ops=2,
            magnitude=9,
            num_magnitude_bins=31,
            interpolation="nearest",
            seed=None,
            data_format=None,
            **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)

        self.value_range = value_range
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)

        augmentation_space = self._augmentation_space(self.num_magnitude_bins)

        random_indices = self.backend.random.randint([11], 0, self.num_magnitude_bins, seed=self.seed)
        self.random_shear = layers.RandomShear(
            x_factor=float(augmentation_space["ShearX"][int(random_indices[0])]),
            y_factor=float(augmentation_space["ShearY"][int(random_indices[1])]),
            seed=self.seed,
            data_format=data_format,
        )

        self.random_translation = layers.RandomTranslation(
            height_factor=float(augmentation_space["TranslateX"][int(random_indices[2])]),
            width_factor=float(augmentation_space["TranslateY"][int(random_indices[3])]),
            seed=self.seed,
            data_format=data_format,
        )

        self.random_rotation = layers.RandomRotation(
            factor=float(augmentation_space["Rotate"][int(random_indices[4])]),
            seed=self.seed,
            data_format=data_format,
        )

        self.random_brightness = layers.RandomBrightness(
            factor=float(augmentation_space["Brightness"][int(random_indices[5])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        self.random_color_degeneration = layers.RandomColorDegeneration(
            factor=float(augmentation_space["Color"][int(random_indices[6])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        self.random_contrast = layers.RandomContrast(
            factor=float(augmentation_space["Contrast"][int(random_indices[7])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        self.random_sharpness = layers.RandomSharpness(
            factor=float(augmentation_space["Sharpness"][int(random_indices[8])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        self.solarization = layers.Solarization(
            addition_factor=int(augmentation_space["Solarize"][int(random_indices[9])]),
            threshold_factor=int(augmentation_space["Solarize"][int(random_indices[10])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        random_indices = self.backend.random.uniform((1,),
                                                     minval=0, maxval=len(augmentation_space["Posterize"]),
                                                     seed=self.seed)
        self.random_posterization = layers.RandomPosterization(
            factor=int(augmentation_space["Posterize"][int(random_indices[0])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        self.auto_contrast = layers.AutoContrast(
            value_range=self.value_range,
            data_format=data_format,
        )

        self.equalization = layers.Equalization(
            value_range=self.value_range,
            data_format=data_format,
        )

    def _augmentation_space(self, num_bins):
        return {
            "ShearX": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "ShearY": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "TranslateX": self.backend.numpy.linspace(-1, 1, num_bins),
            "TranslateY": self.backend.numpy.linspace(-1, 1, num_bins),
            "Rotate": self.backend.numpy.linspace(-1, 1, num_bins),
            "Brightness": self.backend.numpy.linspace(-1, 1, num_bins),
            "Color": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Contrast": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Sharpness": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Solarize": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Posterize": 8. - (self.backend.numpy.arange(num_bins, dtype='float32') / ((num_bins - 1.) / 4)),
        }

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if backend_utils.in_tf_graph():
            self.backend.set_backend("tensorflow")

            for layer_name in self._AUGMENT_LAYERS:
                if layer_name == "Identity":
                    continue
                augmentation_layer = getattr(self, layer_name)
                augmentation_layer.backend.set_backend("tensorflow")

        transformation = {}
        random_indices = self.backend.random.shuffle(
            self.backend.numpy.arange(len(self._AUGMENT_LAYERS)),
            seed=self.seed)[:self.num_ops]
        for layer_idx in random_indices:
            layer_name = self._AUGMENT_LAYERS[layer_idx]
            if layer_name == "Identity":
                continue
            augmentation_layer = getattr(self, layer_name)
            transformation[layer_name] = augmentation_layer.get_random_transformation(data,
                                                                                      training=training,
                                                                                      seed=self._get_seed_generator(
                                                                                          self.backend._backend))

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
            "num_ops": self.num_ops,
            "magnitude": self.magnitude,
            "num_magnitude_bins": self.num_magnitude_bins,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
