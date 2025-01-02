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

    _AUGMENT_LAYERS = ["random_shear", "random_translation", "random_rotation", "random_brightness",
                       "random_color_degeneration", "random_contrast", "random_sharpness",
                       "random_posterization", "solarization", "auto_contrast", "equalization"]

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

        op_index = self.backend.random.uniform((2,),
                                               minval=0, maxval=len(augmentation_space["ShearX"]),
                                               seed=self.seed)
        self.random_shear = layers.RandomShear(
            x_factor=float(augmentation_space["ShearX"][int(op_index[0])]),
            y_factor=float(augmentation_space["ShearY"][int(op_index[1])]),
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((2,),
                                               minval=0, maxval=len(augmentation_space["TranslateX"]),
                                               seed=self.seed)
        self.random_translation = layers.RandomTranslation(
            height_factor=float(augmentation_space["TranslateX"][int(op_index[0])]),
            width_factor=float(augmentation_space["TranslateY"][int(op_index[1])]),
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((1,),
                                               minval=0, maxval=len(augmentation_space["Rotate"]),
                                               seed=self.seed)
        self.random_rotation = layers.RandomRotation(
            factor=float(augmentation_space["Rotate"][int(op_index[0])]),
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((1,),
                                               minval=0, maxval=len(augmentation_space["Brightness"]),
                                               seed=self.seed)
        self.random_brightness = layers.RandomBrightness(
            factor=float(augmentation_space["Brightness"][int(op_index[0])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((1,),
                                               minval=0, maxval=len(augmentation_space["Color"]),
                                               seed=self.seed)
        self.random_color_degeneration = layers.RandomColorDegeneration(
            factor=float(augmentation_space["Color"][int(op_index[0])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((1,),
                                               minval=0, maxval=len(augmentation_space["Contrast"]),
                                               seed=self.seed)
        self.random_contrast = layers.RandomContrast(
            factor=float(augmentation_space["Contrast"][int(op_index[0])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((1,),
                                               minval=0, maxval=len(augmentation_space["Sharpness"]),
                                               seed=self.seed)
        self.random_sharpness = layers.RandomSharpness(
            factor=float(augmentation_space["Sharpness"][int(op_index[0])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((1,),
                                               minval=0, maxval=len(augmentation_space["Posterize"]),
                                               seed=self.seed)
        self.random_posterization = layers.RandomPosterization(
            factor=int(augmentation_space["Posterize"][int(op_index[0])]),
            value_range=self.value_range,
            seed=self.seed,
            data_format=data_format,
        )

        op_index = self.backend.random.uniform((2,),
                                               minval=0, maxval=len(augmentation_space["Solarize"]),
                                               seed=self.seed)
        self.solarization = layers.Solarization(
            addition_factor=int(augmentation_space["Solarize"][int(op_index[0])]),
            threshold_factor=int(augmentation_space["Solarize"][int(op_index[1])]),
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
            "Identity": self.backend.convert_to_tensor(
                0.0, dtype=self.compute_dtype
            ),
            "ShearX": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "ShearY": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "TranslateX": self.backend.numpy.linspace(-1, 1, num_bins),
            "TranslateY": self.backend.numpy.linspace(-1, 1, num_bins),
            "Rotate": self.backend.numpy.linspace(-1, 1, num_bins),
            "Brightness": self.backend.numpy.linspace(-1, 1, num_bins),
            "Color": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Contrast": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Sharpness": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Posterize": 8. - (self.backend.numpy.arange(num_bins, dtype='float32') / ((num_bins - 1.) / 4)),
            "Solarize": self.backend.numpy.linspace(0.0, 1.0, num_bins),
            "Equalize": self.backend.convert_to_tensor(
                0.0, dtype=self.compute_dtype
            ),
        }

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if backend_utils.in_tf_graph():
            self.backend.set_backend("tensorflow")
            self.random_shear.backend.set_backend("tensorflow")
            self.random_translation.backend.set_backend("tensorflow")
            self.random_rotation.backend.set_backend("tensorflow")
            self.random_brightness.backend.set_backend("tensorflow")
            self.random_color_degeneration.backend.set_backend("tensorflow")
            self.random_contrast.backend.set_backend("tensorflow")
            self.random_sharpness.backend.set_backend("tensorflow")
            self.random_posterization.backend.set_backend("tensorflow")
            self.solarization.backend.set_backend("tensorflow")
            self.auto_contrast.backend.set_backend("tensorflow")
            self.equalization.backend.set_backend("tensorflow")

        transformation = {}
        random_index = self.backend.random.randint((1,), 0, len(self._AUGMENT_LAYERS), seed=self.seed)
        for layer_idx in random_index:
            layer_name = self._AUGMENT_LAYERS[layer_idx]
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
