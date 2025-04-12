from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomElasticTransform")
class RandomElasticTransform(BaseImagePreprocessingLayer):
    """A preprocessing layer that applies random elastic transformations.

    This layer distorts input images by applying elastic deformations,
    simulating a physically realistic transformation. The magnitude of the
    distortion is controlled by the `scale` parameter, while the `factor`
    determines the probability of applying the transformation.

    Args:
        factor: A single float or a tuple of two floats.
            `factor` controls the probability of applying the transformation.
            - `factor=0.0` ensures no erasing is applied.
            - `factor=1.0` means erasing is always applied.
            - If a tuple `(min, max)` is provided, a probability value
              is sampled between `min` and `max` for each image.
            - If a single float is provided, a probability is sampled
              between `0.0` and the given float.
            Default is 1.0.
        scale: A float or a tuple of two floats defining the magnitude of
            the distortion applied.
            - If a tuple `(min, max)` is provided, a random scale value is
              sampled within this range.
            - If a single float is provided, a random scale value is sampled
              between `0.0` and the given float.
            Default is 1.0.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about the edge of the last
                pixel.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The input is extended by wrapping around to the opposite edge.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
            Note that when using torch backend, `"reflect"` is redirected to
            `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not
            support `"reflect"`.
            Note that torch backend does not support `"wrap"`.
        fill_value: a float represents the value to be filled outside the
            boundaries when `fill_mode="constant"`.
        value_range: the range of values the incoming images will have.
            Represented as a two-number tuple written `[low, high]`. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        seed: Integer. Used to create a random seed.

    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")
    _SUPPORTED_FILL_MODES = {
        "constant",
        "nearest",
        "wrap",
        "mirror",
        "reflect",
    }

    def __init__(
        self,
        factor=1.0,
        scale=1.0,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        value_range=(0, 255),
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.scale = self._set_factor_by_name(scale, "scale")
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.value_range = value_range
        self.seed = seed
        self.generator = SeedGenerator(seed)

        if interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

        if fill_mode not in self._SUPPORTED_FILL_MODES:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODES}."
            )

        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
            self.channel_axis = -3
        else:
            self.height_axis = -3
            self.width_axis = -2
            self.channel_axis = -1

    def _set_factor_by_name(self, factor, name):
        error_msg = (
            f"The `{name}` argument should be a number "
            "(or a list of two numbers) "
            "in the range "
            f"[{self._FACTOR_BOUNDS[0]}, {self._FACTOR_BOUNDS[1]}]. "
            f"Received: factor={factor}"
        )
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(error_msg)
            if (
                factor[0] > self._FACTOR_BOUNDS[1]
                or factor[1] < self._FACTOR_BOUNDS[0]
            ):
                raise ValueError(error_msg)
            lower, upper = sorted(factor)
        elif isinstance(factor, (int, float)):
            if (
                factor < self._FACTOR_BOUNDS[0]
                or factor > self._FACTOR_BOUNDS[1]
            ):
                raise ValueError(error_msg)
            factor = abs(factor)
            lower, upper = [max(-factor, self._FACTOR_BOUNDS[0]), factor]
        else:
            raise ValueError(error_msg)
        return lower, upper

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if (self.scale[1] == 0) or (self.factor[1] == 0):
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        images_shape = self.backend.shape(images)
        unbatched = len(images_shape) == 3
        if unbatched:
            batch_size = 1
        else:
            batch_size = images_shape[0]

        seed = seed or self._get_seed_generator(self.backend._backend)

        transformation_probability = self.backend.random.uniform(
            shape=(batch_size,),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )

        random_threshold = self.backend.random.uniform(
            shape=(batch_size,),
            minval=0.0,
            maxval=1.0,
            seed=seed,
        )
        apply_transform = random_threshold < transformation_probability

        distortion_factor = self.backend.random.uniform(
            shape=(),
            minval=self.scale[0],
            maxval=self.scale[1],
            seed=seed,
            dtype=self.compute_dtype,
        )

        return {
            "apply_transform": apply_transform,
            "distortion_factor": distortion_factor,
            "seed": seed,
        }

    def get_elastic_transform_params(self, height, width, factor):
        alpha_scale = 0.1 * factor
        sigma_scale = 0.05 * factor

        alpha = max(height, width) * alpha_scale
        sigma = min(height, width) * sigma_scale

        return alpha, sigma

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training and transformation is not None:
            apply_transform = transformation["apply_transform"]
            distortion_factor = transformation["distortion_factor"]
            seed = transformation["seed"]

            height, width = (
                images.shape[self.height_axis],
                images.shape[self.width_axis],
            )

            alpha, sigma = self.get_elastic_transform_params(
                height, width, distortion_factor
            )

            transformed_images = self.backend.image.elastic_transform(
                images,
                alpha=alpha,
                sigma=sigma,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                seed=seed,
                data_format=self.data_format,
            )

            apply_transform = (
                apply_transform[:, None, None]
                if len(images.shape) == 3
                else apply_transform[:, None, None, None]
            )

            images = self.backend.numpy.where(
                apply_transform,
                transformed_images,
                images,
            )

            images = self.backend.numpy.clip(
                images, self.value_range[0], self.value_range[1]
            )

            images = self.backend.cast(images, self.compute_dtype)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "factor": self.factor,
            "scale": self.scale,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        return {**base_config, **config}
