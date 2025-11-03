from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.RandomColorDegeneration")
class RandomColorDegeneration(BaseImagePreprocessingLayer):
    """Randomly performs the color degeneration operation on given images.

    The sharpness operation first converts an image to gray scale, then back to
    color. It then takes a weighted average between original image and the
    degenerated image. This makes colors appear more dull.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        factor: A tuple of two floats or a single float.
            `factor` controls the extent to which the
            image sharpness is impacted. `factor=0.0` makes this layer perform a
            no-op operation, while a value of 1.0 uses the degenerated result
            entirely. Values between 0 and 1 result in linear interpolation
            between the original image and the sharpened image.
            Values should be between `0.0` and `1.0`. If a tuple is used, a
            `factor` is sampled between the two values for every image
            augmented. If a single float is used, a value between `0.0` and the
            passed float is sampled. In order to ensure the value is always the
            same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.
    """

    _VALUE_RANGE_VALIDATION_ERROR = (
        "The `value_range` argument should be a list of two numbers. "
    )

    def __init__(
        self,
        factor,
        value_range=(0, 255),
        data_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self._set_value_range(value_range)
        self.seed = seed
        self.generator = SeedGenerator(seed)

    def _set_value_range(self, value_range):
        if not isinstance(value_range, (tuple, list)):
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        if len(value_range) != 2:
            raise ValueError(
                self._VALUE_RANGE_VALIDATION_ERROR
                + f"Received: value_range={value_range}"
            )
        self.value_range = sorted(value_range)

    def get_random_transformation(self, data, training=True, seed=None):
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        images_shape = self.backend.shape(images)
        rank = len(images_shape)
        if rank == 3:
            batch_size = 1
        elif rank == 4:
            batch_size = images_shape[0]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Received: "
                f"inputs.shape={images_shape}"
            )

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        factor = self.backend.random.uniform(
            (batch_size, 1, 1, 1),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )
        factor = factor
        return {"factor": factor}

    def transform_images(self, images, transformation=None, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)
            factor = self.backend.cast(
                transformation["factor"], self.compute_dtype
            )
            degenerates = self.backend.image.rgb_to_grayscale(
                images, data_format=self.data_format
            )
            images = images + factor * (degenerates - images)
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
        return segmentation_masks

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        return bounding_boxes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "value_range": self.value_range,
                "seed": self.seed,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
