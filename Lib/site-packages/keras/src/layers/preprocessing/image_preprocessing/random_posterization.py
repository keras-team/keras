from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.RandomPosterization")
class RandomPosterization(BaseImagePreprocessingLayer):
    """Reduces the number of bits for each color channel.

    References:
    - [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
    - [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)

    Args:
        value_range: a tuple or a list of two elements. The first value
            represents the lower bound for values in passed images, the second
            represents the upper bound. Images passed to the layer should have
            values within `value_range`. Defaults to `(0, 255)`.
        factor: integer, the number of bits to keep for each channel. Must be a
            value between 1-8.
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (1, 8)
    _MAX_FACTOR = 8
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
        self.generator = self.backend.random.SeedGenerator(seed)

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

        if self.factor[0] != self.factor[1]:
            factor = self.backend.random.randint(
                (batch_size,),
                minval=self.factor[0],
                maxval=self.factor[1],
                seed=seed,
                dtype="uint8",
            )
        else:
            factor = (
                self.backend.numpy.ones((batch_size,), dtype="uint8")
                * self.factor[0]
            )

        shift_factor = self._MAX_FACTOR - factor
        return {"shift_factor": shift_factor}

    def transform_images(self, images, transformation=None, training=True):
        if training:
            shift_factor = transformation["shift_factor"]

            shift_factor = self.backend.numpy.reshape(
                shift_factor, self.backend.shape(shift_factor) + (1, 1, 1)
            )

            images = self._transform_value_range(
                images,
                original_range=self.value_range,
                target_range=(0, 255),
                dtype=self.compute_dtype,
            )

            images = self.backend.cast(images, "uint8")
            images = self.backend.numpy.bitwise_left_shift(
                self.backend.numpy.bitwise_right_shift(images, shift_factor),
                shift_factor,
            )
            images = self.backend.cast(images, self.compute_dtype)

            images = self._transform_value_range(
                images,
                original_range=(0, 255),
                target_range=self.value_range,
                dtype=self.compute_dtype,
            )

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
