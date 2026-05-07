from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.RandomSharpness")
class RandomSharpness(BaseImagePreprocessingLayer):
    """Randomly performs the sharpness operation on given images.

    The sharpness operation first performs a blur, then blends between the
    original image and the processed image. This operation adjusts the clarity
    of the edges in an image, ranging from blurred to enhanced sharpness.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        factor: A tuple of two floats or a single float.
            `factor` controls the extent to which the image sharpness
            is impacted. `factor=0.0` results in a fully blurred image,
            `factor=0.5` applies no operation (preserving the original image),
            and `factor=1.0` enhances the sharpness beyond the original. Values
            should be between `0.0` and `1.0`. If a tuple is used, a `factor`
            is sampled between the two values for every image augmented.
            If a single float is used, a value between `0.0` and the passed
            float is sampled. To ensure the value is always the same,
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        value_range: the range of values the incoming images will have.
            Represented as a two-number tuple written `[low, high]`. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        seed: Integer. Used to create a random seed.
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

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
            (batch_size,),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )
        return {"factor": factor}

    def transform_images(self, images, transformation=None, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            if self.data_format == "channels_first":
                images = self.backend.numpy.swapaxes(images, -3, -1)

            sharpness_factor = self.backend.cast(
                transformation["factor"] * 2, dtype=self.compute_dtype
            )
            sharpness_factor = self.backend.numpy.reshape(
                sharpness_factor, (-1, 1, 1, 1)
            )

            num_channels = self.backend.shape(images)[-1]

            a, b = 1.0 / 13.0, 5.0 / 13.0
            kernel = self.backend.convert_to_tensor(
                [[a, a, a], [a, b, a], [a, a, a]], dtype=self.compute_dtype
            )
            kernel = self.backend.numpy.reshape(kernel, (3, 3, 1, 1))
            kernel = self.backend.numpy.tile(kernel, [1, 1, num_channels, 1])
            kernel = self.backend.cast(kernel, self.compute_dtype)

            smoothed_image = self.backend.nn.depthwise_conv(
                images,
                kernel,
                strides=1,
                padding="same",
                data_format="channels_last",
            )

            smoothed_image = self.backend.cast(
                smoothed_image, dtype=self.compute_dtype
            )
            images = images + (1.0 - sharpness_factor) * (
                smoothed_image - images
            )

            images = self.backend.numpy.clip(
                images, self.value_range[0], self.value_range[1]
            )

            if self.data_format == "channels_first":
                images = self.backend.numpy.swapaxes(images, -3, -1)

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
