from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.RandomGaussianBlur")
class RandomGaussianBlur(BaseImagePreprocessingLayer):
    """Randomly performs the sharpness operation on given images.

    The sharpness operation first performs a blur, then blends between the
    original image and the processed image. This operation adjusts the clarity
    of the edges in an image, ranging from blurred to enhanced sharpness.

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

    def __init__(
        self,
        factor=1.0,
        kernel_size=3,
        sigma=1.0,
        value_range=(0, 255),
        data_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.kernel_size = self._set_kernel_size(kernel_size, "kernel_size")
        self.sigma = self._set_factor_by_name(sigma, "sigma")
        self.value_range = value_range
        self.seed = seed
        self.generator = SeedGenerator(seed)

        if self.data_format == "channels_first":
            self.channel_axis = -3
        else:
            self.channel_axis = -1

    def _set_kernel_size(self, factor, name):
        error_msg = f"{name} must be an odd number. Received: {name}={factor}"
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                error_msg = (
                    f"The `{name}` argument should be a number "
                    "(or a list of two numbers) "
                    f"Received: {name}={factor}"
                )
                raise ValueError(error_msg)
            if (factor[0] % 2 == 0) or (factor[1] % 2 == 0):
                raise ValueError(error_msg)
            lower, upper = factor
        elif isinstance(factor, (int, float)):
            if factor % 2 == 0:
                raise ValueError(error_msg)
            lower, upper = factor, factor
        else:
            raise ValueError(error_msg)

        return lower, upper

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
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        seed = seed or self._get_seed_generator(self.backend._backend)

        blur_probability = self.backend.random.uniform(
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
        should_apply_blur = random_threshold < blur_probability

        blur_factor = self.backend.random.uniform(
            shape=(2,),
            minval=self.sigma[0],
            maxval=self.sigma[1],
            seed=seed,
        )

        return {
            "should_apply_blur": should_apply_blur,
            "blur_factor": blur_factor,
        }

    def create_gaussian_kernel(self, kernel_size, sigma, num_channels):
        def get_gaussian_kernel1d(size, sigma):
            x = (
                self.backend.numpy.arange(size, dtype=self.compute_dtype)
                - (size - 1) / 2
            )
            kernel1d = self.backend.numpy.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / self.backend.numpy.sum(kernel1d)

        def get_gaussian_kernel2d(size, sigma):
            kernel1d_x = get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_y = get_gaussian_kernel1d(size[1], sigma[1])
            return self.backend.numpy.tensordot(kernel1d_y, kernel1d_x, axes=0)

        kernel = get_gaussian_kernel2d(kernel_size, sigma)

        kernel = self.backend.numpy.reshape(
            kernel, (kernel_size[0], kernel_size[1], 1, 1)
        )
        kernel = self.backend.numpy.tile(kernel, [1, 1, num_channels, 1])

        kernel = self.backend.cast(kernel, self.compute_dtype)

        return kernel

    def transform_images(self, images, transformation=None, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            if self.data_format == "channels_first":
                images = self.backend.numpy.swapaxes(images, -3, -1)

            blur_factor = transformation["blur_factor"]
            should_apply_blur = transformation["should_apply_blur"]

            kernel = self.create_gaussian_kernel(
                self.kernel_size,
                blur_factor,
                self.backend.shape(images)[self.channel_axis],
            )

            blur_images = self.backend.nn.depthwise_conv(
                images,
                kernel,
                strides=1,
                padding="same",
                data_format="channels_last",
            )

            if self.data_format == "channels_first":
                images = self.backend.numpy.swapaxes(images, -3, -1)

            images = self.backend.numpy.where(
                should_apply_blur[:, None, None, None],
                blur_images,
                images,
            )

            images = self.backend.numpy.clip(
                images, self.value_range[0], self.value_range[1]
            )

            images = self.backend.cast(images, dtype=self.compute_dtype)

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

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "kernel_size": self.kernel_size,
                "sigma": self.sigma,
                "value_range": self.value_range,
                "seed": self.seed,
            }
        )
        return config
