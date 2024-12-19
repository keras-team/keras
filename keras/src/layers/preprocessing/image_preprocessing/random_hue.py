from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.RandomHue")
class RandomHue(BaseImagePreprocessingLayer):
    """Randomly adjusts the hue on given images.

    This layer will randomly increase/reduce the hue for the input RGB
    images.

    The image hue is adjusted by converting the image(s) to HSV and rotating the
    hue channel (H) by delta. The image is then converted back to RGB.

    Args:
        factor: A single float or a tuple of two floats.
            `factor` controls the extent to which the
            image hue is impacted. `factor=0.0` makes this layer perform a
            no-op operation, while a value of `1.0` performs the most aggressive
            contrast adjustment available. If a tuple is used, a `factor` is
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        value_range: the range of values the incoming images will have.
            Represented as a two-number tuple written `[low, high]`. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        seed: Integer. Used to create a random seed.

    Example:

    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    random_hue = keras.layers.RandomHue(factor=0.5, value_range=[0, 1])
    images = keras.ops.cast(images, "float32")
    augmented_images_batch = random_hue(images[:8])
    ```
    """

    _USE_BASE_FACTOR = True
    _FACTOR_BOUNDS = (0, 1)

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
        self.value_range = value_range
        self.seed = seed
        self.generator = self.backend.random.SeedGenerator(seed)

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
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)
        invert = self.backend.random.uniform((batch_size,), seed=seed)

        invert = self.backend.numpy.where(
            invert > 0.5,
            -self.backend.numpy.ones_like(invert),
            self.backend.numpy.ones_like(invert),
        )
        factor = self.backend.random.uniform(
            (batch_size,),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )
        return {"factor": invert * factor * 0.5}

    def transform_images(self, images, transformation=None, training=True):
        def _apply_random_hue(images, transformation):
            images = self.backend.cast(images, self.compute_dtype)
            images = self._transform_value_range(
                images, self.value_range, (0, 1)
            )
            adjust_factors = transformation["factor"]
            adjust_factors = self.backend.cast(adjust_factors, images.dtype)
            adjust_factors = self.backend.numpy.expand_dims(adjust_factors, -1)
            adjust_factors = self.backend.numpy.expand_dims(adjust_factors, -1)
            images = self.backend.image.rgb_to_hsv(
                images, data_format=self.data_format
            )
            if self.data_format == "channels_first":
                h_channel = images[:, 0, :, :] + adjust_factors
                h_channel = self.backend.numpy.where(
                    h_channel > 1.0, h_channel - 1.0, h_channel
                )
                h_channel = self.backend.numpy.where(
                    h_channel < 0.0, h_channel + 1.0, h_channel
                )
                images = self.backend.numpy.stack(
                    [h_channel, images[:, 1, :, :], images[:, 2, :, :]], axis=1
                )
            else:
                h_channel = images[..., 0] + adjust_factors
                h_channel = self.backend.numpy.where(
                    h_channel > 1.0, h_channel - 1.0, h_channel
                )
                h_channel = self.backend.numpy.where(
                    h_channel < 0.0, h_channel + 1.0, h_channel
                )
                images = self.backend.numpy.stack(
                    [h_channel, images[..., 1], images[..., 2]], axis=-1
                )
            images = self.backend.image.hsv_to_rgb(
                images, data_format=self.data_format
            )
            images = self.backend.numpy.clip(images, 0, 1)
            images = self._transform_value_range(
                images, (0, 1), self.value_range
            )
            images = self.backend.cast(images, self.compute_dtype)
            return images

        if training:
            images = _apply_random_hue(images, transformation)
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
