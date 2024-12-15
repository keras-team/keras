from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random import SeedGenerator


@keras_export("keras.layers.RandomSaturation")
class RandomSaturation(BaseImagePreprocessingLayer):
    """Randomly adjusts the saturation on given images.

    This layer will randomly increase/reduce the saturation for the input RGB
    images.

    Args:
        factor: A tuple of two floats or a single float.
            `factor` controls the extent to which the image saturation
            is impacted. `factor=0.5` makes this layer perform a no-op
            operation. `factor=0.0` makes the image fully grayscale.
            `factor=1.0` makes the image fully saturated. Values should
            be between `0.0` and `1.0`. If a tuple is used, a `factor`
            is sampled between the two values for every image augmented.
            If a single float is used, a value between `0.0` and the passed
            float is sampled. To ensure the value is always the same,
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    images = images.astype("float32") / 255.0
    random_saturation = keras.layers.RandomSaturation(factor=0.2)
    augmented_images = random_saturation(images)
    ```
    """

    def __init__(self, factor, data_format=None, seed=None, **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.seed = seed
        self.generator = SeedGenerator(seed)

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

        factor = self.backend.random.uniform(
            (batch_size,),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )
        factor = factor / (1 - factor)
        return {"factor": factor}

    def transform_images(self, images, transformation=None, training=True):
        if training:
            adjust_factors = transformation["factor"]
            adjust_factors = self.backend.cast(adjust_factors, images.dtype)
            adjust_factors = self.backend.numpy.expand_dims(adjust_factors, -1)
            adjust_factors = self.backend.numpy.expand_dims(adjust_factors, -1)

            images = self.backend.image.rgb_to_hsv(
                images, data_format=self.data_format
            )

            if self.data_format == "channels_first":
                s_channel = self.backend.numpy.multiply(
                    images[:, 1, :, :], adjust_factors
                )
                s_channel = self.backend.numpy.clip(s_channel, 0.0, 1.0)
                images = self.backend.numpy.stack(
                    [images[:, 0, :, :], s_channel, images[:, 2, :, :]], axis=1
                )
            else:
                s_channel = self.backend.numpy.multiply(
                    images[..., 1], adjust_factors
                )
                s_channel = self.backend.numpy.clip(s_channel, 0.0, 1.0)
                images = self.backend.numpy.stack(
                    [images[..., 0], s_channel, images[..., 2]], axis=-1
                )

            images = self.backend.image.hsv_to_rgb(
                images, data_format=self.data_format
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
                "seed": self.seed,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
