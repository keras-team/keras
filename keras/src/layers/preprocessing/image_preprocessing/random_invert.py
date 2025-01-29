from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.RandomInvert")
class RandomInvert(BaseImagePreprocessingLayer):
    """Preprocessing layer for random inversion of image colors.

    This layer randomly inverts the colors of input images with a specified
    probability range. When applied, each image has a chance of having its
    colors inverted, where the pixel values are transformed to their
    complementary values. Images that are not selected for inversion
    remain unchanged.

    Args:
        factor: A single float or a tuple of two floats.
            `factor` controls the probability of inverting the image colors.
            If a tuple is provided, the value is sampled between the two values
            for each image, where `factor[0]` is the minimum and `factor[1]` is
            the maximum probability. If a single float is provided, a value
            between `0.0` and the provided float is sampled.
            Defaults to `(0, 1)`.
        value_range: a tuple or a list of two elements. The first value
            represents the lower bound for values in passed images, the second
            represents the upper bound. Images passed to the layer should have
            values within `value_range`. Defaults to `(0, 255)`.
        seed: Integer. Used to create a random seed.
    """

    _USE_BASE_FACTOR = False
    _FACTOR_BOUNDS = (0, 1)

    def __init__(
        self,
        factor=1.0,
        value_range=(0, 255),
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self._set_factor(factor)
        self.value_range = value_range
        self.seed = seed
        self.generator = self.backend.random.SeedGenerator(seed)

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data

        seed = seed or self._get_seed_generator(self.backend._backend)

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

        invert_probability = self.backend.random.uniform(
            shape=(batch_size,),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=seed,
        )

        random_threshold = self.backend.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=1,
            seed=seed,
        )

        apply_inversion = random_threshold < invert_probability
        return {"apply_inversion": apply_inversion}

    def transform_images(self, images, transformation, training=True):
        if training:
            images = self.backend.cast(images, self.compute_dtype)
            apply_inversion = transformation["apply_inversion"]
            return self.backend.numpy.where(
                apply_inversion[:, None, None, None],
                self.value_range[1] - images,
                images,
            )
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
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
