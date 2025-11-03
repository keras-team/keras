from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.RandomGrayscale")
class RandomGrayscale(BaseImagePreprocessingLayer):
    """Preprocessing layer for random conversion of RGB images to grayscale.

    This layer randomly converts input images to grayscale with a specified
    factor. When applied, it maintains the original number of channels
    but sets all channels to the same grayscale value. This can be useful
    for data augmentation and training models to be robust to color
    variations.

    The conversion preserves the perceived luminance of the original color
    image using standard RGB to grayscale conversion coefficients. Images
    that are not selected for conversion remain unchanged.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        factor: Float between 0 and 1, specifying the factor of
            converting each image to grayscale. Defaults to 0.5. A value of
            1.0 means all images will be converted, while 0.0 means no images
            will be converted.
        data_format: String, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        Same as input shape. The output maintains the same number of channels
        as the input, even for grayscale-converted images where all channels
        will have the same value.
    """

    def __init__(self, factor=0.5, data_format=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        if factor < 0 or factor > 1:
            raise ValueError(
                f"`factor` should be between 0 and 1. Received: factor={factor}"
            )
        self.factor = factor
        self.data_format = backend.standardize_data_format(data_format)
        self.seed = seed
        self.generator = self.backend.random.SeedGenerator(seed)

    def get_random_transformation(self, images, training=True, seed=None):
        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)
        # Base case: Unbatched data
        batch_size = 1
        if len(images.shape) == 4:
            # This is a batch of images (4D input)
            batch_size = self.backend.core.shape(images)[0]

        random_values = self.backend.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=1,
            seed=seed,
        )
        should_apply = self.backend.numpy.expand_dims(
            random_values < self.factor, axis=[1, 2, 3]
        )
        return should_apply

    def transform_images(self, images, transformation, training=True):
        if training:
            should_apply = (
                transformation
                if transformation is not None
                else self.get_random_transformation(images)
            )

            grayscale_images = self.backend.image.rgb_to_grayscale(
                images, data_format=self.data_format
            )
            return self.backend.numpy.where(
                should_apply, grayscale_images, images
            )
        return images

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_spec(self, inputs, **kwargs):
        return backend.KerasTensor(
            inputs.shape, dtype=inputs.dtype, sparse=inputs.sparse
        )

    def transform_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def transform_labels(self, labels, transformations=None, **kwargs):
        return labels

    def transform_segmentation_masks(
        self, segmentation_masks, transformations=None, **kwargs
    ):
        return segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config
