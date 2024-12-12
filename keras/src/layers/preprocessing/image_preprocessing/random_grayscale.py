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

    **Note:** This layer is safe to use inside a `tf.data` pipeline
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

    def __init__(self, factor=0.5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        if factor < 0 or factor > 1:
            raise ValueError(
                "`factor` should be between 0 and 1. "
                f"Received: factor={factor}"
            )
        self.factor = factor
        self.data_format = backend.standardize_data_format(data_format)
        self.random_generator = self.backend.random.SeedGenerator()

    def get_random_transformation(self, images, training=True, seed=None):
        batch_size = self.backend.core.shape(images)[0]
        random_values = self.backend.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=1,
            seed=self.random_generator,
        )
        
        broadcast_shape = (
            [1, 1, 1, 1] if self.data_format == "channels_last" else [1, 1, 1, 1]
        )
        should_apply = self.backend.numpy.reshape(
            random_values < self.factor, (-1,) + tuple(broadcast_shape[1:])
        )
        return should_apply

    def transform_images(self, images, transformations=None, **kwargs):
        should_apply = (
            transformations
            if transformations is not None
            else self.get_random_transformation(images)
        )
        
        grayscale_images = self.backend.image.rgb_to_grayscale(
            images, data_format=self.data_format
        )
        
        if self.data_format == "channels_last":
            grayscale_images = self.backend.numpy.repeat(grayscale_images, 3, axis=-1)
        else:  
            grayscale_images = self.backend.numpy.repeat(grayscale_images, 3, axis=1)

        return self.backend.numpy.where(should_apply, grayscale_images, images)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_spec(self, inputs, **kwargs):
        return inputs

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
