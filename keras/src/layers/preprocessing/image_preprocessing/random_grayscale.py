from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.RandomGrayscale")
class RandomGrayscale(BaseImagePreprocessingLayer):
    """Preprocessing layer for random conversion of RGB images to grayscale.

    This layer randomly converts input images to grayscale with a specified
    probability. When applied, it maintains the original number of channels
    but sets all channels to the same grayscale value. This can be useful
    for data augmentation and training models to be robust to color
    variations.

    The conversion preserves the perceived luminance of the original color
    image using standard RGB to grayscale conversion coefficients. Images
    that are not selected for conversion remain unchanged.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        probability: Float between 0 and 1, specifying the probability of
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

    Example:

    ```python
    # Create a random grayscale layer with 50% probability
    random_gray = keras.layers.RandomGrayscale(probability=0.5)

    # Apply to a single color image
    color_image = [...] # your input RGB image
    output_image = random_gray(color_image)

    # Use in a sequential model for data augmentation
    model = keras.Sequential([
        keras.layers.RandomGrayscale(probability=0.3),
        keras.layers.Conv2D(16, 3, activation='relu'),
        # ... rest of your model
    ])
    ```
    """

    def __init__(self, probability=0.5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        if probability < 0 or probability > 1:
            raise ValueError(
                "`probability` should be between 0 and 1. "
                f"Received: probability={probability}"
            )
        self.probability = probability
        self.data_format = backend.standardize_data_format(data_format)
        self.random_generator = self.backend.random.SeedGenerator()

    def transform_images(self, images, transformations=None, **kwargs):
        # Generate random values for batch
        random_values = self.backend.random.uniform(
            shape=(self.backend.core.shape(images)[0],),
            minval=0,
            maxval=1,
            seed=self.random_generator,
        )
        should_apply = self.backend.numpy.expand_dims(
            random_values < self.probability, axis=[1, 2, 3]
        )

        # Convert selected images to grayscale
        grayscale_images = self.backend.image.rgb_to_grayscale(
            images, data_format=self.data_format
        )

        output_images = self.backend.numpy.where(
            should_apply, grayscale_images, images
        )

        return output_images

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
        config.update({"probability": self.probability})
        return config
