from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)


@keras_export("keras.layers.Grayscale")
class Grayscale(BaseImagePreprocessingLayer):
    """Grayscale is a preprocessing layer that transforms RGB images to
    Grayscale images.
    Input images should have values in the range of [0, 255].

    Args:
        output_channels.
            Number color channels present in the output image.
            The output_channels can be 1 or 3. RGB image with shape
            (..., height, width, 3) will have the following shapes
            after the `Grayscale` operation:
                 a. (..., height, width, 1) if output_channels = 1
                 b. (..., height, width, 3) if output_channels = 3.
    """

    def __init__(
        self,
        output_channels=1,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self._set_output_channels(output_channels)

    def _set_output_channels(self, output_channels):
        if output_channels not in [1, 3]:
            raise ValueError(
                "Received invalid argument output_channels. "
                f"output_channels must be in 1 or 3. Got {output_channels}"
            )
        self.output_channels = output_channels

    def transform_images(self, images, transformation=None, training=True):
        def _grayscale_to_rgb(images):
            images_shape = self.backend.shape(images)
            rank = len(images_shape)
            if rank == 3:
                images = self.backend.numpy.expand_dims(images, axis=0)

            if self.data_format == "channels_last":
                rgb_images = self.backend.numpy.tile(images, (1, 1, 1, 3))
            else:
                rgb_images = self.backend.numpy.tile(images, (1, 3, 1, 1))

            return (
                self.backend.numpy.squeeze(rgb_images, axis=0)
                if rank == 3
                else rgb_images
            )

        grayscale = self.backend.image.rgb_to_grayscale(
            images, data_format=self.data_format
        )
        if self.output_channels == 1:
            return grayscale
        return _grayscale_to_rgb(grayscale)

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

    def get_config(self):
        config = super().get_config()
        config.update({"output_channels": self.output_channels})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
