from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.utils import image_utils


@keras_export("keras.layers.CenterCrop")
class CenterCrop(BaseImagePreprocessingLayer):
    """A preprocessing layer which crops images.

    This layers crops the central portion of the images to a target size. If an
    image is smaller than the target size, it will be resized and cropped
    so as to return the largest possible window in the image that matches
    the target aspect ratio.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`,
        or `(..., channels, target_height, target_width)`,
        in `"channels_first"` format.

    If the input height/width is even and the target height/width is odd (or
    inversely), the input image is left-padded by 1 pixel.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
    """

    _USE_BASE_FACTOR = False

    def __init__(self, height, width, data_format=None, **kwargs):
        super().__init__(data_format=data_format, **kwargs)
        self.height = height
        self.width = width

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        raise NotImplementedError

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def transform_images(self, images, transformation=None, training=True):
        inputs = self.backend.cast(images, self.compute_dtype)
        if self.data_format == "channels_first":
            init_height = inputs.shape[-2]
            init_width = inputs.shape[-1]
        else:
            init_height = inputs.shape[-3]
            init_width = inputs.shape[-2]

        if init_height is None or init_width is None:
            # Dynamic size case. TODO.
            raise ValueError(
                "At this time, CenterCrop can only "
                "process images with a static spatial "
                f"shape. Received: inputs.shape={inputs.shape}"
            )

        h_diff = init_height - self.height
        w_diff = init_width - self.width

        h_start = int(h_diff / 2)
        w_start = int(w_diff / 2)

        if h_diff >= 0 and w_diff >= 0:
            if len(inputs.shape) == 4:
                if self.data_format == "channels_first":
                    return inputs[
                        :,
                        :,
                        h_start : h_start + self.height,
                        w_start : w_start + self.width,
                    ]
                return inputs[
                    :,
                    h_start : h_start + self.height,
                    w_start : w_start + self.width,
                    :,
                ]
            elif len(inputs.shape) == 3:
                if self.data_format == "channels_first":
                    return inputs[
                        :,
                        h_start : h_start + self.height,
                        w_start : w_start + self.width,
                    ]
                return inputs[
                    h_start : h_start + self.height,
                    w_start : w_start + self.width,
                    :,
                ]
        return image_utils.smart_resize(
            inputs,
            [self.height, self.width],
            data_format=self.data_format,
            backend_module=self.backend,
        )

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if isinstance(input_shape[0], (list, tuple)) or len(
            input_shape
        ) not in (3, 4):
            raise ValueError(
                "`input_shape` must be a non-nested tuple or list "
                "of rank-1 with size 3 (unbatched) or 4 (batched). "
            )
        if len(input_shape) == 4:
            if self.data_format == "channels_last":
                input_shape[1] = self.height
                input_shape[2] = self.width
            else:
                input_shape[2] = self.height
                input_shape[3] = self.width
        else:
            if self.data_format == "channels_last":
                input_shape[0] = self.height
                input_shape[1] = self.width
            else:
                input_shape[1] = self.height
                input_shape[2] = self.width
        return tuple(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "height": self.height,
            "width": self.width,
            "data_format": self.data_format,
        }
        return {**base_config, **config}
