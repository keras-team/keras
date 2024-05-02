from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer


@keras_export("keras.layers.Resizing")
class Resizing(TFDataLayer):
    """A preprocessing layer which resizes images.

    This layer resizes an image input to a target height and width. The input
    should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
    format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`,
        or `(..., channels, target_height, target_width)`,
        in `"channels_first"` format.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`,
            `"lanczos3"`, `"lanczos5"`. Defaults to `"bilinear"`.
        crop_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            cropped so as to return the
            largest possible window in the image (of size `(height, width)`)
            that matches the target aspect ratio. By default
            (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
        pad_to_aspect_ratio: If `True`, pad the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            evenly padded on the short side.
        fill_mode: When using `pad_to_aspect_ratio=True`, padded areas
            are filled according to the given mode. Only `"constant"` is
            supported at this time
            (fill with constant value, equal to `fill_value`).
        fill_value: Float. Padding value to use when `pad_to_aspect_ratio=True`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    """

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        fill_mode="constant",
        fill_value=0.0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.data_format = backend.standardize_data_format(data_format)
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def call(self, inputs):
        size = (self.height, self.width)
        return self.backend.image.resize(
            inputs,
            size=size,
            interpolation=self.interpolation,
            data_format=self.data_format,
            crop_to_aspect_ratio=self.crop_to_aspect_ratio,
            pad_to_aspect_ratio=self.pad_to_aspect_ratio,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
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
            "interpolation": self.interpolation,
            "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
            "pad_to_aspect_ratio": self.pad_to_aspect_ratio,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "data_format": self.data_format,
        }
        return {**base_config, **config}
