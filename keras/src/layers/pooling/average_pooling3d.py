from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_pooling import BasePooling


@keras_export(["keras.layers.AveragePooling3D", "keras.layers.AvgPool3D"])
class AveragePooling3D(BasePooling):
    """Average pooling operation for 3D data (spatial or spatio-temporal).

    Downsamples the input along its spatial dimensions (depth, height, and
    width) by taking the average value over an input window (of size defined by
    `pool_size`) for each channel of the input. The window is shifted by
    `strides` along each dimension.

    Args:
        pool_size: int or tuple of 3 integers, factors by which to downscale
            (dim1, dim2, dim3). If only one integer is specified, the same
            window length will be used for all dimensions.
        strides: int or tuple of 3 integers, or None. Strides values. If None,
            it will default to `pool_size`. If only one int is specified, the
            same stride size will be used for all dimensions.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. If you never set it, then it
            will be `"channels_last"`.

    Input shape:

    - If `data_format="channels_last"`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:

    - If `data_format="channels_last"`:
        5D tensor with shape:
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape:
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`

    Example:

    ```python
    depth = 30
    height = 30
    width = 30
    channels = 3

    inputs = keras.layers.Input(shape=(depth, height, width, channels))
    layer = keras.layers.AveragePooling3D(pool_size=3)
    outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
    ```
    """

    def __init__(
        self,
        pool_size,
        strides=None,
        padding="valid",
        data_format=None,
        name=None,
        **kwargs
    ):
        super().__init__(
            pool_size,
            strides,
            pool_dimensions=3,
            pool_mode="average",
            padding=padding,
            data_format=data_format,
            name=name,
            **kwargs,
        )
