from keras.src.api_export import keras_export
from keras.src.layers.pooling.base_pooling import BasePooling


@keras_export(["keras.layers.MaxPooling2D", "keras.layers.MaxPool2D"])
class MaxPooling2D(BasePooling):
    """Max pooling operation for 2D spatial data.

    Downsamples the input along its spatial dimensions (height and width)
    by taking the maximum value over an input window
    (of size defined by `pool_size`) for each channel of the input.
    The window is shifted by `strides` along each dimension.

    The resulting output when using the `"valid"` padding option has a spatial
    shape (number of rows or columns) of:
    `output_shape = math.floor((input_shape - pool_size) / strides) + 1`
    (when `input_shape >= pool_size`)

    The resulting output shape when using the `"same"` padding option is:
    `output_shape = math.floor((input_shape - 1) / strides) + 1`

    Args:
        pool_size: int or tuple of 2 integers, factors by which to downscale
            (dim1, dim2). If only one integer is specified, the same
            window length will be used for all dimensions.
        strides: int or tuple of 2 integers, or None. Strides values. If None,
            it will default to `pool_size`. If only one int is specified, the
            same stride size will be used for all dimensions.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.

    Input shape:

    - If `data_format="channels_last"`:
        4D tensor with shape `(batch_size, height, width, channels)`.
    - If `data_format="channels_first"`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:

    - If `data_format="channels_last"`:
        4D tensor with shape
        `(batch_size, pooled_height, pooled_width, channels)`.
    - If `data_format="channels_first"`:
        4D tensor with shape
        `(batch_size, channels, pooled_height, pooled_width)`.

    Examples:

    `strides=(1, 1)` and `padding="valid"`:

    >>> x = np.array([[1., 2., 3.],
    ...               [4., 5., 6.],
    ...               [7., 8., 9.]])
    >>> x = np.reshape(x, [1, 3, 3, 1])
    >>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding="valid")
    >>> max_pool_2d(x)

    `strides=(2, 2)` and `padding="valid"`:

    >>> x = np.array([[1., 2., 3., 4.],
    ...               [5., 6., 7., 8.],
    ...               [9., 10., 11., 12.]])
    >>> x = np.reshape(x, [1, 3, 4, 1])
    >>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(2, 2), padding="valid")
    >>> max_pool_2d(x)

    `stride=(1, 1)` and `padding="same"`:

    >>> x = np.array([[1., 2., 3.],
    ...               [4., 5., 6.],
    ...               [7., 8., 9.]])
    >>> x = np.reshape(x, [1, 3, 3, 1])
    >>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
    ...    strides=(1, 1), padding="same")
    >>> max_pool_2d(x)
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        name=None,
        **kwargs
    ):
        super().__init__(
            pool_size,
            strides,
            pool_dimensions=2,
            pool_mode="max",
            padding=padding,
            data_format=data_format,
            name=name,
            **kwargs,
        )
