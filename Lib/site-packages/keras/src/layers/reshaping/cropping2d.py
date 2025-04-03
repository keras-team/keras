from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.Cropping2D")
class Cropping2D(Layer):
    """Cropping layer for 2D input (e.g. picture).

    It crops along spatial dimensions, i.e. height and width.

    Example:

    >>> input_shape = (2, 28, 28, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
    >>> y.shape
    (2, 24, 20, 3)

    Args:
        cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric cropping is applied to height and
              width.
            - If tuple of 2 ints: interpreted as two different symmetric
              cropping values for height and width:
              `(symmetric_height_crop, symmetric_width_crop)`.
            - If tuple of 2 tuples of 2 ints: interpreted as
              `((top_crop, bottom_crop), (left_crop, right_crop))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch_size, channels, height, width)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, height, width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, cropped_height, cropped_width, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, cropped_height, cropped_width)`
    """

    def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        if isinstance(cropping, int):
            if cropping < 0:
                raise ValueError(
                    "`cropping` cannot be negative. "
                    f"Received: cropping={cropping}."
                )
            self.cropping = ((cropping, cropping), (cropping, cropping))
        elif hasattr(cropping, "__len__"):
            if len(cropping) != 2:
                raise ValueError(
                    "`cropping` should have two elements. "
                    f"Received: cropping={cropping}."
                )
            height_cropping = argument_validation.standardize_tuple(
                cropping[0], 2, "1st entry of cropping", allow_zero=True
            )
            width_cropping = argument_validation.standardize_tuple(
                cropping[1], 2, "2nd entry of cropping", allow_zero=True
            )
            self.cropping = (height_cropping, width_cropping)
        else:
            raise ValueError(
                "`cropping` should be either an int, a tuple of 2 ints "
                "(symmetric_height_crop, symmetric_width_crop), "
                "or a tuple of 2 tuples of 2 ints "
                "((top_crop, bottom_crop), (left_crop, right_crop)). "
                f"Received: cropping={cropping}."
            )
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            if (
                input_shape[2] is not None
                and sum(self.cropping[0]) >= input_shape[2]
            ) or (
                input_shape[3] is not None
                and sum(self.cropping[1]) >= input_shape[3]
            ):
                raise ValueError(
                    "Values in `cropping` argument should be smaller than the "
                    "corresponding spatial dimension of the input. Received: "
                    f"input_shape={input_shape}, cropping={self.cropping}"
                )
            return (
                input_shape[0],
                input_shape[1],
                (
                    input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
                    if input_shape[2] is not None
                    else None
                ),
                (
                    input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
                    if input_shape[3] is not None
                    else None
                ),
            )
        else:
            if (
                input_shape[1] is not None
                and sum(self.cropping[0]) >= input_shape[1]
            ) or (
                input_shape[2] is not None
                and sum(self.cropping[1]) >= input_shape[2]
            ):
                raise ValueError(
                    "Values in `cropping` argument should be smaller than the "
                    "corresponding spatial dimension of the input. Received: "
                    f"input_shape={input_shape}, cropping={self.cropping}"
                )
            return (
                input_shape[0],
                (
                    input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
                    if input_shape[1] is not None
                    else None
                ),
                (
                    input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
                    if input_shape[2] is not None
                    else None
                ),
                input_shape[3],
            )

    def call(self, inputs):
        if self.data_format == "channels_first":
            if (
                inputs.shape[2] is not None
                and sum(self.cropping[0]) >= inputs.shape[2]
            ) or (
                inputs.shape[3] is not None
                and sum(self.cropping[1]) >= inputs.shape[3]
            ):
                raise ValueError(
                    "Values in `cropping` argument should be smaller than the "
                    "corresponding spatial dimension of the input. Received: "
                    f"inputs.shape={inputs.shape}, cropping={self.cropping}"
                )
            if self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :, :, self.cropping[0][0] :, self.cropping[1][0] :
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                ]
            return inputs[
                :,
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
            ]
        else:
            if (
                inputs.shape[1] is not None
                and sum(self.cropping[0]) >= inputs.shape[1]
            ) or (
                inputs.shape[2] is not None
                and sum(self.cropping[1]) >= inputs.shape[2]
            ):
                raise ValueError(
                    "Values in `cropping` argument should be smaller than the "
                    "corresponding spatial dimension of the input. Received: "
                    f"inputs.shape={inputs.shape}, cropping={self.cropping}"
                )
            if self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :, self.cropping[0][0] :, self.cropping[1][0] :, :
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    :,
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    :,
                ]
            return inputs[
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
                :,
            ]

    def get_config(self):
        config = {"cropping": self.cropping, "data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}
