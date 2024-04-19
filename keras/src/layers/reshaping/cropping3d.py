from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.Cropping3D")
class Cropping3D(Layer):
    """Cropping layer for 3D data (e.g. spatial or spatio-temporal).

    Example:

    >>> input_shape = (2, 28, 28, 10, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = keras.layers.Cropping3D(cropping=(2, 4, 2))(x)
    >>> y.shape
    (2, 24, 20, 6, 3)

    Args:
        cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
            - If int: the same symmetric cropping is applied to depth, height,
              and width.
            - If tuple of 3 ints: interpreted as three different symmetric
              cropping values for depth, height, and width:
              `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
            - If tuple of 3 tuples of 2 ints: interpreted as
              `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
              right_dim2_crop), (left_dim3_crop, right_dim3_crop))`.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            When unspecified, uses `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json` (if exists). Defaults to
            `"channels_last"`.

    Input shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_axis_to_crop, second_axis_to_crop,
          third_axis_to_crop, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, first_axis_to_crop, second_axis_to_crop,
          third_axis_to_crop)`

    Output shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
          `(batch_size, first_cropped_axis, second_cropped_axis,
          third_cropped_axis, channels)`
        - If `data_format` is `"channels_first"`:
          `(batch_size, channels, first_cropped_axis, second_cropped_axis,
          third_cropped_axis)`
    """

    def __init__(
        self, cropping=((1, 1), (1, 1), (1, 1)), data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = backend.standardize_data_format(data_format)
        if isinstance(cropping, int):
            if cropping < 0:
                raise ValueError(
                    "`cropping` cannot be negative. "
                    f"Received: cropping={cropping}."
                )
            self.cropping = (
                (cropping, cropping),
                (cropping, cropping),
                (cropping, cropping),
            )
        elif hasattr(cropping, "__len__"):
            if len(cropping) != 3:
                raise ValueError(
                    f"`cropping` should have 3 elements. Received: {cropping}."
                )
            dim1_cropping = argument_validation.standardize_tuple(
                cropping[0], 2, "1st entry of cropping", allow_zero=True
            )
            dim2_cropping = argument_validation.standardize_tuple(
                cropping[1], 2, "2nd entry of cropping", allow_zero=True
            )
            dim3_cropping = argument_validation.standardize_tuple(
                cropping[2], 2, "3rd entry of cropping", allow_zero=True
            )
            self.cropping = (dim1_cropping, dim2_cropping, dim3_cropping)
        else:
            raise ValueError(
                "`cropping` should be either an int, a tuple of 3 ints "
                "(symmetric_dim1_crop, symmetric_dim2_crop, "
                "symmetric_dim3_crop), "
                "or a tuple of 3 tuples of 2 ints "
                "((left_dim1_crop, right_dim1_crop),"
                " (left_dim2_crop, right_dim2_crop),"
                " (left_dim3_crop, right_dim2_crop)). "
                f"Received: {cropping}."
            )
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            spatial_dims = list(input_shape[2:5])
        else:
            spatial_dims = list(input_shape[1:4])

        for index in range(0, 3):
            if spatial_dims[index] is None:
                continue
            spatial_dims[index] -= sum(self.cropping[index])
            if spatial_dims[index] <= 0:
                raise ValueError(
                    "Values in `cropping` argument should be smaller than the "
                    "corresponding spatial dimension of the input. Received: "
                    f"input_shape={input_shape}, cropping={self.cropping}"
                )

        if self.data_format == "channels_first":
            return (input_shape[0], input_shape[1], *spatial_dims)
        else:
            return (input_shape[0], *spatial_dims, input_shape[4])

    def call(self, inputs):
        if self.data_format == "channels_first":
            spatial_dims = list(inputs.shape[2:5])
        else:
            spatial_dims = list(inputs.shape[1:4])

        for index in range(0, 3):
            if spatial_dims[index] is None:
                continue
            spatial_dims[index] -= sum(self.cropping[index])
            if spatial_dims[index] <= 0:
                raise ValueError(
                    "Values in `cropping` argument should be smaller than the "
                    "corresponding spatial dimension of the input. Received: "
                    f"inputs.shape={inputs.shape}, cropping={self.cropping}"
                )

        if self.data_format == "channels_first":
            if (
                self.cropping[0][1]
                == self.cropping[1][1]
                == self.cropping[2][1]
                == 0
            ):
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                ]
            elif self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                ]
            elif self.cropping[1][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                ]
            elif self.cropping[0][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] : -self.cropping[2][1],
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                ]
            elif self.cropping[2][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                ]
            return inputs[
                :,
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
                self.cropping[2][0] : -self.cropping[2][1],
            ]
        else:
            if (
                self.cropping[0][1]
                == self.cropping[1][1]
                == self.cropping[2][1]
                == 0
            ):
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                    :,
                ]
            elif self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                    :,
                ]
            elif self.cropping[1][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                    :,
                ]
            elif self.cropping[0][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                    :,
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] : -self.cropping[2][1],
                    :,
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                    :,
                ]
            elif self.cropping[2][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                    :,
                ]
            return inputs[
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
                self.cropping[2][0] : -self.cropping[2][1],
                :,
            ]

    def get_config(self):
        config = {"cropping": self.cropping, "data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}
