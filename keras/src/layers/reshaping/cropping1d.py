from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation


@keras_export("keras.layers.Cropping1D")
class Cropping1D(Layer):
    """Cropping layer for 1D input (e.g. temporal sequence).

    It crops along the time dimension (axis 1).

    Example:

    >>> input_shape = (2, 3, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1]
      [ 2  3]
      [ 4  5]]
     [[ 6  7]
      [ 8  9]
      [10 11]]]
    >>> y = keras.layers.Cropping1D(cropping=1)(x)
    >>> y
    [[[2 3]]
     [[8 9]]]

    Args:
        cropping: Int, or tuple of int (length 2), or dictionary.
            - If int: how many units should be trimmed off at the beginning and
              end of the cropping dimension (axis 1).
            - If tuple of 2 ints: how many units should be trimmed off at the
              beginning and end of the cropping dimension
              (`(left_crop, right_crop)`).

    Input shape:
        3D tensor with shape `(batch_size, axis_to_crop, features)`

    Output shape:
        3D tensor with shape `(batch_size, cropped_axis, features)`
    """

    def __init__(self, cropping=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.cropping = argument_validation.standardize_tuple(
            cropping, 2, "cropping", allow_zero=True
        )
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            length = input_shape[1] - self.cropping[0] - self.cropping[1]
            if length <= 0:
                raise ValueError(
                    "`cropping` parameter of `Cropping1D` layer must be "
                    "smaller than the input length. Received: input_shape="
                    f"{input_shape}, cropping={self.cropping}"
                )
        else:
            length = None
        return (input_shape[0], length, input_shape[2])

    def call(self, inputs):
        if (
            inputs.shape[1] is not None
            and sum(self.cropping) >= inputs.shape[1]
        ):
            raise ValueError(
                "`cropping` parameter of `Cropping1D` layer must be "
                "smaller than the input length. Received: inputs.shape="
                f"{inputs.shape}, cropping={self.cropping}"
            )
        if self.cropping[1] == 0:
            return inputs[:, self.cropping[0] :, :]
        else:
            return inputs[:, self.cropping[0] : -self.cropping[1], :]

    def get_config(self):
        config = {"cropping": self.cropping}
        base_config = super().get_config()
        return {**base_config, **config}
