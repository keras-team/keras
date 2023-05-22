from keras_core.api_export import keras_core_export
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Cropping1D")
class Cropping1D(Layer):
    """Cropping layer for 1D input (e.g. temporal sequence).

    It crops along the time dimension (axis 1).

    Examples:

    >>> input_shape = (2, 3, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1]
      [ 2  3]
      [ 4  5]]
     [[ 6  7]
      [ 8  9]
      [10 11]]]
    >>> y = keras_core.layers.Cropping1D(cropping=1)(x)
    >>> y
    [[[2 3]]
     [[8 9]]]

    Args:
        cropping: Integer or tuple of integers of length 2.
            How many units should be trimmed off at the beginning and end of
            the cropping dimension (axis 1).
            If a single int is provided, the same value will be used for both.

    Input shape:
        3D tensor with shape `(batch_size, axis_to_crop, features)`

    Output shape:
        3D tensor with shape `(batch_size, cropped_axis, features)`
    """

    def __init__(self, cropping=(1, 1), name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        if isinstance(cropping, int):
            cropping = (cropping, cropping)
        self.cropping = cropping
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            length = input_shape[1] - self.cropping[0] - self.cropping[1]
            if length <= 0:
                raise ValueError(
                    "`cropping` parameter of `Cropping1D` layer must be "
                    "greater than the input length. Received: input_shape="
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
                "greater than the input length. Received: inputs.shape="
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
