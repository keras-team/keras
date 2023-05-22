from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.ZeroPadding1D")
class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

    Examples:

    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> x
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = keras_core.layers.ZeroPadding1D(padding=2)(x)
    >>> y
    [[[ 0  0  0]
      [ 0  0  0]
      [ 0  1  2]
      [ 3  4  5]
      [ 0  0  0]
      [ 0  0  0]]
     [[ 0  0  0]
      [ 0  0  0]
      [ 6  7  8]
      [ 9 10 11]
      [ 0  0  0]
      [ 0  0  0]]]

    Args:
        padding: Int, or tuple of int (length 2), or dictionary.
            - If int: how many zeros to add at the beginning and end of
              the padding dimension (axis 1).
            - If tuple of 2 ints: how many zeros to add at the beginning and the
              end of the padding dimension (`(left_pad, right_pad)`).

    Input shape:
        3D tensor with shape `(batch_size, axis_to_pad, features)`

    Output shape:
        3D tensor with shape `(batch_size, padded_axis, features)`
    """

    def __init__(self, padding=1, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if input_shape[1] is not None:
            input_shape[1] += self.padding[0] + self.padding[1]
        return tuple(output_shape)

    def call(self, inputs):
        all_dims_padding = ((0, 0), self.padding, (0, 0))
        return ops.pad(inputs, all_dims_padding)

    def get_config(self):
        config = {"padding": self.padding}
        base_config = super().get_config()
        return {**base_config, **config}
