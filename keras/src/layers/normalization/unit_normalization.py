from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.UnitNormalization")
class UnitNormalization(Layer):
    """Unit normalization layer.

    Normalize a batch of inputs so that each input in the batch has a L2 norm
    equal to 1 (across the axes specified in `axis`).

    Example:

    >>> data = np.arange(6).reshape(2, 3)
    >>> normalized_data = keras.layers.UnitNormalization()(data)
    >>> np.sum(normalized_data[0, :] ** 2)
    1.0

    Args:
        axis: Integer or list/tuple. The axis or axes to normalize across.
            Typically, this is the features axis or axes. The left-out axes are
            typically the batch axis or axes. `-1` is the last dimension
            in the input. Defaults to `-1`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Invalid value for `axis` argument: "
                "expected an int or a list/tuple of ints. "
                f"Received: axis={axis}"
            )
        self.supports_masking = True
        self.built = True

    def call(self, inputs):
        return ops.normalize(inputs, axis=self.axis, order=2, epsilon=1e-12)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape
            Shape tuple (tuple of integers) or list of shape tuples (one per
            output tensor of the layer). Shape tuples can include None for free
            dimensions, instead of an integer.

        Returns
        -------
        output_shape
            Shape of the output of Unit Normalization Layer for
            an input of given shape.

        Raises
        ------
        ValueError
            If an axis is out of bounds for the input shape.
        TypeError
            If the input shape is not a tuple or a list of tuples.
        """
        if isinstance(input_shape, (tuple, list)):
            input_shape = input_shape
        else:
            raise TypeError(
                "Invalid input shape type: expected tuple or list. "
                f"Received: {type(input_shape)}"
            )

        for axis in self.axis:
            if axis >= len(input_shape) or axis < -len(input_shape):
                raise ValueError(
                    f"Axis {axis} is out of bounds for "
                    "input shape {input_shape}. "
                    "Ensure axis is within the range of input dimensions."
                )

        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
