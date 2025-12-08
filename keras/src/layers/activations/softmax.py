from keras.src import activations
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


def _large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if backend.standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


@keras_export("keras.layers.Softmax")
class Softmax(Layer):
    """Softmax activation layer.

    Formula:
    ``` python
    exp_x = exp(x - max(x))
    f(x) = exp_x / sum(exp_x)
    ```

    Example:
    >>> softmax_layer = keras.layers.Softmax()
    >>> input = np.array([1.0, 2.0, 1.0])
    >>> result = softmax_layer(input)
    >>> result
    [0.21194157, 0.5761169, 0.21194157]


    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Call arguments:
        inputs: The inputs (logits) to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.

    Returns:
        Softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

        self._build_at_init()

    def call(self, inputs, mask=None):
        if mask is not None:
            # We keep the positions where the mask is True or > 0.5, and set the
            # other (masked) positions to -1e.9.
            if backend.standardize_dtype(mask.dtype) != "bool":
                mask = backend.numpy.greater(
                    mask, backend.cast(0.5, dtype=mask.dtype)
                )
            inputs = backend.numpy.where(
                mask, inputs, _large_negative_number(inputs.dtype)
            )
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                outputs = backend.numpy.exp(
                    inputs
                    - backend.math.logsumexp(
                        inputs, axis=self.axis, keepdims=True
                    )
                )
            else:
                outputs = activations.softmax(inputs, axis=self.axis[0])
        else:
            outputs = activations.softmax(inputs, axis=self.axis)

        if mask is not None:
            # Apply the mask to the softmax output to ensure that masked
            # values are set to 0 in case the entire axis is masked.
            outputs = backend.numpy.multiply(
                outputs, backend.cast(mask, outputs.dtype)
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
