from keras_core import activations
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.ELU")
class ELU(Layer):
    """Applies an Exponential Linear Unit function to an output.

    Formula:
    ```python
    f(x) = (exp(x) - 1.) for x < 0
    f(x) = x for x >= 0
    ```

    Args:
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return activations.elu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
