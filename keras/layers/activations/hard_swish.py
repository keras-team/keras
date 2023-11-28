from keras import activations
from keras.api_export import keras_export
from keras.layers.layer import Layer


@keras_export("keras.layers.HardSwish")
class HardSwish(Layer):
    """Applies a Hard Swish function to an output.

    Formula:
    ``` python
    f(x) = 0 for x < -3
    f(x) = x for x > 3
    f(x) = x * (x + 3) / 6 for -3 <= x <= 3
    ```

    Example:
    ``` python
    hard_swish_layer = keras.layers.activations.HardSwish()
    input = np.array([-10, -2.5, 0.0, 2.5, 10])
    result = hard_swish_layer(input)
    # result = [-0.0, -0.2083,  0.0,  2.2917, 10.0]
    ```

    Args:
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return activations.hard_swish(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
