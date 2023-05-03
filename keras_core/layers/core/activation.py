from keras_core import activations
from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Activation")
class Activation(Layer):
    """Applies an activation function to an input.

    Args:
        activation: Activation function, such as
            `keras_core.operations.relu`, or string name of
            built-in activation function, such as `"relu"`.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.

    Examples:

    >>> layer = keras_core.layers.Activation('relu')
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> output
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = keras_core.layers.Activation(tf.nn.relu)
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> output
    [0.0, 0.0, 0.0, 2.0]
    """

    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"activation": activations.serialize(self.activation)}
        base_config = super().get_config()
        return {**base_config, **config}
