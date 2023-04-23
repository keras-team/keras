from keras_core import activations
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Activation")
class Activation(Layer):
    """Applies an activation function to an output.

    Args:
        activation: Activation function. It could be
            a callable, or the name of an activation
            from the `keras_core.activations` namespace.

    Example:

    >>> layer = keras_core.layers.Activation('relu')
    >>> layer([-3.0, -1.0, 0.0, 2.0])
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = keras_core.layers.Activation(keras_core.activations.relu)
    >>> layer([-3.0, -1.0, 0.0, 2.0])
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
