from keras.src import activations
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.Activation")
class Activation(Layer):
    """Applies an activation function to an output.

    Args:
        activation: Activation function. It could be a callable, or the name of
            an activation from the `keras.activations` namespace.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:

    >>> layer = keras.layers.Activation('relu')
    >>> layer(np.array([-3.0, -1.0, 0.0, 2.0]))
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = keras.layers.Activation(keras.activations.relu)
    >>> layer(np.array([-3.0, -1.0, 0.0, 2.0]))
    [0.0, 0.0, 0.0, 2.0]
    """

    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

        self._build_at_init()

    def call(self, inputs):
        return self.activation(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"activation": activations.serialize(self.activation)}
        base_config = super().get_config()
        return {**base_config, **config}
