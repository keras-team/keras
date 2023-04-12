from keras_core import activations
from keras_core import initializers
from keras_core import operations as ops
from keras_core.layers.layer import Layer


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, name=None):
        # TODO: support all other arguments.
        super().__init__(name=name)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=initializers.GlorotUniform(),
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=initializers.Zeros(),
            )

    def call(self, inputs):
        x = ops.matmul(inputs, self.kernel)
        if self.use_bias:
            x = x + self.bias
        return self.activation(x)
