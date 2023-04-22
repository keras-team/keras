from keras_core import activations
from keras_core import constraints
from keras_core import initializers
from keras_core import operations as ops
from keras_core import regularizers
from keras_core.layers.layer import Layer


class Dense(Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name=None,
    ):
        super().__init__(name=name)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if activity_regularizer:
            # TODO
            raise ValueError("activity_regularizer not yet supported.")

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
            )

    def call(self, inputs):
        x = ops.matmul(inputs, self.kernel)
        if self.use_bias:
            x = x + self.bias
        return self.activation(x)

    def get_config(self):
        base_config = super().get_config()
        # TODO
        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
        }
        return {**base_config, **config}
