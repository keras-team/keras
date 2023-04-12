from keras_core import backend
from keras_core.layers.layer import Layer
from keras_core.backend import KerasTensor
from keras_core.operations.function import Function
from keras_core import initializers
from keras_core.operations import numpy as knp


class MiniDense(Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        w_shape = (input_dim, self.units)
        w_value = initializers.GlorotUniform()(w_shape)
        self.w = backend.Variable(w_value)

        b_shape = (self.units,)
        b_value = initializers.Zeros()(b_shape)
        self.b = backend.Variable(b_value)

    def call(self, inputs):
        return knp.matmul(inputs, self.w) + self.b


class MiniDropout(Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        self.seed_generator = backend.random.RandomSeedGenerator(1337)

    def call(self, inputs):
        return backend.random.dropout(inputs, self.rate, seed=self.seed_generator)


class MiniBatchNorm(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.epsilon = 1e-5
        self.momentum = 0.99

    def build(self, input_shape):
        shape = (input_shape[-1],)
        self.mean = backend.Variable(initializers.Zeros()(shape), trainable=False)
        self.variance = backend.Variable(
            initializers.GlorotUniform()(shape), trainable=False
        )
        self.beta = backend.Variable(initializers.Zeros()(shape))
        self.gamma = backend.Variable(initializers.Ones()(shape))

    def call(self, inputs, training=False):
        if training:
            mean = knp.mean(inputs, axis=(0,))  # TODO: extend to rank 3+
            variance = knp.var(inputs, axis=(0,))
            outputs = (inputs - mean) / (variance + self.epsilon)
            self.variance.assign(
                self.variance * self.momentum + variance * (1.0 - self.momentum)
            )
            self.mean.assign(self.mean * self.momentum + mean * (1.0 - self.momentum))
        else:
            outputs = (inputs - self.mean) / (self.variance + self.epsilon)
        outputs *= self.gamma
        outputs += self.beta
        return outputs


# Eager call
layer = MiniDense(5)
x = knp.zeros((3, 4))
y = layer(x)
y = MiniBatchNorm()(y, training=True)
y = MiniDropout(0.5)(y)
assert y.shape == (3, 5)
assert layer.built
print(layer.variables)
assert len(layer.variables) == 2

# Symbolic call
x = KerasTensor((3, 4))
layer = MiniDense(5)
y = layer(x)
y = MiniBatchNorm()(y, training=True)
y = MiniDropout(0.5)(y)
assert y.shape == (3, 5)
assert layer.built
assert len(layer.variables) == 2

# Symbolic graph building
x = KerasTensor((3, 4))
y = MiniDense(5)(x)
y = MiniBatchNorm()(y, training=True)
y = MiniDropout(0.5)(y)
fn = Function(inputs=x, outputs=y)
y_val = fn(knp.ones((3, 4)))
assert y_val.shape == (3, 5)

print(y_val)
